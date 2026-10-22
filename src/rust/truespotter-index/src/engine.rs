//! The embedded search engine: Tantivy (BM25) for the analyzed text
//! clauses + an in-process record index (filters, keyword terms, aggs,
//! sort) + a brute-force kNN vector store (mmap'd f32 matrix on /data,
//! in-RAM fallback — spec rust-embedded-rag, spec-approved fallback:
//! LanceDB is skipped because it does not build on musl/scratch).
//!
//! Every clause is evaluated against the whole corpus (29k docs — the
//! record-filter scans are microseconds, tantivy term queries single
//! milliseconds), then combined in Rust: bool must/should/filter/
//! minimum_should_match, best_fields multi_match, function_score
//! (native trust-score script + seeded random_score), sort (with
//! missing/_last), search_after cursor, terms/filter/stats aggregations.
//! Ranking is deterministic: the implicit final tie-break is `slug` asc.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, Map, Value};
use tantivy::collector::TopDocs;
use tantivy::query::{PhraseQuery, TermQuery};
use tantivy::schema::{
    Field, IndexRecordOption, Schema, TantivyDocument, TextFieldIndexing, TextOptions, FAST,
};
use tantivy::tokenizer::{Token, TokenStream, Tokenizer};
use tantivy::{DocAddress, Index, IndexReader, ReloadPolicy, Term};

use crate::dsn::{Msm, Operator, Order, Query, SearchRequest, SortSpec, SortValue};

/// One corpus document — the original JSON (returned verbatim as
/// `_source` minus `_source.excludes`) plus the fields the DSL needs.
pub struct EngineDoc {
    pub slug: String,
    pub doc: Value,
    pub embedding: Option<Vec<f32>>,
}

/// Filter/sort/agg metadata extracted once at rebuild time.
#[derive(Default)]
struct DocRec {
    slug: String,
    title: String,
    raw: String,
    status: Option<String>,
    available: Option<bool>,
    price: Option<f64>,
    currency: Option<String>,
    /// the `type` doc field (the public "category")
    category: Option<String>,
    /// the `category` doc field (separate from `type`)
    category2: Option<String>,
    category_path: Option<String>,
    vendor: Option<String>,
    brand: Option<String>,
    tags: Vec<String>,
    author: Option<String>,
    genre: Option<String>,
    trust_score: Option<f64>,
    trust_count: Option<f64>,
    /// epoch millis (doc ISO strings or epoch millis accepted)
    created_at: Option<i64>,
    /// the `id` doc field (external ref, e.g. "gid://shopify/…")
    external_id: Option<String>,
    embedding: Option<Vec<f32>>,
    /// tombstone flag — deleted slots keep the docno mapping stable until
    /// the next full rebuild compacts the corpus.
    deleted: bool,
}

const TEXT_FIELDS: [&str; 7] = [
    "title",
    "vendor",
    "brand",
    "description",
    "tags",
    "specs_text",
    "reviews_text",
];

struct Fields {
    text: HashMap<&'static str, Field>,
    /// exact-slug field (raw tokenizer) — tantivy delete_term handle for
    /// incremental writes
    slug: Field,
    docno: Field,
}

/// Lowercase + Latin-diacritics fold (the .NET analyzer semantics the
/// OpenSearch index used — see `fold_char` in truespotter-catalog).
fn fold_char(c: char) -> Vec<char> {
    let s = match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => "a",
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => "c",
        'ď' | 'đ' => "d",
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ę' | 'ě' => "e",
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => "g",
        'ĥ' | 'ħ' => "h",
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => "i",
        'ĵ' => "j",
        'ķ' => "k",
        'ĺ' | 'ļ' | 'ľ' | 'ł' => "l",
        'ñ' | 'ń' | 'ņ' | 'ň' => "n",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => "o",
        'ŕ' | 'ŗ' | 'ř' => "r",
        'ś' | 'ŝ' | 'ş' | 'š' => "s",
        'ţ' | 'ť' => "t",
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => "u",
        'ŵ' => "w",
        'ý' | 'ÿ' => "y",
        'ź' | 'ż' | 'ž' => "z",
        _ => return vec![c],
    };
    s.chars().collect()
}

pub fn fold_str(s: &str) -> String {
    s.to_lowercase().chars().flat_map(fold_char).collect()
}

/// Tokens of folded text — identical tokenization to the registered
/// `ts` tokenizer (alphanumeric runs), so in-memory term sets match the
/// tantivy dictionaries exactly.
fn folded_tokens(s: &str) -> Vec<String> {
    let folded = fold_str(s);
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in folded.chars() {
        if c.is_alphanumeric() {
            cur.push(c);
        } else if !cur.is_empty() {
            tokens.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

// ---------------------------------------------------------------------------
// Custom tantivy tokenizer ("ts") — SimpleTokenizer + LowerCaser +
// Polish-diacritics fold. Implemented against the trait directly so the
// analyzer semantics are fully under our control (no API guessing).
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct TsTokenizer;

impl Tokenizer for TsTokenizer {
    type TokenStream<'a> = TsTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> TsTokenStream {
        TsTokenStream {
            tokens: folded_tokens(text),
            pos: 0,
            token: Token::default(),
        }
    }
}

struct TsTokenStream {
    tokens: Vec<String>,
    pos: usize,
    token: Token,
}

impl TokenStream for TsTokenStream {
    fn advance(&mut self) -> bool {
        if self.pos >= self.tokens.len() {
            return false;
        }
        let t = &self.tokens[self.pos];
        self.token = Token {
            position: self.pos,
            offset_from: 0,
            offset_to: 0,
            text: t.clone(),
            position_length: 1,
        };
        self.pos += 1;
        true
    }

    fn token(&self) -> &Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

// ---------------------------------------------------------------------------
// Vector store — brute-force kNN over a raw LE f32 matrix. Persisted to
// /data/vectors.f32 and mmap'd read-only (RAM budget); in-RAM fallback
// when /data is unavailable. 29k × 4096-dim ≈ 10–50 ms per query
// (spec-approved LanceDB fallback).
// ---------------------------------------------------------------------------

enum VectorStore {
    Empty,
    Ram {
        data: Vec<f32>,
        n: usize,
        dim: usize,
        has_vec: Vec<bool>,
    },
    Mmap {
        map: memmap2::Mmap,
        n: usize,
        dim: usize,
        has_vec: Vec<bool>,
    },
}

impl VectorStore {
    /// Build from the records; persist+mmmap into `dir` when possible.
    fn build(records: &[DocRec], dir: Option<&std::path::Path>) -> Self {
        let dim = records
            .iter()
            .find_map(|r| r.embedding.as_ref().map(|e| e.len()))
            .unwrap_or(0);
        if dim == 0 {
            return VectorStore::Empty;
        }
        let n = records.len();
        let has_vec: Vec<bool> = records
            .iter()
            .map(|r| r.embedding.as_ref().is_some_and(|e| e.len() == dim))
            .collect();
        let mut data = vec![0f32; n * dim];
        for (i, r) in records.iter().enumerate() {
            if let Some(e) = r.embedding.as_ref().filter(|e| e.len() == dim) {
                data[i * dim..(i + 1) * dim].copy_from_slice(e);
            }
        }
        if let Some(dir) = dir {
            let path = dir.join("vectors.f32");
            let mut buf: Vec<u8> = Vec::with_capacity(8 + data.len() * 4);
            buf.extend_from_slice(&(dim as u32).to_le_bytes());
            buf.extend_from_slice(&(n as u32).to_le_bytes());
            for f in &data {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            if fs::create_dir_all(dir).is_ok() && fs::write(&path, &buf).is_ok() {
                if let Ok(bytes) = fs::read(&path) {
                    if bytes.len() >= 8 {
                        let fdim = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
                        let fnn = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
                        if fdim == dim && fnn == n {
                            if let Ok(map) = unsafe { memmap2::Mmap::map(&path_as_file(&path)) } {
                                return VectorStore::Mmap {
                                    map,
                                    n,
                                    dim,
                                    has_vec,
                                };
                            }
                        }
                    }
                }
            }
            tracing::warn!("vector mmap failed — falling back to in-RAM vectors");
        }
        VectorStore::Ram {
            data,
            n,
            dim,
            has_vec,
        }
    }

    fn row(&self, i: usize) -> Option<&[f32]> {
        match self {
            VectorStore::Empty => None,
            VectorStore::Ram { data, dim, .. } => Some(&data[i * dim..(i + 1) * dim]),
            VectorStore::Mmap { map, dim, .. } => {
                let off = (i * dim) * 4 + 8;
                let bytes = &map[off..off + dim * 4];
                // Safety-free view: copy is avoided by transmute-free re-reading below.
                Some(vector_slice(bytes, *dim))
            }
        }
    }
}

/// Reinterpret a LE f32 byte slice without copying.
fn vector_slice(bytes: &[u8], dim: usize) -> &[f32] {
    // SAFETY-backed via memmap2's as_ptr; validated length by caller.
    assert_eq!(bytes.len(), dim * 4);
    let ptr = bytes.as_ptr() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, dim) }
}

fn path_as_file(path: &std::path::Path) -> fs::File {
    fs::File::open(path).expect("file just written")
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct Engine {
    /// interior mutability: readers take the read lock per query, the
    /// (serialized) write path takes the write lock — records/docnos stay
    /// consistent while the engine is shared behind an Arc.
    records: std::sync::RwLock<Vec<DocRec>>,
    reader: IndexReader,
    /// tantivy incremental writer (kept alive — deletes + re-adds commit
    /// immediately; the reader reloads synchronously after each commit).
    writer: std::sync::Mutex<tantivy::IndexWriter<TantivyDocument>>,
    fields: Fields,
    /// in-memory dictionaries per text field (prefix/fuzzy enumeration) —
    /// extended on every write so new tokens are findable without a rebuild
    terms: std::sync::RwLock<HashMap<&'static str, BTreeSet<String>>>,
    vectors: VectorStore,
    _tmp: Option<tempfile::TempDir>,
}

pub struct Hit {
    pub slug: String,
    pub score: Option<f64>,
    pub sort: Vec<Value>,
    pub source: Value,
}

pub struct SearchOutcome {
    pub total: usize,
    pub hits: Vec<Hit>,
    pub aggregations: Option<Value>,
    pub took_ms: u64,
}

#[derive(Clone)]
enum SortKey {
    Missing,
    Num(f64),
    Str(String),
}

fn get_f64(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|p| {
        p.as_f64()
            .or_else(|| p.as_i64().map(|i| i as f64))
            .or_else(|| p.as_str().and_then(|s| s.parse().ok()))
    })
}

/// Ad-hoc field lookup against the raw doc JSON (write-path fields the
/// .NET scans filter on that are not in the fast record set).
fn raw_value(rec: &DocRec, field: &str) -> Option<Value> {
    serde_json::from_str::<Value>(&rec.raw)
        .ok()
        .and_then(|v| v.get(field.trim_end_matches(".keyword")).cloned())
}

fn json_bool(b: bool) -> Value {
    serde_json::Value::Bool(b)
}

fn parse_created_at(v: &Value) -> Option<i64> {
    v.as_str()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&chrono::Utc).timestamp_millis())
        .or_else(|| v.as_i64())
}

fn extract_rec(slug: &str, doc: &Value) -> DocRec {
    DocRec {
        slug: slug.to_string(),
        title: doc
            .get("title")
            .and_then(|t| t.as_str())
            .unwrap_or_default()
            .to_string(),
        raw: doc.to_string(),
        status: doc.get("status").and_then(|s| s.as_str()).map(String::from),
        available: doc.get("available").and_then(|a| a.as_bool()),
        price: get_f64(doc, "price"),
        currency: doc
            .get("currency")
            .and_then(|c| c.as_str())
            .map(String::from),
        category: doc.get("type").and_then(|t| t.as_str()).map(String::from),
        category2: doc
            .get("category")
            .and_then(|c| c.as_str())
            .map(String::from),
        category_path: doc
            .get("category_path")
            .and_then(|c| c.as_str())
            .map(String::from),
        vendor: doc.get("vendor").and_then(|v| v.as_str()).map(String::from),
        brand: doc.get("brand").and_then(|b| b.as_str()).map(String::from),
        tags: doc
            .get("tags")
            .and_then(|t| t.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        author: doc.get("author").and_then(|a| a.as_str()).map(String::from),
        genre: doc.get("genre").and_then(|g| g.as_str()).map(String::from),
        trust_score: get_f64(doc, "trust_score"),
        trust_count: get_f64(doc, "trust_count"),
        created_at: doc.get("created_at").and_then(parse_created_at),
        external_id: doc.get("id").and_then(|i| i.as_str()).map(String::from),
        embedding: None,
        deleted: false,
    }
}

impl Engine {
    /// Build the engine from corpus docs (PG rows or synthetic test data).
    /// `data_dir` is used for the vector matrix persistence (mmap); the
    /// Tantivy index is always built fresh in a temp dir (29k docs →
    /// seconds) — no load/persist dance, `slug` is the id everywhere.
    pub fn build(docs: Vec<EngineDoc>, data_dir: Option<PathBuf>) -> Result<Self, String> {
        // scratch containers have no /tmp — tempfile::tempdir needs an env
        // TMPDIR that exists. Prefer the data dir, else create one under the
        // first writable base (./, /data, / — scratch has no writable /
        // without a dir baked into the image, so the Dockerfile pre-creates
        // /data; local runs get an OS temp dir).
        let tmp = tempfile::tempdir_in(
            data_dir
                .clone()
                .filter(|d| std::fs::create_dir_all(d).is_ok())
                .unwrap_or_else(std::env::temp_dir),
        )
        .or_else(|_| tempfile::tempdir())
        .map_err(|e| format!("tempdir: {e}"))?;
        let mut schema = Schema::builder();
        let text_opts = || {
            TextOptions::default().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("ts")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
        };
        let mut text = HashMap::new();
        for name in TEXT_FIELDS {
            text.insert(name, schema.add_text_field(name, text_opts()));
        }
        let slug_field = schema.add_text_field(
            "__slug",
            TextOptions::default().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_index_option(IndexRecordOption::Basic),
            ),
        );
        let docno = schema.add_u64_field("docno", FAST);
        let schema = schema.build();
        let index = Index::create_in_dir(tmp.path(), schema)
            .map_err(|e| format!("create tantivy index: {e}"))?;
        index.tokenizers().register("ts", TsTokenizer);

        let mut writer = index
            .writer::<TantivyDocument>(100_000_000)
            .map_err(|e| format!("tantivy writer: {e}"))?;

        let mut records: Vec<DocRec> = Vec::with_capacity(docs.len());
        let mut terms: HashMap<&'static str, BTreeSet<String>> = HashMap::new();
        for name in TEXT_FIELDS {
            terms.insert(name, BTreeSet::new());
        }
        for (i, d) in docs.into_iter().enumerate() {
            let mut rec = extract_rec(&d.slug, &d.doc);
            rec.embedding = d.embedding.clone();
            let mut tdoc = TantivyDocument::default();
            for name in TEXT_FIELDS {
                let texts = match name {
                    "tags" => rec.tags.to_vec(),
                    other => vec![d
                        .doc
                        .get(other)
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string()],
                };
                let field = text.get(name).copied().expect("field");
                for text_val in texts {
                    for token in folded_tokens(&text_val) {
                        terms.get_mut(name).expect("terms").insert(token);
                    }
                    if !text_val.trim().is_empty() {
                        tdoc.add_text(field, text_val);
                    }
                }
            }
            tdoc.add_text(slug_field, &d.slug);
            tdoc.add_u64(docno, i as u64);
            writer
                .add_document(tdoc)
                .map_err(|e| format!("tantivy add_document: {e}"))?;
            records.push(rec);
        }
        writer
            .commit()
            .map_err(|e| format!("tantivy commit: {e}"))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| format!("tantivy reader: {e}"))?;
        let _ = reader.reload();
        let vectors = VectorStore::build(&records, data_dir.as_deref());
        let fields = Fields {
            text,
            slug: slug_field,
            docno,
        };
        Ok(Engine {
            records: std::sync::RwLock::new(records),
            reader,
            writer: std::sync::Mutex::new(writer),
            fields,
            terms: std::sync::RwLock::new(terms),
            vectors,
            _tmp: Some(tmp),
        })
    }

    pub fn num_docs(&self) -> usize {
        self.recs().iter().filter(|r| !r.deleted).count()
    }

    fn recs(&self) -> std::sync::RwLockReadGuard<'_, Vec<DocRec>> {
        self.records.read().expect("records lock poisoned")
    }

    // -- record field accessors ---------------------------------------------

    fn str_field<'a>(rec: &'a DocRec, field: &str) -> Option<&'a str> {
        match field {
            "slug" | "slug.keyword" => Some(&rec.slug),
            "title" | "title.keyword" => Some(&rec.title),
            "vendor" | "vendor.keyword" => rec.vendor.as_deref(),
            "brand" | "brand.keyword" => rec.brand.as_deref(),
            "type" | "type.keyword" => rec.category.as_deref(),
            "category" | "category.keyword" => rec.category2.as_deref(),
            "category_path" | "category_path.keyword" => rec.category_path.as_deref(),
            "currency" | "currency.keyword" => rec.currency.as_deref(),
            "status" | "status.keyword" => rec.status.as_deref(),
            "author" | "author.keyword" => rec.author.as_deref(),
            "genre" | "genre.keyword" => rec.genre.as_deref(),
            "id" | "id.keyword" => rec.external_id.as_deref(),
            _ => None,
        }
    }

    fn num_field(rec: &DocRec, field: &str) -> Option<f64> {
        match field {
            "price" => rec.price,
            "trust_score" => rec.trust_score,
            "trust_count" => rec.trust_count,
            "created_at" => rec.created_at.map(|m| m as f64),
            "available" => rec.available.map(|b| if b { 1.0 } else { 0.0 }),
            // generic fallback: ad-hoc numeric fields the .NET scans use
            // (category_attempts, embedding_pending…) live only in raw JSON
            _ => raw_value(rec, field).and_then(|v| {
                v.as_f64()
                    .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
            }),
        }
    }

    /// Fast known string fields, falling back to the raw JSON for ad-hoc
    /// fields (exists/term/prefix on category, model_key, external_ids…).
    fn str_field_dyn(rec: &DocRec, field: &str) -> Option<String> {
        match field.trim_end_matches(".keyword") {
            "slug" => Some(rec.slug.clone()),
            "title" => Some(rec.title.clone()),
            "vendor" | "brand" | "type" | "category" | "category_path" | "currency" | "status"
            | "author" | "genre" | "id" => Self::str_field(rec, field).map(String::from),
            _ => raw_value(rec, field).and_then(|v| v.as_str().map(String::from)),
        }
    }

    fn exists_field(rec: &DocRec, field: &str) -> bool {
        match field {
            "tags" => !rec.tags.is_empty(),
            "embedding" => rec.embedding.is_some(),
            _ => match Self::str_field_dyn(rec, field) {
                Some(s) => !s.is_empty(),
                None => raw_value(rec, field)
                    .map(|v| !v.is_null())
                    .is_some_and(|v| v),
            },
        }
    }

    // -- query evaluation ----------------------------------------------------

    /// Evaluate a query → matching docno → score. Filter-position clauses
    /// get uniform scores (the caller ignores them).
    fn eval(&self, q: &Query) -> Result<BTreeMap<usize, f64>, String> {
        match q {
            Query::MatchAll => Ok({
                let records = self.recs();
                records
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| !r.deleted)
                    .map(|(i, _)| (i, 1.0))
                    .collect()
            }),
            Query::Term { field, value } => Ok(self.eval_term(field, value)),
            Query::Terms { field, values } => {
                let mut out = BTreeMap::new();
                for v in values {
                    for docno in self.eval_term(field, v).keys() {
                        out.insert(*docno, 1.0);
                    }
                }
                Ok(out)
            }
            Query::Prefix { field, value } => Ok({
                let records = self.recs();
                records
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| !r.deleted)
                    .filter(|(_, r)| {
                        Self::str_field_dyn(r, field).is_some_and(|s| s.starts_with(value.as_str()))
                    })
                    .map(|(i, _)| (i, 1.0))
                    .collect()
            }),
            Query::Range { field, gte, lte } => Ok({
                let records = self.recs();
                records
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| !r.deleted)
                    .filter(|(_, r)| {
                        let Some(v) = Self::num_field(r, field) else {
                            return false;
                        };
                        gte.is_none_or(|g| v >= g) && lte.is_none_or(|l| v <= l)
                    })
                    .map(|(i, _)| (i, 1.0))
                    .collect()
            }),
            Query::Exists { field } => {
                let records = self.recs();
                Ok(records
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| !r.deleted && Self::exists_field(r, field))
                    .map(|(i, _)| (i, 1.0))
                    .collect())
            }
            Query::Bool(b) => self.eval_bool(b),
            Query::Match {
                field,
                text,
                fuzziness,
                prefix_length,
                boost,
                operator,
            } => Ok(self
                .eval_text_match(field, text, fuzziness.is_some(), *prefix_length, *operator)
                .into_iter()
                .map(|(k, v)| (k, v * boost))
                .collect()),
            Query::MatchPhrase { field, text, boost } => Ok(self
                .eval_phrase(field, text, false)
                .into_iter()
                .map(|(k, v)| (k, v * boost))
                .collect()),
            Query::MatchPhrasePrefix { field, text, boost } => Ok(self
                .eval_phrase(field, text, true)
                .into_iter()
                .map(|(k, v)| (k, v * boost))
                .collect()),
            Query::MultiMatch {
                text,
                fields,
                boost,
                operator,
                minimum_should_match,
            } => self.eval_multi_match(text, fields, *boost, *operator, minimum_should_match),
            Query::Knn { field, vector, k } => self.eval_knn(field, vector, *k),
            Query::FunctionScore {
                query,
                script_score,
                random_seed,
                boost_mode_sum,
            } => {
                let inner = self.eval(query)?;
                let use_script = script_score.is_some();
                let mut out = BTreeMap::new();
                for (docno, score) in inner {
                    let mut s = score;
                    if use_script {
                        // The recognized trust script: trust_score * ln1p(trust_count)
                        // (Math.log1p(x) = ln(1+x)), floored at 0 — evaluated
                        // natively, boost_mode sum.
                        let records = self.recs();
                        let rec = &records[docno];
                        let trust = match (rec.trust_score, rec.trust_count) {
                            (Some(ts), Some(tc)) => (ts * (1.0f64 + tc).ln()).max(0.0),
                            (Some(ts), None) => (ts * (1.0f64 + 0.0).ln()).max(0.0),
                            _ => 0.0,
                        };
                        s = if *boost_mode_sum {
                            s + trust
                        } else {
                            s * trust
                        };
                    }
                    if let Some(seed) = random_seed {
                        let r = seeded_random(*seed, docno);
                        s = if *boost_mode_sum { s + r } else { s * r };
                    }
                    out.insert(docno, s);
                }
                Ok(out)
            }
        }
    }

    fn eval_term(&self, field: &str, value: &crate::dsn::TermValue) -> BTreeMap<usize, f64> {
        use crate::dsn::TermValue;
        let records = self.recs();
        let mut out = BTreeMap::new();
        for (i, r) in records.iter().enumerate() {
            if r.deleted {
                continue;
            }
            let matched = match value {
                TermValue::Str(s) => {
                    if field == "tags" || field == "tags.keyword" {
                        r.tags.iter().any(|t| t == s)
                    } else {
                        Self::str_field(r, field).is_some_and(|f| f == s)
                            || Self::str_field_dyn(r, field).as_deref() == Some(s)
                    }
                }
                TermValue::Bool(b) => {
                    r.available == Some(*b) || raw_value(r, field).as_ref() == Some(&json_bool(*b))
                }
                TermValue::Num(n) => {
                    Self::num_field(r, field).is_some_and(|v| (v - n).abs() < f64::EPSILON * 4.0)
                }
            };
            if matched {
                out.insert(i, 1.0);
            }
        }
        out
    }

    fn eval_bool(&self, b: &crate::dsn::BoolQuery) -> Result<BTreeMap<usize, f64>, String> {
        // must: intersect, summing scores.
        let mut must: Option<BTreeMap<usize, f64>> = None;
        for clause in &b.must {
            let r = self.eval(clause)?;
            must = Some(match must {
                None => r,
                Some(acc) => {
                    let mut out = BTreeMap::new();
                    for (doc, s) in acc {
                        if let Some(s2) = r.get(&doc) {
                            out.insert(doc, s + s2);
                        }
                    }
                    out
                }
            });
        }
        // filter: intersect as a set.
        let mut filter: Option<Vec<usize>> = None;
        for clause in &b.filter {
            let r = self.eval(clause)?;
            filter = Some(match filter {
                None => r.keys().copied().collect(),
                Some(acc) => acc.into_iter().filter(|d| r.contains_key(d)).collect(),
            });
        }
        // base doc set = must ∩ filter (or whichever is present, else all).
        let mut base: Vec<usize> = match (&must, &filter) {
            (Some(m), Some(f)) => {
                let fset: std::collections::HashSet<usize> = f.iter().copied().collect();
                m.keys().copied().filter(|d| fset.contains(d)).collect()
            }
            (Some(m), None) => m.keys().copied().collect(),
            (None, Some(f)) => f.clone(),
            (None, None) => {
                let records = self.recs();
                (0..records.len())
                    .filter(|i| !records[*i].deleted)
                    .collect()
            }
        };
        // must_not: subtract.
        for clause in &b.must_not {
            let r = self.eval(clause)?;
            base.retain(|d| !r.contains_key(d));
        }
        // should: count matched clauses per doc, sum scores.
        let should_maps: Vec<BTreeMap<usize, f64>> = {
            let mut maps = Vec::new();
            for clause in &b.should {
                maps.push(self.eval(clause)?);
            }
            maps
        };
        let mut out = BTreeMap::new();
        if b.should.is_empty() {
            for doc in base {
                let score = must
                    .as_ref()
                    .and_then(|m| m.get(&doc))
                    .copied()
                    .unwrap_or(1.0);
                out.insert(doc, score);
            }
            return Ok(out);
        }
        let required = b
            .minimum_should_match
            .as_ref()
            .map(|m| m.required(b.should.len()))
            .unwrap_or(if must.is_some() || filter.is_some() {
                0
            } else {
                1
            });
        for doc in base {
            let mut matched = 0usize;
            let mut score = must
                .as_ref()
                .and_then(|m| m.get(&doc))
                .copied()
                .unwrap_or(0.0);
            for map in &should_maps {
                if let Some(s) = map.get(&doc) {
                    matched += 1;
                    score += s;
                }
            }
            if matched >= required {
                if must.is_none() && filter.is_none() && score == 0.0 {
                    score = 1.0;
                }
                out.insert(doc, score);
            }
        }
        Ok(out)
    }

    // -- tantivy text evaluation ----------------------------------------------

    fn text_field(&self, field: &str) -> Option<Field> {
        // Only the analyzed text fields live in tantivy.
        let name = match field {
            "title" | "vendor" | "brand" | "description" | "tags" | "specs_text"
            | "reviews_text" => field,
            _ => return None,
        };
        self.fields.text.get(name).copied()
    }

    fn searcher(&self) -> tantivy::Searcher {
        self.reader.searcher()
    }

    /// Map tantivy DocAddresses to our record ordinals.
    fn docnos(
        &self,
        hits: &[(f32, DocAddress)],
        searcher: &tantivy::Searcher,
    ) -> Vec<(usize, f64)> {
        let mut out = Vec::with_capacity(hits.len());
        for (score, addr) in hits {
            let seg = searcher.segment_reader(addr.segment_ord);
            let docno = seg
                .fast_fields()
                .u64("docno")
                .ok()
                .and_then(|col| col.first(addr.doc_id))
                .unwrap_or_default() as usize;
            out.push((docno, *score as f64));
        }
        out
    }

    /// All (docno, bm25) pairs for one term.
    fn term_scores(&self, field: Field, term: &str) -> Vec<(usize, f64)> {
        let searcher = self.searcher();
        let limit = searcher.num_docs() as usize + 1;
        let query = TermQuery::new(
            Term::from_field_text(field, term),
            IndexRecordOption::WithFreqs,
        );
        let Ok(hits) = searcher.search(&query, &TopDocs::with_limit(limit)) else {
            return Vec::new();
        };
        self.docnos(&hits, &searcher)
    }

    /// Sum term scores across query tokens (operator or → all matching
    /// docs, and → docs matching every token).
    fn eval_text_match(
        &self,
        field: &str,
        text: &str,
        fuzzy: bool,
        prefix_length: Option<usize>,
        operator: Operator,
    ) -> BTreeMap<usize, f64> {
        let Some(field) = self.text_field(field) else {
            return BTreeMap::new(); // lenient: unmapped field → no hits
        };
        let tokens = folded_tokens(text);
        if tokens.is_empty() {
            return BTreeMap::new();
        }
        let mut per_token: Vec<BTreeMap<usize, f64>> = Vec::with_capacity(tokens.len());
        for token in &tokens {
            let mut m = BTreeMap::new();
            if fuzzy {
                // AUTO: len<=2 → 0, 3..=5 → 1, >5 → 2. Candidates share the
                // first `prefix_length` chars (Lucene semantics).
                let d = if token.len() <= 2 {
                    0
                } else if token.len() <= 5 {
                    1
                } else {
                    2
                };
                let prefix: String = match prefix_length {
                    Some(p) if p > 0 => token.chars().take(p).collect(),
                    _ => String::new(),
                };
                if d == 0 {
                    for (doc, s) in self.term_scores(field, token) {
                        m.insert(doc, s);
                    }
                } else {
                    let dict = {
                        let terms = self.terms.read().expect("terms lock");
                        terms[self.field_name(field)].clone()
                    };
                    for cand in dict.iter() {
                        if !prefix.is_empty() && !cand.starts_with(&prefix) {
                            continue;
                        }
                        if levenshtein_within(token, cand, d) {
                            for (doc, s) in self.term_scores(field, cand) {
                                *m.entry(doc).or_insert(0.0) += s;
                            }
                        }
                    }
                }
            } else {
                for (doc, s) in self.term_scores(field, token) {
                    m.insert(doc, s);
                }
            }
            per_token.push(m);
        }
        let mut out = BTreeMap::new();
        match operator {
            Operator::Or => {
                for m in &per_token {
                    for (doc, s) in m {
                        *out.entry(*doc).or_insert(0.0) += s;
                    }
                }
            }
            Operator::And => {
                let mut docs: Vec<usize> = per_token[0].keys().copied().collect();
                for m in &per_token[1..] {
                    docs.retain(|d| m.contains_key(d));
                }
                for doc in docs {
                    out.insert(
                        doc,
                        per_token
                            .iter()
                            .map(|m| m.get(&doc).copied().unwrap_or(0.0))
                            .sum(),
                    );
                }
            }
        }
        out
    }

    fn field_name(&self, field: Field) -> &'static str {
        for (name, f) in &self.fields.text {
            if *f == field {
                return name;
            }
        }
        "title"
    }

    /// match_phrase (and match_phrase_prefix when `prefix`).
    fn eval_phrase(&self, field: &str, text: &str, prefix: bool) -> BTreeMap<usize, f64> {
        let Some(tfield) = self.text_field(field) else {
            return BTreeMap::new();
        };
        let tokens = folded_tokens(text);
        if tokens.is_empty() {
            return BTreeMap::new();
        }
        let mut candidates: Vec<Vec<String>> = vec![tokens[..tokens.len() - 1].to_vec()];
        let last = tokens.last().expect("non-empty");
        if prefix {
            let dict = {
                let terms = self.terms.read().expect("terms lock");
                terms[self.field_name(tfield)].clone()
            };
            let mut tail: Vec<String> = dict
                .iter()
                .filter(|t| t.starts_with(last.as_str()))
                .take(50)
                .cloned()
                .collect();
            if tail.is_empty() {
                return BTreeMap::new();
            }
            candidates = tail
                .drain(..)
                .map(|t| {
                    let mut v = tokens[..tokens.len() - 1].to_vec();
                    v.push(t);
                    v
                })
                .collect();
        } else {
            candidates[0].push(last.clone());
        }
        let mut out = BTreeMap::new();
        let searcher = self.searcher();
        let limit = searcher.num_docs() as usize + 1;
        for toks in candidates {
            let hits = if toks.len() == 1 {
                searcher
                    .search(
                        &TermQuery::new(
                            Term::from_field_text(tfield, &toks[0]),
                            IndexRecordOption::WithFreqs,
                        ),
                        &TopDocs::with_limit(limit),
                    )
                    .unwrap_or_default()
            } else {
                searcher
                    .search(&phrase_query(&toks, tfield), &TopDocs::with_limit(limit))
                    .unwrap_or_default()
            };
            for (docno, score) in self.docnos(&hits, &searcher) {
                *out.entry(docno).or_insert(0.0) += score;
            }
        }
        out
    }

    fn eval_multi_match(
        &self,
        text: &str,
        fields: &[(String, f64)],
        boost: f64,
        operator: Operator,
        msm: &Option<Msm>,
    ) -> Result<BTreeMap<usize, f64>, String> {
        let tokens = folded_tokens(text);
        let mut best: BTreeMap<usize, f64> = BTreeMap::new();
        for (name, fboost) in fields {
            // Per-field score: sum over matched tokens (operator), with the
            // per-field minimum_should_match applied to the matched-token count.
            let per_field = self.eval_text_match_with_msm(name, &tokens, operator, msm);
            for (doc, s) in per_field {
                let v = s * fboost;
                let e = best.entry(doc).or_insert(0.0);
                if v > *e {
                    *e = v;
                }
            }
        }
        Ok(best.into_iter().map(|(k, v)| (k, v * boost)).collect())
    }

    fn eval_text_match_with_msm(
        &self,
        field: &str,
        tokens: &[String],
        operator: Operator,
        msm: &Option<Msm>,
    ) -> BTreeMap<usize, f64> {
        let Some(tfield) = self.text_field(field) else {
            return BTreeMap::new();
        };
        if tokens.is_empty() {
            return BTreeMap::new();
        }
        let required = msm
            .as_ref()
            .map(|m| m.required(tokens.len()))
            .unwrap_or(match operator {
                Operator::And => tokens.len(),
                Operator::Or => 0,
            });
        let mut matched_counts: BTreeMap<usize, usize> = BTreeMap::new();
        let mut sums: BTreeMap<usize, f64> = BTreeMap::new();
        for token in tokens {
            for (doc, s) in self.term_scores(tfield, token) {
                *matched_counts.entry(doc).or_insert(0) += 1;
                *sums.entry(doc).or_insert(0.0) += s;
            }
        }
        sums.retain(|doc, _| matched_counts[doc] >= required);
        sums
    }

    fn eval_knn(
        &self,
        field: &str,
        vector: &[f32],
        k: usize,
    ) -> Result<BTreeMap<usize, f64>, String> {
        if field != "embedding" {
            return Err(format!(
                "knn on field '{field}' is not supported (only 'embedding')"
            ));
        }
        let (n, dim, has_vec) = match &self.vectors {
            VectorStore::Empty => return Ok(BTreeMap::new()),
            VectorStore::Ram {
                n, dim, has_vec, ..
            } => (*n, *dim, has_vec),
            VectorStore::Mmap {
                n, dim, has_vec, ..
            } => (*n, *dim, has_vec),
        };
        if vector.len() != dim {
            return Err(format!(
                "knn vector dimension {} does not match the index dimension {dim}",
                vector.len()
            ));
        }
        // cosine similarity, OpenSearch lucene-engine score = (1+cos)/2.
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n);
        for (i, has) in has_vec.iter().enumerate() {
            if !has {
                continue;
            }
            let row = self.vectors.row(i).expect("row present");
            let mut dot = 0f32;
            let mut na = 0f32;
            let mut nb = 0f32;
            for j in 0..dim {
                dot += vector[j] * row[j];
                na += vector[j] * vector[j];
                nb += row[j] * row[j];
            }
            let cos = if na == 0.0 || nb == 0.0 {
                0.0
            } else {
                dot / (na.sqrt() * nb.sqrt())
            };
            scored.push((i, (1.0 + cos) / 2.0));
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        Ok(scored
            .into_iter()
            .take(k)
            .map(|(docno, sim)| (docno, sim as f64))
            .collect())
    }

    // -- sort / paginate / aggs ------------------------------------------------

    fn sort_key(&self, rec: &DocRec, docno: usize, spec: &SortSpec, score: f64) -> SortKey {
        match spec.key.as_str() {
            "_score" => SortKey::Num(score),
            "_doc" | "_seq_no" => SortKey::Num(docno as f64),
            "price" | "trust_score" | "created_at" | "trust_count" => {
                Self::num_field(rec, &spec.key)
                    .map(SortKey::Num)
                    .unwrap_or(SortKey::Missing)
            }
            "available" => rec
                .available
                .map(|b| SortKey::Num(if b { 1.0 } else { 0.0 }))
                .unwrap_or(SortKey::Missing),
            _ => Self::str_field(rec, &spec.key)
                .map(|s| SortKey::Str(s.to_string()))
                .unwrap_or(SortKey::Missing),
        }
    }

    /// Pairwise comparison honoring order + missing/_last. Returns Ordering
    /// of a relative to b in OUTPUT order (i.e., already direction-corrected).
    fn cmp_keys(a: &SortKey, b: &SortKey, spec: &SortSpec) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        let a_missing = matches!(a, SortKey::Missing);
        let b_missing = matches!(b, SortKey::Missing);
        if a_missing || b_missing {
            if a_missing && b_missing {
                return Ordering::Equal;
            }
            // missing_last in BOTH directions (Lucene `_last`); missing_first
            // flips to before every value.
            let missing_order = if spec.missing_last {
                Ordering::Greater
            } else {
                Ordering::Less
            };
            return if a_missing {
                missing_order
            } else {
                missing_order.reverse()
            };
        }
        let ord = match (a, b) {
            (SortKey::Num(x), SortKey::Num(y)) => x.total_cmp(y),
            (SortKey::Str(x), SortKey::Str(y)) => x.cmp(y),
            _ => Ordering::Equal,
        };
        if spec.order == Order::Asc {
            ord
        } else {
            ord.reverse()
        }
    }

    fn cursor_after(&self, keys: &[SortKey], cursor: &[SortValue], sorts: &[SortSpec]) -> bool {
        // Compare the doc's sort tuple against the cursor; the doc is "after"
        // when it compares Greater on the first differing value.
        for (i, cur) in cursor.iter().enumerate() {
            let Some(spec) = sorts.get(i) else {
                return true;
            };
            let doc = &keys[i];
            let doc_missing = matches!(doc, SortKey::Missing);
            let cur_missing = matches!(cur, SortValue::Missing);
            if doc_missing || cur_missing {
                if doc_missing && cur_missing {
                    continue;
                }
                // missing sorts LAST in both directions → the doc is after
                // the cursor exactly when the doc is the missing one.
                return doc_missing;
            }
            let cur_key = match cur {
                SortValue::Num(n) => SortKey::Num(*n),
                SortValue::Str(s) => SortKey::Str(s.clone()),
                SortValue::Missing => unreachable!("handled above"),
            };
            match Self::cmp_keys(doc, &cur_key, spec) {
                std::cmp::Ordering::Greater => return true,
                std::cmp::Ordering::Less => return false,
                std::cmp::Ordering::Equal => continue,
            }
        }
        // Fully equal to the cursor tuple → this IS the cursor doc (or a
        // duplicate of it); skip (clients include the slug tie-break in the
        // cursor, making positions deterministic).
        false
    }

    /// Aggregation keyword values of a record (tags are multi-valued).
    fn agg_values(rec: &DocRec, field: &str) -> Vec<String> {
        if rec.deleted {
            return Vec::new();
        }
        match field {
            "tags" | "tags.keyword" => rec.tags.clone(),
            "slug" | "slug.keyword" => vec![rec.slug.clone()],
            "title" | "title.keyword" => vec![rec.title.clone()],
            other => Self::str_field(rec, other)
                .map(|s| vec![s.to_string()])
                .unwrap_or_default(),
        }
    }

    fn eval_aggs(
        &self,
        aggs: &[(String, crate::dsn::Agg)],
        docs: &[usize],
    ) -> Result<Value, String> {
        let mut out = Map::new();
        for (name, agg) in aggs {
            match agg {
                crate::dsn::Agg::Terms {
                    field,
                    size,
                    exclude,
                } => {
                    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
                    for d in docs {
                        for v in Self::agg_values(&self.recs()[*d], field) {
                            *counts.entry(v).or_insert(0) += 1;
                        }
                    }
                    let mut buckets: Vec<(String, usize)> = counts
                        .into_iter()
                        .filter(|(k, _)| {
                            !exclude
                                .iter()
                                .any(|e| e.as_str() == k.as_str() || regex_is_match(e, k))
                        })
                        .collect();
                    buckets.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                    buckets.truncate(*size);
                    out.insert(
                        name.clone(),
                        json!({
                            "doc_count_error_upper_bound": 0,
                            "sum_other_doc_count": 0,
                            "buckets": buckets
                                .into_iter()
                                .map(|(key, doc_count)| json!({ "key": key, "doc_count": doc_count }))
                                .collect::<Vec<_>>()
                        }),
                    );
                }
                crate::dsn::Agg::Stats { field } => {
                    let vals: Vec<f64> = docs
                        .iter()
                        .filter_map(|d| {
                            let records = self.recs();
                            Self::num_field(&records[*d], field)
                        })
                        .collect();
                    let count = vals.len();
                    let sum: f64 = vals.iter().sum();
                    out.insert(
                        name.clone(),
                        json!({
                            "count": count,
                            "min": if count == 0 { Value::Null } else { json!(vals.iter().cloned().fold(f64::INFINITY, f64::min)) },
                            "max": if count == 0 { Value::Null } else { json!(vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)) },
                            "avg": if count == 0 { Value::Null } else { json!(sum / count as f64) },
                            "sum": sum
                        }),
                    );
                }
                crate::dsn::Agg::Filter { filter, aggs } => {
                    let r = self.eval(filter)?;
                    let sub_docs: Vec<usize> =
                        docs.iter().copied().filter(|d| r.contains_key(d)).collect();
                    let mut body = Map::new();
                    body.insert("doc_count".into(), json!(sub_docs.len()));
                    let sub = self.eval_aggs(aggs, &sub_docs)?;
                    if let Value::Object(so) = sub {
                        for (k, v) in so {
                            body.insert(k, v);
                        }
                    }
                    out.insert(name.clone(), Value::Object(body));
                }
            }
        }
        Ok(Value::Object(out))
    }

    // -- the search ----------------------------------------------------------

    pub fn search(&self, req: &SearchRequest) -> Result<SearchOutcome, String> {
        let start = Instant::now();
        let matched = self.eval(&req.query)?;
        let mut docnos: Vec<usize> = matched.keys().copied().collect();
        let has_score_sort = req.sort.iter().any(|s| s.key == "_score");

        // Sort with explicit specs + implicit slug tie-break.
        let has_slug_sort = req
            .sort
            .iter()
            .any(|s| s.key == "slug.keyword" || s.key == "slug");
        let mut sorts = req.sort.clone();
        if !has_slug_sort {
            sorts.push(SortSpec {
                key: "slug.keyword".into(),
                order: Order::Asc,
                missing_last: true,
            });
        }
        let mut keyed: Vec<(usize, Vec<SortKey>)> = {
            let records = self.recs();
            docnos
                .drain(..)
                .map(|docno| {
                    let rec = &records[docno];
                    (
                        docno,
                        sorts
                            .iter()
                            .map(|s| self.sort_key(rec, docno, s, matched[&docno]))
                            .collect(),
                    )
                })
                .collect()
        };
        keyed.sort_by(|a, b| {
            for (i, spec) in sorts.iter().enumerate() {
                let ord = Self::cmp_keys(&a.1[i], &b.1[i], spec);
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });

        // search_after cursor: drop everything up to and including the cursor
        // tuple (slug tie-break makes equality positions deterministic).
        let mut start_idx = 0usize;
        if let Some(cursor) = &req.search_after {
            for (i, (docno, _)) in keyed.iter().enumerate() {
                if self.cursor_after_keys(docno, cursor, &sorts, &matched) {
                    start_idx = i;
                    break;
                }
                start_idx = i + 1;
            }
        }

        let total = keyed.len();
        let skip = req.from + start_idx;
        let page: Vec<usize> = keyed
            .iter()
            .skip(skip)
            .take(req.size)
            .map(|(docno, _)| *docno)
            .collect();

        let mut hits = Vec::with_capacity(page.len());
        {
            let records = self.recs();
            for docno in &page {
                let rec = &records[*docno];
                let score = matched[docno];
                let sort_vals: Vec<Value> = req
                    .sort
                    .iter()
                    .map(|spec| match self.sort_key(rec, *docno, spec, score) {
                        SortKey::Num(n) => json!(n),
                        SortKey::Str(s) => json!(s),
                        SortKey::Missing => Value::Null,
                    })
                    .collect();
                let mut source: Value = serde_json::from_str(&rec.raw).unwrap_or(Value::Null);
                if let Value::Object(o) = &mut source {
                    for ex in &req.source_excludes {
                        o.remove(ex);
                    }
                }
                hits.push(Hit {
                    slug: rec.slug.clone(),
                    score: if has_score_sort || req.sort.is_empty() {
                        Some(score)
                    } else {
                        None
                    },
                    sort: sort_vals,
                    source,
                });
            }
        }

        let aggregations = if req.aggs.is_empty() {
            None
        } else {
            let all: Vec<usize> = keyed.iter().map(|(d, _)| *d).collect();
            Some(self.eval_aggs(&req.aggs, &all)?)
        };

        Ok(SearchOutcome {
            total,
            hits,
            aggregations,
            took_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// cursor comparison used by search (keys precomputed per doc).
    fn cursor_after_keys(
        &self,
        docno: &usize,
        cursor: &[SortValue],
        sorts: &[SortSpec],
        matched: &BTreeMap<usize, f64>,
    ) -> bool {
        let keys: Vec<SortKey> = {
            let records = self.recs();
            let rec = &records[*docno];
            sorts
                .iter()
                .map(|s| self.sort_key(rec, *docno, s, matched[docno]))
                .collect()
        };
        self.cursor_after(&keys, cursor, sorts)
    }

    pub fn get_by_id(&self, id: &str) -> Option<Value> {
        self.get_doc(id).map(|(doc, _)| doc)
    }

    /// Full doc fetch for the write path: (source JSON, stored embedding).
    pub fn get_doc(&self, id: &str) -> Option<(Value, Option<Vec<f32>>)> {
        self.recs()
            .iter()
            .find(|r| !r.deleted && r.slug == id)
            .map(|r| {
                (
                    serde_json::from_str(&r.raw).unwrap_or(Value::Null),
                    r.embedding.clone(),
                )
            })
    }

    /// Slugs matching a query (the _update_by_query/_delete_by_query
    /// selection set).
    pub fn select_slugs(&self, q: &crate::dsn::Query) -> Result<Vec<String>, String> {
        let matched = self.eval(q)?;
        let records = self.recs();
        Ok(matched
            .keys()
            .filter(|d| !records[**d].deleted)
            .map(|d| records[*d].slug.clone())
            .collect())
    }

    /// Build a tantivy doc for a JSON source (mirrors the build loop).
    fn tantivy_of(&self, slug: &str, doc: &Value, docno: usize) -> TantivyDocument {
        let mut tdoc = TantivyDocument::default();
        let tags = doc
            .get("tags")
            .and_then(|t| t.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for name in TEXT_FIELDS {
            let text_val = match name {
                "tags" => tags.join("\u{1}"),
                other => doc
                    .get(other)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
            };
            if !text_val.trim().is_empty() {
                if let Some(field) = self.fields.text.get(name).copied() {
                    tdoc.add_text(field, text_val);
                }
            }
        }
        tdoc.add_text(self.fields.slug, slug);
        tdoc.add_u64(self.fields.docno, docno as u64);
        tdoc
    }

    /// Incremental upsert: delete the tantivy doc by slug term, re-add with
    /// the (stable) docno, commit + reload — immediately searchable. The
    /// vector matrix is NOT touched here (background rebuild picks it up).
    /// Returns whether the doc already existed.
    pub fn upsert_doc(&self, slug: &str, doc: Value, embedding: Option<Vec<f32>>) -> bool {
        {
            let mut writer = self.writer.lock().expect("writer lock");
            let _ = writer.delete_term(Term::from_field_text(self.fields.slug, slug));
            let docno = {
                let records = self.recs();
                match records.iter().position(|r| !r.deleted && r.slug == slug) {
                    Some(slot) => slot,
                    None => {
                        // slot index of the (tombstoned or absent) old entry
                        // is unknown — the new record goes to the end
                        records.len()
                    }
                }
            };
            let _ = writer.add_document(self.tantivy_of(slug, &doc, docno));
            let _ = writer.commit();
        }
        let _ = self.reader.reload();
        // extend the prefix/fuzzy dictionaries with the new tokens
        {
            let tags = doc
                .get("tags")
                .and_then(|t| t.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let mut terms = self.terms.write().expect("terms lock");
            for name in TEXT_FIELDS {
                let text_val = match name {
                    "tags" => tags.join(" "),
                    other => doc
                        .get(other)
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                };
                if let Some(set) = terms.get_mut(name) {
                    for token in folded_tokens(&text_val) {
                        set.insert(token);
                    }
                }
            }
        }
        let mut records = self.records.write().expect("records lock");
        let slot = records.iter().position(|r| !r.deleted && r.slug == slug);
        let existed = slot.is_some();
        let mut rec = extract_rec(slug, &doc);
        rec.embedding = match slot {
            Some(i) => embedding.or_else(|| records[i].embedding.clone()),
            None => embedding,
        };
        match slot {
            Some(i) => records[i] = rec,
            None => records.push(rec),
        }
        existed
    }

    /// Incremental delete (tombstone + tantivy delete_term + commit).
    /// Returns whether the doc was present.
    pub fn delete_doc(&self, slug: &str) -> bool {
        {
            let mut writer = self.writer.lock().expect("writer lock");
            let _ = writer.delete_term(Term::from_field_text(self.fields.slug, slug));
            let _ = writer.commit();
        }
        let _ = self.reader.reload();
        let mut records = self.records.write().expect("records lock");
        let Some(slot) = records.iter().position(|r| !r.deleted && r.slug == slug) else {
            return false;
        };
        records[slot] = DocRec {
            slug: String::new(),
            title: String::new(),
            raw: "{}".into(),
            status: None,
            available: None,
            price: None,
            currency: None,
            category: None,
            category2: None,
            category_path: None,
            vendor: None,
            brand: None,
            tags: Vec::new(),
            author: None,
            genre: None,
            trust_score: None,
            trust_count: None,
            created_at: None,
            external_id: None,
            embedding: None,
            deleted: true,
        };
        true
    }
}

/// Deterministic seeded random in [0,1) — splitmix64 over (seed, docno).
fn seeded_random(seed: i64, docno: usize) -> f64 {
    let mut x = (seed as u64 ^ (docno as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_add(0x9E3779B97F4A7C15);
    x = x.wrapping_add(x >> 30);
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x = x.wrapping_add(x >> 27);
    x = x.wrapping_mul(0x94D049BB133111EB);
    x = x ^ (x >> 31);
    (x >> 11) as f64 / (1u64 << 53) as f64
}

fn levenshtein_within(a: &str, b: &str, d: usize) -> bool {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > d {
        return false;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] <= d
}

/// Regex-or-literal match for terms-agg exclude. OpenSearch/Lucene regexp
/// semantics are FULL-match (anchored) and an empty pattern excludes only
/// empty keys. Invalid patterns degrade to literal equality, never panic.
fn regex_is_match(pattern: &str, key: &str) -> bool {
    if pattern.is_empty() {
        return key.is_empty();
    }
    let anchored = format!("^(?:{})$", pattern);
    match regex::Regex::new(&anchored) {
        Ok(re) => re.is_match(key),
        Err(_) => pattern == key,
    }
}

/// Build a tantivy PhraseQuery from tokens (offsets = token positions).
fn phrase_query(tokens: &[String], field: Field) -> PhraseQuery {
    let terms: Vec<(usize, Term)> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (i, Term::from_field_text(field, t)))
        .collect();
    PhraseQuery::new_with_offset(terms)
}
