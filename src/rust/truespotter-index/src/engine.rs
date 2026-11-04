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

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
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
    /// Pre-computed function_score trust boost (spec
    /// index-rs-trust-precompute W1): `(ts * ln(1+tc)).max(0)` when both
    /// fields are present, else 0.0 (trust_score without trust_count →
    /// ln(1) = 0). Evaluated once per write instead of per doc per query.
    trust_boost: f64,
    /// epoch millis (doc ISO strings or epoch millis accepted)
    created_at: Option<i64>,
    /// the `id` doc field (external ref, e.g. "gid://shopify/…")
    external_id: Option<String>,
    /// (embeddings live in the VectorStore mmap + the small `pending_vectors`
    /// sidecar — NOT per-record; 32k x 16KB duplicates cost ~512MB RSS)
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
    // Spec index-rs-pl-stemming: conservative Polish case-ending strip,
    // applied AFTER the fold (suffixes are matched in folded form —
    // "ów" → "ow"). One place only — TsTokenizer, the term dictionaries,
    // fuzzy enumeration, phrase and multi_match all read through this
    // function, so index and query semantics stay identical.
    tokens.iter().map(|t| stem_pl(t).to_string()).collect()
}

/// Polish inflectional suffixes in FOLDED (post-diacritic-fold) form,
/// longest first. Case endings only (noun/adjective declension) — no
/// derivational stripping (spec index-rs-pl-stemming W3).
const PL_SUFFIXES: [&str; 22] = [
    "iami", "ymi", "imi", "ego", "ach", "ami", "owi", "emu", "ych", "ich", "iej", "ow", "om", "em",
    "ie", "ej", "y", "i", "a", "e", "u", "o",
];

/// Strip at most TWO inflectional suffixes (longest first per pass) from a
/// folded token, keeping at least 3 characters of the stem after each
/// pass (W2). The second pass converges the -nie/-nia gerund patterns
/// ("bieganie" → "biegani" → "biegan" ≡ "biegania" → "biegan"). Words
/// without a known ending and stems that would get too short pass through
/// unchanged. Over-stemming hurts precision only — the same function
/// tokenizes index and query, so recall never breaks.
pub(crate) fn stem_pl(word: &str) -> &str {
    let mut w = word;
    for _ in 0..2 {
        let mut stripped = w;
        for suf in PL_SUFFIXES {
            if w.len() > suf.len() && w.ends_with(suf) {
                // suffixes are ASCII — the boundary is a valid char boundary
                let stem = &w[..w.len() - suf.len()];
                if stem.chars().count() >= 3 {
                    stripped = stem;
                    break;
                }
            }
        }
        if stripped == w {
            break;
        }
        w = stripped;
    }
    w
}

// ---------------------------------------------------------------------------
// Synonym dictionary (specs index-rs-dynamic-synonyms +
// index-rs-llm-synonym-mining) — query-time expansion for
// match/multi_match. DATA, not code: loaded from the synonym_rules table
// (the durable cache for admin-curated AND LLM-generated pairs), held in
// RAM behind an RwLock, hot-swappable without a rebuild.
//
// Every term is stored RAW (with diacritics) and stemmed at load through
// fold_str + stem_pl — the single source of truth for the term space —
// so index and query semantics match by construction. A rule side is a
// token SEQUENCE: len 1 -> Term, len > 1 -> adjacent Phrase.
// ---------------------------------------------------------------------------

pub struct SynonymDict {
    /// bidirectional (side_a, side_b) stemmed token sequences
    pairs: Vec<(Vec<String>, Vec<String>)>,
}

impl SynonymDict {
    pub fn empty() -> Self {
        Self { pairs: Vec::new() }
    }

    /// Build from raw (unstemmed) rule pairs — each token folded+stemmed
    /// individually (spec index-rs-dynamic-synonyms W2).
    pub fn from_raw(pairs: impl IntoIterator<Item = (Vec<String>, Vec<String>)>) -> Self {
        let stem_seq = |v: &[String]| -> Vec<String> {
            v.iter().map(|w| stem_pl(&fold_str(w)).to_string()).collect()
        };
        Self {
            pairs: pairs
                .into_iter()
                .map(|(a, b)| (stem_seq(&a), stem_seq(&b)))
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// One member of a synonym group: a single term or an adjacent phrase
/// (verified via tantivy PhraseQuery — adjacency strict, W3).
enum GroupMember {
    Term(String),
    Phrase(Vec<String>),
}

/// Group tokens with their dictionary synonyms: each group is one msm
/// UNIT — a doc matching any group member counts once, and
/// `minimum_should_match` keeps its per-original-token semantics
/// (expansion must not inflate the required count). A matched
/// multi-word phrase MERGES its token positions into ONE group
/// (semantically one concept). Longest phrase match wins (greedy).
fn expand_synonym_groups(dict: &SynonymDict, tokens: Vec<String>) -> Vec<Vec<GroupMember>> {
    let push_alt = |g: &mut Vec<GroupMember>, alt: &[String]| {
        if alt.len() == 1 {
            g.push(GroupMember::Term(alt[0].clone()));
        } else {
            g.push(GroupMember::Phrase(alt.to_vec()));
        }
    };
    let mut groups: Vec<Vec<GroupMember>> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        // longest phrase side match at the current position (greedy)
        let mut matched: Option<(usize, usize)> = None;
        for (pi, (a, b)) in dict.pairs.iter().enumerate() {
            for (seq, _alt) in [(a.as_slice(), b.as_slice()), (b.as_slice(), a.as_slice())] {
                let len = seq.len();
                if len >= 2
                    && i + len <= tokens.len()
                    && matched.is_none_or(|(_, l)| len > l)
                    && seq.iter().zip(&tokens[i..i + len]).all(|(s, t)| s == t)
                {
                    matched = Some((pi, len));
                }
            }
        }
        if let Some((pi, _len)) = matched {
            let (a, b) = &dict.pairs[pi];
            // the matched side becomes the Phrase; the other side joins
            // the group as the alternative (term or phrase)
            let side_fits = |seq: &[String]| {
                i + seq.len() <= tokens.len()
                    && seq.iter().zip(&tokens[i..i + seq.len()]).all(|(s, t)| s == t)
            };
            let (seq, alt) = if side_fits(a) {
                (a.as_slice(), b.as_slice())
            } else {
                (b.as_slice(), a.as_slice())
            };
            let mut g = vec![GroupMember::Phrase(seq.to_vec())];
            push_alt(&mut g, alt);
            groups.push(g);
            i += seq.len();
            continue;
        }
        // single token: original + pairs where either side is exactly
        // this token (the other side joins the group — term or phrase)
        let t = &tokens[i];
        let mut g = vec![GroupMember::Term(t.clone())];
        for (a, b) in &dict.pairs {
            for (seq, alt) in [(a.as_slice(), b.as_slice()), (b.as_slice(), a.as_slice())] {
                if seq.len() == 1 && &seq[0] == t {
                    push_alt(&mut g, alt);
                }
            }
        }
        groups.push(g);
        i += 1;
    }
    groups
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
// Vector store — brute-force kNN over an int8-quantized matrix. Persisted
// to /data/vectors.f32 and mmap'd read-only (RAM budget); in-RAM fallback
// when /data is unavailable. Rows are L2-normalized (W2) + symmetrically
// quantized per-vector to i8 (spec index-rs-int8-quantization): 4× less
// memory and scan bandwidth than f32 at ≤1% cosine error.
// ---------------------------------------------------------------------------

/// Persisted-matrix header magic ("TSIN" LE) — guards the binary format.
/// Files written by older builds (pre-W2 no header; v2 f32) start with a
/// dim u32 / v2 marker instead and are ignored (the build always rewrites
/// the file fresh from PG rows).
const VECTORS_MAGIC: u32 = 0x5453_494E;
/// v3: header (magic, version, dim, n) + n×f32 per-row scales +
/// n×dim×i8 quantized matrix.
const VECTORS_VERSION: u32 = 3;

/// L2-normalize a vector (W2): stored embeddings are unit-length so the
/// kNN scan is a plain dot product (one pass instead of three). A
/// zero-norm vector passes through unchanged (dot = 0 → neutral score
/// 0.5, same as the old cos=0 branch).
pub(crate) fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 && (norm - 1.0).abs() > 1e-6 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Symmetric per-vector int8 quantization (W1): scale = max|x|, values
/// clamp to [-127, 127]. Zero vectors get an all-zero row and scale 0.0
/// (dot = 0 → neutral score 0.5, same semantics as before).
fn quantize_row(v: &[f32]) -> (Vec<i8>, f32) {
    let scale = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if scale <= 0.0 {
        return (vec![0i8; v.len()], 0.0);
    }
    let q = v
        .iter()
        .map(|x| (x / scale * 127.0).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

/// Approximate dequantization (`x = q/127 * scale`, W3) — used only by
/// `get_doc`'s embedding return; every production call site discards it.
fn dequantize_row(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|c| *c as f32 / 127.0 * scale).collect()
}

enum VectorStore {
    Empty,
    Ram {
        data: Vec<i8>,
        scales: Vec<f32>,
        n: usize,
        dim: usize,
        has_vec: Vec<bool>,
    },
    Mmap {
        map: memmap2::Mmap,
        scales: Vec<f32>,
        n: usize,
        dim: usize,
        has_vec: Vec<bool>,
    },
}

impl VectorStore {
    /// Build from the per-doc embeddings (parallel to records); persist +
    /// mmap into `dir` when possible.
    fn build(embeddings: &[Option<Vec<f32>>], dir: Option<&std::path::Path>) -> Self {
        let dim = embeddings
            .iter()
            .find_map(|e| e.as_ref().map(|v| v.len()))
            .unwrap_or(0);
        if dim == 0 {
            return VectorStore::Empty;
        }
        let n = embeddings.len();
        let has_vec: Vec<bool> = embeddings
            .iter()
            .map(|e| e.as_ref().is_some_and(|v| v.len() == dim))
            .collect();
        // W1 (spec index-rs-int8-quantization): normalize (W2) then
        // quantize each row symmetrically to i8 with a per-row scale.
        let mut data = vec![0i8; n * dim];
        let mut scales = vec![0f32; n];
        for (i, e) in embeddings.iter().enumerate() {
            if let Some(e) = e.as_ref().filter(|e| e.len() == dim) {
                let normed = l2_normalize(e);
                let (q, s) = quantize_row(&normed);
                data[i * dim..(i + 1) * dim].copy_from_slice(&q);
                scales[i] = s;
            }
        }
        if let Some(dir) = dir {
            let path = dir.join("vectors.f32");
            // v3 layout: 16-byte header + n×f32 scales + n×dim×i8 matrix
            let mut buf: Vec<u8> = Vec::with_capacity(16 + n * 4 + data.len());
            buf.extend_from_slice(&VECTORS_MAGIC.to_le_bytes());
            buf.extend_from_slice(&VECTORS_VERSION.to_le_bytes());
            buf.extend_from_slice(&(dim as u32).to_le_bytes());
            buf.extend_from_slice(&(n as u32).to_le_bytes());
            for s in &scales {
                buf.extend_from_slice(&s.to_le_bytes());
            }
            buf.extend(data.iter().map(|b| *b as u8));
            if fs::create_dir_all(dir).is_ok() && fs::write(&path, &buf).is_ok() {
                if let Ok(bytes) = fs::read(&path) {
                    if bytes.len() >= 16 + n * 4 + n * dim {
                        let fmagic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
                        let fversion = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
                        let fdim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
                        let fnn = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
                        if fmagic == VECTORS_MAGIC
                            && fversion == VECTORS_VERSION
                            && fdim == dim
                            && fnn == n
                        {
                            if let Ok(map) = unsafe { memmap2::Mmap::map(&path_as_file(&path)) } {
                                // Spec index-rs-knn-warm-start W1: fault in
                                // ALL pages NOW — the first kNN query after
                                // boot would otherwise pay seconds of volume
                                // reads on the matrix (prod measured 10.4 s
                                // pre-int8, 2026-11-04). One touch per 4K
                                // page; best-effort, never fails the build.
                                let sum: u64 = map.chunks(4096).map(|c| c[0] as u64).sum();
                                std::hint::black_box(sum);
                                // scales live in RAM (n × 4 B — tiny)
                                let scales: Vec<f32> = bytes[16..16 + n * 4]
                                    .chunks_exact(4)
                                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                                    .collect();
                                return VectorStore::Mmap {
                                    map,
                                    scales,
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
            scales,
            n,
            dim,
            has_vec,
        }
    }

    fn has_at(&self, i: usize) -> bool {
        match self {
            VectorStore::Empty => false,
            VectorStore::Ram { n, has_vec, .. } | VectorStore::Mmap { n, has_vec, .. } => {
                i < *n && has_vec[i]
            }
        }
    }

    /// Owned (approximately dequantized) copy for the write path
    /// (partial-update "keep embedding", W3).
    fn row_vec(&self, i: usize) -> Option<Vec<f32>> {
        let (q, scale) = self.row8(i)?;
        Some(dequantize_row(q, scale))
    }

    /// Quantized row + its scale. `i` can exceed the build-time matrix
    /// (docs upserted after build — their vectors live in pending_vectors,
    /// not the matrix).
    fn row8(&self, i: usize) -> Option<(&[i8], f32)> {
        if i >= self.len() {
            return None;
        }
        match self {
            VectorStore::Empty => None,
            VectorStore::Ram {
                data, scales, dim, ..
            } => Some((&data[i * dim..(i + 1) * dim], scales[i])),
            VectorStore::Mmap {
                map,
                scales,
                n,
                dim,
                ..
            } => {
                // v3 layout: 16-byte header + n×f32 scales, then the i8 matrix
                let off = 16 + n * 4 + i * dim;
                let bytes = &map[off..off + dim];
                Some((i8_slice(bytes, *dim), scales[i]))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            VectorStore::Empty => 0,
            VectorStore::Ram { n, .. } | VectorStore::Mmap { n, .. } => *n,
        }
    }
}

/// Reinterpret a byte slice as the i8 quantized row (no copy).
fn i8_slice(bytes: &[u8], dim: usize) -> &[i8] {
    assert_eq!(bytes.len(), dim);
    let ptr = bytes.as_ptr() as *const i8;
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
    /// Embeddings upserted since the last full rebuild (the mmap matrix is
    /// immutable). Keyed by docno — rebuilt with a fresh Engine at swap.
    pending_vectors: std::sync::RwLock<HashMap<usize, Vec<f32>>>,
    /// Dynamic synonym dictionary (spec index-rs-dynamic-synonyms W3) —
    /// hot-swappable without a rebuild; expansion reads it per query.
    synonyms: std::sync::RwLock<SynonymDict>,
    /// Query-term frequency counter (spec index-rs-query-mining W1) —
    /// RAM only, per-boot; feeds the LLM synonym miner with the
    /// vocabulary users actually type.
    query_log: std::sync::Mutex<BTreeMap<String, u64>>,
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

/// Intentional per-clause weight for hybrid fusion (spec
/// index-rs-hybrid-normalization W2): the clause boost, kNN = 1.0.
fn clause_weight(q: &Query) -> f64 {
    match q {
        Query::Match { boost, .. }
        | Query::MatchPhrase { boost, .. }
        | Query::MatchPhrasePrefix { boost, .. }
        | Query::MultiMatch { boost, .. } => *boost,
        _ => 1.0,
    }
}

/// Min-max score normalization to [0,1] per clause (spec
/// index-rs-hybrid-normalization W1). A degenerate (constant) scale maps
/// every matched doc to 1.0 — it matched maximally within its clause.
fn min_max_normalize(m: &BTreeMap<usize, f64>) -> BTreeMap<usize, f64> {
    if m.is_empty() {
        return BTreeMap::new();
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for s in m.values() {
        if *s < min {
            min = *s;
        }
        if *s > max {
            max = *s;
        }
    }
    if max - min <= f64::EPSILON {
        return m.keys().map(|d| (*d, 1.0)).collect();
    }
    m.iter()
        .map(|(d, s)| (*d, (s - min) / (max - min)))
        .collect()
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
        // W1 (spec index-rs-trust-precompute): computed once at write
        trust_boost: match (get_f64(doc, "trust_score"), get_f64(doc, "trust_count")) {
            (Some(ts), Some(tc)) => (ts * (1.0f64 + tc).ln()).max(0.0),
            _ => 0.0,
        },
        created_at: doc.get("created_at").and_then(parse_created_at),
        external_id: doc.get("id").and_then(|i| i.as_str()).map(String::from),
        deleted: false,
    }
}

/// Boot-time sweep: remove leaked tantivy temp dirs left by previous
/// crashed/killed builds. SIGKILL never runs `Drop` (TempDir), so every hard
/// restart leaked a full index on /data (72 leaks found on prod). Dirs
/// younger than 10 minutes are left alone (a concurrent build in the same
/// container may own them); failures only warn — never fail the boot.
pub(crate) fn sweep_stale_tempdirs(dir: &std::path::Path) {
    sweep_stale_tempdirs_with(dir, std::time::Duration::from_secs(600));
}

pub(crate) fn sweep_stale_tempdirs_with(dir: &std::path::Path, max_age: std::time::Duration) {
    let cutoff = std::time::SystemTime::now() - max_age;
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(n) => n,
            None => continue,
        };
        if !name.starts_with(".tmp") {
            continue;
        }
        let is_stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .map(|mt| mt < cutoff)
            .unwrap_or(false);
        if !is_stale {
            continue;
        }
        match fs::remove_dir_all(entry.path()) {
            Ok(()) => tracing::info!(dir = name, "swept leaked tantivy temp dir"),
            Err(e) => tracing::warn!(dir = name, error = %e, "temp dir sweep failed"),
        }
    }
}

impl Engine {
    /// Build the engine from corpus docs (PG rows or synthetic test data).
    /// `data_dir` is used for the vector matrix persistence (mmap); the
    /// Tantivy index is always built fresh in a temp dir (29k docs →
    /// seconds) — no load/persist dance, `slug` is the id everywhere.
    pub fn build(docs: Vec<EngineDoc>, data_dir: Option<PathBuf>) -> Result<Self, String> {
        Self::build_with_synonyms(docs, data_dir, SynonymDict::empty())
    }

    /// Build with a dynamic synonym dictionary (spec
    /// index-rs-dynamic-synonyms W2): boot/rebuild paths load it from the
    /// synonym_rules table; tests inject it directly.
    pub fn build_with_synonyms(
        docs: Vec<EngineDoc>,
        data_dir: Option<PathBuf>,
        synonyms: SynonymDict,
    ) -> Result<Self, String> {
        if let Some(d) = data_dir.as_ref() {
            sweep_stale_tempdirs(d);
        }
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
        // embeddings travel parallel to records (extract_rec no longer
        // stores them — the mmap matrix is their only long-term home)
        let mut embeddings: Vec<Option<Vec<f32>>> = Vec::with_capacity(docs.len());
        for (i, d) in docs.into_iter().enumerate() {
            let rec = extract_rec(&d.slug, &d.doc);
            embeddings.push(d.embedding);
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
        let vectors = VectorStore::build(&embeddings, data_dir.as_deref());
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
            pending_vectors: std::sync::RwLock::new(HashMap::new()),
            synonyms: std::sync::RwLock::new(synonyms),
            query_log: std::sync::Mutex::new(BTreeMap::new()),
            _tmp: Some(tmp),
        })
    }

    pub fn num_docs(&self) -> usize {
        self.recs().iter().filter(|r| !r.deleted).count()
    }

    /// Hot-swap the synonym dictionary (spec index-rs-dynamic-synonyms W4):
    /// takes effect on the very next query — no reindex, no rebuild.
    pub fn set_synonyms(&self, dict: SynonymDict) {
        *self.synonyms.write().expect("synonyms lock poisoned") = dict;
    }

    /// Record query tokens into the frequency counter (spec
    /// index-rs-query-mining W1) — called pre-expansion, once per
    /// match/multi_match clause. Amortized cap: past 20k unique terms the
    /// map prunes to the top 10k by count.
    fn record_query_tokens(&self, tokens: &[String]) {
        let mut log = self.query_log.lock().expect("query log lock poisoned");
        for t in tokens {
            *log.entry(t.clone()).or_insert(0) += 1;
        }
        if log.len() > 20_000 {
            let mut counts: Vec<(String, u64)> =
                log.iter().map(|(k, v)| (k.clone(), *v)).collect();
            counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            counts.truncate(10_000);
            *log = counts.into_iter().collect();
        }
    }

    /// Top query-log terms by frequency (W2) — count desc, alpha tie-break.
    pub fn query_terms(&self, limit: usize) -> Vec<(String, u64)> {
        let log = self.query_log.lock().expect("query log lock poisoned");
        let mut out: Vec<(String, u64)> =
            log.iter().map(|(k, v)| (k.clone(), *v)).collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(limit);
        out
    }

    /// Top dictionary terms of a text field by document frequency (spec
    /// index-rs-llm-synonym-mining — candidate vocabulary for the LLM
    /// miner). Count collector hits the term-doc_freq fast path, so a
    /// full dictionary sweep is cheap and runs once per mining pass.
    pub fn top_terms(&self, field_name: &str, limit: usize) -> Vec<(String, usize)> {
        let Some(field) = self.fields.text.get(field_name).copied() else {
            return Vec::new();
        };
        let searcher = self.searcher();
        let terms = self.terms.read().expect("terms lock poisoned");
        let Some(set) = terms.get(field_name) else {
            return Vec::new();
        };
        let mut out: Vec<(String, usize)> = Vec::with_capacity(set.len());
        for t in set.iter() {
            let count = searcher
                .search(
                    &TermQuery::new(
                        Term::from_field_text(field, t),
                        IndexRecordOption::Basic,
                    ),
                    &tantivy::collector::Count,
                )
                .unwrap_or(0);
            out.push((t.clone(), count));
        }
        out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(limit);
        out
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
    /// W4 (spec index-rs-optimization): returns a borrow instead of an
    /// owned String — the filter clauses scan the whole corpus per query
    /// and the old per-record clone was tens of thousands of allocations.
    fn str_field_cow<'a>(rec: &'a DocRec, field: &str) -> Option<Cow<'a, str>> {
        let f = field.trim_end_matches(".keyword");
        match f {
            "slug" => Some(Cow::Borrowed(&rec.slug)),
            "title" => Some(Cow::Borrowed(&rec.title)),
            "vendor" => rec.vendor.as_deref().map(Cow::Borrowed),
            "brand" => rec.brand.as_deref().map(Cow::Borrowed),
            "type" => rec.category.as_deref().map(Cow::Borrowed),
            "category" => rec.category2.as_deref().map(Cow::Borrowed),
            "category_path" => rec.category_path.as_deref().map(Cow::Borrowed),
            "currency" => rec.currency.as_deref().map(Cow::Borrowed),
            "status" => rec.status.as_deref().map(Cow::Borrowed),
            "author" => rec.author.as_deref().map(Cow::Borrowed),
            "genre" => rec.genre.as_deref().map(Cow::Borrowed),
            "id" => rec.external_id.as_deref().map(Cow::Borrowed),
            _ => raw_value(rec, f).and_then(|v| v.as_str().map(|s| Cow::Owned(String::from(s)))),
        }
    }

    fn exists_field(&self, rec: &DocRec, docno: usize, field: &str) -> bool {
        match field {
            "tags" => !rec.tags.is_empty(),
            "embedding" => {
                self.vectors.has_at(docno)
                    || self
                        .pending_vectors
                        .read()
                        .expect("pending lock poisoned")
                        .contains_key(&docno)
            }
            _ => match Self::str_field_cow(rec, field) {
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
                        Self::str_field_cow(r, field).is_some_and(|s| s.starts_with(value.as_str()))
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
                    .filter(|(i, r)| !r.deleted && self.exists_field(r, *i, field))
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
                        // — evaluated natively, boost_mode sum. Pre-computed at
                        // write time (spec index-rs-trust-precompute W2).
                        let trust = self.recs()[docno].trust_boost;
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
                            || Self::str_field_cow(r, field).as_deref() == Some(s.as_str())
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
        // Spec index-rs-hybrid-normalization: when a should clause is kNN,
        // the bool is a "hybrid" query. Raw BM25 (unbounded) and kNN
        // ((1+cos)/2 in [0,1]) scales are incomparable — the naive sum lets
        // one modality drown the other. Normalize each clause min-max to
        // [0,1] and weight it by the clause boost (OpenSearch min_max
        // normalization + weighted arithmetic combination, subset). Boosts
        // are not double-applied: normalization erases the raw scale
        // (boost included), the weight reintroduces it as intent.
        let hybrid = b.should.iter().any(|c| matches!(c, Query::Knn { .. }));
        let weights: Vec<f64> = b.should.iter().map(clause_weight).collect();
        let should_maps: Vec<BTreeMap<usize, f64>> = if hybrid {
            should_maps.iter().map(min_max_normalize).collect()
        } else {
            should_maps
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
            for (ci, map) in should_maps.iter().enumerate() {
                if let Some(s) = map.get(&doc) {
                    matched += 1;
                    score += if hybrid { weights[ci] * s } else { *s };
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
        // spec index-rs-query-mining W1: record the vocabulary users type
        self.record_query_tokens(&tokens);
        // Spec index-rs-synonyms W2: non-fuzzy match queries expand each
        // token with its synonyms (group = original + synonyms, one map
        // per GROUP). Fuzzy stays literal — typo tolerance is orthogonal.
        let per_token: Vec<BTreeMap<usize, f64>> = if fuzzy {
            tokens
                .iter()
                .map(|token| {
                    let mut m = BTreeMap::new();
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
                        // W1 (spec index-rs-optimization): borrow the dictionary
                        // under the read guard instead of cloning the whole
                        // BTreeSet — the old clone allocated tens of thousands
                        // of Strings on EVERY fuzzy query. read-read RwLock
                        // doesn't block; term_scores takes no terms lock.
                        let terms = self.terms.read().expect("terms lock");
                        let dict = &terms[self.field_name(field)];
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
                    m
                })
                .collect()
        } else {
            let groups = {
                let dict = self.synonyms.read().expect("synonyms lock poisoned");
                expand_synonym_groups(&dict, tokens)
            };
            groups
                .iter()
                .map(|group| {
                    // union over the group members (terms + phrases)
                    let mut m = BTreeMap::new();
                    for member in group {
                        let pairs = match member {
                            GroupMember::Term(t) => self.term_scores(field, t),
                            GroupMember::Phrase(toks) => self.phrase_scores(field, toks),
                        };
                        for (doc, s) in pairs {
                            *m.entry(doc).or_insert(0.0) += s;
                        }
                    }
                    m
                })
                .collect()
        };
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

    /// All (docno, bm25) pairs for an adjacent phrase in a text field
    /// (spec index-rs-phrase-synonyms W3 — extracted from the
    /// eval_phrase body pattern; multi-token phrases only).
    fn phrase_scores(&self, field: Field, tokens: &[String]) -> Vec<(usize, f64)> {
        let searcher = self.searcher();
        let limit = searcher.num_docs() as usize + 1;
        let hits = searcher
            .search(&phrase_query(tokens, field), &TopDocs::with_limit(limit))
            .unwrap_or_default();
        self.docnos(&hits, &searcher)
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
            // W1: borrow the dictionary under the read guard instead of
            // cloning the whole BTreeSet per autocomplete query.
            // W5 (spec index-rs-optimization): rank the prefix tail by
            // document frequency (Count collector — cheap term-metadata
            // path) instead of taking the first 50 alphabetically:
            // frequent terms are the better autocomplete suggestions.
            // Frequency ties keep the old alphabetical order so ranking
            // stays deterministic.
            let mut cands: Vec<String> = {
                let terms = self.terms.read().expect("terms lock");
                terms[self.field_name(tfield)]
                    .iter()
                    .filter(|t| t.starts_with(last.as_str()))
                    .take(1000)
                    .cloned()
                    .collect()
            };
            if cands.is_empty() {
                return BTreeMap::new();
            }
            let searcher = self.searcher();
            let mut tail: Vec<(String, usize)> = cands
                .drain(..)
                .map(|t| {
                    let count = searcher
                        .search(
                            &TermQuery::new(
                                Term::from_field_text(tfield, &t),
                                IndexRecordOption::Basic,
                            ),
                            &tantivy::collector::Count,
                        )
                        .unwrap_or(0);
                    (t, count)
                })
                .collect();
            tail.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            tail.truncate(50);
            candidates = tail
                .into_iter()
                .map(|(t, _)| {
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
        // Spec index-rs-synonyms W2: synonym groups — the GROUP is the
        // msm unit, so minimum_should_match keeps its per-original-token
        // semantics (W3).
        let query_tokens = folded_tokens(text);
        // spec index-rs-query-mining W1: record the vocabulary users type
        self.record_query_tokens(&query_tokens);
        let groups = {
            let dict = self.synonyms.read().expect("synonyms lock poisoned");
            expand_synonym_groups(&dict, query_tokens)
        };
        let mut best: BTreeMap<usize, f64> = BTreeMap::new();
        for (name, fboost) in fields {
            // Per-field score: sum over matched groups (operator), with the
            // per-field minimum_should_match applied to the matched-group count.
            let per_field = self.eval_text_match_with_msm(name, &groups, operator, msm);
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
        groups: &[Vec<GroupMember>],
        operator: Operator,
        msm: &Option<Msm>,
    ) -> BTreeMap<usize, f64> {
        let Some(tfield) = self.text_field(field) else {
            return BTreeMap::new();
        };
        if groups.is_empty() {
            return BTreeMap::new();
        }
        // required counts original TOKENS (== groups), not the expanded
        // synonym members (spec index-rs-synonyms W3).
        let required = msm
            .as_ref()
            .map(|m| m.required(groups.len()))
            .unwrap_or(match operator {
                Operator::And => groups.len(),
                Operator::Or => 0,
            });
        let mut matched_counts: BTreeMap<usize, usize> = BTreeMap::new();
        let mut sums: BTreeMap<usize, f64> = BTreeMap::new();
        for group in groups {
            // union over the group members (terms + phrases)
            let mut m: BTreeMap<usize, f64> = BTreeMap::new();
            for member in group {
                let pairs = match member {
                    GroupMember::Term(t) => self.term_scores(tfield, t),
                    GroupMember::Phrase(toks) => self.phrase_scores(tfield, toks),
                };
                for (doc, s) in pairs {
                    *m.entry(doc).or_insert(0.0) += s;
                }
            }
            for (doc, s) in m {
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
        // W2 (spec index-rs-optimization): the query is L2-normalized so
        // cosine collapses to a plain dot product. Pending rows are f32
        // (small sidecar) — dotted in f32 below.
        let q = l2_normalize(vector);
        // W1/W2 (spec index-rs-int8-quantization): quantize the query to
        // i8 as well — the scan is an i8×i8 integer dot with i32
        // accumulation, rescaled by the per-row/query scales.
        let (q8, s_q) = quantize_row(&q);
        // pending upserts override their (stale) matrix row until the next
        // full rebuild swaps in a fresh Engine
        let pending = self
            .pending_vectors
            .read()
            .expect("pending lock poisoned")
            .clone();
        // W3: parallel score scan (rayon) over the matrix rows. Rows
        // without a vector and rows overridden by pending upserts score
        // -inf and are filtered below — the pending loop re-scores those
        // (and rows beyond the matrix, which are pending-only).
        let scores: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                if !has_vec[i] || pending.contains_key(&i) {
                    return f32::NEG_INFINITY;
                }
                let (r8, s_r) = self.vectors.row8(i).expect("row present");
                let mut acc: i32 = 0;
                for j in 0..dim {
                    acc += q8[j] as i32 * r8[j] as i32;
                }
                // 127² = 16129; clamp guards rounding overshoot
                (acc as f32 * s_q * s_r / 16129.0).clamp(-1.0, 1.0)
            })
            .collect();
        let mut scored: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .filter(|(_, s)| *s > f32::NEG_INFINITY)
            .map(|(docno, dot)| (docno, (1.0 + dot) / 2.0))
            .collect();
        // Score (1+cos)/2 semantics preserved for pending rows too; the
        // defensive re-normalization covers rows upserted by pre-W2 engine
        // instances that may still be unnormalized.
        for (&docno, v) in pending.iter() {
            if v.len() != dim {
                continue;
            }
            let row = l2_normalize(v);
            let mut dot = 0f32;
            for j in 0..dim {
                dot += q[j] * row[j];
            }
            scored.push((docno, (1.0 + dot) / 2.0));
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
        let records = self.recs();
        let (i, r) = records
            .iter()
            .enumerate()
            .find(|(_, r)| !r.deleted && r.slug == id)?;
        let emb = self
            .pending_vectors
            .read()
            .expect("pending lock poisoned")
            .get(&i)
            .cloned()
            .or_else(|| self.vectors.row_vec(i));
        Some((serde_json::from_str(&r.raw).unwrap_or(Value::Null), emb))
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
        let rec = extract_rec(slug, &doc);
        {
            // New embeddings land in the pending sidecar (the mmap matrix is
            // immutable until the next full rebuild); `embedding == None`
            // keeps the stored one — get_doc falls back to the matrix row.
            let docno = slot.unwrap_or(records.len());
            if let Some(e) = embedding {
                // W2: pending rows normalized at insert so the kNN scan is
                // a plain dot product for them too.
                self.pending_vectors
                    .write()
                    .expect("pending lock poisoned")
                    .insert(docno, l2_normalize(&e));
            }
        }
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
            trust_boost: 0.0,
            created_at: None,
            external_id: None,
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
