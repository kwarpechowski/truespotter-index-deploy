//! Postgres `catalog_documents` — the source of truth for the embedded
//! index — plus the `--backfill` mode: scroll the EXISTING OpenSearch
//! `products` index (OPENSEARCH_URL) and upsert docs in batches of
//! BACKFILL_BATCH (multi-row INSERT, one COMMIT per batch) into Postgres.
//! Existing rows are skipped unless `--force`.

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

use crate::engine::EngineDoc;

/// f32 slice → LE bytes (the bytea storage format).
pub fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// bytea → f32 slice (LE).
pub fn bytes_to_vec(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The table is owned by this service (CREATE IF NOT EXISTS).
pub async fn ensure_table(pool: &PgPool) -> Result<(), String> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS catalog_documents (
            slug text PRIMARY KEY,
            doc jsonb NOT NULL,
            embedding bytea NULL,
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        "#,
    )
    .execute(pool)
    .await
    .map_err(|e| format!("ensure catalog_documents: {e}"))?;
    // External-version tracking for _bulk index ops (KAN-0162 idempotency).
    sqlx::query("ALTER TABLE catalog_documents ADD COLUMN IF NOT EXISTS version bigint")
        .execute(pool)
        .await
        .map_err(|e| format!("ensure catalog_documents.version: {e}"))?;
    // Dynamic synonym dictionary (spec index-rs-dynamic-synonyms W1) —
    // the durable CACHE for LLM-generated and admin-curated pairs.
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS synonym_rules (
            id bigserial PRIMARY KEY,
            terms_a text[] NOT NULL,
            terms_b text[] NOT NULL,
            category text,
            enabled boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        "#,
    )
    .execute(pool)
    .await
    .map_err(|e| format!("ensure synonym_rules: {e}"))?;
    // LLM mining dedup: terms already attempted (generated or not).
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS synonym_gen_log (
            term text PRIMARY KEY,
            last_run timestamptz NOT NULL DEFAULT now()
        )
        "#,
    )
    .execute(pool)
    .await
    .map_err(|e| format!("ensure synonym_gen_log: {e}"))?;
    // Seed only when the dictionary is empty (W6) — curated pairs
    // covering the real prod categories (books/footwear/pet food/apparel).
    sqlx::query(SCHEMA_SEED_SYNONYMS)
        .execute(pool)
        .await
        .map_err(|e| format!("seed synonym_rules: {e}"))?;
    Ok(())
}

const SCHEMA_SEED_SYNONYMS: &str = r#"
INSERT INTO synonym_rules (terms_a, terms_b)
SELECT v.a, v.b
FROM (VALUES
    (ARRAY['buty'],            ARRAY['obuwie']),
    (ARRAY['smartfon'],        ARRAY['telefon']),
    (ARRAY['smartfon'],        ARRAY['smartphone']),
    (ARRAY['laptop'],          ARRAY['notebook']),
    (ARRAY['klawiatura'],      ARRAY['keyboard']),
    (ARRAY['mysz'],            ARRAY['mouse']),
    (ARRAY['słuchawki'],       ARRAY['headphones']),
    (ARRAY['słuchawki'],       ARRAY['headset']),
    (ARRAY['zegarek'],         ARRAY['watch']),
    (ARRAY['rower'],           ARRAY['bike']),
    (ARRAY['telewizor'],       ARRAY['tv']),
    (ARRAY['pilot'],           ARRAY['remote']),
    (ARRAY['kamera'],          ARRAY['camera']),
    (ARRAY['kawa'],            ARRAY['coffee']),
    (ARRAY['rękawiczki'],      ARRAY['rękawice']),
    (ARRAY['spodenki'],        ARRAY['szorty']),
    (ARRAY['bluza'],           ARRAY['hoodie']),
    (ARRAY['koszulki'],        ARRAY['t-shirty']),
    (ARRAY['torba'],           ARRAY['torebka']),
    (ARRAY['klapki'],          ARRAY['sandały']),
    (ARRAY['karta','graficzna'], ARRAY['gpu']),
    (ARRAY['karta','graficzna'], ARRAY['grafika']),
    (ARRAY['ekspres','do','kawy'], ARRAY['ekspres']),
    (ARRAY['pilot','zdalnego','sterowania'], ARRAY['pilot']),
    (ARRAY['buty','sportowe'], ARRAY['sneakersy']),
    (ARRAY['buty','do','biegania'], ARRAY['buty','biegowe']),
    (ARRAY['karma','dla','psów'], ARRAY['karma','psia']),
    (ARRAY['przysmaki','dla','psów'], ARRAY['smakołyki']),
    (ARRAY['przysmaki','dla','kotów'], ARRAY['smakołyki']),
    (ARRAY['bielizna','termoaktywna'], ARRAY['termobielizna'])
) AS v(a, b)
WHERE NOT EXISTS (SELECT 1 FROM synonym_rules)
"#;

/// One enabled synonym rule — raw words, both sides token sequences.
pub struct SynonymRule {
    pub id: i64,
    pub terms_a: Vec<String>,
    pub terms_b: Vec<String>,
}

/// All enabled rules (spec index-rs-dynamic-synonyms W2 loading).
pub async fn load_synonym_rules(pool: &PgPool) -> Result<Vec<SynonymRule>, String> {
    let rows = sqlx::query_as::<_, (i64, Vec<String>, Vec<String>)>(
        "SELECT id, terms_a, terms_b FROM synonym_rules WHERE enabled ORDER BY id",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("load synonym_rules: {e}"))?;
    Ok(rows
        .into_iter()
        .map(|(id, terms_a, terms_b)| SynonymRule {
            id,
            terms_a,
            terms_b,
        })
        .collect())
}

/// Insert a rule and return its id.
pub async fn insert_synonym_rule(
    pool: &PgPool,
    terms_a: &[String],
    terms_b: &[String],
    category: Option<&str>,
) -> Result<i64, String> {
    let id = sqlx::query_scalar::<_, i64>(
        "INSERT INTO synonym_rules (terms_a, terms_b, category) VALUES ($1, $2, $3) RETURNING id",
    )
    .bind(terms_a)
    .bind(terms_b)
    .bind(category)
    .fetch_one(pool)
    .await
    .map_err(|e| format!("insert synonym_rules: {e}"))?;
    Ok(id)
}

/// Delete a rule; returns whether it existed.
pub async fn delete_synonym_rule(pool: &PgPool, id: i64) -> Result<bool, String> {
    let n = sqlx::query("DELETE FROM synonym_rules WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await
        .map_err(|e| format!("delete synonym_rules: {e}"))?;
    Ok(n.rows_affected() > 0)
}

/// Mining dedup (spec index-rs-llm-synonym-mining): terms already covered
/// by any rule side (either side of any pair) — skip mining them again.
pub async fn covered_synonym_terms(pool: &PgPool) -> Result<Vec<String>, String> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT unnest(terms_a) FROM synonym_rules UNION SELECT DISTINCT unnest(terms_b) FROM synonym_rules",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("covered synonym terms: {e}"))?;
    Ok(rows.into_iter().map(|(t,)| t).collect())
}

/// Mark a term as mined (attempted), regardless of LLM result.
pub async fn log_synonym_gen(pool: &PgPool, term: &str) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO synonym_gen_log (term) VALUES ($1) ON CONFLICT (term) DO UPDATE SET last_run = now()",
    )
    .bind(term)
    .execute(pool)
    .await
    .map_err(|e| format!("log synonym gen: {e}"))?;
    Ok(())
}

/// Load the synonym dictionary for an engine build (spec
/// index-rs-dynamic-synonyms W5) — errors degrade to an empty dict
/// (the engine must still build).
pub async fn load_synonym_dict(pool: &PgPool) -> Result<crate::engine::SynonymDict, String> {
    let rules = load_synonym_rules(pool).await?;
    Ok(crate::engine::SynonymDict::from_raw(
        rules.into_iter().map(|r| (r.terms_a, r.terms_b)),
    ))
}

/// Terms already attempted by the miner.
pub async fn mined_synonym_terms(pool: &PgPool) -> Result<Vec<String>, String> {
    let rows: Vec<(String,)> = sqlx::query_as("SELECT term FROM synonym_gen_log")
        .fetch_all(pool)
        .await
        .map_err(|e| format!("mined synonym terms: {e}"))?;
    Ok(rows.into_iter().map(|(t,)| t).collect())
}

pub async fn connect(url: &str) -> Result<PgPool, String> {
    PgPoolOptions::new()
        .max_connections(4)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .connect(url)
        .await
        .map_err(|e| format!("connect postgres: {e}"))
}

/// All rows → corpus docs (ordered by slug for deterministic builds).
pub async fn load_all(pool: &PgPool) -> Result<Vec<EngineDoc>, String> {
    let rows = sqlx::query_as::<_, (String, serde_json::Value, Option<Vec<u8>>)>(
        "SELECT slug, doc, embedding FROM catalog_documents ORDER BY slug",
    )
    .fetch_all(pool)
    .await
    .map_err(|e| format!("load catalog_documents: {e}"))?;
    Ok(rows
        .into_iter()
        .map(|(slug, doc, emb)| EngineDoc {
            slug,
            doc,
            embedding: emb.map(|b| bytes_to_vec(&b)),
        })
        .collect())
}

/// Rows per multi-row INSERT + COMMIT during `--backfill`. One round trip
/// per batch instead of one per row (34,925 docs → ~140 transactions).
pub const BACKFILL_BATCH: usize = 250;

/// Build the multi-row upsert SQL for `n` rows
/// (`VALUES ($1,$2,$3),($4,$5,$6),...` + ON CONFLICT clause).
fn multi_upsert_sql(n: usize, force: bool) -> String {
    let mut sql = String::from("INSERT INTO catalog_documents (slug, doc, embedding) VALUES ");
    for i in 0..n {
        if i > 0 {
            sql.push(',');
        }
        let b = i * 3 + 1;
        sql.push_str(&format!("(${b}, ${}, ${})", b + 1, b + 2));
    }
    if force {
        sql.push_str(
            " ON CONFLICT (slug) DO UPDATE SET doc = EXCLUDED.doc, embedding = EXCLUDED.embedding, updated_at = now()",
        );
    } else {
        sql.push_str(" ON CONFLICT (slug) DO NOTHING");
    }
    sql
}

/// Multi-row batch upsert (backfill path): ONE statement + ONE commit per
/// batch. `rows_affected` counts inserted + updated (DO UPDATE) or only
/// inserted (DO NOTHING) — same semantics as the old per-row sum.
pub async fn upsert_multi(pool: &PgPool, docs: &[EngineDoc], force: bool) -> Result<usize, String> {
    if docs.is_empty() {
        return Ok(0);
    }
    let sql = multi_upsert_sql(docs.len(), force);
    let mut q = sqlx::query(&sql);
    for d in docs {
        let emb: Option<Vec<u8>> = d.embedding.as_ref().map(|v| vec_to_bytes(v));
        q = q.bind(&d.slug).bind(&d.doc).bind(emb);
    }
    let res = q
        .execute(pool)
        .await
        .map_err(|e| format!("multi upsert ({} docs): {e}", docs.len()))?;
    Ok(res.rows_affected() as usize)
}

/// Buffered chunking for the backfill scroll: emits full batches of `cap`
/// docs across scroll-page boundaries; the caller flushes the final partial
/// batch with [`BatchBuffer::take`].
pub struct BatchBuffer {
    cap: usize,
    buf: Vec<EngineDoc>,
}

impl BatchBuffer {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            buf: Vec::new(),
        }
    }

    /// Push a doc; returns `Some(full_batch)` when the buffer reached `cap`.
    pub fn push(&mut self, d: EngineDoc) -> Option<Vec<EngineDoc>> {
        self.buf.push(d);
        if self.buf.len() >= self.cap {
            Some(std::mem::take(&mut self.buf))
        } else {
            None
        }
    }

    /// Flush the final (possibly partial) batch.
    pub fn take(&mut self) -> Vec<EngineDoc> {
        std::mem::take(&mut self.buf)
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

/// Parse one OpenSearch scroll hit → EngineDoc (`_id` wins over `source.slug`;
/// the `embedding` float array is pulled out of the doc into the bytea side).
/// Returns None when the hit carries no usable id.
pub fn hit_to_doc(hit: &serde_json::Value) -> Option<EngineDoc> {
    let source = hit.get("_source")?;
    let slug = hit
        .get("_id")
        .and_then(|i| i.as_str())
        .or_else(|| source.get("slug").and_then(|s| s.as_str()))
        .map(String::from);
    let slug = slug.filter(|s| !s.is_empty())?;
    let mut doc = source.clone();
    // The embedding travels as a float array in _source; pull it out
    // (it is NOT part of the indexed doc payload — stored separately).
    let embedding = doc.get("embedding").and_then(|e| e.as_array()).map(|a| {
        a.iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<f32>>()
    });
    if let Some(obj) = doc.as_object_mut() {
        obj.remove("embedding");
    }
    Some(EngineDoc {
        slug,
        doc,
        embedding,
    })
}

/// Single-doc upsert (the write path). `embedding == None` keeps the stored
/// embedding (partial-update semantics — .NET doc merges never carry it);
/// `version == None` keeps the stored external version.
pub async fn upsert_doc(
    pool: &PgPool,
    slug: &str,
    doc: &serde_json::Value,
    embedding: Option<&[u8]>,
    version: Option<i64>,
) -> Result<(), String> {
    sqlx::query(
        r#"INSERT INTO catalog_documents (slug, doc, embedding, version) VALUES ($1, $2, $3, $4)
           ON CONFLICT (slug) DO UPDATE SET
             doc = EXCLUDED.doc,
             embedding = COALESCE(EXCLUDED.embedding, catalog_documents.embedding),
             version = COALESCE(EXCLUDED.version, catalog_documents.version),
             updated_at = now()"#,
    )
    .bind(slug)
    .bind(doc)
    .bind(embedding)
    .bind(version)
    .execute(pool)
    .await
    .map_err(|e| format!("upsert doc {slug}: {e}"))?;
    Ok(())
}

/// Single-doc hard delete. Returns false when the row was not present.
pub async fn delete_doc(pool: &PgPool, slug: &str) -> Result<bool, String> {
    let res = sqlx::query("DELETE FROM catalog_documents WHERE slug = $1")
        .bind(slug)
        .execute(pool)
        .await
        .map_err(|e| format!("delete doc {slug}: {e}"))?;
    Ok(res.rows_affected() > 0)
}

/// Stored external version of a doc (bulk version_type=external checks).
pub async fn get_version(pool: &PgPool, slug: &str) -> Result<Option<i64>, String> {
    let row: Option<(Option<i64>,)> =
        sqlx::query_as("SELECT version FROM catalog_documents WHERE slug = $1")
            .bind(slug)
            .fetch_optional(pool)
            .await
            .map_err(|e| format!("get version {slug}: {e}"))?;
    Ok(row.and_then(|(v,)| v))
}

/// Scroll the existing OpenSearch index (backfill source). Returns
/// (upserted, scanned).
pub async fn backfill(
    pool: &PgPool,
    os_base: &str,
    os_index: &str,
    os_user: Option<&str>,
    os_pass: Option<&str>,
    force: bool,
) -> Result<(usize, usize), String> {
    let http = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .read_timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let base = os_base.trim_end_matches('/');
    let mut scroll_id: Option<String> = None;
    let mut scanned = 0usize;
    let mut upserted = 0usize;
    let mut buf = BatchBuffer::new(BACKFILL_BATCH);
    loop {
        let (url, body) = match &scroll_id {
            None => (
                format!("{base}/{os_index}/_search?scroll=2m"),
                serde_json::json!({ "size": 500, "query": { "match_all": {} } }),
            ),
            Some(sid) => (
                format!("{base}/_search/scroll"),
                serde_json::json!({ "scroll": "2m", "scroll_id": sid }),
            ),
        };
        let mut req = http.post(&url).json(&body);
        if let (Some(u), Some(p)) = (os_user, os_pass) {
            req = req.basic_auth(u, Some(p));
        }
        let resp = req
            .send()
            .await
            .map_err(|e| format!("opensearch scroll: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!(
                "opensearch scroll status {status}: {}",
                &text[..text.len().min(300)]
            ));
        }
        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("opensearch scroll decode: {e}"))?;
        let hits = data
            .pointer("/hits/hits")
            .and_then(|h| h.as_array())
            .cloned()
            .unwrap_or_default();
        if hits.is_empty() {
            break;
        }
        for hit in &hits {
            if let Some(d) = hit_to_doc(hit) {
                scanned += 1;
                if let Some(full) = buf.push(d) {
                    upserted += upsert_multi(pool, &full, force).await?;
                    tracing::info!(scanned, upserted, "backfill progress (batch committed)");
                }
            }
        }
        scroll_id = data
            .get("_scroll_id")
            .and_then(|s| s.as_str())
            .map(String::from);
        if scroll_id.is_none() {
            break;
        }
    }
    // final partial batch — one last flush + commit
    let rest = buf.take();
    if !rest.is_empty() {
        upserted += upsert_multi(pool, &rest, force).await?;
        tracing::info!(
            scanned,
            upserted,
            "backfill progress (final batch committed)"
        );
    }
    if let Some(sid) = scroll_id {
        let _ = http
            .post(format!("{base}/_search/scroll"))
            .json(&serde_json::json!({ "scroll_id": sid }))
            .send()
            .await;
    }
    Ok((upserted, scanned))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_bytes_roundtrip() {
        let v = vec![0.5f32, -1.25, 3.0, 0.0];
        let b = vec_to_bytes(&v);
        assert_eq!(b.len(), 16);
        assert_eq!(bytes_to_vec(&b), v);
    }

    fn edoc(slug: &str) -> EngineDoc {
        EngineDoc {
            slug: slug.into(),
            doc: serde_json::json!({ "name": slug }),
            embedding: None,
        }
    }

    #[test]
    fn batch_buffer_flushes_at_capacity() {
        let mut buf = BatchBuffer::new(3);
        assert!(buf.push(edoc("a")).is_none());
        assert!(buf.push(edoc("b")).is_none());
        let full = buf.push(edoc("c")).expect("flush at cap");
        assert_eq!(full.len(), 3);
        assert_eq!(full[0].slug, "a");
        assert!(buf.is_empty());
        // buffer restarts cleanly after a flush
        assert!(buf.push(edoc("d")).is_none());
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn batch_buffer_exact_multiple_of_capacity() {
        // 6 docs, cap 3 → two full flushes, empty final take()
        let mut buf = BatchBuffer::new(3);
        let mut flushed = 0usize;
        for i in 0..6 {
            if buf.push(edoc(&format!("d{i}"))).is_some() {
                flushed += 1;
            }
        }
        assert_eq!(flushed, 2);
        assert!(buf.take().is_empty());
    }

    #[test]
    fn batch_buffer_final_partial_batch() {
        // 5 docs, cap 3 → one full flush + partial take of 2
        let mut buf = BatchBuffer::new(3);
        let mut total = 0usize;
        for i in 0..5 {
            if let Some(full) = buf.push(edoc(&format!("d{i}"))) {
                total += full.len();
            }
        }
        assert_eq!(total, 3);
        let rest = buf.take();
        assert_eq!(rest.len(), 2);
        assert_eq!(rest[0].slug, "d3");
        assert!(buf.take().is_empty());
    }

    #[test]
    fn batch_buffer_chunks_across_scroll_pages() {
        // scroll pages of 500 with cap 250 → full batches mid-page,
        // never a flush per page boundary
        let mut buf = BatchBuffer::new(250);
        let mut batches = 0usize;
        for p in 0..2 {
            for i in 0..500 {
                let d = edoc(&format!("p{p}-{i}"));
                if buf.push(d).is_some() {
                    batches += 1;
                }
            }
        }
        assert_eq!(batches, 4);
        assert!(buf.take().is_empty());
    }

    #[test]
    fn multi_upsert_sql_shape() {
        let force = multi_upsert_sql(2, true);
        assert!(force.starts_with(
            "INSERT INTO catalog_documents (slug, doc, embedding) VALUES ($1, $2, $3),($4, $5, $6)"
        ));
        assert!(force.ends_with(
            "ON CONFLICT (slug) DO UPDATE SET doc = EXCLUDED.doc, embedding = EXCLUDED.embedding, updated_at = now()"
        ));
        let skip = multi_upsert_sql(3, false);
        assert!(skip.contains("($7, $8, $9)"));
        assert!(skip.ends_with("ON CONFLICT (slug) DO NOTHING"));
    }

    #[test]
    fn hit_to_doc_parses_id_and_extract_embedding() {
        let hit = serde_json::json!({
            "_id": "asics-sonicblast",
            "_source": { "slug": "wrong", "name": "Shoe", "embedding": [1.0, 2.5] }
        });
        let d = hit_to_doc(&hit).expect("doc");
        assert_eq!(d.slug, "asics-sonicblast");
        assert_eq!(d.embedding, Some(vec![1.0f32, 2.5]));
        assert!(d.doc.get("embedding").is_none(), "embedding stripped");
        assert_eq!(d.doc["name"], "Shoe");
    }

    #[test]
    fn hit_to_doc_falls_back_to_source_slug_and_skips_bad_hits() {
        let hit = serde_json::json!({ "_source": { "slug": "from-source" } });
        assert_eq!(hit_to_doc(&hit).expect("doc").slug, "from-source");
        // no _source at all
        assert!(hit_to_doc(&serde_json::json!({ "_id": "x" })).is_none());
        // no usable id
        assert!(hit_to_doc(&serde_json::json!({ "_source": { "name": "x" } })).is_none());
        // empty id
        assert!(hit_to_doc(&serde_json::json!({ "_id": "", "_source": {} })).is_none());
    }
}
