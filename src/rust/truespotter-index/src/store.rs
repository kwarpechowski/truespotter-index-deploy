//! Postgres `catalog_documents` — the source of truth for the embedded
//! index — plus the `--backfill` mode: scroll the EXISTING OpenSearch
//! `products` index (OPENSEARCH_URL) and upsert every doc (embedding
//! array → LE f32 bytea) into Postgres. Existing rows are skipped unless
//! `--force`.

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
    Ok(())
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

pub async fn upsert_batch(pool: &PgPool, docs: &[EngineDoc], force: bool) -> Result<usize, String> {
    let mut tx = pool
        .begin()
        .await
        .map_err(|e| format!("begin upsert: {e}"))?;
    let mut n = 0usize;
    for d in docs {
        let emb: Option<Vec<u8>> = d.embedding.as_ref().map(|v| vec_to_bytes(v));
        let sql = if force {
            "INSERT INTO catalog_documents (slug, doc, embedding) VALUES ($1, $2, $3)
             ON CONFLICT (slug) DO UPDATE SET doc = EXCLUDED.doc, embedding = EXCLUDED.embedding, updated_at = now()"
        } else {
            "INSERT INTO catalog_documents (slug, doc, embedding) VALUES ($1, $2, $3)
             ON CONFLICT (slug) DO NOTHING"
        };
        let res = sqlx::query(sql)
            .bind(&d.slug)
            .bind(&d.doc)
            .bind(emb)
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("upsert {}: {e}", d.slug))?;
        n += res.rows_affected() as usize;
    }
    tx.commit()
        .await
        .map_err(|e| format!("commit upsert: {e}"))?;
    Ok(n)
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
        let mut batch: Vec<EngineDoc> = Vec::with_capacity(hits.len());
        for hit in &hits {
            let Some(source) = hit.get("_source") else {
                continue;
            };
            let slug = hit
                .get("_id")
                .and_then(|i| i.as_str())
                .or_else(|| source.get("slug").and_then(|s| s.as_str()))
                .map(String::from);
            let Some(slug) = slug.filter(|s| !s.is_empty()) else {
                continue;
            };
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
            batch.push(EngineDoc {
                slug,
                doc,
                embedding,
            });
            scanned += 1;
        }
        upserted += upsert_batch(pool, &batch, force).await?;
        scroll_id = data
            .get("_scroll_id")
            .and_then(|s| s.as_str())
            .map(String::from);
        if scroll_id.is_none() {
            break;
        }
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
}
