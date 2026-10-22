//! The OpenSearch-wire HTTP API (axum). Response shapes are 1:1 with the
//! OpenSearch subset our clients parse (hits.total.value, hits.hits[]._source
//! verbatim minus _source.excludes, hits.hits[].sort, aggregations incl.
//! the `filtered` wrapper, GET /{index}/_doc/{id}, GET /{index}/_count,
//! and — since the write API landed — _update, _update_by_query, _doc,
//! _bulk, _delete_by_query, _refresh, _mapping, _mget, scroll).
//! Unknown DSL constructs → an OpenSearch-shaped 400 parsing_exception.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Map, Value};
use tokio::sync::{Mutex as AsyncMutex, RwLock};

use crate::engine::{Engine, Hit, SearchOutcome};

pub struct ScrollEntry {
    pub index: String,
    pub remaining: Vec<Value>,
    pub page: usize,
}

pub type ScrollRegistry = std::collections::HashMap<String, ScrollEntry>;

pub struct AppState {
    pub engine: RwLock<Arc<Engine>>,
    pub pool: Option<sqlx::PgPool>,
    pub rebuild_secret: Option<String>,
    /// serializes writers (PG write + engine mutation per request)
    pub write_lock: AsyncMutex<()>,
    /// live scroll contexts (POST /_search/scroll continuation)
    pub scrolls: std::sync::Mutex<ScrollRegistry>,
    /// set after every write — the background task rebuilds the vector
    /// matrix (full engine rebuild from Postgres) with debounce
    pub vectors_dirty: std::sync::atomic::AtomicBool,
}

impl AppState {
    pub fn mark_vectors_dirty(&self) {
        self.vectors_dirty
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// OpenSearch-shaped parsing error (status 400).
fn parsing_error(reason: &str, index: &str) -> Response {
    let body = json!({
        "error": {
            "root_cause": [{
                "type": "parsing_exception",
                "reason": reason,
                "index": index
            }],
            "type": "parsing_exception",
            "reason": reason,
            "index": index
        },
        "status": 400
    });
    (StatusCode::BAD_REQUEST, Json(body)).into_response()
}

fn shards() -> Value {
    json!({ "total": 1, "successful": 1, "skipped": 0, "failed": 0 })
}

pub fn hit_entry(index: &str, h: &Hit) -> Value {
    let mut o = Map::new();
    o.insert("_index".into(), json!(index));
    o.insert("_id".into(), json!(h.slug));
    o.insert(
        "_score".into(),
        match h.score {
            Some(s) => json!(s),
            None => Value::Null,
        },
    );
    o.insert("_source".into(), h.source.clone());
    if !h.sort.is_empty() {
        o.insert("sort".into(), json!(h.sort));
    }
    Value::Object(o)
}

pub fn search_response(index: &str, outcome: &SearchOutcome) -> Value {
    let hits: Vec<Value> = outcome.hits.iter().map(|h| hit_entry(index, h)).collect();
    let max_score = outcome
        .hits
        .iter()
        .filter_map(|h| h.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut body = Map::new();
    body.insert("took".into(), json!(outcome.took_ms));
    body.insert("timed_out".into(), json!(false));
    body.insert("_shards".into(), shards());
    body.insert(
        "hits".into(),
        json!({
            "total": { "value": outcome.total, "relation": "eq" },
            "max_score": if max_score.is_finite() { json!(max_score) } else { Value::Null },
            "hits": hits
        }),
    );
    if let Some(aggs) = &outcome.aggregations {
        body.insert("aggregations".into(), aggs.clone());
    }
    Value::Object(body)
}

#[derive(serde::Deserialize, Default)]
pub struct SearchParams {
    pub scroll: Option<String>,
}

pub async fn search_handler(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    Query(params): Query<SearchParams>,
    Json(body): Json<Value>,
) -> Response {
    let engine = state.engine.read().await.clone();
    let req = match crate::dsn::parse_search_body(&body) {
        Ok(r) => r,
        Err(e) => return parsing_error(&e.0, &index),
    };
    let scroll_requested = params.scroll.is_some();
    // Scroll: materialize the whole result set once (capped), page it out
    // of the scroll context. Non-scroll requests behave exactly as before.
    let mut req = req;
    let page = req.size;
    if scroll_requested {
        req.size = 100_000;
    }
    match engine.search(&req) {
        Ok(outcome) => {
            if !scroll_requested {
                return Json(search_response(&index, &outcome)).into_response();
            }
            let all: Vec<Value> = outcome.hits.iter().map(|h| hit_entry(&index, h)).collect();
            static SCROLL_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
            let scroll_id = format!(
                "ts-scroll-{}",
                SCROLL_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            );
            let (first, rest) = all.split_at(page.min(all.len()));
            state.scrolls.lock().expect("scrolls lock").insert(
                scroll_id.clone(),
                ScrollEntry {
                    index: index.clone(),
                    remaining: rest.to_vec(),
                    page,
                },
            );
            let mut resp = search_response(&index, &outcome);
            if let Value::Object(o) = &mut resp {
                o.insert("_scroll_id".into(), json!(scroll_id));
                if let Some(h) = o.get_mut("hits").and_then(|h| h.as_object_mut()) {
                    h.insert("hits".into(), json!(first.to_vec()));
                }
            }
            Json(resp).into_response()
        }
        Err(e) => parsing_error(&e, &index),
    }
}

/// POST /_search/scroll — continuation page.
pub async fn scroll_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    let Some(scroll_id) = body.get("scroll_id").and_then(|s| s.as_str()) else {
        return parsing_error("scroll_id is required", "_all");
    };
    let _index;
    let page_hits: Vec<Value>;
    {
        let mut scrolls = state.scrolls.lock().expect("scrolls lock");
        let Some(ctx) = scrolls.get_mut(scroll_id) else {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": {
                        "root_cause": [{
                            "type": "search_context_missing_exception",
                            "reason": format!("No search context found for id {scroll_id}")
                        }],
                        "type": "search_context_missing_exception",
                        "reason": format!("No search context found for id {scroll_id}")
                    },
                    "status": 404
                })),
            )
                .into_response();
        };
        _index = ctx.index.clone();
        let n = ctx.page.min(ctx.remaining.len());
        page_hits = ctx.remaining.drain(..n).collect();
        if ctx.remaining.is_empty() {
            scrolls.remove(scroll_id);
        }
    }
    let remaining = state
        .scrolls
        .lock()
        .expect("scrolls lock")
        .get(scroll_id)
        .is_some();
    let mut resp = Map::new();
    resp.insert("took".into(), json!(1));
    resp.insert("timed_out".into(), json!(false));
    resp.insert("_shards".into(), shards());
    resp.insert(
        "hits".into(),
        json!({
            "total": { "value": page_hits.len(), "relation": "eq" },
            "max_score": Value::Null,
            "hits": page_hits
        }),
    );
    if remaining {
        resp.insert("_scroll_id".into(), json!(scroll_id));
    }
    Json(Value::Object(resp)).into_response()
}

/// DELETE /_search/scroll?scroll_id=... (or body {scroll_id}) — release.
pub async fn scroll_delete_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
    body: Option<Json<Value>>,
) -> Response {
    let from_query = params.get("scroll_id").cloned();
    let from_body = body.and_then(|Json(b)| {
        b.get("scroll_id")
            .and_then(|s| s.as_str())
            .map(String::from)
    });
    let mut scrolls = state.scrolls.lock().expect("scrolls lock");
    let freed = match from_query.or(from_body) {
        Some(id) if !id.is_empty() => usize::from(scrolls.remove(&id).is_some()),
        _ => {
            let n = scrolls.len();
            scrolls.clear();
            n
        }
    };
    Json(json!({ "succeeded": true, "num_freed": freed })).into_response()
}

pub async fn doc_handler(
    State(state): State<Arc<AppState>>,
    Path((index, id)): Path<(String, String)>,
) -> Response {
    let engine = state.engine.read().await.clone();
    match engine.get_by_id(&id) {
        Some(source) => Json(json!({
            "_index": index,
            "_id": id,
            "_version": 1,
            "_seq_no": 0,
            "_primary_term": 1,
            "found": true,
            "_source": source
        }))
        .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "_index": index,
                "_id": id,
                "found": false
            })),
        )
            .into_response(),
    }
}

pub async fn count_handler(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    body: Option<Json<Value>>,
) -> Response {
    let engine = state.engine.read().await.clone();
    let count = match body.map(|Json(b)| b) {
        Some(body) => {
            if let Ok(req) = crate::dsn::parse_search_body(&body) {
                match engine.search(&crate::dsn::SearchRequest {
                    size: 0,
                    from: 0,
                    source_excludes: vec![],
                    query: req.query,
                    sort: vec![],
                    search_after: None,
                    aggs: vec![],
                }) {
                    Ok(o) => o.total,
                    Err(e) => return parsing_error(&e, &index),
                }
            } else {
                engine.num_docs()
            }
        }
        None => engine.num_docs(),
    };
    Json(json!({ "count": count, "_shards": shards() })).into_response()
}

pub async fn rebuild_handler(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let expected = state.rebuild_secret.clone();
    let given = headers
        .get("x-rebuild-secret")
        .and_then(|v| v.to_str().ok());
    match (&expected, given) {
        (Some(exp), Some(g)) if exp == g => {}
        (None, _) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": "REBUILD_SECRET is not configured" })),
            )
                .into_response()
        }
        _ => {
            return (
                StatusCode::FORBIDDEN,
                Json(json!({ "error": "invalid x-rebuild-secret" })),
            )
                .into_response()
        }
    }
    let Some(pool) = state.pool.clone() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "no database configured" })),
        )
            .into_response();
    };
    // Load → build (blocking) → swap into the shared engine lock.
    let docs = match crate::store::load_all(&pool).await {
        Ok(d) => d,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "rebuild": false, "error": e })),
            )
                .into_response()
        }
    };
    let n = docs.len();
    let data_dir = std::env::var("DATA_DIR").map(std::path::PathBuf::from).ok();
    let engine =
        match tokio::task::spawn_blocking(move || crate::engine::Engine::build(docs, data_dir))
            .await
        {
            Ok(Ok(e)) => e,
            Ok(Err(e)) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "rebuild": false, "error": e })),
                )
                    .into_response()
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "rebuild": false, "error": e.to_string() })),
                )
                    .into_response()
            }
        };
    *state.engine.write().await = Arc::new(engine);
    Json(json!({ "rebuild": true, "docs": n })).into_response()
}

pub async fn health_handler(State(state): State<Arc<AppState>>) -> Response {
    let engine = state.engine.read().await.clone();
    Json(json!({ "status": "ok", "docs": engine.num_docs() })).into_response()
}

pub async fn root_handler() -> Response {
    Json(json!({
        "name": "truespotter-index",
        "cluster_name": "truespotter",
        "version": { "number": "2.18.0-index", "distribution": "truespotter-index" },
        "tagline": "You Know, for Search"
    }))
    .into_response()
}

// ---------------------------------------------------------------------------
// WRITE API - OpenSearch replacement for the .NET side (raw-HttpClient
// repositories + Temporal worker activities; they keep working UNCHANGED
// with OPENSEARCH_URL pointed here). Durability: every write goes to
// Postgres catalog_documents FIRST, then the live tantivy index (delete
// term slug + re-add + commit). The vector matrix is rebuilt in the
// background (dirty flag + debounce) - see main.rs.
// ---------------------------------------------------------------------------

/// OpenSearch-shaped generic error.
fn os_error(status: StatusCode, err_type: &str, reason: &str, index: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "root_cause": [{ "type": err_type, "reason": reason, "index": index }],
                "type": err_type,
                "reason": reason,
                "index": index
            },
            "status": status.as_u16()
        })),
    )
        .into_response()
}

fn document_missing(index: &str, id: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "root_cause": [{
                    "type": "document_missing_exception",
                    "reason": format!("[{index}][{id}]: document missing"),
                    "index_uuid": "na",
                    "shard": "0",
                    "index": index
                }],
                "type": "document_missing_exception",
                "reason": format!("[{index}][{id}]: document missing"),
                "index_uuid": "na",
                "shard": "0",
                "index": index
            },
            "status": 404
        })),
    )
        .into_response()
}

fn script_error(reason: &str, index: &str) -> Response {
    os_error(
        StatusCode::BAD_REQUEST,
        "script_compile_exception",
        reason,
        index,
    )
}

pub fn write_response(index: &str, id: &str, result: &str) -> Value {
    json!({
        "_index": index,
        "_id": id,
        "_version": 1,
        "result": result,
        "_shards": shards()
    })
}

/// Ensure the doc carries its own slug (search-by-slug consistency).
fn ensure_slug(doc: &mut Map<String, Value>, id: &str) {
    doc.entry("slug".to_string()).or_insert_with(|| json!(id));
}

/// Persist a computed update to Postgres (source of truth) - embedding None
/// keeps the stored one (partial-update semantics).
async fn persist_upsert(
    state: &AppState,
    slug: &str,
    doc: &Map<String, Value>,
    embedding: Option<&[f32]>,
    version: Option<i64>,
) -> Result<(), String> {
    let Some(pool) = state.pool.clone() else {
        return Err("no database configured".into());
    };
    let emb: Option<Vec<u8>> = embedding.map(crate::store::vec_to_bytes);
    crate::store::upsert_doc(
        &pool,
        slug,
        &Value::Object(doc.clone()),
        emb.as_deref(),
        version,
    )
    .await
}

/// POST /{index}/_update/{id} - {doc, doc_as_upsert} merge or {script}.
pub async fn update_handler(
    State(state): State<Arc<AppState>>,
    Path((index, id)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Response {
    let _wl = state.write_lock.lock().await;
    let engine = state.engine.read().await.clone();
    let existing = engine.get_doc(&id).map(|(d, _)| d);
    let outcome = match crate::write::compute_update(existing.as_ref(), &body) {
        Ok(o) => o,
        Err(crate::write::UpdateError::MissingDoc) => return document_missing(&index, &id),
        Err(crate::write::UpdateError::Script(e)) => return script_error(&e, &index),
        Err(crate::write::UpdateError::Parse(e)) => return parsing_error(&e, &index),
    };
    state.mark_vectors_dirty();
    if let Err(e) = persist_upsert(
        &state,
        &id,
        &outcome.doc,
        outcome.embedding.as_deref(),
        None,
    )
    .await
    {
        return os_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable_exception",
            &e,
            &index,
        );
    }
    engine.upsert_doc(&id, Value::Object(outcome.doc.clone()), outcome.embedding);
    Json(write_response(&index, &id, outcome.result)).into_response()
}

/// POST /{index}/_update_by_query[?refresh=true&conflicts=proceed] - query
/// selection (the dsn parser) + script application per matched doc.
/// .NET parses the `updated` field (OpenSearchCatalogCategoryApplier).
pub async fn update_by_query_handler(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let query = match body.get("query") {
        Some(q) => match crate::dsn::parse_query(q) {
            Ok(q) => q,
            Err(e) => return parsing_error(&e.0, &index),
        },
        None => crate::dsn::Query::MatchAll,
    };
    let Some(script) = body.get("script") else {
        return parsing_error("update_by_query requires a script", &index);
    };
    let Some(source) = script.get("source").and_then(|s| s.as_str()) else {
        return parsing_error("script.source must be a string", &index);
    };
    let params = script.get("params").cloned().unwrap_or(json!({}));
    let script_body = json!({ "script": { "source": source, "params": params } });

    let _wl = state.write_lock.lock().await;
    let engine = state.engine.read().await.clone();
    let slugs = match engine.select_slugs(&query) {
        Ok(s) => s,
        Err(e) => return parsing_error(&e, &index),
    };
    let total = slugs.len();
    let mut updated = 0usize;
    let mut failures: Vec<Value> = Vec::new();
    state.mark_vectors_dirty();
    for slug in &slugs {
        let existing = engine.get_doc(slug).map(|(d, _)| d);
        match crate::write::compute_update(existing.as_ref(), &script_body) {
            Ok(outcome) => {
                if let Err(e) = persist_upsert(
                    &state,
                    slug,
                    &outcome.doc,
                    outcome.embedding.as_deref(),
                    None,
                )
                .await
                {
                    failures.push(json!({
                        "type": "unavailable_exception",
                        "reason": e,
                        "index": index,
                        "shard": 0,
                        "node_id": "local"
                    }));
                    continue;
                }
                engine.upsert_doc(slug, Value::Object(outcome.doc.clone()), outcome.embedding);
                updated += 1;
            }
            Err(crate::write::UpdateError::MissingDoc) => {
                failures.push(json!({
                    "type": "document_missing_exception",
                    "reason": format!("[{index}][{slug}]: document missing"),
                    "index": index, "shard": 0, "node_id": "local"
                }));
            }
            Err(e) => {
                failures.push(json!({
                    "type": "script_compile_exception",
                    "reason": format!("{e:?}"),
                    "index": index, "shard": 0, "node_id": "local"
                }));
            }
        }
    }
    Json(json!({
        "took": 1,
        "timed_out": false,
        "total": total,
        "updated": updated,
        "deleted": 0,
        "batches": 1,
        "version_conflicts": 0,
        "noops": 0,
        "failures": failures
    }))
    .into_response()
}

/// POST /{index}/_delete_by_query - selection + count (PG delete first).
pub async fn delete_by_query_handler(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let query = match body.get("query") {
        Some(q) => match crate::dsn::parse_query(q) {
            Ok(q) => q,
            Err(e) => return parsing_error(&e.0, &index),
        },
        None => return parsing_error("delete_by_query requires a query", &index),
    };
    let _wl = state.write_lock.lock().await;
    let engine = state.engine.read().await.clone();
    let slugs = match engine.select_slugs(&query) {
        Ok(s) => s,
        Err(e) => return parsing_error(&e, &index),
    };
    let total = slugs.len();
    let mut deleted = 0usize;
    state.mark_vectors_dirty();
    for slug in &slugs {
        if let Some(pool) = state.pool.clone() {
            if let Err(e) = crate::store::delete_doc(&pool, slug).await {
                return os_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "unavailable_exception",
                    &e,
                    &index,
                );
            }
        }
        if engine.delete_doc(slug) {
            deleted += 1;
        }
    }
    Json(json!({
        "took": 1,
        "timed_out": false,
        "total": total,
        "deleted": deleted,
        "batches": 1,
        "version_conflicts": 0,
        "noops": 0,
        "failures": []
    }))
    .into_response()
}

/// POST/PUT /{index}/_doc/{id} - full index/replace.
pub async fn index_doc_handler(
    State(state): State<Arc<AppState>>,
    Path((index, id)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Response {
    index_doc(state, index, id, body).await
}

/// POST /{index}/_doc - auto-generated id (slug from the doc when present).
pub async fn auto_doc_handler(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let id = body
        .get("slug")
        .and_then(|s| s.as_str())
        .map(String::from)
        .unwrap_or_else(|| format!("auto-{}", std::process::id()));
    index_doc(state, index, id, body).await
}

async fn index_doc(state: Arc<AppState>, index: String, id: String, body: Value) -> Response {
    let _wl = state.write_lock.lock().await;
    let engine = state.engine.read().await.clone();
    let mut doc = match body {
        Value::Object(o) => o,
        _ => return parsing_error("doc must be an object", &index),
    };
    ensure_slug(&mut doc, &id);
    let embedding = crate::write::extract_embedding(&mut doc);
    let existed = engine.get_doc(&id).is_some();
    state.mark_vectors_dirty();
    if let Err(e) = persist_upsert(&state, &id, &doc, embedding.as_deref(), None).await {
        return os_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable_exception",
            &e,
            &index,
        );
    }
    engine.upsert_doc(&id, Value::Object(doc), embedding);
    Json(write_response(
        &index,
        &id,
        if existed { "updated" } else { "created" },
    ))
    .into_response()
}

/// DELETE /{index}/_doc/{id}.
pub async fn delete_doc_handler(
    State(state): State<Arc<AppState>>,
    Path((index, id)): Path<(String, String)>,
) -> Response {
    let _wl = state.write_lock.lock().await;
    let engine = state.engine.read().await.clone();
    if let Some(pool) = state.pool.clone() {
        if let Err(e) = crate::store::delete_doc(&pool, &id).await {
            return os_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "unavailable_exception",
                &e,
                &index,
            );
        }
    }
    let deleted = engine.delete_doc(&id);
    if deleted {
        state.mark_vectors_dirty();
        Json(write_response(&index, &id, "deleted")).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "_index": index,
                "_id": id,
                "_version": 1,
                "result": "not_found",
                "_shards": shards()
            })),
        )
            .into_response()
    }
}

fn bulk_item(op: &str, index: &str, id: &str, status: u16, result: Option<&str>) -> Value {
    let mut o = Map::new();
    o.insert("_index".into(), json!(index));
    o.insert("_id".into(), json!(id));
    o.insert("_version".into(), json!(1));
    if let Some(result) = result {
        o.insert("result".into(), json!(result));
    }
    o.insert("status".into(), json!(status));
    json!({ op: Value::Object(o) })
}

fn bulk_item_error(
    op: &str,
    index: &str,
    id: &str,
    status: u16,
    err_type: &str,
    reason: &str,
) -> Value {
    let mut o = Map::new();
    o.insert("_index".into(), json!(index));
    o.insert("_id".into(), json!(id));
    o.insert("status".into(), json!(status));
    o.insert(
        "error".into(),
        json!({ "type": err_type, "reason": reason, "index": index, "shard": 0 }),
    );
    json!({ op: Value::Object(o) })
}

/// POST /_bulk - NDJSON (the .NET side posts to /{index}/_bulk; both wired).
pub async fn bulk_handler(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    bulk_apply(state, "products".into(), body).await
}

pub async fn index_bulk_handler(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    body: Bytes,
) -> Response {
    bulk_apply(state, index, body).await
}

async fn bulk_apply(state: Arc<AppState>, default_index: String, body: Bytes) -> Response {
    let text = String::from_utf8_lossy(&body);
    let actions = match crate::write::parse_bulk(&text, &default_index) {
        Ok(a) => a,
        Err(e) => return parsing_error(&e, &default_index),
    };
    let _wl = state.write_lock.lock().await;
    let engine = state.engine.read().await.clone();
    let mut items: Vec<Value> = Vec::with_capacity(actions.len());
    let mut had_error = false;
    state.mark_vectors_dirty();
    for action in actions {
        match action {
            crate::write::BulkAction::Index {
                index,
                id,
                version_type,
                version,
                source,
            } => {
                // version_type=external idempotency (KAN-0162): an older
                // re-run must never overwrite a newer doc (item 409).
                if version_type.as_deref() == Some("external") {
                    let stored = match &state.pool {
                        Some(pool) => crate::store::get_version(pool, &id).await,
                        None => Ok(None),
                    };
                    match stored {
                        Ok(Some(cur)) if version.is_some_and(|v| v <= cur) => {
                            had_error = true;
                            items.push(bulk_item_error(
                                "index",
                                &index,
                                &id,
                                409,
                                "version_conflict_engine_exception",
                                &format!(
                                    "[{id}]: version conflict, required [{:?}] (or higher), current version [{cur}]",
                                    version
                                ),
                            ));
                            continue;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            had_error = true;
                            items.push(bulk_item_error(
                                "index",
                                &index,
                                &id,
                                503,
                                "unavailable_exception",
                                &e,
                            ));
                            continue;
                        }
                    }
                }
                let mut doc = match source {
                    Value::Object(o) => o,
                    _ => {
                        had_error = true;
                        items.push(bulk_item_error(
                            "index",
                            &index,
                            &id,
                            400,
                            "parse_exception",
                            "doc must be an object",
                        ));
                        continue;
                    }
                };
                ensure_slug(&mut doc, &id);
                let embedding = crate::write::extract_embedding(&mut doc);
                let existed = engine.get_doc(&id).is_some();
                match persist_upsert(&state, &id, &doc, embedding.as_deref(), version).await {
                    Ok(()) => {
                        engine.upsert_doc(&id, Value::Object(doc), embedding);
                        items.push(bulk_item(
                            "index",
                            &index,
                            &id,
                            if existed { 200 } else { 201 },
                            Some(if existed { "updated" } else { "created" }),
                        ));
                    }
                    Err(e) => {
                        had_error = true;
                        items.push(bulk_item_error(
                            "index",
                            &index,
                            &id,
                            503,
                            "unavailable_exception",
                            &e,
                        ));
                    }
                }
            }
            crate::write::BulkAction::Update { index, id, body } => {
                let existing = engine.get_doc(&id).map(|(d, _)| d);
                match crate::write::compute_update(existing.as_ref(), &body) {
                    Ok(outcome) => {
                        match persist_upsert(
                            &state,
                            &id,
                            &outcome.doc,
                            outcome.embedding.as_deref(),
                            None,
                        )
                        .await
                        {
                            Ok(()) => {
                                engine.upsert_doc(
                                    &id,
                                    Value::Object(outcome.doc.clone()),
                                    outcome.embedding,
                                );
                                items.push(bulk_item("update", &index, &id, 200, Some("updated")));
                            }
                            Err(e) => {
                                had_error = true;
                                items.push(bulk_item_error(
                                    "update",
                                    &index,
                                    &id,
                                    503,
                                    "unavailable_exception",
                                    &e,
                                ));
                            }
                        }
                    }
                    Err(crate::write::UpdateError::MissingDoc) => {
                        had_error = true;
                        items.push(bulk_item_error(
                            "update",
                            &index,
                            &id,
                            404,
                            "document_missing_exception",
                            &format!("[{index}][{id}]: document missing"),
                        ));
                    }
                    Err(crate::write::UpdateError::Script(e)) => {
                        had_error = true;
                        items.push(bulk_item_error(
                            "update",
                            &index,
                            &id,
                            400,
                            "script_compile_exception",
                            &e,
                        ));
                    }
                    Err(crate::write::UpdateError::Parse(e)) => {
                        had_error = true;
                        items.push(bulk_item_error(
                            "update",
                            &index,
                            &id,
                            400,
                            "parse_exception",
                            &e,
                        ));
                    }
                }
            }
            crate::write::BulkAction::Delete { index, id } => {
                if let Some(pool) = state.pool.clone() {
                    if let Err(e) = crate::store::delete_doc(&pool, &id).await {
                        had_error = true;
                        items.push(bulk_item_error(
                            "delete",
                            &index,
                            &id,
                            503,
                            "unavailable_exception",
                            &e,
                        ));
                        continue;
                    }
                }
                if engine.delete_doc(&id) {
                    items.push(bulk_item("delete", &index, &id, 200, Some("deleted")));
                } else {
                    had_error = true;
                    items.push(bulk_item("delete", &index, &id, 404, Some("not_found")));
                }
            }
        }
    }
    Json(json!({ "took": 1, "errors": had_error, "items": items })).into_response()
}

/// POST /{index}/_refresh - no-op (writes are synchronous + committed).
pub async fn refresh_handler(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
) -> Response {
    let engine = state.engine.read().await.clone();
    tracing::debug!(index = %index, docs = engine.num_docs(), "refresh (no-op)");
    Json(json!({ "_shards": shards() })).into_response()
}

/// PUT/POST /{index}/_mapping - no-op (the index-rs schema is fixed; new
/// fields are dynamically present in the stored JSON anyway).
pub async fn mapping_handler(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
) -> Response {
    let _ = (state, index);
    Json(json!({ "acknowledged": true })).into_response()
}

/// POST /{index}/_mget - multi-get (OpenSearchBulkIndexer + EnrichmentActivities).
pub async fn mget_handler(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let engine = state.engine.read().await.clone();
    let mut ids: Vec<String> = Vec::new();
    if let Some(docs) = body.get("docs").and_then(|d| d.as_array()) {
        for d in docs {
            let id = d
                .get("_id")
                .and_then(|v| v.as_str())
                .or_else(|| d.get("slug").and_then(|v| v.as_str()));
            if let Some(id) = id {
                ids.push(id.to_string());
            }
        }
    } else if let Some(raw) = body.get("ids").and_then(|i| i.as_array()) {
        for v in raw {
            if let Some(id) = v.as_str() {
                ids.push(id.to_string());
            }
        }
    }
    let docs: Vec<Value> = ids
        .iter()
        .map(|id| match engine.get_doc(id) {
            Some((source, _)) => json!({
                "_index": index,
                "_id": id,
                "_version": 1,
                "found": true,
                "_source": source
            }),
            None => json!({ "_index": index, "_id": id, "found": false }),
        })
        .collect();
    Json(json!({ "docs": docs })).into_response()
}
