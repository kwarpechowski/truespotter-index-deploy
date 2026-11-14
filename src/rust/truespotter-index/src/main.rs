//! truespotter-index — the TrueSpotter embedded search engine (spec
//! rust-embedded-rag). Serves the OpenSearch REST subset the three Rust
//! services consume (catalog-rs / chat-rs / admin-rs switch by pointing
//! OPENSEARCH_URL at this service — wire compatible, no code changes).
//!
//! Modes:
//!   (default)   serve HTTP on PORT (8090) — SERVE-FIRST: the listener
//!               binds and /health answers immediately; the index build
//!               from Postgres runs in a background task and swaps in
//!               atomically (/health reports ready=false until then)
//!   --backfill  [--force]  serve, then scroll the EXISTING OpenSearch
//!               (OPENSEARCH_URL) in the background and upsert docs +
//!               embeddings into catalog_documents (resumable upserts)
//!   --backfill-only  [--force]  one-off backfill without the server:
//!               same flow, exits when done ("backfill complete")
//!   --rebuild             rebuild the embedded index from Postgres
//!   --healthcheck         GET /health self-probe (scratch image HEALTHCHECK)

mod api;
mod dsn;
mod engine;
mod query_cache;
mod store;
mod synonym_gen;
mod write;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct Config {
    pub database_url: Option<String>,
    pub opensearch_url: String,
    pub opensearch_index: String,
    pub opensearch_user: Option<String>,
    pub opensearch_password: Option<String>,
    pub rebuild_secret: Option<String>,
    pub data_dir: std::path::PathBuf,
    pub port: u16,
}

fn non_empty(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            database_url: non_empty("DATABASE_URL").or_else(|| non_empty("POSTGRES_URL")),
            opensearch_url: non_empty("OPENSEARCH_URL")
                .unwrap_or_else(|| "http://opensearch:9200".into()),
            opensearch_index: non_empty("OPENSEARCH_INDEX").unwrap_or_else(|| "products".into()),
            opensearch_user: non_empty("OPENSEARCH_USER"),
            opensearch_password: non_empty("OPENSEARCH_PASSWORD"),
            rebuild_secret: non_empty("REBUILD_SECRET"),
            data_dir: std::path::PathBuf::from(
                non_empty("DATA_DIR").unwrap_or_else(|| "/data".into()),
            ),
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.trim().parse::<u16>().ok())
                .unwrap_or(8090),
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let config = Config::from_env();

    if args.contains(&"--healthcheck".into()) {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/health", config.port);
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => std::process::exit(0),
            _ => std::process::exit(1),
        }
    }

    if args.contains(&"--backfill-only".into()) {
        // One-off backfill without the server: the same flow the serve
        // mode runs in the background — run it, log, exit.
        let db = config
            .database_url
            .clone()
            .expect("DATABASE_URL is required for --backfill-only");
        let pool = store::connect(&db).await.expect("connect postgres");
        store::ensure_table(&pool).await.expect("ensure table");
        let force = args.contains(&"--force".into());
        match store::backfill(
            &pool,
            &config.opensearch_url,
            &config.opensearch_index,
            config.opensearch_user.as_deref(),
            config.opensearch_password.as_deref(),
            force,
        )
        .await
        {
            Ok((upserted, scanned)) => {
                tracing::info!(upserted, scanned, "backfill complete");
                println!("backfill: {upserted} upserted of {scanned} scanned (--force={force})");
            }
            Err(e) => {
                tracing::error!(error = %e, "backfill failed");
                eprintln!("backfill failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.contains(&"--rebuild".into()) {
        let db = config
            .database_url
            .clone()
            .expect("DATABASE_URL is required for --rebuild");
        let pool = store::connect(&db).await.expect("connect postgres");
        store::ensure_table(&pool).await.expect("ensure table");
        match engine_rebuild_standalone(&pool).await {
            Ok(n) => println!("rebuild: {n} docs indexed"),
            Err(e) => {
                eprintln!("rebuild failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // `--backfill` in serve mode: the server boots FIRST (healthcheck-safe),
    // the initial index build + backfill run in the background.
    serve(
        config,
        args.contains(&"--backfill".into()),
        args.contains(&"--force".into()),
    )
    .await;
}

/// `--rebuild` standalone mode: stream → build → report.
async fn engine_rebuild_standalone(pool: &sqlx::PgPool) -> Result<usize, String> {
    let data_dir = std::env::var("DATA_DIR").map(std::path::PathBuf::from).ok();
    let synonyms = store::load_synonym_dict(pool).await.unwrap_or_else(|e| {
        tracing::warn!(error = %e, "synonym dict load failed — building without");
        engine::SynonymDict::empty()
    });
    let (_engine, n) = build_engine_streaming(pool, data_dir, synonyms).await?;
    Ok(n)
}

/// Memory-flat engine build from Postgres (spec index-rs-scale-100x
/// R2/R3): keyset pages flow through a BOUNDED channel into a blocking
/// builder task — backpressure caps in-flight batches, so peak RSS is
/// O(tantivy heap + batch) regardless of corpus size. Both sides of the
/// pipeline surface their errors; a failed DB read aborts the build even
/// if the builder finished on the partial corpus (the engine is discarded).
pub(crate) async fn build_engine_streaming(
    pool: &sqlx::PgPool,
    data_dir: Option<std::path::PathBuf>,
    synonyms: engine::SynonymDict,
) -> Result<(engine::Engine, usize), String> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<engine::EngineDoc>>(2);
    let dd = data_dir.clone();
    let build = tokio::task::spawn_blocking(move || {
        let mut b = engine::Engine::streaming_build(dd, synonyms, engine::heap_budget_bytes())?;
        while let Some(batch) = rx.blocking_recv() {
            b.add_batch(batch)?;
        }
        b.finish()
    });
    // Feed keyset pages into the channel. A send error means the builder
    // already failed — stop early and report the builder's error.
    //
    // OWNERSHIP (prod hang 2026-11-14): tx must be MOVED into the feed
    // closure. When stream_pages consumes and drops the closure, the last
    // sender drops and the builder's blocking_recv returns None. Borrowing
    // tx instead deadlocks: it would stay alive in this frame past
    // build.await, so the builder would wait for a next page forever.
    let fed = {
        let feed = move |page: Vec<engine::EngineDoc>| {
            let tx = tx.clone();
            async move {
                if tx.send(page).await.is_err() {
                    Err("streaming build stopped mid-feed (builder failed)".to_string())
                } else {
                    Ok(())
                }
            }
        };
        store::stream_pages(pool, feed).await
    };
    match build.await {
        Ok(Ok(built)) => match fed {
            Ok(n) => Ok((built, n)),
            Err(e) => Err(e), // DB read failed — discard the partial build
        },
        Ok(Err(e)) => Err(e),
        Err(e) => Err(format!("streaming build join failure: {e}")),
    }
}

/// Background-backfill options (serve mode with `--backfill`).
struct BackfillOpts {
    url: String,
    index: String,
    user: Option<String>,
    pass: Option<String>,
    force: bool,
}

/// Serve-first boot task: (1) STREAM the PG docs into a memory-flat build
/// (spec index-rs-scale-100x R2/R3) → atomically swap into the serving
/// engine and flip `ready`; (2) optionally run the OpenSearch backfill in
/// the background. Neither phase may crash the server — a failed load
/// leaves the empty index serving (ready=false), a failed backfill leaves
/// the partial upserts in place (backfill is resumable — upserts).
fn spawn_boot_builder(
    state: Arc<api::AppState>,
    pool: sqlx::PgPool,
    data_dir: std::path::PathBuf,
    backfill: Option<BackfillOpts>,
) {
    tokio::spawn(async move {
        // phase 1 — initial index build from Postgres (streaming)
        let synonyms = store::load_synonym_dict(&pool).await.unwrap_or_else(|e| {
            tracing::warn!(error = %e, "synonym dict load failed — building without");
            engine::SynonymDict::empty()
        });
        match build_engine_streaming(&pool, Some(data_dir), synonyms).await {
            Ok((built, n)) => {
                *state.engine.write().await = Arc::new(built);
                state
                    .ready
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                if n == 0 {
                    tracing::warn!(
                        "catalog_documents is empty — serving an empty index; --backfill imports OpenSearch"
                    );
                }
                tracing::info!(docs = n, "initial index build complete (ready)");
            }
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "initial index build failed — serving empty index, /health stays ready=false"
                );
                return;
            }
        }

        // phase 2 — background backfill (resumable; failures never kill the server)
        if let Some(bf) = backfill {
            match store::backfill(
                &pool,
                &bf.url,
                &bf.index,
                bf.user.as_deref(),
                bf.pass.as_deref(),
                bf.force,
            )
            .await
            {
                Ok((upserted, scanned)) => {
                    tracing::info!(upserted, scanned, "background backfill complete");
                    // index the newly imported docs: the existing debounced
                    // vector-rebuilder loop reloads from PG and swaps.
                    state.mark_vectors_dirty();
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "background backfill failed — continuing to serve (upserts are resumable)"
                    );
                }
            }
        }
    });
}

async fn serve(config: Config, backfill: bool, force: bool) {
    let Some(db_url) = config.database_url.clone() else {
        eprintln!("DATABASE_URL is required (Postgres catalog_documents is the source of truth)");
        std::process::exit(1);
    };
    let pool = match store::connect(&db_url).await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = store::ensure_table(&pool).await {
        eprintln!("{e}");
        std::process::exit(1);
    }

    // SERVE-FIRST: start with an empty engine. The real index (PG docs →
    // BM25 + vector matrix) is built in a background task and swapped in
    // atomically (the same Arc-swap /admin/rebuild uses). /health answers
    // 200 with ready=false until the swap lands.
    let data_dir = config.data_dir.clone();
    let engine = match tokio::task::spawn_blocking(move || {
        engine::Engine::build(Vec::new(), Some(data_dir))
    })
    .await
    {
        Ok(Ok(e)) => e,
        Ok(Err(e)) => {
            eprintln!("empty engine build failed: {e}");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("empty engine build join failure: {e}");
            std::process::exit(1);
        }
    };

    let state = Arc::new(api::AppState {
        engine: RwLock::new(Arc::new(engine)),
        pool: Some(pool.clone()),
        rebuild_secret: config.rebuild_secret.clone(),
        write_lock: tokio::sync::Mutex::new(()),
        scrolls: std::sync::Mutex::new(std::collections::HashMap::new()),
        search_cache: query_cache::SearchCache::new(),
        vectors_dirty: std::sync::atomic::AtomicBool::new(false),
        ready: std::sync::atomic::AtomicBool::new(false),
    });

    // Initial index build (+ optional backfill) in the background — the
    // listener below binds immediately, errors here never kill the server.
    let backfill_opts = backfill.then(|| {
        // explicit OPENSEARCH_URL is required for the background backfill
        // (the Config default points at a compose-internal hostname).
        non_empty("OPENSEARCH_URL").map(|_| BackfillOpts {
            url: config.opensearch_url.clone(),
            index: config.opensearch_index.clone(),
            user: config.opensearch_user.clone(),
            pass: config.opensearch_password.clone(),
            force,
        })
    });
    if backfill && backfill_opts.is_none() {
        tracing::warn!(
            "--backfill passed but OPENSEARCH_URL is not set — skipping background backfill"
        );
    }
    spawn_boot_builder(
        state.clone(),
        pool,
        config.data_dir.clone(),
        backfill_opts.flatten(),
    );

    // Background vector-matrix rebuild: writes mark the engine dirty; this
    // task debounces and rebuilds from Postgres (the brute-force kNN matrix
    // is not updated incrementally — a full rebuild takes seconds at 29k).
    spawn_vector_rebuilder(state.clone(), config.data_dir.clone());

    // LLM synonym mining (spec index-rs-llm-synonym-mining): mines the
    // title vocabulary, asks GLM for pairs, caches into synonym_rules and
    // hot-swaps the dictionary. No-op when LLM_* is not configured.
    synonym_gen::spawn(state.clone());

    let app = Router::new()
        .route("/", get(api::root_handler))
        .route("/health", get(api::health_handler))
        .route("/admin/rebuild", post(api::rebuild_handler))
        .route("/:index/_synonyms", get(api::synonyms_list_handler))
        .route("/:index/_synonyms", post(api::synonyms_insert_handler))
        .route(
            "/:index/_synonyms/_reload",
            post(api::synonyms_reload_handler),
        )
        .route(
            "/:index/_synonyms/_candidates",
            get(api::synonyms_candidates_handler),
        )
        .route(
            "/:index/_synonyms/:id",
            axum::routing::delete(api::synonyms_delete_handler),
        )
        .route("/:index/_search", post(api::search_handler))
        .route(
            "/_search/scroll",
            post(api::scroll_handler).delete(api::scroll_delete_handler),
        )
        .route(
            "/:index/_count",
            get(api::count_handler).post(api::count_handler),
        )
        .route(
            "/:index/_doc/:id",
            get(api::doc_handler)
                .post(api::index_doc_handler)
                .put(api::index_doc_handler)
                .delete(api::delete_doc_handler),
        )
        .route("/:index/_doc", post(api::auto_doc_handler))
        .route("/:index/_update/:id", post(api::update_handler))
        .route(
            "/:index/_update_by_query",
            post(api::update_by_query_handler),
        )
        .route(
            "/:index/_delete_by_query",
            post(api::delete_by_query_handler),
        )
        .route("/_bulk", post(api::bulk_handler))
        .route("/:index/_bulk", post(api::index_bulk_handler))
        .route("/:index/_refresh", post(api::refresh_handler))
        .route(
            "/:index/_mapping",
            axum::routing::put(api::mapping_handler).post(api::mapping_handler),
        )
        .route("/:index/_mget", post(api::mget_handler))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", config.port);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!("listening on {addr}");
    let _ = axum::serve(listener, app).await;
}

/// Debounced full engine rebuild (vector matrix + tantivy compaction) from
/// Postgres. Runs only when writes marked the engine dirty; if writes land
/// during a rebuild the loop runs again so nothing is lost at the swap.
fn spawn_vector_rebuilder(state: Arc<api::AppState>, data_dir: std::path::PathBuf) {
    tokio::spawn(async move {
        // W2 (spec index-rs-rebuild-churn): the full rebuild is COMPACTION
        // only — pending_vectors serves correct kNN reads until then. Gate
        // by pending overflow (RAM drift) or a minimum interval; without
        // this, routine script traffic (offers/status/category) triggered
        // a ~90s full rebuild every ~90s, saturating the CPU and pushing
        // search latency to seconds (prod measured 2026-11-05).
        let pending_threshold: usize = std::env::var("VECTOR_REBUILD_PENDING_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2000);
        let min_interval = std::time::Duration::from_secs(
            std::env::var("VECTOR_REBUILD_MIN_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(900),
        );
        let mut last_rebuild = std::time::Instant::now();
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            if !state
                .vectors_dirty
                .swap(false, std::sync::atomic::Ordering::Relaxed)
            {
                continue;
            }
            // debounce: coalesce write bursts into one rebuild
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            state
                .vectors_dirty
                .store(false, std::sync::atomic::Ordering::Relaxed);
            // gates: rebuild only when the pending sidecar is large or the
            // minimum interval has passed; otherwise re-queue the flag
            let pending = state.engine.read().await.pending_count();
            if pending < pending_threshold && last_rebuild.elapsed() < min_interval {
                state
                    .vectors_dirty
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                continue;
            }
            let Some(pool) = state.pool.clone() else {
                continue;
            };
            // streaming build (spec index-rs-scale-100x R3) — compaction
            // no longer materializes the corpus
            let synonyms = store::load_synonym_dict(&pool).await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "synonym dict load failed — building without");
                engine::SynonymDict::empty()
            });
            let built = match build_engine_streaming(&pool, Some(data_dir.clone()), synonyms)
                .await
            {
                Ok((e, n)) => {
                    tracing::info!(docs = n, "vector-matrix rebuild complete");
                    e
                }
                Err(e) => {
                    tracing::warn!(error = %e, "vector rebuild failed");
                    state
                        .vectors_dirty
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
            };
            *state.engine.write().await = Arc::new(built);
            last_rebuild = std::time::Instant::now();
        }
    });
}
