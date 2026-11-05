//! LLM synonym mining (spec index-rs-llm-synonym-mining).
//!
//! GLM generates the dictionary — PG (`synonym_rules`) is the durable
//! cache, the engine's in-RAM `SynonymDict` the hot copy (swapped after
//! each batch, no reindex — expansion is query-time).
//!
//! Loop: mine the engine's title vocabulary (top terms by doc frequency)
//! → skip already-covered/mined terms → ask the LLM for synonyms →
//! validate + persist valid pairs → hot-swap the dict. Runs at boot and
//! then every `SYNONYM_GEN_INTERVAL_SECS` (default 6h); each run processes
//! at most `SYNONYM_GEN_BATCH` terms (default 20).

use std::collections::HashSet;
use std::sync::Arc;

use crate::api::AppState;
use crate::engine::Engine;

pub struct MinerConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub batch: usize,
    pub interval: std::time::Duration,
}

impl MinerConfig {
    /// None when LLM is not configured or mining is explicitly disabled.
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var("LLM_BASE_URL")
            .unwrap_or_default()
            .trim_end_matches('/')
            .to_string();
        let api_key = std::env::var("LLM_API_KEY").unwrap_or_default();
        let model = std::env::var("LLM_MODEL").unwrap_or_default();
        if base_url.is_empty() || api_key.is_empty() || model.is_empty() {
            return None;
        }
        if let Ok(v) = std::env::var("SYNONYM_GEN_ENABLED") {
            if v.eq_ignore_ascii_case("false") || v == "0" {
                tracing::info!("synonym mining disabled via SYNONYM_GEN_ENABLED");
                return None;
            }
        }
        Some(Self {
            base_url,
            api_key,
            model,
            batch: std::env::var("SYNONYM_GEN_BATCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20),
            interval: std::time::Duration::from_secs(
                std::env::var("SYNONYM_GEN_INTERVAL_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(6 * 3600),
            ),
        })
    }
}

fn stem_of(raw: &str) -> String {
    crate::engine::fold_str(raw)
        .split_whitespace()
        .map(|w| crate::engine::stem_pl(w).to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parse the LLM answer into validated raw-word pairs (pure — unit tested).
/// Accepts a bare JSON object with {"pairs": [[side_a, side_b], ...]};
/// markdown fences tolerated. Each side: 1-4 words, 2-40 chars each;
/// sides must differ after stemming.
pub fn parse_llm_pairs(content: &str, term: &str) -> Vec<(Vec<String>, Vec<String>, Option<String>)> {
    // strip markdown code fences if the model added them
    let body = content
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let start = match body.find('{') {
        Some(s) => s,
        None => return Vec::new(),
    };
    let end = match body.rfind('}') {
        Some(e) => e,
        None => return Vec::new(),
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&body[start..=end]) else {
        return Vec::new();
    };
    let Some(arr) = parsed.get("pairs").and_then(|p| p.as_array()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in arr.iter().take(8) {
        let Some(pair) = entry.as_array() else { continue };
        if pair.len() != 2 {
            continue;
        }
        let split_words = |v: &serde_json::Value| -> Option<Vec<String>> {
            let s = v.as_str()?.trim().to_lowercase();
            let words: Vec<String> = s.split_whitespace().map(str::to_string).collect();
            if words.is_empty() || words.len() > 4 {
                return None;
            }
            if words.iter().any(|w| w.len() < 2 || w.len() > 40 || !w.chars().all(char::is_alphanumeric)) {
                return None;
            }
            Some(words)
        };
        let (Some(a), Some(b)) = (split_words(&pair[0]), split_words(&pair[1])) else {
            continue;
        };
        // skip self-synonyms (same stem sequence on both sides)
        if stem_of(&a.join(" ")) == stem_of(&b.join(" ")) {
            continue;
        }
        let _ = term;
        // spec index-rs-category-scoped-synonyms W4: mined rules are global
        out.push((a, b, None));
    }
    out
}

async fn generate_pairs(
    cfg: &MinerConfig,
    term: &str,
) -> Result<Vec<(Vec<String>, Vec<String>, Option<String>)>, String> {
    let http = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .read_timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| e.to_string())?;
    let prompt = format!(
        "Jesteś ekspertem polskich katalogów e-commerce. Dla produktowego terminu \"{term}\" podaj maksymalnie 6 synonimów lub potocznych nazw, których klienci wpisują w wyszukiwarkę sklepu (po polsku, w formie naturalnej z polskimi znakami; dopuszczalne popularne angielskie odpowiedniki). Odpowiedz WYŁĄCZNIE jednym obiektem JSON bez markdownu, w formacie: {{\"pairs\": [[\"{term}\", \"synonim pierwszy\"], [\"{term}\", \"synonim drugi\"]]}}"
    );
    let body = serde_json::json!({
        "model": cfg.model,
        "messages": [{ "role": "user", "content": prompt }],
        "temperature": 0.2,
        "max_tokens": 400,
    });
    let resp = http
        .post(format!("{}/chat/completions", cfg.base_url))
        .bearer_auth(&cfg.api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("LLM status {}", resp.status()));
    }
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let content = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or_default();
    Ok(parse_llm_pairs(content, term))
}

/// One mining pass. Returns the number of NEW rules persisted.
pub async fn run_mining_once(state: &Arc<AppState>, cfg: &MinerConfig) -> usize {
    let Some(pool) = state.pool.clone() else { return 0 };
    let engine: Arc<Engine> = state.engine.read().await.clone();

    // candidate vocabulary: top title terms by doc frequency
    let covered: HashSet<String> = match crate::store::covered_synonym_terms(&pool).await {
        Ok(v) => v.iter().map(|t| stem_of(t)).collect(),
        Err(_) => return 0,
    };
    let mined: HashSet<String> = match crate::store::mined_synonym_terms(&pool).await {
        Ok(v) => v.iter().map(|t| t.clone()).collect(),
        Err(_) => return 0,
    };
    // spec index-rs-query-mining W2: query-log terms FIRST — the
    // vocabulary users actually type (incl. terms absent from the
    // catalog — the biggest recall gaps); title terms only fill the rest.
    let pass_filters = |t: &str| -> bool {
        t.chars().count() >= 5
            && t.chars().all(|c| c.is_ascii_alphanumeric())
            && !covered.contains(t)
            && !mined.contains(t)
    };
    let mut candidates: Vec<String> = Vec::with_capacity(cfg.batch);
    let mut seen: HashSet<String> = HashSet::new();
    // zero-hit terms FIRST (miss desc — their queries return nothing:
    // the catalog does not know the user's word), then frequency
    for (t, count, _miss) in engine.query_terms(500) {
        if count >= 3 && pass_filters(&t) && seen.insert(t.clone()) {
            candidates.push(t);
            if candidates.len() >= cfg.batch {
                break;
            }
        }
    }
    for (t, freq) in engine.top_terms("title", 2000) {
        if candidates.len() >= cfg.batch {
            break;
        }
        if freq >= 20 && pass_filters(&t) && seen.insert(t.clone()) {
            candidates.push(t);
        }
    }

    let mut new_rules = 0usize;
    for term in &candidates {
        match generate_pairs(cfg, term).await {
            Ok(pairs) => {
                let n_pairs = pairs.len();
                for (a, b, category) in pairs {
                    match crate::store::insert_synonym_rule(&pool, &a, &b, category.as_deref()).await {
                        Ok(_) => new_rules += 1,
                        Err(e) => tracing::warn!(error = %e, term = %term, "synonym insert failed"),
                    }
                }
                tracing::info!(term = %term, pairs = n_pairs, "mined synonyms");
            }
            Err(e) => tracing::warn!(error = %e, term = %term, "synonym LLM call failed"),
        }
        // log the attempt regardless of outcome — mining never retries a term
        if let Err(e) = crate::store::log_synonym_gen(&pool, term).await {
            tracing::warn!(error = %e, "synonym gen log failed");
        }
    }

    // hot-swap the dictionary (takes effect on the next query)
    if new_rules > 0 {
        match crate::store::load_synonym_dict(&pool).await {
            Ok(dict) => {
                let n = dict.len();
                engine.set_synonyms(dict);
                tracing::info!(new_rules, total_rules = n, "synonym dictionary hot-swapped");
            }
            Err(e) => tracing::warn!(error = %e, "synonym dict reload failed"),
        }
    }
    new_rules
}

/// Spawn the background mining loop (call once at boot after the engine exists).
pub fn spawn(state: Arc<AppState>) {
    let Some(cfg) = MinerConfig::from_env() else {
        tracing::info!("synonym mining not configured (LLM_BASE_URL/LLM_API_KEY/LLM_MODEL) — skipping");
        return;
    };
    tokio::spawn(async move {
        // let the boot engine build settle first
        tokio::time::sleep(std::time::Duration::from_secs(120)).await;
        loop {
            let added = run_mining_once(&state, &cfg).await;
            tracing::info!(added, "synonym mining pass finished");
            tokio::time::sleep(cfg.interval).await;
        }
    });
}
