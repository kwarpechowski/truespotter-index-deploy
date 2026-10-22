//! OpenSearch write-path semantics:
//!  - `doc` deep-merge (recursive object merge; arrays/scalars/nulls replace)
//!    with `doc_as_upsert` upsert semantics,
//!  - a PATTERN INTERPRETER for the exact Painless update scripts the .NET
//!    side sends (transcribed literally from
//!    TrueSpotter.Infrastructure: ReviewUpdateScript.cs,
//!    OpenSearchBulkIndexer.AppendOfferScript, OpenSearchCatalogCategoryApplier,
//!    ProductCategoryBackfillActivities, OpenSearchActivities.AddCommentToProduct).
//!    A script is recognized by whitespace-normalized equality against the
//!    catalog and then evaluated NATIVELY in Rust — no Painless engine.
//!    Unrecognized scripts fail LOUDLY (OpenSearch-shaped 400) — drift is
//!    never silently wrong.
//!  - _bulk NDJSON parsing.

use serde_json::{json, Map, Value};

use crate::dsn::normalize_ws;

// ---------------------------------------------------------------------------
// doc merge
// ---------------------------------------------------------------------------

/// OpenSearch partial-update merge: recursive object merge; arrays and
/// scalars (including nulls) replace.
pub fn merge_doc(target: &mut Value, partial: &Value) {
    match (target, partial) {
        (Value::Object(t), Value::Object(p)) => {
            for (k, v) in p {
                match t.get_mut(k) {
                    Some(tv) if tv.is_object() && v.is_object() => merge_doc(tv, v),
                    _ => {
                        t.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (t, p) => *t = p.clone(),
    }
}

/// Pull the `embedding` float array out of a doc (it travels in _source on
/// the wire but is stored in the embedding column, never in doc jsonb).
pub fn extract_embedding(doc: &mut Map<String, Value>) -> Option<Vec<f32>> {
    match doc.remove("embedding") {
        Some(Value::Array(a)) => Some(
            a.into_iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect(),
        ),
        _ => None,
    }
}

/// Result of computing an update against the stored doc.
pub struct UpdateOutcome {
    /// merged/new doc WITHOUT the embedding key
    pub doc: Map<String, Value>,
    /// embedding extracted from the merged doc (None → keep stored)
    pub embedding: Option<Vec<f32>>,
    /// "created" | "updated"
    pub result: &'static str,
}

#[derive(Debug, PartialEq)]
pub enum UpdateError {
    /// 400 — unrecognized/invalid update body (OpenSearch parsing_exception)
    Parse(String),
    /// 400 — unrecognized script (OpenSearch script_compile / illegal_argument)
    Script(String),
    /// 404 — document_missing_exception
    MissingDoc,
}

/// Compute a POST /{index}/_update/{id} result — either a `doc` merge
/// (doc_as_upsert semantics) or a `script` application.
pub fn compute_update(
    existing: Option<&Value>,
    body: &Value,
) -> Result<UpdateOutcome, UpdateError> {
    let obj = body
        .as_object()
        .ok_or_else(|| UpdateError::Parse("update body must be an object".into()))?;
    let doc_as_upsert = obj
        .get("doc_as_upsert")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if let Some(script) = obj.get("script") {
        if !obj.get("doc").is_some_and(|d| !d.is_null()) {
            let Some(stored) = existing else {
                return Err(UpdateError::MissingDoc);
            };
            let mut doc = stored
                .as_object()
                .cloned()
                .ok_or_else(|| UpdateError::Parse("stored doc is not an object".into()))?;
            let source = script
                .get("source")
                .and_then(|s| s.as_str())
                .ok_or_else(|| UpdateError::Parse("script.source must be a string".into()))?;
            let params = script.get("params").cloned().unwrap_or(json!({}));
            apply_script(source, &params, &mut doc).map_err(UpdateError::Script)?;
            let embedding = extract_embedding(&mut doc);
            return Ok(UpdateOutcome {
                doc,
                embedding,
                result: "updated",
            });
        }
    }
    if let Some(partial) = obj.get("doc") {
        let partial = partial
            .as_object()
            .ok_or_else(|| UpdateError::Parse("doc must be an object".into()))?
            .clone();
        match existing {
            Some(stored) => {
                let mut stored_val = stored.clone();
                merge_doc(&mut stored_val, &Value::Object(partial));
                let merged = match stored_val {
                    Value::Object(m) => m,
                    _ => return Err(UpdateError::Parse("merged doc is not an object".into())),
                };
                let embedding = extract_embedding_map(&merged);
                let mut doc = merged;
                doc.remove("embedding");
                return Ok(UpdateOutcome {
                    doc,
                    embedding,
                    result: "updated",
                });
            }
            None => {
                if !doc_as_upsert {
                    return Err(UpdateError::MissingDoc);
                }
                let mut doc = partial;
                let embedding = extract_embedding(&mut doc);
                return Ok(UpdateOutcome {
                    doc,
                    embedding,
                    result: "created",
                });
            }
        }
    }
    Err(UpdateError::Parse(
        "update body must contain 'doc' or 'script'".into(),
    ))
}

fn extract_embedding_map(doc: &Map<String, Value>) -> Option<Vec<f32>> {
    doc.get("embedding").and_then(|v| match v {
        Value::Array(a) => Some(
            a.iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect(),
        ),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// Painless pattern catalog — literal bodies from TrueSpotter.Infrastructure
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// OpenSearchCatalogCategoryApplier (update_by_query)
    SetType,
    /// ProductCategoryBackfillActivities — attempt counter
    IncCategoryAttempts,
    /// OpenSearchActivities.AddCommentToProduct
    AddComment,
    /// OpenSearchBulkIndexer.AppendOfferScript
    AppendOffer,
    /// ReviewUpdateScript.Source
    ReviewSource,
    /// ReviewUpdateScript.RecomputeTrustSource
    RecomputeTrust,
    /// ReviewUpdateScript.ReplaceReviewsSource
    ReplaceReviews,
    /// ReviewUpdateScript.AspectBackfillSource
    AspectBackfill,
    /// ReviewUpdateScript.CommentAspectBackfillSource
    CommentAspectBackfill,
    /// ReviewUpdateScript.SetCategorySource
    SetCategory,
    /// ReviewUpdateScript.SetCategoryAndPathSource
    SetCategoryAndPath,
    /// ReviewUpdateScript.SetModelKeySource
    SetModelKey,
    /// ReviewUpdateScript.ApplyModelTrustSource
    ApplyModelTrust,
}

/// The shared RecomputeAspectStats body (embedded inside several scripts —
/// transcribed from ReviewUpdateScript.cs).
const ASPECT_STATS_BODY: &str = "if (ctx._source.aspect_stats == null) { ctx._source.aspect_stats = new HashMap(); } ctx._source.aspect_stats.clear(); def firstQuotes = new HashMap(); if (ctx._source.youtube_reviews != null) { for (def r : ctx._source.youtube_reviews) { if (r.aspects == null) { continue; } def rSource = r.channel != null && !r.channel.isEmpty() ? r.channel : 'recenzja'; for (def a : r.aspects) { if (a.name == null) { continue; } if (!ctx._source.aspect_stats.containsKey(a.name)) { def c = new HashMap(); c.put('positive', 0); c.put('negative', 0); c.put('mixed', 0); c.put('best_quote', null); c.put('sources', new ArrayList()); ctx._source.aspect_stats.put(a.name, c); } if (a.sentiment == 'positive') { ctx._source.aspect_stats[a.name].positive = (int)ctx._source.aspect_stats[a.name].positive + 1; } else if (a.sentiment == 'negative') { ctx._source.aspect_stats[a.name].negative = (int)ctx._source.aspect_stats[a.name].negative + 1; } else { ctx._source.aspect_stats[a.name].mixed = (int)ctx._source.aspect_stats[a.name].mixed + 1; } if (a.quote != null && !a.quote.isEmpty()) { if (a.sentiment == 'positive' && ctx._source.aspect_stats[a.name].best_quote == null) { ctx._source.aspect_stats[a.name].best_quote = a.quote; } if (!firstQuotes.containsKey(a.name)) { firstQuotes.put(a.name, a.quote); } } if (!ctx._source.aspect_stats[a.name].sources.contains(rSource)) { ctx._source.aspect_stats[a.name].sources.add(rSource); } } } } if (ctx._source.comments != null) { for (def c : ctx._source.comments) { if (c.aspects == null) { continue; } def cSource = c.source != null && !c.source.isEmpty() ? c.source : 'opinia klienta'; for (def a : c.aspects) { if (a.name == null) { continue; } if (!ctx._source.aspect_stats.containsKey(a.name)) { def m = new HashMap(); m.put('positive', 0); m.put('negative', 0); m.put('mixed', 0); m.put('best_quote', null); m.put('sources', new ArrayList()); ctx._source.aspect_stats.put(a.name, m); } if (a.sentiment == 'positive') { ctx._source.aspect_stats[a.name].positive = (int)ctx._source.aspect_stats[a.name].positive + 1; } else if (a.sentiment == 'negative') { ctx._source.aspect_stats[a.name].negative = (int)ctx._source.aspect_stats[a.name].negative + 1; } else { ctx._source.aspect_stats[a.name].mixed = (int)ctx._source.aspect_stats[a.name].mixed + 1; } if (a.quote != null && !a.quote.isEmpty()) { if (a.sentiment == 'positive' && ctx._source.aspect_stats[a.name].best_quote == null) { ctx._source.aspect_stats[a.name].best_quote = a.quote; } if (!firstQuotes.containsKey(a.name)) { firstQuotes.put(a.name, a.quote); } } if (!ctx._source.aspect_stats[a.name].sources.contains(cSource)) { ctx._source.aspect_stats[a.name].sources.add(cSource); } } } } for (def entry : ctx._source.aspect_stats.entrySet()) { if (entry.getValue().best_quote == null && firstQuotes.containsKey(entry.getKey())) { entry.getValue().best_quote = firstQuotes.get(entry.getKey()); } }";

const INDEPENDENCE_PRELUDE: &str = "def iw = new HashMap(); iw.put('laboratory', 1.0); iw.put('press', 0.75); iw.put('affiliate', 0.4); iw.put('unknown', 0.25); if (params.independenceWeights != null) { for (def e : params.independenceWeights.entrySet()) { iw.put(e.getKey(), e.getValue()); } }";

const TRUST_WILSON_BODY: &str = "def z = 1.96; def p = weightedTotal > 0 ? weightedPositive / weightedTotal : 0.0; ctx._source.trust_score = weightedTotal > 0 ? Math.max(0.0, 5.0 * ((p + z*z/(2*weightedTotal) - z*Math.sqrt(p*(1-p)/weightedTotal + z*z/(4*weightedTotal*weightedTotal))) / (1 + z*z/weightedTotal))) : 0.0;";

/// ReviewUpdateScript.Source (literal).
pub const SCRIPT_REVIEW_SOURCE: &str = "if (ctx._source.youtube_reviews != null && ctx._source.youtube_reviews.any(r -> r.video_url == params.review.video_url)) { return; } if (ctx._source.youtube_reviews == null) { ctx._source.youtube_reviews = new ArrayList(); } ctx._source.youtube_reviews.add(params.review); if (ctx._source.reviews_text == null) { ctx._source.reviews_text = ''; } ctx._source.reviews_text = ctx._source.reviews_text + (ctx._source.reviews_text.length() > 0 ? ' ' : '') + params.summary; def total = ctx._source.youtube_reviews.size(); def positive = 0; def halfLife = params.halfLifeDays; def typeKey = ctx._source.type != null ? ctx._source.type : ''; if (params.halfLifeByType != null && params.halfLifeByType.containsKey(typeKey)) { halfLife = params.halfLifeByType[typeKey]; } def weightedTotal = 0.0; def weightedPositive = 0.0; def sources = new ArrayList(); {INDEPENDENCE} def trustSources = new HashMap(); for (def r : ctx._source.youtube_reviews) { if (r.sentiment == 'positive') { positive++; } def w = 0.5; if (r.published_at != null) { def ageDays = (params.nowMs - r.published_at) / 86400000.0; if (ageDays < 0) { ageDays = 0; } if (halfLife <= 0) { w = 1.0; } else { w = Math.pow(0.5, ageDays / halfLife); } } def tk = r.source_tier != null && iw.containsKey(r.source_tier) ? r.source_tier : 'unknown'; def tv = iw.get(tk); if (tv == null || tv < 0.01) { tv = 0.01; } if (tv > 10.0) { tv = 10.0; } w = w * tv; def sn = r.source_name != null && !r.source_name.isEmpty() ? r.source_name : r.channel; if (sn != null && !sn.isEmpty() && !trustSources.containsKey(sn)) { trustSources.put(sn, tk); } weightedTotal += w; if (r.sentiment == 'positive') { weightedPositive += w; } if (r.channel != null && !r.channel.isEmpty() && !sources.contains(r.channel)) { sources.add(r.channel); } } if (ctx._source.comments != null) { for (def c : ctx._source.comments) { if (c.source != null && !c.source.isEmpty() && !sources.contains(c.source)) { sources.add(c.source); } if (c.source != null && !c.source.isEmpty() && !trustSources.containsKey(c.source)) { trustSources.put(c.source, 'unknown'); } } } {WILSON} ctx._source.trust_count = total; ctx._source.source_count = sources.size(); ctx._source.trust_sources = trustSources; ctx._source.trust_score_computed_at = params.nowMs; {ASPECTS}";

/// ReviewUpdateScript.RecomputeTrustSource (literal).
pub const SCRIPT_RECOMPUTE_TRUST: &str = "def positive = 0; def halfLife = params.halfLifeDays; def typeKey = ctx._source.type != null ? ctx._source.type : ''; if (params.halfLifeByType != null && params.halfLifeByType.containsKey(typeKey)) { halfLife = params.halfLifeByType[typeKey]; } {INDEPENDENCE} def weightedTotal = 0.0; def weightedPositive = 0.0; def sources = new ArrayList(); def trustSources = new HashMap(); if (ctx._source.youtube_reviews != null) { for (def r : ctx._source.youtube_reviews) { if (r.sentiment == 'positive') { positive++; } def w = 0.5; if (r.published_at != null) { def ageDays = (params.nowMs - r.published_at) / 86400000.0; if (ageDays < 0) { ageDays = 0; } if (halfLife <= 0) { w = 1.0; } else { w = Math.pow(0.5, ageDays / halfLife); } } def tk = r.source_tier != null && iw.containsKey(r.source_tier) ? r.source_tier : 'unknown'; def tv = iw.get(tk); if (tv == null || tv < 0.01) { tv = 0.01; } if (tv > 10.0) { tv = 10.0; } w = w * tv; def sn = r.source_name != null && !r.source_name.isEmpty() ? r.source_name : r.channel; if (sn != null && !sn.isEmpty() && !trustSources.containsKey(sn)) { trustSources.put(sn, tk); } weightedTotal += w; if (r.sentiment == 'positive') { weightedPositive += w; } if (r.channel != null && !r.channel.isEmpty() && !sources.contains(r.channel)) { sources.add(r.channel); } } } if (ctx._source.comments != null) { for (def c : ctx._source.comments) { if (c.source != null && !c.source.isEmpty() && !sources.contains(c.source)) { sources.add(c.source); } if (c.source != null && !c.source.isEmpty() && !trustSources.containsKey(c.source)) { trustSources.put(c.source, 'unknown'); } } } {WILSON} ctx._source.trust_count = ctx._source.youtube_reviews != null ? ctx._source.youtube_reviews.size() : 0; ctx._source.source_count = sources.size(); ctx._source.trust_sources = trustSources; ctx._source.trust_score_computed_at = params.nowMs; ";

/// ReviewUpdateScript.ReplaceReviewsSource (literal).
pub const SCRIPT_REPLACE_REVIEWS: &str = "ctx._source.youtube_reviews = params.reviews; ctx._source.reviews_text = params.reviewsText; {TRUST} {ASPECTS}";

/// ReviewUpdateScript.AspectBackfillSource (literal).
pub const SCRIPT_ASPECT_BACKFILL: &str = "if (ctx._source.youtube_reviews != null) { for (def r : ctx._source.youtube_reviews) { if (params.aspectsMap.containsKey(r.video_url) && (r.aspects == null || r.aspects.isEmpty())) { r.aspects = params.aspectsMap.get(r.video_url); } } } {ASPECTS}";

/// ReviewUpdateScript.CommentAspectBackfillSource (literal).
pub const SCRIPT_COMMENT_ASPECT_BACKFILL: &str = "if (ctx._source.comments != null) { for (def c : ctx._source.comments) { if (params.aspectsMap.containsKey(c.id) && (c.aspects == null || c.aspects.isEmpty())) { c.aspects = params.aspectsMap.get(c.id); } } } {ASPECTS}";

/// OpenSearchBulkIndexer.AppendOfferScript (literal).
pub const SCRIPT_APPEND_OFFER: &str = "if (ctx._source.offers == null) { ctx._source.offers = new ArrayList(); } def exists = false; for (def o : ctx._source.offers) { if (o.shop_slug == params.offer.shop_slug) { def stored = 0; if (o.updated_at != null && o.updated_at > 0) { stored = o.updated_at; } if (params.ingestedAt >= stored) { o.price = params.offer.price; o.regular_price = params.offer.regular_price; o.checkout_url = params.offer.checkout_url; o.currency = params.offer.currency; o.in_stock = params.offer.in_stock; o.shop_name = params.offer.shop_name; o.updated_at = params.ingestedAt; } exists = true; } } if (!exists) { ctx._source.offers.add(params.offer); } if (ctx._source.external_ids == null) { ctx._source.external_ids = new ArrayList(); } if (!ctx._source.external_ids.contains(params.externalId)) { ctx._source.external_ids.add(params.externalId); } if (ctx._source.offers.size() > 0) { def minPrice = ctx._source.offers[0].price; for (def o : ctx._source.offers) { if (o.price < minPrice) { minPrice = o.price; } } ctx._source.price = minPrice; }";

/// OpenSearchCatalogCategoryApplier (literal).
pub const SCRIPT_SET_TYPE: &str = "ctx._source.type = params.category";

/// ProductCategoryBackfillActivities — attempt counter (literal).
pub const SCRIPT_INC_CATEGORY_ATTEMPTS: &str = "ctx._source.category_attempts = (ctx._source.category_attempts == null ? 0 : ctx._source.category_attempts) + 1";

/// OpenSearchActivities.AddCommentToProduct (literal).
pub const SCRIPT_ADD_COMMENT: &str = "if (ctx._source.comments == null) { ctx._source.comments = new ArrayList() } ctx._source.comments.add(params.comment)";

/// ReviewUpdateScript.SetCategorySource / SetCategoryAndPathSource /
/// SetModelKeySource / ApplyModelTrustSource (literal).
pub const SCRIPT_SET_CATEGORY: &str = "ctx._source.category = params.category; ";
pub const SCRIPT_SET_CATEGORY_AND_PATH: &str =
    "ctx._source.category = params.category; ctx._source.category_path = params.categoryPath; ";
pub const SCRIPT_SET_MODEL_KEY: &str = "ctx._source.model_key = params.modelKey; ";
pub const SCRIPT_APPLY_MODEL_TRUST: &str = "ctx._source.trust_score = params.trustScore; ctx._source.trust_count = params.trustCount; ctx._source.source_count = params.sourceCount; ctx._source.trust_sources = params.trustSources; ctx._source.trust_score_computed_at = params.nowMs; ";

fn catalog_entry(pattern: Pattern, source: &str) -> (Pattern, String) {
    (pattern, normalize_ws(source))
}

fn render(template: &str) -> String {
    template
        .replace("{INDEPENDENCE}", INDEPENDENCE_PRELUDE)
        .replace("{WILSON}", TRUST_WILSON_BODY)
        .replace("{ASPECTS}", ASPECT_STATS_BODY)
        .replace(
            "{TRUST}",
            &SCRIPT_RECOMPUTE_TRUST
                .replace("{INDEPENDENCE}", INDEPENDENCE_PRELUDE)
                .replace("{WILSON}", TRUST_WILSON_BODY),
        )
}

/// Whitespace-normalized catalog — (pattern, normalized source).
pub fn catalog() -> Vec<(Pattern, String)> {
    vec![
        catalog_entry(Pattern::SetType, SCRIPT_SET_TYPE),
        catalog_entry(Pattern::IncCategoryAttempts, SCRIPT_INC_CATEGORY_ATTEMPTS),
        catalog_entry(Pattern::AddComment, SCRIPT_ADD_COMMENT),
        catalog_entry(Pattern::AppendOffer, SCRIPT_APPEND_OFFER),
        catalog_entry(Pattern::ReviewSource, &render(SCRIPT_REVIEW_SOURCE)),
        catalog_entry(Pattern::RecomputeTrust, &render(SCRIPT_RECOMPUTE_TRUST)),
        catalog_entry(Pattern::ReplaceReviews, &render(SCRIPT_REPLACE_REVIEWS)),
        catalog_entry(Pattern::AspectBackfill, &render(SCRIPT_ASPECT_BACKFILL)),
        catalog_entry(
            Pattern::CommentAspectBackfill,
            &render(SCRIPT_COMMENT_ASPECT_BACKFILL),
        ),
        catalog_entry(Pattern::SetCategory, SCRIPT_SET_CATEGORY),
        catalog_entry(Pattern::SetCategoryAndPath, SCRIPT_SET_CATEGORY_AND_PATH),
        catalog_entry(Pattern::SetModelKey, SCRIPT_SET_MODEL_KEY),
        catalog_entry(Pattern::ApplyModelTrust, SCRIPT_APPLY_MODEL_TRUST),
    ]
}

/// Match a (whitespace-normalized) script against the catalog.
pub fn match_pattern(normalized: &str) -> Result<Pattern, String> {
    for (p, src) in catalog() {
        if src == normalized {
            return Ok(p);
        }
    }
    Err(format!(
        "unrecognized script source (not in the supported pattern catalog): '{}'",
        normalized.chars().take(120).collect::<String>()
    ))
}

/// Apply a script from the catalog to a doc. `Ok(())` even when the script
/// is a semantic no-op (idempotent review append).
pub fn apply_script(
    source: &str,
    params: &Value,
    doc: &mut Map<String, Value>,
) -> Result<(), String> {
    let normalized = normalize_ws(source);
    let pattern = match_pattern(&normalized)?;
    run(pattern, params, doc)
}

// ---------------------------------------------------------------------------
// pattern implementations
// ---------------------------------------------------------------------------

fn arr_mut<'a>(doc: &'a mut Map<String, Value>, key: &str) -> &'a mut Vec<Value> {
    let entry = doc.entry(key).or_insert_with(|| Value::Array(Vec::new()));
    if !entry.is_array() {
        *entry = Value::Array(Vec::new());
    }
    entry.as_array_mut().expect("array")
}

fn get_str<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(|s| s.as_str())
}

fn non_empty(s: Option<&str>) -> Option<&str> {
    s.filter(|s| !s.is_empty())
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_i64().map(|i| i as f64))
}

fn run(pattern: Pattern, params: &Value, doc: &mut Map<String, Value>) -> Result<(), String> {
    match pattern {
        Pattern::SetType => {
            doc.insert("type".into(), params["category"].clone());
        }
        Pattern::SetCategory => {
            doc.insert("category".into(), params["category"].clone());
        }
        Pattern::SetCategoryAndPath => {
            doc.insert("category".into(), params["category"].clone());
            doc.insert("category_path".into(), params["categoryPath"].clone());
        }
        Pattern::SetModelKey => {
            doc.insert("model_key".into(), params["modelKey"].clone());
        }
        Pattern::ApplyModelTrust => {
            doc.insert("trust_score".into(), params["trustScore"].clone());
            doc.insert("trust_count".into(), params["trustCount"].clone());
            doc.insert("source_count".into(), params["sourceCount"].clone());
            doc.insert("trust_sources".into(), params["trustSources"].clone());
            doc.insert("trust_score_computed_at".into(), params["nowMs"].clone());
        }
        Pattern::IncCategoryAttempts => {
            // painless: (x == null ? 0 : x) + 1 — stays integer
            let cur = doc.get("category_attempts");
            let next = match cur.and_then(|v| v.as_i64()) {
                Some(i) => json!(i + 1),
                None if cur.and_then(num).is_none() => json!(1),
                None => json!(cur.and_then(num).unwrap_or(0.0) + 1.0),
            };
            doc.insert("category_attempts".into(), next);
        }
        Pattern::AddComment => {
            arr_mut(doc, "comments").push(params["comment"].clone());
        }
        Pattern::AppendOffer => append_offer(params, doc),
        Pattern::ReviewSource => {
            // Idempotency backstop: same video_url → no-op.
            let review = params["review"].clone();
            let already = doc
                .get("youtube_reviews")
                .and_then(|r| r.as_array())
                .is_some_and(|reviews| {
                    reviews
                        .iter()
                        .any(|r| get_str(r, "video_url") == get_str(&review, "video_url"))
                });
            if already {
                return Ok(());
            }
            arr_mut(doc, "youtube_reviews").push(review);
            let summary = params["summary"].as_str().unwrap_or_default();
            let mut text = doc
                .get("reviews_text")
                .and_then(|t| t.as_str())
                .unwrap_or_default()
                .to_string();
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(summary);
            doc.insert("reviews_text".into(), json!(text));
            recompute_trust(params, doc);
            recompute_aspect_stats(doc);
        }
        Pattern::RecomputeTrust => {
            recompute_trust(params, doc);
        }
        Pattern::ReplaceReviews => {
            doc.insert("youtube_reviews".into(), params["reviews"].clone());
            doc.insert("reviews_text".into(), params["reviewsText"].clone());
            recompute_trust(params, doc);
            recompute_aspect_stats(doc);
        }
        Pattern::AspectBackfill => {
            let aspects_map = params["aspectsMap"].clone();
            let entry_set: Vec<(String, Value)> = match aspects_map.as_object() {
                Some(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                None => Vec::new(),
            };
            let reviews = arr_mut(doc, "youtube_reviews");
            for r in reviews.iter_mut() {
                let url = match get_str(r, "video_url") {
                    Some(u) => u.to_string(),
                    None => continue,
                };
                let has_aspects = r
                    .get("aspects")
                    .is_some_and(|a| a.as_array().is_some_and(|a| !a.is_empty()));
                if !has_aspects {
                    if let Some((_, v)) = entry_set.iter().find(|(k, _)| k == &url) {
                        if let Value::Object(ro) = r {
                            ro.insert("aspects".into(), v.clone());
                        }
                    }
                }
            }
            recompute_aspect_stats(doc);
        }
        Pattern::CommentAspectBackfill => {
            let aspects_map = params["aspectsMap"].clone();
            let entry_set: Vec<(String, Value)> = match aspects_map.as_object() {
                Some(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                None => Vec::new(),
            };
            let comments = arr_mut(doc, "comments");
            for c in comments.iter_mut() {
                let id = match get_str(c, "id") {
                    Some(i) => i.to_string(),
                    None => continue,
                };
                let has_aspects = c
                    .get("aspects")
                    .is_some_and(|a| a.as_array().is_some_and(|a| !a.is_empty()));
                if !has_aspects {
                    if let Some((_, v)) = entry_set.iter().find(|(k, _)| k == &id) {
                        if let Value::Object(co) = c {
                            co.insert("aspects".into(), v.clone());
                        }
                    }
                }
            }
            recompute_aspect_stats(doc);
        }
    }
    Ok(())
}

/// OpenSearchBulkIndexer.AppendOfferScript — offer dedup by shop_slug with
/// the KAN-0162 staleness guard, external_ids dedup append, min-price recompute.
fn append_offer(params: &Value, doc: &mut Map<String, Value>) {
    let offer = params["offer"].clone();
    let shop_slug = get_str(&offer, "shop_slug").unwrap_or_default().to_string();
    let ingested = num(&params["ingestedAt"]).unwrap_or(0.0);
    let mut exists = false;
    for o in arr_mut(doc, "offers").iter_mut() {
        if get_str(o, "shop_slug") != Some(shop_slug.as_str()) {
            continue;
        }
        let stored = o
            .get("updated_at")
            .and_then(num)
            .filter(|v| *v > 0.0)
            .unwrap_or(0.0);
        if ingested >= stored {
            if let Value::Object(oo) = o {
                for key in [
                    "price",
                    "regular_price",
                    "checkout_url",
                    "currency",
                    "in_stock",
                    "shop_name",
                ] {
                    oo.insert(key.into(), offer.get(key).cloned().unwrap_or(Value::Null));
                }
                oo.insert("updated_at".into(), params["ingestedAt"].clone());
            }
        }
        exists = true;
    }
    if !exists {
        arr_mut(doc, "offers").push(offer);
    }
    let external_id = params["externalId"].clone();
    let ext = arr_mut(doc, "external_ids");
    if !ext.iter().any(|e| e == &external_id) {
        ext.push(external_id);
    }
    // min-price loop over offers
    let offers = doc.get("offers").and_then(|o| o.as_array());
    if let Some(offers) = offers.filter(|o| !o.is_empty()) {
        let mut min_price = num(&offers[0]["price"]).unwrap_or(0.0);
        for o in offers {
            if let Some(p) = num(&o["price"]) {
                if p < min_price {
                    min_price = p;
                }
            }
        }
        doc.insert("price".into(), json!(min_price));
    }
}

/// The shared trust recompute (decay × independence-tier weighted Wilson
/// lower bound, z=1.96, ×5, floored at 0) — ReviewUpdateScript semantics.
fn recompute_trust(params: &Value, doc: &mut Map<String, Value>) {
    let now_ms = num(&params["nowMs"]).unwrap_or(0.0);
    let mut half_life = num(&params["halfLifeDays"]).unwrap_or(0.0);
    let type_key = doc.get("type").and_then(|t| t.as_str()).unwrap_or("");
    if let Some(h) = params
        .get("halfLifeByType")
        .and_then(|m| m.get(type_key))
        .and_then(num)
    {
        half_life = h;
    }
    // Independence weights: defaults overlaid with params.independenceWeights.
    let mut iw: Vec<(String, f64)> = vec![
        ("laboratory".into(), 1.0),
        ("press".into(), 0.75),
        ("affiliate".into(), 0.4),
        ("unknown".into(), 0.25),
    ];
    if let Some(overrides) = params
        .get("independenceWeights")
        .and_then(|v| v.as_object())
    {
        for (k, v) in overrides {
            if let Some(w) = num(v) {
                if let Some(e) = iw.iter_mut().find(|(name, _)| name == k) {
                    e.1 = w;
                } else {
                    iw.push((k.clone(), w));
                }
            }
        }
    }
    let tier_weight = |tier: &str| -> f64 {
        iw.iter()
            .find(|(name, _)| name == tier)
            .map(|(_, w)| *w)
            .unwrap_or(0.25)
            .clamp(0.01, 10.0)
    };

    let mut sources: Vec<Value> = Vec::new();
    let mut trust_sources = Map::new();
    let mut weighted_total = 0.0f64;
    let mut weighted_positive = 0.0f64;

    let reviews = doc.get("youtube_reviews").cloned();
    if let Some(reviews) = reviews.and_then(|r| r.as_array().cloned()) {
        for r in &reviews {
            let sentiment_positive = get_str(r, "sentiment") == Some("positive");
            let mut w = 0.5f64;
            if let Some(published) = r.get("published_at").and_then(num) {
                let mut age_days = (now_ms - published) / 86_400_000.0;
                if age_days < 0.0 {
                    age_days = 0.0;
                }
                w = if half_life <= 0.0 {
                    1.0
                } else {
                    0.5f64.powf(age_days / half_life)
                };
            }
            let tier = get_str(r, "source_tier")
                .filter(|t| iw.iter().any(|(name, _)| name == t))
                .unwrap_or("unknown");
            let tv = tier_weight(tier);
            w *= tv;
            let sn =
                non_empty(get_str(r, "source_name")).or_else(|| non_empty(get_str(r, "channel")));
            if let Some(sn) = sn {
                trust_sources
                    .entry(sn.to_string())
                    .or_insert_with(|| json!(tier));
            }
            weighted_total += w;
            if sentiment_positive {
                weighted_positive += w;
            }
            if let Some(ch) = non_empty(get_str(r, "channel")) {
                let v = json!(ch);
                if !sources.contains(&v) {
                    sources.push(v);
                }
            }
        }
        doc.insert("trust_count".into(), json!(reviews.len()));
    } else {
        doc.insert("trust_count".into(), json!(0));
    }

    if let Some(comments) = doc.get("comments").and_then(|c| c.as_array().cloned()) {
        for c in &comments {
            if let Some(s) = non_empty(get_str(c, "source")) {
                let v = json!(s);
                if !sources.contains(&v) {
                    sources.push(v);
                }
                trust_sources
                    .entry(s.to_string())
                    .or_insert_with(|| json!("unknown"));
            }
        }
    }

    // Wilson lower bound (z=1.96) on the decay-weighted positive proportion, ×5.
    let z = 1.96f64;
    let p = if weighted_total > 0.0 {
        weighted_positive / weighted_total
    } else {
        0.0
    };
    let trust = if weighted_total > 0.0 {
        (5.0 * ((p + z * z / (2.0 * weighted_total)
            - z * (p * (1.0 - p) / weighted_total
                + z * z / (4.0 * weighted_total * weighted_total))
                .sqrt())
            / (1.0 + z * z / weighted_total)))
            .max(0.0)
    } else {
        0.0
    };
    doc.insert("trust_score".into(), json!(trust));
    doc.insert("source_count".into(), json!(sources.len()));
    doc.insert("trust_sources".into(), Value::Object(trust_sources));
    doc.insert("trust_score_computed_at".into(), params["nowMs"].clone());
}

/// The shared RecomputeAspectStats body — aspect_stats rebuilt from the
/// per-review/comment aspects; sentiment buckets, best_quote (first
/// positive, fallback first quote), deduplicated sources.
fn recompute_aspect_stats(doc: &mut Map<String, Value>) {
    let mut stats: Map<String, Value> = Map::new();
    let mut first_quotes: Map<String, Value> = Map::new();

    let ensure_entry = |stats: &mut Map<String, Value>, name: &str| -> usize {
        if !stats.contains_key(name) {
            stats.insert(
                name.to_string(),
                json!({
                    "positive": 0,
                    "negative": 0,
                    "mixed": 0,
                    "best_quote": null,
                    "sources": []
                }),
            );
        }
        let keys: Vec<String> = stats.keys().cloned().collect();
        keys.iter().position(|k| k == name).expect("just inserted")
    };

    let process_aspects = |stats: &mut Map<String, Value>,
                           first_quotes: &mut Map<String, Value>,
                           aspects: Option<&Vec<Value>>,
                           source: &str| {
        let Some(aspects) = aspects else { return };
        for a in aspects {
            let Some(name) = non_empty(a.get("name").and_then(|n| n.as_str())) else {
                continue;
            };
            let idx = ensure_entry(stats, name);
            let sentiment = a.get("sentiment").and_then(|s| s.as_str()).unwrap_or("");
            let bucket = if sentiment == "positive" {
                "positive"
            } else if sentiment == "negative" {
                "negative"
            } else {
                "mixed"
            };
            if let Some(entry) = stats.get_mut(name).and_then(|e| e.as_object_mut()) {
                let cur = entry.get(bucket).and_then(|v| v.as_i64()).unwrap_or(0);
                entry.insert(bucket.into(), json!(cur + 1));
            }
            let quote = non_empty(a.get("quote").and_then(|q| q.as_str()));
            if let Some(quote) = quote {
                if sentiment == "positive" {
                    if let Some(entry) = stats.get_mut(name).and_then(|e| e.as_object_mut()) {
                        if entry.get("best_quote").map(|b| b.is_null()).unwrap_or(true) {
                            entry.insert("best_quote".into(), json!(quote));
                        }
                    }
                }
                first_quotes
                    .entry(name.to_string())
                    .or_insert_with(|| json!(quote));
            }
            if let Some(entry) = stats.get_mut(name).and_then(|e| e.as_object_mut()) {
                let src = json!(source);
                let list = entry
                    .entry("sources")
                    .or_insert_with(|| Value::Array(Vec::new()));
                if let Some(arr) = list.as_array_mut() {
                    if !arr.contains(&src) {
                        arr.push(src);
                    }
                }
            }
            let _ = idx;
        }
    };

    let reviews = doc
        .get("youtube_reviews")
        .and_then(|r| r.as_array())
        .cloned();
    if let Some(reviews) = reviews {
        for r in &reviews {
            let Some(aspects) = r.get("aspects").and_then(|a| a.as_array()) else {
                continue;
            };
            let aspects = aspects.clone();
            let r_source = non_empty(get_str(r, "channel"))
                .unwrap_or("recenzja")
                .to_string();
            process_aspects(&mut stats, &mut first_quotes, Some(&aspects), &r_source);
        }
    }
    let comments = doc.get("comments").and_then(|c| c.as_array()).cloned();
    if let Some(comments) = comments {
        for c in &comments {
            let Some(aspects) = c.get("aspects").and_then(|a| a.as_array()) else {
                continue;
            };
            let aspects = aspects.clone();
            let c_source = non_empty(get_str(c, "source"))
                .unwrap_or("opinia klienta")
                .to_string();
            process_aspects(&mut stats, &mut first_quotes, Some(&aspects), &c_source);
        }
    }

    // Fallback: aspects with no positive quote get the first quote seen.
    for (name, entry) in stats.iter_mut() {
        let needs = entry.get("best_quote").map(|b| b.is_null()).unwrap_or(true);
        if needs {
            if let Some(q) = first_quotes.get(name) {
                if let Value::Object(eo) = entry {
                    eo.insert("best_quote".into(), q.clone());
                }
            }
        }
    }

    doc.insert("aspect_stats".into(), Value::Object(stats));
}

// ---------------------------------------------------------------------------
// _bulk NDJSON parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum BulkAction {
    Index {
        index: String,
        id: String,
        version_type: Option<String>,
        version: Option<i64>,
        source: Value,
    },
    Update {
        index: String,
        id: String,
        body: Value,
    },
    Delete {
        index: String,
        id: String,
    },
}

/// Parse an NDJSON _bulk body. `default_index` is used when an action line
/// omits `_index` (route /_bulk).
pub fn parse_bulk(body: &str, default_index: &str) -> Result<Vec<BulkAction>, String> {
    let mut lines = body
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str::<Value>(l).map_err(|e| format!("bulk line parse: {e}")));
    let mut out = Vec::new();
    while let Some(meta) = lines.next() {
        let meta = meta?;
        let (op, meta_obj) = meta
            .as_object()
            .and_then(|o| {
                o.iter()
                    .next()
                    .map(|(k, v)| (k.as_str(), v.as_object().cloned()))
            })
            .ok_or("bulk action line must have one op")?;
        let meta_obj = meta_obj.ok_or("bulk op must be an object")?;
        let get_id = |m: &Map<String, Value>| -> Result<String, String> {
            m.get("_id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .ok_or_else(|| format!("bulk op '{op}' requires _id"))
        };
        let index_of = |m: &Map<String, Value>| -> String {
            m.get("_index")
                .and_then(|v| v.as_str())
                .unwrap_or(default_index)
                .to_string()
        };
        match op {
            "index" => {
                let id = get_id(&meta_obj)?;
                let source = lines
                    .next()
                    .transpose()?
                    .ok_or("index action requires a source line")?;
                out.push(BulkAction::Index {
                    index: index_of(&meta_obj),
                    id,
                    version_type: meta_obj
                        .get("version_type")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    version: meta_obj.get("version").and_then(|v| v.as_i64()),
                    source,
                });
            }
            "update" => {
                let id = get_id(&meta_obj)?;
                let payload = lines
                    .next()
                    .transpose()?
                    .ok_or("update action requires a payload line")?;
                out.push(BulkAction::Update {
                    index: index_of(&meta_obj),
                    id,
                    body: payload,
                });
            }
            "delete" => {
                out.push(BulkAction::Delete {
                    index: index_of(&meta_obj),
                    id: get_id(&meta_obj)?,
                });
            }
            other => return Err(format!("unsupported bulk op '{other}'")),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc_of(pairs: Value) -> Map<String, Value> {
        pairs.as_object().cloned().unwrap()
    }

    // ---- doc merge ----------------------------------------------------------

    #[test]
    fn merge_doc_recurses_objects_replaces_arrays_and_overwrites_nulls() {
        let mut stored = json!({
            "title": "old",
            "specs": { "weight": "250", "drop": "8" },
            "tags": ["a", "b"],
            "image_url": "https://keep.png",
            "price": 100.0
        });
        merge_doc(
            &mut stored,
            &json!({
                "title": "new",
                "specs": { "weight": "300" },
                "tags": ["x"],
                "price": null
            }),
        );
        assert_eq!(stored["title"], "new");
        // recursive object merge: drop survives, weight replaced
        assert_eq!(stored["specs"]["weight"], "300");
        assert_eq!(stored["specs"]["drop"], "8");
        // arrays replace wholesale
        assert_eq!(stored["tags"], json!(["x"]));
        // nulls overwrite (OpenSearch doc-merge removes the field value)
        assert!(stored["price"].is_null());
        // untouched fields survive
        assert_eq!(stored["image_url"], "https://keep.png");
    }

    #[test]
    fn compute_update_doc_as_upsert_semantics() {
        let stored = json!({ "slug": "a", "title": "A", "embedding_stays": true });
        // merge into existing -> updated, embedding key untouched when absent
        let out = compute_update(
            Some(&stored),
            &json!({ "doc": { "title": "A2" }, "doc_as_upsert": true }),
        )
        .unwrap();
        assert_eq!(out.result, "updated");
        assert_eq!(out.doc["title"], "A2");
        assert_eq!(out.doc["slug"], "a");
        assert!(out.embedding.is_none());

        // missing doc + doc_as_upsert -> created
        let out = compute_update(
            None,
            &json!({ "doc": { "slug": "new", "title": "N" }, "doc_as_upsert": true }),
        )
        .unwrap();
        assert_eq!(out.result, "created");
        assert_eq!(out.doc["title"], "N");

        // missing doc without doc_as_upsert -> 404 (document_missing_exception)
        assert!(matches!(
            compute_update(None, &json!({ "doc": { "title": "x" } })),
            Err(UpdateError::MissingDoc)
        ));

        // script on missing doc -> 404
        assert!(matches!(
            compute_update(None, &json!({ "script": { "source": SCRIPT_SET_TYPE } })),
            Err(UpdateError::MissingDoc)
        ));

        // embedding travels in doc -> extracted out of the stored jsonb
        let out = compute_update(
            None,
            &json!({
                "doc": { "slug": "e", "embedding": [0.5, -0.5, 1.0] },
                "doc_as_upsert": true
            }),
        )
        .unwrap();
        assert_eq!(out.embedding, Some(vec![0.5, -0.5, 1.0]));
        assert!(!out.doc.contains_key("embedding"));

        // no doc and no script -> 400 parse
        assert!(matches!(
            compute_update(Some(&stored), &json!({})),
            Err(UpdateError::Parse(_))
        ));
    }

    // ---- pattern catalog ----------------------------------------------------

    #[test]
    fn all_13_wire_scripts_recognized_including_whitespace_drift() {
        let scripts: Vec<String> = vec![
            SCRIPT_SET_TYPE.into(),
            SCRIPT_INC_CATEGORY_ATTEMPTS.into(),
            SCRIPT_ADD_COMMENT.into(),
            SCRIPT_APPEND_OFFER.into(),
            render(SCRIPT_REVIEW_SOURCE),
            render(SCRIPT_RECOMPUTE_TRUST),
            render(SCRIPT_REPLACE_REVIEWS),
            render(SCRIPT_ASPECT_BACKFILL),
            render(SCRIPT_COMMENT_ASPECT_BACKFILL),
            SCRIPT_SET_CATEGORY.into(),
            SCRIPT_SET_CATEGORY_AND_PATH.into(),
            SCRIPT_SET_MODEL_KEY.into(),
            SCRIPT_APPLY_MODEL_TRUST.into(),
        ];
        assert_eq!(scripts.len(), 13);
        for s in &scripts {
            // whitespace drift (newlines, double spaces) must still match —
            // the .NET single-line concatenations are the wire form
            let drifted = s.replace("; ", ";\n  ").replace("if (", "if  (");
            assert!(
                match_pattern(&normalize_ws(&drifted)).is_ok(),
                "script not recognized: {}",
                &drifted[..40]
            );
        }
        // unknown script -> loud failure
        assert!(match_pattern("ctx._source.hack = 1").is_err());
        assert!(apply_script("ctx._source.hack = 1", &json!({}), &mut Map::new()).is_err());
    }

    #[test]
    fn simple_param_assignment_patterns() {
        let mut doc = doc_of(json!({}));
        apply_script(SCRIPT_SET_TYPE, &json!({ "category": "Buty" }), &mut doc).unwrap();
        assert_eq!(doc["type"], "Buty");

        apply_script(
            SCRIPT_SET_CATEGORY_AND_PATH,
            &json!({ "category": "buty-do-biegania", "categoryPath": "sport/bieganie/buty" }),
            &mut doc,
        )
        .unwrap();
        assert_eq!(doc["category"], "buty-do-biegania");
        assert_eq!(doc["category_path"], "sport/bieganie/buty");

        apply_script(
            SCRIPT_SET_MODEL_KEY,
            &json!({ "modelKey": "nike-pegasus-40" }),
            &mut doc,
        )
        .unwrap();
        assert_eq!(doc["model_key"], "nike-pegasus-40");

        apply_script(
            SCRIPT_APPLY_MODEL_TRUST,
            &json!({ "trustScore": 4.1, "trustCount": 12, "sourceCount": 3, "trustSources": { "Kanal": "laboratory" }, "nowMs": 1735689600000i64 }),
            &mut doc,
        )
        .unwrap();
        assert_eq!(doc["trust_score"], 4.1);
        assert_eq!(doc["trust_count"], 12);
        assert_eq!(doc["source_count"], 3);
        assert_eq!(doc["trust_sources"]["Kanal"], "laboratory");
        assert_eq!(doc["trust_score_computed_at"], 1735689600000i64);
    }

    #[test]
    fn inc_category_attempts_null_guard() {
        let mut doc = doc_of(json!({ "slug": "x" }));
        apply_script(SCRIPT_INC_CATEGORY_ATTEMPTS, &json!({}), &mut doc).unwrap();
        assert_eq!(doc["category_attempts"], 1);
        apply_script(SCRIPT_INC_CATEGORY_ATTEMPTS, &json!({}), &mut doc).unwrap();
        assert_eq!(doc["category_attempts"], 2);
    }

    #[test]
    fn add_comment_appends_to_array() {
        let mut doc = doc_of(json!({ "slug": "x" }));
        let comment = json!({ "id": "c1", "author": "Ala", "body": "super", "source": "admin" });
        apply_script(SCRIPT_ADD_COMMENT, &json!({ "comment": comment }), &mut doc).unwrap();
        assert_eq!(doc["comments"][0]["id"], "c1");
        let c2 = json!({ "id": "c2", "author": "Bob", "body": "ok", "source": "external" });
        apply_script(SCRIPT_ADD_COMMENT, &json!({ "comment": c2 }), &mut doc).unwrap();
        assert_eq!(doc["comments"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn append_offer_dedup_staleness_guard_and_min_price() {
        let params = |ingested: i64, price: f64, slug: &str| {
            json!({
                "offer": {
                    "shop_slug": slug,
                    "shop_name": "Sklep",
                    "price": price,
                    "regular_price": null,
                    "currency": "PLN",
                    "in_stock": true,
                    "checkout_url": "https://s/1",
                    "updated_at": ingested
                },
                "externalId": "ext-1",
                "ingestedAt": ingested
            })
        };
        let mut doc = doc_of(json!({
            "slug": "p", "offers": [
                { "shop_slug": "shop-a", "price": 200.0, "updated_at": 5000 }
            ]
        }));
        // new shop -> appended
        apply_script(
            SCRIPT_APPEND_OFFER,
            &params(4000, 100.0, "shop-b"),
            &mut doc,
        )
        .unwrap();
        assert_eq!(doc["offers"].as_array().unwrap().len(), 2);
        assert_eq!(doc["external_ids"][0], "ext-1");
        // min price recomputed
        assert_eq!(doc["price"], 100.0);
        // older re-run for shop-a (4000 < 5000) -> rejected (staleness guard)
        apply_script(SCRIPT_APPEND_OFFER, &params(4000, 50.0, "shop-a"), &mut doc).unwrap();
        assert_eq!(doc["offers"][0]["price"], 200.0);
        // newer run overwrites
        apply_script(SCRIPT_APPEND_OFFER, &params(6000, 90.0, "shop-a"), &mut doc).unwrap();
        assert_eq!(doc["offers"][0]["price"], 90.0);
        assert_eq!(doc["offers"][0]["updated_at"], 6000);
        assert_eq!(doc["price"], 90.0);
        // dedup: same externalId never appended twice
        assert_eq!(doc["external_ids"].as_array().unwrap().len(), 1);
    }

    fn review_params(now_ms: i64, review: Value, summary: &str) -> Value {
        json!({
            "review": review,
            "summary": summary,
            "nowMs": now_ms,
            "halfLifeDays": 365.0,
            "halfLifeByType": { "Buty do biegania": 180.0 },
            "independenceWeights": null
        })
    }

    #[test]
    fn review_source_appends_recomputes_and_is_idempotent() {
        let mut doc = doc_of(json!({
            "slug": "asics-sonicblast",
            "type": "Buty do biegania",
            "reviews_text": "stara recenzja"
        }));
        let now = 1_760_000_000_000i64;
        let review = json!({
            "video_url": "https://youtu.be/1", "channel": "TestTube",
            "sentiment": "positive", "summary": "super buty", "published_at": now - 86400000i64 * 30,
            "source_tier": "press", "source_name": "Portal Biegowy"
        });
        apply_script(
            &render(SCRIPT_REVIEW_SOURCE),
            &review_params(now, review.clone(), "super buty"),
            &mut doc,
        )
        .unwrap();
        // append + reviews_text join with space
        assert_eq!(doc["youtube_reviews"].as_array().unwrap().len(), 1);
        assert_eq!(doc["reviews_text"], "stara recenzja super buty");
        // trust math (decay x tier weighted Wilson, z=1.96, x5, floored at 0)
        assert_eq!(doc["trust_count"], 1);
        assert!(doc["trust_score"].as_f64().unwrap() > 0.0);
        assert!(doc["trust_score"].as_f64().unwrap() < 5.0);
        // source_count: review channel + comment sources
        assert_eq!(doc["source_count"], 1);
        // trust_sources: source_name preferred over channel, tier value
        assert_eq!(doc["trust_sources"]["Portal Biegowy"], "press");
        assert_eq!(doc["trust_score_computed_at"], now);
        // aspect_stats written (empty here — no aspects on the review)
        assert!(doc["aspect_stats"].as_object().unwrap().is_empty());

        // idempotency: same video_url -> no-op
        let before = doc.clone();
        apply_script(
            &render(SCRIPT_REVIEW_SOURCE),
            &review_params(now, review, "super buty"),
            &mut doc,
        )
        .unwrap();
        assert_eq!(doc, before);
    }

    #[test]
    fn review_source_decay_older_review_scores_lower() {
        let now = 1_760_000_000_000i64;
        let mut fresh = doc_of(json!({ "type": "Inne" }));
        let mut stale = doc_of(json!({ "type": "Inne" }));
        let reviews = [(&mut fresh, 0i64), (&mut stale, 3000i64)];
        for (target, age_days) in reviews {
            let review = json!({
                "video_url": format!("https://youtu.be/{age_days}"), "channel": "C",
                "sentiment": "positive", "published_at": now - 86400000i64 * age_days
            });
            apply_script(
                &render(SCRIPT_REVIEW_SOURCE),
                &review_params(now, review, "s"),
                target,
            )
            .unwrap();
        }
        assert!(
            fresh["trust_score"].as_f64().unwrap() > stale["trust_score"].as_f64().unwrap(),
            "fresh {} vs stale {}",
            fresh["trust_score"],
            stale["trust_score"]
        );
    }

    #[test]
    fn recompute_trust_appends_nothing_and_replaces_counts() {
        let mut doc = doc_of(json!({
            "slug": "p", "type": "Buty do biegania",
            "youtube_reviews": [
                { "video_url": "a", "channel": "C1", "sentiment": "positive", "published_at": 1750000000000i64 },
                { "video_url": "b", "channel": "C2", "sentiment": "negative" }
            ],
            "comments": [ { "id": "c1", "source": "ceneo" } ],
            "trust_count": 99, "source_count": 99
        }));
        let n_reviews = doc["youtube_reviews"].as_array().unwrap().len();
        apply_script(
            &render(SCRIPT_RECOMPUTE_TRUST),
            &json!({ "nowMs": 1_760_000_000_000i64, "halfLifeDays": 365.0, "halfLifeByType": { "Buty do biegania": 180.0 }, "independenceWeights": { "laboratory": 2.0 } }),
            &mut doc,
        )
        .unwrap();
        // nothing appended, counts recomputed
        assert_eq!(doc["youtube_reviews"].as_array().unwrap().len(), n_reviews);
        assert_eq!(doc["trust_count"], 2);
        // channels C1,C2 + comment source ceneo
        assert_eq!(doc["source_count"], 3);
        assert!(doc["trust_score"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn replace_reviews_replaces_and_recomputes() {
        let mut doc = doc_of(json!({
            "slug": "p", "type": "Inne",
            "youtube_reviews": [ { "video_url": "old" } ],
            "reviews_text": "old text"
        }));
        let reviews = json!([
            { "video_url": "n1", "channel": "NC", "sentiment": "positive", "published_at": null,
              "aspects": [ { "name": "amortyzacja", "sentiment": "positive", "quote": "miekka" } ] }
        ]);
        apply_script(
            &render(SCRIPT_REPLACE_REVIEWS),
            &json!({ "reviews": reviews, "reviewsText": "new text", "nowMs": 1000i64, "halfLifeDays": 365.0, "halfLifeByType": null, "independenceWeights": null }),
            &mut doc,
        )
        .unwrap();
        assert_eq!(doc["youtube_reviews"][0]["video_url"], "n1");
        assert_eq!(doc["reviews_text"], "new text");
        // aspect stats recomputed from the NEW list
        assert_eq!(doc["aspect_stats"]["amortyzacja"]["positive"], 1);
        assert_eq!(doc["aspect_stats"]["amortyzacja"]["best_quote"], "miekka");
        assert_eq!(doc["trust_count"], 1);
    }

    #[test]
    fn aspect_backfill_sets_aspects_and_stats() {
        let mut doc = doc_of(json!({
            "slug": "p",
            "youtube_reviews": [
                { "video_url": "v1", "channel": "Chan", "sentiment": "positive" },
                { "video_url": "v2", "channel": "Chan", "sentiment": "negative",
                  "aspects": [ { "name": "waga", "sentiment": "positive", "quote": "lekka" } ] }
            ]
        }));
        let aspects = json!([
            { "name": "waga", "sentiment": "negative", "quote": "ciezka" },
            { "name": "waga", "sentiment": "mixed", "quote": "srednia" },
            { "name": "trwalosc", "sentiment": "positive" }
        ]);
        apply_script(
            &render(SCRIPT_ASPECT_BACKFILL),
            &json!({ "aspectsMap": { "v1": aspects } }),
            &mut doc,
        )
        .unwrap();
        // v1 got its aspects (idempotent skip for v2 which has some)
        assert_eq!(
            doc["youtube_reviews"][0]["aspects"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            doc["youtube_reviews"][1]["aspects"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        // aspect_stats: waga = negative 1 (from v1) + positive 1 (from v2) + mixed 1
        assert_eq!(doc["aspect_stats"]["waga"]["negative"], 1);
        assert_eq!(doc["aspect_stats"]["waga"]["positive"], 1);
        assert_eq!(doc["aspect_stats"]["waga"]["mixed"], 1);
        assert_eq!(doc["aspect_stats"]["trwalosc"]["positive"], 1);
        // best_quote: the positive quote for waga (v2's "lekka")
        assert_eq!(doc["aspect_stats"]["waga"]["best_quote"], "lekka");
        // trwalosc: no quote at all -> null best_quote (no fallback quote seen)
        assert!(doc["aspect_stats"]["trwalosc"]["best_quote"].is_null());
        // sources dedup: both from Chan
        assert_eq!(doc["aspect_stats"]["waga"]["sources"], json!(["Chan"]));
        assert_eq!(doc["aspect_stats"]["trwalosc"]["sources"], json!(["Chan"]));
    }

    #[test]
    fn comment_aspect_backfill_keyed_by_comment_id() {
        let mut doc = doc_of(json!({
            "comments": [ { "id": "cid-1", "source": "ceneo" } ]
        }));
        apply_script(
            &render(SCRIPT_COMMENT_ASPECT_BACKFILL),
            &json!({ "aspectsMap": { "cid-1": [ { "name": "cena", "sentiment": "negative", "quote": "drogo" } ] } }),
            &mut doc,
        )
        .unwrap();
        assert_eq!(doc["comments"][0]["aspects"][0]["name"], "cena");
        // comment source default is the stored source ("ceneo")
        assert_eq!(doc["aspect_stats"]["cena"]["negative"], 1);
        assert_eq!(doc["aspect_stats"]["cena"]["sources"], json!(["ceneo"]));
        // best_quote fallback: first quote seen — even a negative-only one
        assert_eq!(doc["aspect_stats"]["cena"]["best_quote"], "drogo");
    }

    #[test]
    fn first_quote_fallback_for_aspect_without_positive_quote() {
        let mut doc = doc_of(json!({
            "youtube_reviews": [
                { "video_url": "v1", "channel": "C", "sentiment": "negative",
                  "aspects": [ { "name": "waga", "sentiment": "negative", "quote": "ciezka" } ] }
            ]
        }));
        apply_script(
            &render(SCRIPT_ASPECT_BACKFILL),
            &json!({ "aspectsMap": {} }),
            &mut doc,
        )
        .unwrap();
        // no positive quote -> fallback to the first quote seen at all
        assert_eq!(doc["aspect_stats"]["waga"]["best_quote"], "ciezka");
    }

    // ---- bulk ndjson --------------------------------------------------------

    #[test]
    fn bulk_parse_mixed_ops() {
        let ndjson = concat!(
            r#"{"index": {"_index": "products", "_id": "a", "version_type": "external", "version": 123}}"#,
            "\n",
            r#"{"slug": "a", "title": "A"}"#,
            "\n",
            r#"{"update": {"_index": "products", "_id": "b"}}"#,
            "\n",
            r#"{"doc": {"available": false}}"#,
            "\n",
            r#"{"update": {"_index": "products", "_id": "c"}}"#,
            "\n",
            r#"{"script": {"source": "ctx._source.type = params.category", "params": {"category": "X"}}}"#,
            "\n",
            r#"{"delete": {"_index": "products", "_id": "d"}}"#,
            "\n",
        );
        let actions = parse_bulk(ndjson, "products").unwrap();
        assert_eq!(actions.len(), 4);
        match &actions[0] {
            BulkAction::Index {
                id,
                version_type,
                version,
                source,
                ..
            } => {
                assert_eq!(id, "a");
                assert_eq!(version_type.as_deref(), Some("external"));
                assert_eq!(*version, Some(123));
                assert_eq!(source["title"], "A");
            }
            _ => panic!("expected index"),
        }
        match &actions[1] {
            BulkAction::Update { id, body, .. } => {
                assert_eq!(id, "b");
                assert_eq!(body["doc"]["available"], false);
            }
            _ => panic!("expected update"),
        }
        match &actions[2] {
            BulkAction::Update { id, body, .. } => {
                assert_eq!(id, "c");
                assert!(body["script"]["source"].is_string());
            }
            _ => panic!("expected update"),
        }
        match &actions[3] {
            BulkAction::Delete { id, .. } => assert_eq!(id, "d"),
            _ => panic!("expected delete"),
        }
        // default index applies when _index missing
        let a2 = parse_bulk(
            "{\"index\": {\"_id\": \"z\"}}\n{\"slug\": \"z\"}\n",
            "products",
        )
        .unwrap();
        match &a2[0] {
            BulkAction::Index { index, .. } => assert_eq!(index, "products"),
            _ => panic!(),
        }
        // unknown op rejected
        assert!(parse_bulk("{\"create\": {\"_id\": \"x\"}}\n", "products").is_err());
        // missing _id rejected
        assert!(parse_bulk("{\"index\": {}}\n", "products").is_err());
    }

    #[test]
    fn extract_embedding_pulls_float_array() {
        let mut doc = doc_of(json!({ "slug": "s", "embedding": [1.5, 2.5] }));
        assert_eq!(extract_embedding(&mut doc), Some(vec![1.5, 2.5]));
        assert!(!doc.contains_key("embedding"));
        assert!(extract_embedding(&mut doc).is_none());
    }
}
