//! Engine integration tests — a synthetic corpus running the REAL query
//! bodies the three Rust services send (copied verbatim from
//! truespotter-catalog/src/opensearch.rs, truespotter-chat/src/opensearch.rs
//! and truespotter-admin/src/products.rs), asserting hit structure,
//! facet counts, search_after determinism, backward-compatible
//! approved/available filters and kNN ordering. Plus the response-shape
//! golden-keys test.

use crate::api::{search_response, AppState};
use crate::dsn::parse_search_body;
use crate::engine::{Engine, EngineDoc};
use serde_json::{json, Value};
use std::sync::Arc;

fn doc(slug: &str, extra: Value) -> EngineDoc {
    let mut base = json!({
        "slug": slug,
        "title": slug,
        "vendor": "Test vendor",
        "type": "Inne",
        "price": 0.0,
        "currency": "PLN",
    });
    if let (Some(b), Value::Object(e)) = (base.as_object_mut(), extra) {
        for (k, v) in e {
            b.insert(k, v);
        }
    }
    EngineDoc {
        slug: slug.to_string(),
        doc: base,
        embedding: None,
    }
}

fn corpus() -> Vec<EngineDoc> {
    let mut docs = vec![
        doc(
            "asics-sonicblast",
            json!({
                "title": "ASICS Sonicblast", "vendor": "top4running.pl", "brand": "ASICS",
                "type": "Buty do biegania", "category_path": "Buty do biegania/Buty męskie",
                "price": 447.3, "currency": "PLN", "status": "approved", "available": true,
                "tags": ["bieganie", "bieganie-miejskie"],
                "description": "Buty do biegania po mieście",
                "specs_text": "Waga: 250 g\nDrop: 8 mm",
                "reviews_text": "świetne buty do biegania duży komfort",
                "trust_score": 0.8, "trust_count": 5,
                "created_at": "2026-01-10T00:00:00Z",
                "image_url": "https://i.example.com/1.png"
            }),
        ),
        doc(
            "asics-gel-kayano",
            json!({
                "title": "ASICS Gel-Kayano 30", "vendor": "top4running.pl", "brand": "ASICS",
                "type": "Buty do biegania", "price": 600.0, "currency": "PLN",
                "status": "approved", "available": true,
                "reviews_text": "stabilne buty do biegania",
                "trust_score": 0.9, "trust_count": 10,
                "created_at": "2026-02-10T00:00:00Z"
            }),
        ),
        doc(
            "nike-air-max",
            json!({
                "title": "Nike Air Max", "vendor": "Ceneo",
                "type": "Buty sportowe", "price": 300.0, "currency": "PLN",
                "created_at": "2026-03-10T00:00:00Z"
            }),
        ),
        doc(
            "nike-pegasus",
            json!({
                "title": "Nike Pegasus 40", "vendor": "Nike", "brand": "Nike",
                "type": "Buty sportowe", "price": 500.0, "currency": "PLN",
                "status": "approved", "available": false,
                "created_at": "2026-04-10T00:00:00Z"
            }),
        ),
        doc(
            "lumina-tv-4k",
            json!({
                "title": "Lumina TV 4K", "vendor": "Media Expert",
                "type": "Elektronika", "price": 2000.0, "currency": "EUR",
                "status": "approved", "available": true, "tags": ["tv"],
                "created_at": "2026-05-10T00:00:00Z"
            }),
        ),
        doc(
            "lumina-tv-8k",
            json!({
                "title": "Lumina TV 8K", "vendor": "Media Expert",
                "type": "Elektronika", "price": 5000.0, "currency": "EUR",
                "status": "approved", "available": true,
                "created_at": "2026-06-10T00:00:00Z"
            }),
        ),
        doc(
            "ksiazka-fantastyka-1",
            json!({
                "title": "Powieść pierwsza", "vendor": "Empik",
                "type": "Książki", "author": "Autor X", "genre": "fantastyka",
                "price": 49.9, "currency": "PLN", "status": "approved", "available": true,
                "created_at": "2026-07-10T00:00:00Z"
            }),
        ),
        doc(
            "ksiazka-fantastyka-2",
            json!({
                "title": "Powieść druga", "vendor": "Empik",
                "type": "Książki", "author": "Autor X", "genre": "fantastyka",
                "price": 59.9, "currency": "PLN", "status": "approved", "available": true,
                "created_at": "2026-08-10T00:00:00Z"
            }),
        ),
        doc(
            "telefon-super",
            json!({
                "title": "Telefon Super", "vendor": "Sklep GSM",
                "type": "Telefony", "price": 1500.0, "currency": "PLN",
                "status": "approved", "available": true,
                "created_at": "2026-09-10T00:00:00Z"
            }),
        ),
        doc(
            "telefon-super-pro",
            json!({
                "title": "Telefon Super Pro", "vendor": "Sklep GSM",
                "type": "Telefony", "price": 3000.0, "currency": "PLN",
                "status": "approved", "available": true,
                "created_at": "2026-09-11T00:00:00Z"
            }),
        ),
        doc(
            "numeric-category-doc",
            json!({
                "title": "Numeric Category", "vendor": "Test",
                "type": "123", "price": 100.0, "currency": "PLN",
                "status": "approved", "available": true
            }),
        ),
        doc(
            "sluchawki-bedace",
            json!({
                "title": "Słuchawki Będące Świetne", "vendor": "Sklep Audio", "brand": "Lumina",
                "type": "Elektronika", "price": 100.0, "currency": "PLN",
                "status": "approved", "available": true,
                "reviews_text": "świetny dźwięk",
                "embedding": [0.8, 0.2, 0.0, 0.0],
                "created_at": "2026-09-12T00:00:00Z"
            }),
        ),
    ];
    // Test-dim (4) vectors — ordering: sonicblast > gel-kayano > sluchawki.
    let with_emb = [
        ("asics-sonicblast", vec![1.0, 0.0, 0.0, 0.0]),
        ("asics-gel-kayano", vec![0.9, 0.1, 0.0, 0.0]),
        ("sluchawki-bedace", vec![0.8, 0.2, 0.0, 0.0]),
    ];
    for (slug, emb) in with_emb {
        if let Some(d) = docs.iter_mut().find(|d| d.slug == slug) {
            d.embedding = Some(emb);
        }
    }
    docs
}

fn engine() -> Engine {
    Engine::build(corpus(), None).expect("build engine")
}

fn search(e: &Engine, body: &Value) -> Value {
    let req = parse_search_body(body).expect("parse body");
    let outcome = e.search(&req).expect("search");
    search_response("products", &outcome)
}

fn hit_slugs(data: &Value) -> Vec<String> {
    data.pointer("/hits/hits")
        .and_then(|h| h.as_array())
        .map(|arr| {
            arr.iter()
                .map(|h| {
                    h.get("_id")
                        .and_then(|i| i.as_str())
                        .unwrap_or_default()
                        .to_string()
                })
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// catalog-rs bodies
// ---------------------------------------------------------------------------

/// .NET `ApprovedFilter()` — verbatim from catalog-rs.
pub fn approved_filter() -> Value {
    json!({
        "bool": {
            "should": [
                { "term": { "status.keyword": "approved" } },
                { "bool": { "must_not": { "exists": { "field": "status" } } } }
            ],
            "minimum_should_match": 1
        }
    })
}

fn available_filter() -> Value {
    json!({
        "bool": {
            "should": [
                { "term": { "available": true } },
                { "bool": { "must_not": { "exists": { "field": "available" } } } }
            ],
            "minimum_should_match": 1
        }
    })
}

const TRUST_SCRIPT: &str =
    "doc.containsKey('trust_score') && doc['trust_score'].size() != 0 ? Math.max(0.0, doc['trust_score'].value * Math.log1p(doc.containsKey('trust_count') && doc['trust_count'].size() != 0 ? doc['trust_count'].value : 0)) : 0";

#[test]
fn catalog_get_by_slug_body() {
    let e = engine();
    // Verbatim get_by_slug body (catalog-rs OpenSearchClient::get_by_slug).
    let body = json!({
        "size": 1,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "must": [{ "term": { "slug.keyword": "asics-sonicblast" } }],
                "filter": [approved_filter()]
            }
        }
    });
    let resp = search(&e, &body);
    assert_eq!(
        resp.pointer("/hits/total/value").and_then(Value::as_i64),
        Some(1)
    );
    let slug = resp
        .pointer("/hits/hits/0/_source/slug")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(slug, "asics-sonicblast");
    // _source excludes removed the embedding key.
    assert!(resp
        .pointer("/hits/hits/0/_source")
        .unwrap()
        .get("embedding")
        .is_none());
}

#[test]
fn catalog_get_by_slugs_terms_body() {
    let e = engine();
    let body = json!({
        "size": 2,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "must": [{ "terms": { "slug.keyword": ["asics-sonicblast", "lumina-tv-4k"] } }],
                "filter": [approved_filter()]
            }
        }
    });
    let resp = search(&e, &body);
    assert_eq!(
        resp.pointer("/hits/total/value").and_then(Value::as_i64),
        Some(2)
    );
}

#[test]
fn catalog_full_bm25_search_with_facets_and_price_sort() {
    let e = engine();
    // The SearchAsync body: bool should (multi_match + phrase + fuzzy) msm=1,
    // filters (category/price/available/approved), function_score trust,
    // sort price-asc + slug tie-break, aggs with the `filtered` wrapper +
    // price_stats. Category filter: Elektronika → 3 docs (4k, 8k, słuchawki).
    let body = json!({
        "size": 10,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [{ "match_all": {} }],
                        "filter": [
                            { "term": { "type.keyword": "Elektronika" } },
                            available_filter(),
                            approved_filter()
                        ]
                    }
                },
                "script_score": { "script": { "source": TRUST_SCRIPT } },
                "boost_mode": "sum"
            }
        },
        "sort": [
            { "price": { "order": "asc", "missing": "_last" } },
            { "slug.keyword": { "order": "asc" } }
        ],
        "aggs": {
            "filtered": {
                "filter": { "term": { "type.keyword": "Elektronika" } },
                "aggs": {
                    "categories": { "terms": { "field": "type.keyword", "size": 20 } },
                    "brands": { "terms": { "field": "brand.keyword", "size": 20 } },
                    "currencies": { "terms": { "field": "currency.keyword", "size": 10 } },
                    "price_stats": {
                        "filter": { "term": { "currency.keyword": "PLN" } },
                        "aggs": { "stats": { "stats": { "field": "price" } } }
                    }
                }
            }
        }
    });
    let resp = search(&e, &body);
    assert_eq!(
        resp.pointer("/hits/total/value").and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        hit_slugs(&resp),
        vec!["sluchawki-bedace", "lumina-tv-4k", "lumina-tv-8k"]
    );
    // hit.sort = [price, slug] as requested.
    assert_eq!(
        resp.pointer("/hits/hits/0/sort").unwrap(),
        &json!([100.0, "sluchawki-bedace"])
    );
    // sorted by field → _score is null.
    assert!(resp.pointer("/hits/hits/0/_score").unwrap().is_null());
    // Facets: filtered wrapper shape + counts.
    let cats = resp
        .pointer("/aggregations/filtered/categories/buckets")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(cats.len(), 1);
    assert_eq!(cats[0]["key"], "Elektronika");
    assert_eq!(cats[0]["doc_count"], 3);
    // price_stats over the PLN subset (słuchawki only): min=max=100.
    let stats = resp.pointer("/aggregations/filtered/price_stats").unwrap();
    assert_eq!(stats["doc_count"], 1);
    assert_eq!(stats["stats"]["min"], 100.0);
    assert_eq!(stats["stats"]["max"], 100.0);
    assert_eq!(stats["stats"]["count"], 1);
}

#[test]
fn catalog_bm25_relevance_ranks_the_exact_phrase_first() {
    let e = engine();
    // build_bool_object should-clauses (verbatim shape) + function_score.
    let body = json!({
        "size": 5,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [{ "match_all": {} }],
                        "should": [
                            {
                                "multi_match": {
                                    "query": "asics sonicblast",
                                    "fields": ["title^3", "vendor^2", "description", "tags", "specs_text", "reviews_text"],
                                    "type": "best_fields",
                                    "operator": "or",
                                    "minimum_should_match": "2<75%",
                                    "lenient": true
                                }
                            },
                            { "match_phrase": { "title": { "query": "asics sonicblast", "boost": 6 } } },
                            { "match": { "title": { "query": "asics sonicblast", "fuzziness": "AUTO", "prefix_length": 2 } } }
                        ],
                        "minimum_should_match": 1,
                        "filter": [available_filter(), approved_filter()]
                    }
                },
                "script_score": { "script": { "source": TRUST_SCRIPT } },
                "boost_mode": "sum"
            }
        },
        "sort": [
            { "_score": { "order": "desc" } },
            { "slug.keyword": { "order": "asc" } }
        ]
    });
    let resp = search(&e, &body);
    let slugs = hit_slugs(&resp);
    assert_eq!(slugs[0], "asics-sonicblast");
    assert!(slugs.contains(&"asics-gel-kayano".to_string()));
    assert!(
        resp.pointer("/hits/hits/0/_score")
            .unwrap()
            .as_f64()
            .unwrap()
            > 0.0
    );
}

#[test]
fn catalog_search_after_pagination_is_deterministic_and_deduped() {
    let e = engine();
    let base = json!({
        "size": 4,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "must": [{ "match_all": {} }],
                "filter": [available_filter(), approved_filter()]
            }
        },
        "sort": [
            { "price": { "order": "asc", "missing": "_last" } },
            { "slug.keyword": { "order": "asc" } }
        ]
    });
    let page1 = search(&e, &base);
    assert_eq!(hit_slugs(&page1).len(), 4);
    let last_sort = page1
        .pointer("/hits/hits/3/sort")
        .unwrap()
        .as_array()
        .unwrap()
        .clone();
    let mut body2 = base.clone();
    body2["search_after"] = Value::Array(last_sort.clone());
    let page2 = search(&e, &body2);
    let s1 = hit_slugs(&page1);
    let s2 = hit_slugs(&page2);
    // No overlap between pages.
    for s in &s2 {
        assert!(!s1.contains(s), "duplicate slug {s} across pages");
    }
    // Determinism: identical query twice → identical order.
    let page2b = search(&e, &body2);
    assert_eq!(s2, hit_slugs(&page2b));
    // Cursor mid-sequence: sorted price asc the docs are
    // ksiazka-1(49.9), ksiazka-2(59.9), numeric(100), sluchawki(100),
    // nike-air-max(300)... → page 2 starts at nike-air-max.
    assert_eq!(s2[0], "nike-air-max");
}

#[test]
fn catalog_approved_available_backward_compatibility() {
    let e = engine();
    // nike-air-max has NO status and NO available → must still match both.
    let body = json!({
        "size": 20,
        "query": {
            "bool": {
                "must": [
                    { "term": { "slug.keyword": "nike-air-max" } }
                ],
                "filter": [available_filter(), approved_filter()]
            }
        }
    });
    let resp = search(&e, &body);
    assert_eq!(
        resp.pointer("/hits/total/value").and_then(Value::as_i64),
        Some(1)
    );
    // available=false doc excluded by the available filter.
    let body = json!({
        "size": 20,
        "query": {
            "bool": {
                "must": [{ "term": { "slug.keyword": "nike-pegasus" } }],
                "filter": [available_filter(), approved_filter()]
            }
        }
    });
    assert_eq!(
        search(&e, &body)
            .pointer("/hits/total/value")
            .and_then(Value::as_i64),
        Some(0)
    );
}

#[test]
fn catalog_get_by_category_body_with_prefix() {
    let e = engine();
    // Verbatim get_by_category should-clause: prefix category_path + terms.
    let body = json!({
        "size": 10,
        "from": 0,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "must": [{
                    "bool": {
                        "should": [
                            { "prefix": { "category_path.keyword": "Buty do biegania/" } },
                            { "term": { "category_path.keyword": "Buty do biegania" } },
                            { "term": { "category.keyword": "Buty do biegania" } },
                            { "term": { "type.keyword": "Buty do biegania" } }
                        ],
                        "minimum_should_match": 1
                    }
                }],
                "filter": [available_filter(), approved_filter()]
            }
        }
    });
    let resp = search(&e, &body);
    assert_eq!(
        resp.pointer("/hits/total/value").and_then(Value::as_i64),
        Some(2)
    );
    assert!(hit_slugs(&resp).contains(&"asics-sonicblast".to_string()));
}

#[test]
fn catalog_newest_sorts_by_doc_order_desc() {
    let e = engine();
    // get_newest: _seq_no desc + _doc desc (most recently indexed first).
    let body = json!({
        "size": 3,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "must": [{ "match_all": {} }],
                "filter": [available_filter(), approved_filter()]
            }
        },
        "sort": [
            { "_seq_no": { "order": "desc" } },
            { "_doc": { "order": "desc" } }
        ]
    });
    let resp = search(&e, &body);
    assert_eq!(hit_slugs(&resp)[0], "sluchawki-bedace");
}

#[test]
fn catalog_list_categories_aggs_exclude_numeric_regex() {
    let e = engine();
    // Verbatim list_categories body — exclude ["", "[0-9]+", "[0-9][0-9 -.]*"].
    let body = json!({
        "size": 0,
        "aggs": {
            "cats": {
                "terms": {
                    "field": "type.keyword",
                    "size": 500,
                    "exclude": ["", "[0-9]+", "[0-9][0-9 -.]*"]
                }
            }
        },
        "query": { "bool": { "filter": [approved_filter()] } }
    });
    let resp = search(&e, &body);
    let buckets = resp
        .pointer("/aggregations/cats/buckets")
        .unwrap()
        .as_array()
        .unwrap()
        .clone();
    let names: Vec<&str> = buckets.iter().map(|b| b["key"].as_str().unwrap()).collect();
    assert!(!names.contains(&"123"), "numeric category excluded");
    assert_eq!(buckets[0]["key"], "Elektronika");
    assert_eq!(buckets[0]["doc_count"], 3);
    // Approved docs (incl. legacy nike-air-max without status):
    // Buty do biegania 2, Buty sportowe 2, Elektronika 3, Książki 2,
    // Telefony 2 — numeric "123" excluded by the regex. 5 buckets.
    assert_eq!(buckets.len(), 5);
}

#[test]
fn catalog_list_shops_vendor_agg() {
    let e = engine();
    let body = json!({
        "size": 0,
        "aggs": { "shops": { "terms": { "field": "vendor.keyword", "size": 1000 } } },
        "query": { "bool": { "filter": [approved_filter()] } }
    });
    let resp = search(&e, &body);
    let buckets = resp
        .pointer("/aggregations/shops/buckets")
        .unwrap()
        .as_array()
        .unwrap();
    let top = buckets[0].clone();
    assert!(top["doc_count"].as_i64().unwrap() >= 2); // Empik or Sklep GSM or Media Expert
}

#[test]
fn catalog_autocomplete_phrase_prefix_and_brand_match() {
    let e = engine();
    // Verbatim autocomplete body.
    let body = json!({
        "size": 5,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "should": [
                    { "match_phrase_prefix": { "title": { "query": "asics son", "boost": 2 } } },
                    { "match": { "brand": "lumina" } }
                ],
                "minimum_should_match": 1,
                "filter": [approved_filter()]
            }
        }
    });
    let resp = search(&e, &body);
    let slugs = hit_slugs(&resp);
    assert!(slugs.contains(&"asics-sonicblast".to_string()));
    assert!(slugs.contains(&"sluchawki-bedace".to_string())); // brand "Lumina" match
}

#[test]
fn catalog_related_function_score_random_score_seeded() {
    let e = engine();
    let body = json!({
        "size": 2,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [{
                            "bool": {
                                "should": [
                                    { "term": { "type.keyword": "Buty do biegania" } }
                                ],
                                "minimum_should_match": 1
                            }
                        }],
                        "must_not": [{ "term": { "slug.keyword": "asics-sonicblast" } }],
                        "filter": [available_filter(), approved_filter()]
                    }
                },
                "functions": [{ "random_score": { "seed": 42, "field": "_seq_no" } }]
            }
        }
    });
    let resp = search(&e, &body);
    assert_eq!(
        resp.pointer("/hits/total/value").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(hit_slugs(&resp)[0], "asics-gel-kayano");
    // Same seed → same order (deterministic).
    assert_eq!(hit_slugs(&search(&e, &body))[0], "asics-gel-kayano");
}

#[test]
fn trust_script_scores_higher_trust_above_match_all() {
    let e = engine();
    let body = json!({
        "size": 10,
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [{ "match_all": {} }],
                        "filter": [available_filter(), approved_filter()]
                    }
                },
                "script_score": { "script": { "source": TRUST_SCRIPT } },
                "boost_mode": "sum"
            }
        },
        "sort": [{ "_score": { "order": "desc" } }]
    });
    let resp = search(&e, &body);
    // gel-kayano (0.9×ln11) > sonicblast (0.8×ln6) > everything else (+0).
    assert_eq!(hit_slugs(&resp)[0], "asics-gel-kayano");
    assert_eq!(hit_slugs(&resp)[1], "asics-sonicblast");
}

#[test]
fn polish_diacritics_fold_in_text_search() {
    let e = engine();
    // "świetny dźwięk" indexed with diacritics — query without them folds.
    let body = json!({
        "size": 5,
        "query": {
            "bool": {
                "must": [{ "match": { "reviews_text": { "query": "swietny dzwiek" } } }],
                "filter": [approved_filter()]
            }
        }
    });
    let resp = search(&e, &body);
    assert!(hit_slugs(&resp).contains(&"sluchawki-bedace".to_string()));
    // Phrase with diacritics also works (fold at index time).
    let body = json!({
        "size": 5,
        "query": { "match_phrase": { "title": { "query": "Słuchawki Będące" } } }
    });
    let resp = search(&e, &body);
    assert!(hit_slugs(&resp).contains(&"sluchawki-bedace".to_string()));
}

#[test]
fn fuzzy_match_finds_one_edit_distance_titles() {
    let e = engine();
    // "sonicblust" is 1 edit from "sonicblast".
    let body = json!({
        "size": 5,
        "query": {
            "match": { "title": { "query": "sonicblust", "fuzziness": "AUTO", "prefix_length": 2 } }
        }
    });
    let resp = search(&e, &body);
    assert!(hit_slugs(&resp).contains(&"asics-sonicblast".to_string()));
}

// ---------------------------------------------------------------------------
// chat-rs bodies
// ---------------------------------------------------------------------------

/// W2 (spec index-rs-optimization): kNN ranking must be invariant to the
/// SCALE of both stored and query vectors — cosine semantics. A stored
/// vector scaled 10× and an unnormalized query must hit exactly the same
/// top hit with the same score as the unit-length pair.
#[test]
fn knn_scale_invariant_after_normalization() {
    let mut docs = corpus();
    // scale sonicblast's vector 10× at build time (build normalizes)
    for d in docs.iter_mut() {
        if d.slug == "asics-sonicblast" {
            d.embedding = Some(vec![10.0, 0.0, 0.0, 0.0]);
        }
    }
    let e = Engine::build(docs, None).expect("build engine");
    // unnormalized query (norm 10) — same direction as [1,0,0,0]
    let res = search(
        &e,
        &json!({
            "size": 3,
            "query": { "bool": { "must": [ { "knn": { "embedding": { "vector": [10.0, 0.0, 0.0, 0.0], "k": 3 } } } ] } }
        }),
    );
    let slugs = hit_slugs(&res);
    assert_eq!(slugs[0], "asics-sonicblast", "direction beats magnitude");
    let score = res.pointer("/hits/hits/0/_score").unwrap().as_f64().unwrap();
    assert!(score > 0.999, "cosine 1.0 → score (1+cos)/2 ≈ 1, got {score}");
}

/// Spec index-rs-hybrid-normalization (W4): in a hybrid bool (kNN among
/// the should clauses) a doc that is rank-1 of the kNN clause with NO text
/// match must score at least 50% of a doc that is rank-1 of the text clause
/// with no vector — the raw BM25 scale (unbounded) must not drown the kNN
/// contribution ((1+cos)/2 ∈ [0,1]).
#[test]
fn hybrid_knn_not_drowned_by_bm25_scale() {
    let mut docs = Vec::new();
    // 40 filler docs make "telefon" rare → high-idf BM25 (raw score ~5+)
    for i in 0..40 {
        docs.push(doc(&format!("filler-{i}"), json!({ "title": "filler" })));
    }
    docs.push(doc(
        "telefon-t",
        json!({ "title": "Telefon", "embedding": [0.0, 0.0, 0.0, 0.0] }),
    ));
    docs.push(doc(
        "xyz-k",
        json!({ "title": "xyzxyz", "embedding": [1.0, 0.0, 0.0, 0.0] }),
    ));
    for d in docs.iter_mut() {
        if d.doc.get("embedding").is_some() {
            d.embedding = d
                .doc
                .get("embedding")
                .and_then(|e| e.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                });
        }
    }
    let e = Engine::build(docs, None).expect("build engine");
    let res = search(
        &e,
        &json!({
            "size": 10,
            "query": {
                "bool": {
                    "should": [
                        { "match": { "title": { "query": "telefon" } } },
                        { "knn": { "embedding": { "vector": [1.0, 0.0, 0.0, 0.0], "k": 2 } } }
                    ],
                    "minimum_should_match": 1
                }
            }
        }),
    );
    let score_of = |slug: &str| -> f64 {
        res.pointer("/hits/hits")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .find(|h| h["_id"] == slug)
            .and_then(|h| h["_score"].as_f64())
            .unwrap_or(0.0)
    };
    let t = score_of("telefon-t"); // text rank-1, knn bottom
    let k = score_of("xyz-k"); // knn rank-1, no text match
    assert!(t > 0.0 && k > 0.0, "both hybrid docs must match: t={t} k={k}");
    assert!(
        k >= 0.5 * t,
        "kNN rank-1 contribution must survive the BM25 scale: k={k} t={t}"
    );
}

/// Spec index-rs-pl-stemming (W5): inflected Polish query forms match the
/// indexed base forms — both sides tokenize through the same stemmer.
#[test]
fn polish_stemming_matches_inflections() {
    let docs = vec![
        doc(
            "buty-bieganie",
            json!({ "title": "Buty do biegania", "description": "sukienka letnia" }),
        ),
        doc("telefon", json!({ "title": "Telefon" })),
        doc("kawa", json!({ "title": "Kawa" })),
    ];
    let e = Engine::build(docs, None).expect("build engine");
    // genitive plural of "buty" ("butów" → fold "butow" → strip "ow" → "but")
    let res = search(&e, &json!({ "size": 5, "query": { "match": { "title": "butów" } } }));
    assert!(hit_slugs(&res).contains(&"buty-bieganie".to_string()));
    // gerund inflection ("bieganie" ≡ "biegania" → both stem to "biegani")
    let res = search(&e, &json!({ "size": 5, "query": { "match": { "title": "bieganie" } } }));
    assert!(hit_slugs(&res).contains(&"buty-bieganie".to_string()));
    // singular/plural noun ("sukienki" ≡ "sukienka" → "sukienk")
    let res = search(&e, &json!({ "size": 5, "query": { "match": { "description": "sukienki" } } }));
    assert!(hit_slugs(&res).contains(&"buty-bieganie".to_string()));
    // short stems pass through — "kaw" (from "kawa") does not collide with
    // an unrelated word, and unstemed queries on no-suffix words still hit
    let res = search(&e, &json!({ "size": 5, "query": { "match": { "title": "kawy" } } }));
    assert!(hit_slugs(&res).contains(&"kawa".to_string()));
    // "telefonu" (genitive) matches "Telefon" ("telefonu" → "telefon")
    let res = search(&e, &json!({ "size": 5, "query": { "match": { "title": "telefonu" } } }));
    assert!(hit_slugs(&res).contains(&"telefon".to_string()));
}


/// Spec index-rs-int8-quantization (W5): i8-quantized kNN scores stay
/// within 1% of the exact f32 cosine, and ranking is preserved.
#[test]
fn int8_knn_scores_within_tolerance_of_cosine() {
    let raw_a = [0.3, 0.5, 0.8, 0.2];
    let raw_b = [0.2, 0.4, 0.9, 0.1];
    let mut docs = vec![
        doc("vec-a", json!({ "title": "A", "embedding": raw_a.to_vec() })),
        doc("vec-b", json!({ "title": "B", "embedding": raw_b.to_vec() })),
    ];
    for d in docs.iter_mut() {
        d.embedding = d
            .doc
            .get("embedding")
            .and_then(|e| e.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            });
    }
    let e = Engine::build(docs, None).expect("build engine");
    let query = [0.1, 0.3, 0.7, 0.4];
    let res = search(
        &e,
        &json!({
            "size": 5,
            "query": { "bool": { "must": [ { "knn": { "embedding": { "vector": query.to_vec(), "k": 2 } } } ] } }
        }),
    );
    // exact f64 cosine reference
    let cos = |x: &[f64], y: &[f64]| -> f64 {
        let dot: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
        let nx: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let ny: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        dot / (nx * ny)
    };
    for hit in res.pointer("/hits/hits").unwrap().as_array().unwrap() {
        let slug = hit["_id"].as_str().unwrap();
        let row: Vec<f64> = if slug == "vec-a" {
            raw_a.iter().map(|f| *f as f64).collect()
        } else {
            raw_b.iter().map(|f| *f as f64).collect()
        };
        let expected = (1.0 + cos(&query.iter().map(|f| *f as f64).collect::<Vec<_>>(), &row)) / 2.0;
        let got = hit["_score"].as_f64().unwrap();
        assert!(
            (got - expected).abs() < 0.01,
            "{slug}: int8 score {got} within 1% of f32 cosine {expected}"
        );
    }
    // ranking preserved: a (closer to the query direction, cos 0.938 vs
    // b's 0.926) ranks first — identical to the f32 ranking
    assert_eq!(res.pointer("/hits/hits/0/_id").unwrap(), "vec-a");
}

/// Spec index-rs-dynamic-synonyms (W4/W7): the dictionary is hot-swappable
/// — a freshly set dict takes effect on the very next query, no rebuild.
#[test]
fn synonym_dict_hot_swap_applies_to_next_query() {
    let docs = vec![
        doc("kawa", json!({ "title": "Kawa ziarnista" })),
        doc("czajnik", json!({ "title": "Czajnik" })),
    ];
    let e = Engine::build(docs, None).expect("build engine");
    // no dict — "coffee" matches nothing
    let res = search(&e, &json!({ "size": 5, "query": { "match": { "title": "coffee" } } }));
    assert!(hit_slugs(&res).is_empty(), "{:?}", hit_slugs(&res));
    // hot-swap in a pair — immediately effective
    e.set_synonyms(crate::engine::SynonymDict::from_raw([
        (vec!["kawa".into()], vec!["coffee".into()]),
    ]));
    let res = search(&e, &json!({ "size": 5, "query": { "match": { "title": "coffee" } } }));
    assert!(hit_slugs(&res).contains(&"kawa".to_string()), "{:?}", hit_slugs(&res));
}

/// Spec index-rs-llm-synonym-mining: the LLM answer parser accepts a bare
/// JSON object (fences tolerated), validates sides (1-4 words, alnum) and
/// drops self-synonyms (same stem sequence).
#[test]
fn llm_pair_parser_validates_and_filters() {
    let content = r#"```json
{"pairs": [["kawa", "coffee"], ["kawa", "herbata zbożowa"], ["czajnik", "czajnik"], ["kawa", "a"], ["pralka", "P,RL.KA"]]}
```"#;
    let pairs = crate::synonym_gen::parse_llm_pairs(content, "kawa");
    assert_eq!(pairs.len(), 2, "{pairs:?}");
    assert_eq!(pairs[0], (vec!["kawa".to_string()], vec!["coffee".to_string()]));
    // self-synonym (same stems) dropped; too-short side dropped; punctuation dropped
    let from_raw = crate::engine::SynonymDict::from_raw(pairs);
    assert_eq!(from_raw.len(), 2);
    // garbage → empty, never a panic
    assert!(crate::synonym_gen::parse_llm_pairs("kawa to kawa", "kawa").is_empty());
}

/// Spec index-rs-query-mining (W1/W2): match/multi_match queries feed the
/// in-RAM term counter — the miner's priority candidate source.
#[test]
fn query_log_counts_tokens_per_match() {
    let docs = vec![
        doc("buty", json!({ "title": "Buty do biegania" })),
        doc("telefon", json!({ "title": "Telefon" })),
    ];
    let e = Engine::build(docs, None).expect("build engine");
    for _ in 0..5 {
        search(&e, &json!({ "size": 1, "query": { "multi_match": { "query": "buty do biegania", "fields": ["title"] } } }));
    }
    search(&e, &json!({ "size": 1, "query": { "match": { "title": "telefon" } } }));
    let terms = e.query_terms(10);
    // every multi_match token counted 5x (stems: but, do, biegan)
    let by_term: std::collections::HashMap<String, (u64, u64)> =
        terms.into_iter().map(|(t, c, m)| (t, (c, m))).collect();
    assert_eq!(by_term.get("but"), Some(&(5, 0)), "{by_term:?}");
    assert_eq!(by_term.get("biegan"), Some(&(5, 0)), "{by_term:?}");
    assert_eq!(by_term.get("telefon"), Some(&(1, 0)), "{by_term:?}");
    // fuzzy match queries feed the counter too
    search(&e, &json!({ "size": 1, "query": { "match": { "title": { "query": "telefun", "fuzziness": "AUTO" } } } }));
    let by_term: std::collections::HashMap<String, (u64, u64)> =
        e.query_terms(10).into_iter().map(|(t, c, m)| (t, (c, m))).collect();
    assert_eq!(by_term.get("telefun"), Some(&(1, 0)), "{by_term:?}");
}

/// Spec index-rs-zero-hit-mining (W1): a clause that matches NOTHING
/// increments the miss counter for all its tokens — and miss-first
/// ordering puts those terms at the top of the miner's candidate list.
#[test]
fn zero_hit_queries_increment_miss_counter() {
    let docs = vec![
        doc("buty", json!({ "title": "Buty do biegania" })),
    ];
    let e = Engine::build(docs, None).expect("build engine");
    // 2 misses + 1 hit for "buty"-like queries
    for _ in 0..2 {
        search(&e, &json!({ "size": 1, "query": { "multi_match": { "query": "adidasy tanie", "fields": ["title"] } } }));
    }
    search(&e, &json!({ "size": 1, "query": { "match": { "title": "buty" } } }));
    // one match-query miss
    search(&e, &json!({ "size": 1, "query": { "match": { "title": "krakers" } } }));

    let terms = e.query_terms(10);
    // miss-first ordering: "adidas" (2 misses) and "krakers" (1 miss) on top
    assert_eq!(terms[0].0, "adidas", "{terms:?}");
    assert_eq!(terms[0].2, 2, "{terms:?}");
    let by_term: std::collections::HashMap<String, (u64, u64)> =
        terms.into_iter().map(|(t, c, m)| (t, (c, m))).collect();
    assert_eq!(by_term.get("krakers"), Some(&(1, 1)), "{by_term:?}");
    // hits never increment miss
    assert_eq!(by_term.get("but"), Some(&(1, 0)), "{by_term:?}");
}

/// Spec index-rs-dynamic-synonyms (W7): raw (diacritic) rule words stem at
/// load — "słuchawki" and "sluchawki" queries both hit the same pair.
#[test]
fn synonym_dict_stems_raw_words_at_load() {
    let docs = vec![
        doc("sluch", json!({ "title": "Słuchawki bezprzewodowe" })),
        doc("hp", json!({ "title": "Headphones HP" })),
    ];
    let dict = crate::engine::SynonymDict::from_raw([
        (vec!["słuchawki".into()], vec!["headphones".into()]),
    ]);
    let e = Engine::build_with_synonyms(docs, None, dict).expect("build engine");
    // "headphones" query hits the Słuchawki doc (query also stems/folds)
    let res = search(&e, &json!({ "size": 5, "query": { "match": { "title": "headphones" } } }));
    assert!(hit_slugs(&res).contains(&"sluch".to_string()), "{:?}", hit_slugs(&res));
    // and the reverse direction with diacritics folded in the query
    let res = search(&e, &json!({ "size": 5, "query": { "match": { "title": "sluchawki" } } }));
    assert!(hit_slugs(&res).contains(&"hp".to_string()), "{:?}", hit_slugs(&res));
}

/// Spec index-rs-phrase-synonyms (W5): multi-word synonyms work in both
/// directions — a query with the phrase hits docs using the alternative
/// token, and a query with the alternative token hits docs with the full
/// phrase (adjacency strict).
#[test]
fn phrase_synonyms_match_in_both_directions() {
    let docs = vec![
        doc("gpu-card", json!({ "title": "GPU RTX 4070", "description": "wydajna" })),
        doc("karta", json!({ "title": "Karta graficzna do gier", "description": " szybka" })),
    ];
    let dict = crate::engine::SynonymDict::from_raw([
        // a multi-token side is an adjacent PHRASE; alternatives are
        // SEPARATE rules (same convention as the PG seed)
        (vec!["karta".into(), "graficzna".into()], vec!["gpu".into()]),
        (vec!["karta".into(), "graficzna".into()], vec!["grafika".into()]),
    ]);
    let e = Engine::build_with_synonyms(docs, None, dict).expect("build engine");
    // phrase query → hits the "GPU" doc (alternative token member)
    let res = search(&e, &json!({ "size": 10, "query": { "match": { "title": "karta graficzna" } } }));
    assert!(hit_slugs(&res).contains(&"gpu-card".to_string()), "{:?}", hit_slugs(&res));
    // alternative-token query → hits the "Karta graficzna" doc (reverse
    // phrase member, adjacency strict)
    let res = search(&e, &json!({ "size": 10, "query": { "match": { "title": "gpu" } } }));
    assert!(hit_slugs(&res).contains(&"karta".to_string()), "{:?}", hit_slugs(&res));
    // adjacency: "graficzna karta" (reversed) is NOT the phrase — docs
    // still match their own tokens, but the phrase member does not fire
    // for a non-adjacent query… (query-side: reversed order means no
    // phrase detection — only "graficzn"/"kart" single-token groups)
    let res = search(&e, &json!({ "size": 10, "query": { "match": { "title": "grafika" } } }));
    assert!(hit_slugs(&res).contains(&"karta".to_string()), "{:?}", hit_slugs(&res));
}

/// Spec index-rs-trust-precompute (W4): identical BM25 docs with different
/// trust_score — relevance sort (function_score sum) ranks the higher-trust
/// doc first via the pre-computed boost.
#[test]
fn precomputed_trust_boost_ranks_higher_trust_first() {
    let docs = vec![
        doc("trust-low", json!({ "title": "Widget Pro", "trust_score": 0.1, "trust_count": 3 })),
        doc("trust-high", json!({ "title": "Widget Pro", "trust_score": 0.9, "trust_count": 40 })),
    ];
    let e = Engine::build(docs, None).expect("build engine");
    let res = search(
        &e,
        &json!({
            "size": 10,
            "query": {
                "function_score": {
                    "query": { "match": { "title": "widget" } },
                    "script_score": { "script": { "source": crate::dsn::TRUST_SCRIPT } },
                    "boost_mode": "sum"
                }
            }
        }),
    );
    let slugs = hit_slugs(&res);
    assert_eq!(slugs.first().map(String::as_str), Some("trust-high"), "{slugs:?}");
}


/// Spec index-rs-synonyms (W4): a query term matches documents that use
/// its synonym — both directions — without inflating msm (the synonym
/// group is the msm unit, so a doc matching only one member still counts).
#[test]
fn synonyms_expand_match_queries_bidirectionally() {
    let docs = vec![
        doc("telefon", json!({ "title": "Telefon", "description": "klasyczne urządzenie" })),
        doc("smartfon", json!({ "title": "Smartfon", "description": "nowoczesny" })),
        doc("obuwie", json!({ "title": "Obuwie sportowe", "description": "dobre" })),
        doc("buty", json!({ "title": "Buty do biegania", "description": "szybkie" })),
    ];
    // spec index-rs-dynamic-synonyms: the dictionary is DATA — injected
    // like the PG loader would (raw words, stemmed at load)
    let dict = crate::engine::SynonymDict::from_raw([
        (vec!["buty".into()], vec!["obuwie".into()]),
        (vec!["smartfon".into()], vec!["telefon".into()]),
    ]);
    let e = Engine::build_with_synonyms(docs, None, dict).expect("build engine");
    // "smartfon" query → hits the doc titled "Telefon"
    let res = search(&e, &json!({ "size": 10, "query": { "match": { "title": "smartfon" } } }));
    assert!(hit_slugs(&res).contains(&"telefon".to_string()), "{:?}", hit_slugs(&res));
    // and the reverse direction: "telefon" query → hits "Smartfon"
    let res = search(&e, &json!({ "size": 10, "query": { "match": { "title": "telefon" } } }));
    assert!(hit_slugs(&res).contains(&"smartfon".to_string()), "{:?}", hit_slugs(&res));
    // "obuwie" (stemmed "obuw") ↔ "buty" (stemmed "but")
    let res = search(&e, &json!({ "size": 10, "query": { "match": { "title": "buty" } } }));
    assert!(hit_slugs(&res).contains(&"obuwie".to_string()), "{:?}", hit_slugs(&res));
    // multi_match with msm "1<100%": one real token must stay sufficient —
    // the synonym expansion must NOT raise the required count (W3)
    let res = search(
        &e,
        &json!({
            "size": 10,
            "query": { "multi_match": { "query": "obuwie", "fields": ["title^3", "description"], "operator": "or", "minimum_should_match": "1<100%" } }
        }),
    );
    assert!(hit_slugs(&res).contains(&"buty".to_string()), "{:?}", hit_slugs(&res));
    // no synonym → no phantom matches: a doc about kettles never surfaces
    let res = search(&e, &json!({ "size": 10, "query": { "match": { "title": "czajnik" } } }));
    assert!(hit_slugs(&res).is_empty(), "{:?}", hit_slugs(&res));
}


/// W5 (spec index-rs-optimization): the autocomplete prefix tail is ranked
/// by document frequency, not alphabetically. With 60 prefix candidates
/// the old alphabetical take(50) cut "zz59" (last alphabetically) even
/// though it is the most frequent term — the freq-ranked tail must keep it.
#[test]
fn autocomplete_prefix_tail_prefers_frequent_terms() {
    let mut docs = Vec::new();
    // 59 alphabetically-early candidates, each freq 1
    for i in 0..59 {
        docs.push(doc(
            &format!("zz{i:02}"),
            json!({ "title": format!("prod zz{i:02}") }),
        ));
    }
    // zz59: freq 3 — would be cut by the alphabetical take(50)
    for n in 0..3 {
        docs.push(doc(
            &format!("zz59-doc{n}"),
            json!({ "title": format!("prod zz59 nr {n}") }),
        ));
    }
    let e = Engine::build(docs, None).expect("build engine");
    let res = search(
        &e,
        &json!({
            "size": 100,
            "query": { "match_phrase_prefix": { "title": { "query": "zz" } } }
        }),
    );
    let slugs = hit_slugs(&res);
    assert!(
        slugs.iter().any(|s| s.starts_with("zz59")),
        "frequent tail term zz59 must survive the 50-candidate cut: {slugs:?}"
    );
}

#[test]
fn chat_hybrid_bm25_and_knn_bodies() {
    let e = engine();
    // (a) BM25 body (verbatim chat-rs shape, boost 2 on multi_match).
    let bm25_body = json!({
        "size": 5,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": "asics",
                            "fields": ["title^3", "vendor^2", "description", "tags", "specs_text", "reviews_text"],
                            "type": "best_fields",
                            "operator": "or",
                            "minimum_should_match": "2<75%",
                            "lenient": true,
                            "boost": 2
                        }
                    },
                    { "match": { "title": { "query": "asics", "fuzziness": "AUTO", "prefix_length": 2 } } },
                    { "match_phrase": { "title": { "query": "asics", "boost": 6 } } }
                ],
                "minimum_should_match": 1,
                "filter": [available_filter(), approved_filter()]
            }
        }
    });
    let resp = search(&e, &bm25_body);
    let slugs = hit_slugs(&resp);
    assert!(slugs.contains(&"asics-sonicblast".to_string()));
    assert!(slugs.contains(&"asics-gel-kayano".to_string()));

    // (b) kNN body (verbatim): knn embedding k=50 inside bool.must + filter.
    let knn_body = json!({
        "size": 5,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "must": [ { "knn": { "embedding": { "vector": [1.0, 0.0, 0.0, 0.0], "k": 50 } } } ],
                "filter": [available_filter(), approved_filter()]
            }
        }
    });
    let resp = search(&e, &knn_body);
    let slugs = hit_slugs(&resp);
    assert_eq!(slugs.len(), 3); // 3 docs carry embeddings
    assert_eq!(slugs[0], "asics-sonicblast"); // exact cosine 1.0
    assert_eq!(slugs[1], "asics-gel-kayano");
    assert_eq!(slugs[2], "sluchawki-bedace");
    assert!(
        resp.pointer("/hits/hits/0/_score")
            .unwrap()
            .as_f64()
            .unwrap()
            > 0.99
    );
}

#[test]
fn chat_get_product_details_body() {
    let e = engine();
    let body = json!({
        "size": 1,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "must": [ { "term": { "slug.keyword": "lumina-tv-8k" } } ],
                "filter": [ approved_filter() ]
            }
        }
    });
    let resp = search(&e, &body);
    assert_eq!(
        resp.pointer("/hits/total/value").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(resp.pointer("/hits/hits/0/_source/price").unwrap(), 5000.0);
}

#[test]
fn chat_review_search_body() {
    let e = engine();
    // search_review_products: match on reviews_text + filters.
    let body = json!({
        "size": 5,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "must": [ { "match": { "reviews_text": { "query": "stabilne buty do biegania" } } } ],
                "filter": [ available_filter(), approved_filter() ]
            }
        }
    });
    let resp = search(&e, &body);
    assert!(hit_slugs(&resp).contains(&"asics-gel-kayano".to_string()));
}

// ---------------------------------------------------------------------------
// admin-rs body
// ---------------------------------------------------------------------------

#[test]
fn admin_product_list_body() {
    let e = engine();
    // Verbatim build_search_body shape: multi_match must + vendor/category
    // term filters + created_at sort + brands/categories aggs + from.
    let body = json!({
        "size": 20,
        "from": 0,
        "track_total_hits": true,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "must": [
                    { "multi_match": { "query": "telefon",
                        "fields": ["title^3", "vendor^2", "description", "tags", "specs_text"] } }
                ],
                "filter": []
            }
        },
        "sort": [{ "created_at": { "order": "desc" } }],
        "aggs": {
            "brands": { "terms": { "field": "vendor.keyword", "size": 100 } },
            "categories": { "terms": { "field": "type.keyword", "size": 100 } }
        }
    });
    let resp = search(&e, &body);
    assert_eq!(
        resp.pointer("/hits/total/value").and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(hit_slugs(&resp), vec!["telefon-super-pro", "telefon-super"]); // created_at desc
    let brands = resp
        .pointer("/aggregations/brands/buckets")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(brands[0]["key"], "Sklep GSM");
    // from=1 skips the first hit.
    let mut page2 = body.clone();
    page2["from"] = json!(1);
    page2["size"] = json!(1);
    assert_eq!(hit_slugs(&search(&e, &page2)), vec!["telefon-super"]);
    // Unfiltered default body (match_all + created_at desc + aggs).
    let default_body = json!({
        "size": 20, "from": 0, "track_total_hits": true,
        "_source": { "excludes": ["embedding"] },
        "query": { "bool": { "must": [{ "match_all": {} }], "filter": [] } },
        "sort": [{ "created_at": { "order": "desc" } }],
        "aggs": {
            "brands": { "terms": { "field": "vendor.keyword", "size": 100 } },
            "categories": { "terms": { "field": "type.keyword", "size": 100 } }
        }
    });
    let resp = search(&e, &default_body);
    assert_eq!(
        resp.pointer("/hits/total/value").and_then(Value::as_i64),
        Some(12)
    );
    let cats = resp
        .pointer("/aggregations/categories/buckets")
        .unwrap()
        .as_array()
        .unwrap();
    assert!(cats
        .iter()
        .any(|b| b["key"] == "Elektronika" && b["doc_count"] == 3));
}

// ---------------------------------------------------------------------------
// wire-shape golden keys
// ---------------------------------------------------------------------------

#[test]
fn response_shape_golden_keys() {
    let e = engine();
    let body = json!({
        "size": 2,
        "_source": { "excludes": ["embedding"] },
        "query": {
            "bool": {
                "must": [{ "match_all": {} }],
                "filter": [available_filter(), approved_filter()]
            }
        },
        "sort": [{ "price": { "order": "asc", "missing": "_last" } }],
        "aggs": { "cats": { "terms": { "field": "type.keyword", "size": 3 } } }
    });
    let resp = search(&e, &body);
    // Top-level keys — the OpenSearch search response shape.
    let keys: Vec<&str> = resp
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str())
        .collect();
    assert!(keys.contains(&"took"));
    assert!(keys.contains(&"timed_out"));
    assert!(keys.contains(&"_shards"));
    assert!(keys.contains(&"hits"));
    assert!(keys.contains(&"aggregations"));
    // hits keys.
    let hits = resp.pointer("/hits").unwrap().as_object().unwrap();
    assert!(hits.contains_key("total"));
    assert!(hits.contains_key("max_score"));
    assert!(hits.contains_key("hits"));
    assert_eq!(
        resp.pointer("/hits/total/relation").and_then(Value::as_str),
        Some("eq")
    );
    // per-hit keys.
    let hit = resp.pointer("/hits/hits/0").unwrap().as_object().unwrap();
    for key in ["_index", "_id", "_score", "_source", "sort"] {
        assert!(hit.contains_key(key), "hit key {key}");
    }
    // terms-agg bucket keys.
    let bucket = resp
        .pointer("/aggregations/cats/buckets/0")
        .unwrap()
        .as_object()
        .unwrap();
    assert!(bucket.contains_key("key"));
    assert!(bucket.contains_key("doc_count"));
}

#[tokio::test(flavor = "current_thread")]
async fn unknown_clause_returns_400_shaped_error_via_api() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::routing::{get, post};
    use axum::Router;
    use tower::ServiceExt; // dev-dependency of the other crates

    let e = engine();
    let state = Arc::new(AppState {
        engine: tokio::sync::RwLock::new(Arc::new(e)),
        pool: None,
        rebuild_secret: None,
        write_lock: tokio::sync::Mutex::new(()),
        scrolls: std::sync::Mutex::new(std::collections::HashMap::new()),
        vectors_dirty: std::sync::atomic::AtomicBool::new(false),
        ready: std::sync::atomic::AtomicBool::new(false),
    });
    let app = Router::new()
        .route("/:index/_search", post(crate::api::search_handler))
        .route("/health", get(crate::api::health_handler))
        .with_state(state);
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/products/_search")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"query":{"wildcard":{"slug":"*x*"}}}"#.to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], 400);
    assert_eq!(v["error"]["type"], "parsing_exception");
    assert!(v["error"]["reason"].as_str().unwrap().contains("wildcard"));

    // health reports the corpus size
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test(flavor = "current_thread")]
async fn health_reports_ready_flag() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    let e = engine();
    let state = Arc::new(AppState {
        engine: tokio::sync::RwLock::new(Arc::new(e)),
        pool: None,
        rebuild_secret: None,
        write_lock: tokio::sync::Mutex::new(()),
        scrolls: std::sync::Mutex::new(std::collections::HashMap::new()),
        vectors_dirty: std::sync::atomic::AtomicBool::new(false),
        ready: std::sync::atomic::AtomicBool::new(false),
    });
    let app = Router::new()
        .route("/health", get(crate::api::health_handler))
        .with_state(state.clone());

    // serve-first boot: listener up (200) but index not built yet
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["docs"], 12);
    assert_eq!(v["ready"], false);

    // after the initial build the flag flips (health stays 200 throughout)
    state
        .ready
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["ready"], true);
}

#[tokio::test(flavor = "current_thread")]
async fn count_and_doc_endpoints() {
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    let e = engine();
    let state = Arc::new(AppState {
        engine: tokio::sync::RwLock::new(Arc::new(e)),
        pool: None,
        rebuild_secret: None,
        write_lock: tokio::sync::Mutex::new(()),
        scrolls: std::sync::Mutex::new(std::collections::HashMap::new()),
        vectors_dirty: std::sync::atomic::AtomicBool::new(false),
        ready: std::sync::atomic::AtomicBool::new(false),
    });
    let app = Router::new()
        .route("/:index/_count", get(crate::api::count_handler))
        .route("/:index/_doc/:id", get(crate::api::doc_handler))
        .with_state(state);
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/products/_count")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["count"], 12);
    assert!(v["_shards"]["successful"] == 1);

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/products/_doc/asics-sonicblast")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["found"], true);
    assert_eq!(v["_id"], "asics-sonicblast");
    assert_eq!(v["_source"]["title"], "ASICS Sonicblast");

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/products/_doc/does-not-exist")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[test]
fn knn_dimension_mismatch_is_a_parsing_error() {
    let e = engine();
    let body = json!({
        "size": 5,
        "query": {
            "bool": {
                "must": [ { "knn": { "embedding": { "vector": [1.0, 2.0], "k": 10 } } } ],
                "filter": [available_filter(), approved_filter()]
            }
        }
    });
    let req = parse_search_body(&body).unwrap();
    let err = match e.search(&req) {
        Ok(_) => panic!("expected knn dim error"),
        Err(e) => e,
    };
    assert!(err.contains("dimension"));
}

// ---------------------------------------------------------------------------
// WRITE API integration tests — the literal .NET wire bodies applied to a
// live engine (OpenSearchProductRepository upsert, category applier
// update_by_query, DeleteProducts delete_by_query, bulk indexer ops, _doc
// put/delete, refresh/mapping/mget no-op shapes, scroll).
// ---------------------------------------------------------------------------

/// OpenSearchProductRepository.SaveAsync — the literal {doc, doc_as_upsert}
/// body shape (MapProductToDoc excerpt) posted to POST /{index}/_update/{id}.
#[test]
fn update_doc_as_upsert_searches_immediately_and_preserves_embedding() {
    // the stored doc has an embedding (backfill put it there)
    let mut corpus_doc = doc("asics-sonicblast", json!({}));
    corpus_doc.embedding = Some(vec![0.1, 0.2, 0.3]);
    let e = Engine::build(vec![corpus_doc], None).expect("engine");

    let stored = e.get_doc("asics-sonicblast").unwrap().0;
    let body = json!({
        "doc": {
            "id": "asics-sonicblast",
            "slug": "asics-sonicblast",
            "title": "ASICS Sonicblast PRO",
            "vendor": "top4running.pl",
            "type": "Buty do biegania",
            "tags": ["bieganie"],
            "status": "approved",
            "price": null,
            "offers": []
        },
        "doc_as_upsert": true
    });
    let out = crate::write::compute_update(Some(&stored), &body).expect("merge");
    // partial doc has no embedding → the stored one is kept (doc_as_upsert
    // semantics the .NET partial update relies on)
    assert!(out.embedding.is_none());
    e.upsert_doc(
        "asics-sonicblast",
        Value::Object(out.doc.clone()),
        out.embedding,
    );

    // read-your-writes: the new title is immediately searchable (BM25)
    let data = search(
        &e,
        &json!({ "query": { "multi_match": { "query": "sonicblast pro", "fields": ["title^3", "vendor"] } } }),
    );
    assert!(hit_slugs(&data).contains(&"asics-sonicblast".to_string()));
    // the stored embedding survived the merge — in its normalized+int8
    // DEQUANTIZED stored form (spec index-rs-int8-quantization W3: rows
    // are unit-length i8-quantized; get_doc returns the approximate
    // reconstruction, direction preserved within ~1%)
    let (_, emb) = e.get_doc("asics-sonicblast").unwrap();
    let expected = crate::engine::l2_normalize(&[0.1, 0.2, 0.3]);
    let emb = emb.expect("embedding preserved");
    assert_eq!(emb.len(), expected.len());
    for (a, b) in emb.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < 0.01,
            "dequantized embedding within 1% of the normalized original: {emb:?} vs {expected:?}"
        );
    }
    // _doc GET reflects the merged field
    let src = e.get_by_id("asics-sonicblast").unwrap();
    assert_eq!(src["title"], "ASICS Sonicblast PRO");
}

/// A brand-NEW product via doc_as_upsert (SaveAsync on a fresh slug).
#[test]
fn update_doc_as_upsert_creates_missing_doc() {
    let e = engine();
    assert!(e.get_doc("nowy-produkt").is_none());
    let body = json!({
        "doc": { "slug": "nowy-produkt", "title": "Nowy Produkt XYZ", "vendor": "Sklep" },
        "doc_as_upsert": true
    });
    let out = crate::write::compute_update(None, &body).expect("create");
    assert_eq!(out.result, "created");
    e.upsert_doc("nowy-produkt", Value::Object(out.doc), out.embedding);
    assert_eq!(e.num_docs(), 13);
    let data = search(
        &e,
        &json!({ "query": { "match": { "title": "Nowy Produkt" } } }),
    );
    assert!(hit_slugs(&data).contains(&"nowy-produkt".to_string()));
}

/// ProductCategoryBackfillActivities — the literal attempt-counter script.
#[test]
fn script_update_category_attempts_is_searchable() {
    let e = engine();
    let stored = e.get_doc("asics-sonicblast").unwrap().0;
    let body = json!({
        "script": {
            "source": "ctx._source.category_attempts = (ctx._source.category_attempts == null ? 0 : ctx._source.category_attempts) + 1",
            "lang": "painless"
        }
    });
    let out = crate::write::compute_update(Some(&stored), &body).expect("script");
    assert_eq!(out.doc["category_attempts"], 1);
    e.upsert_doc("asics-sonicblast", Value::Object(out.doc), out.embedding);

    // the scan query (FindUncategorizedBatchAsync must_not exists) can see it
    let data = search(
        &e,
        &json!({ "query": { "range": { "category_attempts": { "gte": 1 } } } }),
    );
    assert!(hit_slugs(&data).contains(&"asics-sonicblast".to_string()));
}

/// OpenSearchCatalogCategoryApplier.ApplyCategoryAsync — the LITERAL
/// update_by_query body (terms on slug + should[not exists type | empty type]).
#[test]
fn update_by_query_category_applier_literal_body() {
    let e = engine();
    let body = json!({
        "query": {
            "bool": {
                "filter": [
                    { "terms": { "slug.keyword": ["asics-sonicblast", "nike-air-max"] } },
                    {
                        "bool": {
                            "should": [
                                { "bool": { "must_not": { "exists": { "field": "type" } } } },
                                { "term": { "type.keyword": "" } }
                            ],
                            "minimum_should_match": 1
                        }
                    }
                ]
            }
        },
        "script": {
            "lang": "painless",
            "source": "ctx._source.type = params.category",
            "params": { "category": "Buty sportowe" }
        }
    });
    // selection via the dsn parser (update_by_query semantics)
    let q = crate::dsn::parse_query(body.get("query").unwrap()).expect("query");
    let slugs = e.select_slugs(&q).expect("select");
    // both have a non-empty type → W6 never-overwrite filter matches nothing
    assert!(
        slugs.is_empty(),
        "non-empty types must not match: {slugs:?}"
    );

    // now blank one type and re-run: the doc matches and gets the category
    let stored = e.get_doc("asics-sonicblast").unwrap().0;
    let mut blanked = stored.as_object().cloned().unwrap();
    blanked.insert("type".into(), json!(""));
    e.upsert_doc("asics-sonicblast", Value::Object(blanked), None);
    let slugs = e.select_slugs(&q).expect("select");
    assert_eq!(slugs, vec!["asics-sonicblast".to_string()]);

    let script_body = json!({ "script": body["script"] });
    for slug in &slugs {
        let doc = e.get_doc(slug).unwrap().0;
        let out = crate::write::compute_update(Some(&doc), &script_body).expect("script");
        e.upsert_doc(slug, Value::Object(out.doc), out.embedding);
    }
    // the .NET side parses response.updated == 1
    assert_eq!(slugs.len(), 1);
    let src = e.get_by_id("asics-sonicblast").unwrap();
    assert_eq!(src["type"], "Buty sportowe");
}

/// OpenSearchActivities.DeleteProductsAsync — the literal bool/prefix body.
#[test]
fn delete_by_query_prefix_id_literal_body() {
    let mut corpus = corpus();
    // a Shopify-fed doc (id = gid://shopify/…)
    corpus.push(doc(
        "shopify-widget",
        json!({ "id": "gid://shopify/Product/1", "vendor": "Sklep ABC", "type": "Akcesoria" }),
    ));
    let e = Engine::build(corpus, None).expect("engine");
    assert_eq!(e.num_docs(), 13);

    let query = json!({
        "bool": {
            "must": [
                { "prefix": { "id": "gid://shopify/" } },
                { "term": { "vendor.keyword": "Sklep ABC" } }
            ]
        }
    });
    let q = crate::dsn::parse_query(&query).expect("query");
    let slugs = e.select_slugs(&q).expect("select");
    assert_eq!(slugs, vec!["shopify-widget".to_string()]);
    for slug in &slugs {
        assert!(e.delete_doc(slug));
    }
    assert_eq!(e.num_docs(), 12);
    assert!(e.get_doc("shopify-widget").is_none());
    // delete of a missing slug → false (DELETE /_doc → 404 result not_found)
    assert!(!e.delete_doc("shopify-widget"));
    // search no longer finds it
    let data = search(
        &e,
        &json!({ "query": { "prefix": { "id": "gid://shopify/" } } }),
    );
    assert!(hit_slugs(&data).is_empty());
}

/// OpenSearchBulkIndexer + MarkStale — the bulk ops against a live engine:
/// index (new doc), update (AppendOfferScript), update (doc merge),
/// delete of a missing id (404 item).
#[test]
fn bulk_mixed_ops_on_live_engine() {
    let e = engine();
    let ndjson = concat!(
        "{\"index\": {\"_index\": \"products\", \"_id\": \"bulk-new\", \"version_type\": \"external\", \"version\": 42}}\n",
        "{\"id\": \"gid://shopify/9\", \"slug\": \"bulk-new\", \"title\": \"Bulk Widget Mocny\", \"vendor\": \"BulkShop\", \"type\": \"Akcesoria\", \"price\": 10.0, \"currency\": \"PLN\"}\n",
        "{\"update\": {\"_index\": \"products\", \"_id\": \"bulk-new\"}}\n",
        "{\"script\": {\"source\": \"if (ctx._source.offers == null) { ctx._source.offers = new ArrayList(); } def exists = false; for (def o : ctx._source.offers) { if (o.shop_slug == params.offer.shop_slug) { def stored = 0; if (o.updated_at != null && o.updated_at > 0) { stored = o.updated_at; } if (params.ingestedAt >= stored) { o.price = params.offer.price; o.regular_price = params.offer.regular_price; o.checkout_url = params.offer.checkout_url; o.currency = params.offer.currency; o.in_stock = params.offer.in_stock; o.shop_name = params.offer.shop_name; o.updated_at = params.ingestedAt; } exists = true; } } if (!exists) { ctx._source.offers.add(params.offer); } if (ctx._source.external_ids == null) { ctx._source.external_ids = new ArrayList(); } if (!ctx._source.external_ids.contains(params.externalId)) { ctx._source.external_ids.add(params.externalId); } if (ctx._source.offers.size() > 0) { def minPrice = ctx._source.offers[0].price; for (def o : ctx._source.offers) { if (o.price < minPrice) { minPrice = o.price; } } ctx._source.price = minPrice; }\", \"params\": {\"offer\": {\"shop_slug\": \"bulkshop\", \"shop_name\": \"BulkShop\", \"price\": 7.5, \"currency\": \"PLN\", \"in_stock\": true, \"checkout_url\": \"https://b/1\"}, \"externalId\": \"ext-9\", \"ingestedAt\": 100}}}\n",
        "{\"update\": {\"_index\": \"products\", \"_id\": \"asics-sonicblast\"}}\n",
        "{\"doc\": {\"available\": false}}\n",
        "{\"delete\": {\"_index\": \"products\", \"_id\": \"nonexistent-slug\"}}\n",
    );
    let actions = crate::write::parse_bulk(ndjson, "products").expect("ndjson");
    assert_eq!(actions.len(), 4);
    let mut not_found = 0usize;
    for action in actions {
        match action {
            crate::write::BulkAction::Index { id, source, .. } => {
                let mut doc = source.as_object().cloned().unwrap();
                let embedding = crate::write::extract_embedding(&mut doc);
                e.upsert_doc(&id, Value::Object(doc), embedding);
            }
            crate::write::BulkAction::Update { id, body, .. } => {
                let existing = e.get_doc(&id).map(|(d, _)| d);
                let out = crate::write::compute_update(existing.as_ref(), &body)
                    .expect("bulk update applies");
                e.upsert_doc(&id, Value::Object(out.doc), out.embedding);
            }
            crate::write::BulkAction::Delete { id, .. } => {
                if !e.delete_doc(&id) {
                    not_found += 1;
                }
            }
        }
    }
    assert_eq!(not_found, 1);
    // created doc is searchable by its new title
    let data = search(
        &e,
        &json!({ "query": { "match": { "title": "widget mocny" } } }),
    );
    assert!(hit_slugs(&data).contains(&"bulk-new".to_string()));
    // offer appended + min price recomputed by the literal script
    let src = e.get_by_id("bulk-new").unwrap();
    assert_eq!(src["offers"][0]["shop_slug"], "bulkshop");
    assert_eq!(src["price"], 7.5);
    assert_eq!(src["external_ids"][0], "ext-9");
    // MarkStale-style doc merge visible immediately
    let src = e.get_by_id("asics-sonicblast").unwrap();
    assert_eq!(src["available"], false);
}

/// Wire shapes the .NET side parses: _update response, bulk items, 404s.
#[test]
fn write_response_wire_shapes() {
    let r = crate::api::write_response("products", "slug-x", "updated");
    assert_eq!(r["_index"], "products");
    assert_eq!(r["_id"], "slug-x");
    assert_eq!(r["_version"], 1);
    assert_eq!(r["result"], "updated");
    assert_eq!(r["_shards"]["successful"], 1);
}

/// EnrichmentActivities batch: {doc: {embedding, embedding_pending}} —
/// the embedding is extracted out of the doc (stored in the embedding
/// column, never in doc jsonb).
#[test]
fn embedding_update_extracted_from_doc() {
    let e = engine();
    let stored = e.get_doc("asics-sonicblast").unwrap().0;
    let body = json!({ "doc": { "embedding": [0.4, 0.5], "embedding_pending": false } });
    let out = crate::write::compute_update(Some(&stored), &body).expect("embedding merge");
    assert_eq!(out.embedding, Some(vec![0.4, 0.5]));
    assert!(!out.doc.contains_key("embedding"));
    assert_eq!(out.doc["embedding_pending"], false);
    e.upsert_doc("asics-sonicblast", Value::Object(out.doc), out.embedding);
    let (src, emb) = e.get_doc("asics-sonicblast").unwrap();
    assert!(src.get("embedding").is_none());
    // W2 (spec index-rs-optimization): embeddings are STORED normalized —
    // the direction matches the written vector, the norm is 1.
    let expected = crate::engine::l2_normalize(&[0.4, 0.5]);
    assert_eq!(emb, Some(expected));
}

/// _refresh, _mapping, _mget and scroll over the axum router (no PG needed).
#[tokio::test(flavor = "current_thread")]
async fn refresh_mapping_mget_and_scroll_endpoints() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::routing::{get, post, put};
    use axum::Router;
    use tower::ServiceExt;

    let e = engine();
    let state = Arc::new(AppState {
        engine: tokio::sync::RwLock::new(Arc::new(e)),
        pool: None,
        rebuild_secret: None,
        write_lock: tokio::sync::Mutex::new(()),
        scrolls: std::sync::Mutex::new(std::collections::HashMap::new()),
        vectors_dirty: std::sync::atomic::AtomicBool::new(false),
        ready: std::sync::atomic::AtomicBool::new(false),
    });
    let app = Router::new()
        .route("/:index/_search", post(crate::api::search_handler))
        .route(
            "/_search/scroll",
            post(crate::api::scroll_handler).delete(crate::api::scroll_delete_handler),
        )
        .route("/:index/_count", get(crate::api::count_handler))
        .route("/:index/_refresh", post(crate::api::refresh_handler))
        .route(
            "/:index/_mapping",
            put(crate::api::mapping_handler).post(crate::api::mapping_handler),
        )
        .route("/:index/_mget", post(crate::api::mget_handler))
        .with_state(state);

    // _refresh → {"_shards":{...}} 200
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/products/_refresh")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(v["_shards"].is_object());

    // PUT _mapping (BookMetadataRepository.EnsureMappingAsync) → acknowledged
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/products/_mapping")
                .header("content-type", "application/json")
                .body(Body::from("{\"properties\": {}}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(v["acknowledged"], true);

    // _mget — the OpenSearchBulkIndexer shape (docs[].found + _id)
    let mget_body = serde_json::json!({
        "docs": [
            { "_index": "products", "_id": "asics-sonicblast" },
            { "_index": "products", "_id": "missing-slug" }
        ]
    })
    .to_string();
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/products/_mget")
                .header("content-type", "application/json")
                .body(Body::from(mget_body))
                .unwrap(),
        )
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(v["docs"][0]["found"], true);
    assert_eq!(v["docs"][0]["_id"], "asics-sonicblast");
    assert!(v["docs"][0]["_source"]["slug"].is_string());
    assert_eq!(v["docs"][1]["found"], false);

    // scroll: page through the whole corpus 3 at a time
    let first = serde_json::json!({ "size": 3, "query": { "match_all": {} } }).to_string();
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/products/_search?scroll=1m")
                .header("content-type", "application/json")
                .body(Body::from(first))
                .unwrap(),
        )
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(v["hits"]["hits"].as_array().unwrap().len(), 3);
    let scroll_id = v["_scroll_id"].as_str().unwrap().to_string();

    let mut total = 3usize;
    loop {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/_search/scroll")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "scroll": "1m", "scroll_id": scroll_id }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let page = v["hits"]["hits"].as_array().unwrap().len();
        total += page;
        if page == 0 || v["_scroll_id"].is_null() {
            break;
        }
    }
    assert_eq!(total, 12, "scrolled the whole corpus");
}

// -- index-efficiency spec: sweep + pending embeddings ------------------------

#[test]
fn upsert_without_embedding_keeps_stored_vector() {
    let e = engine();
    e.upsert_doc(
        "asics-sonicblast",
        json!({"title": "Asics Sonic-Blast v2", "type": "Inne", "currency": "PLN"}),
        None,
    );
    let (_src, emb) = e.get_doc("asics-sonicblast").unwrap();
    assert_eq!(emb, Some(vec![1.0, 0.0, 0.0, 0.0]));
}

#[test]
fn knn_sees_pending_embedding_upsert() {
    let e = engine();
    // swap sonicblast's vector to a NEW direction — knn must score the
    // pending row; with the stale matrix row it would rank last
    e.upsert_doc(
        "asics-sonicblast",
        json!({"title": "Asics Sonic-Blast"}),
        Some(vec![0.0, 1.0, 0.0, 0.0]),
    );
    let res = search(
        &e,
        &json!({
            "query": {
                "bool": {
                    "must": [ { "knn": { "embedding": { "vector": [0.0, 1.0, 0.0, 0.0], "k": 3 } } } ]
                }
            },
            "_source": { "excludes": ["embedding"] }
        }),
    );
    let hits = res["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 3, "kNN k=3 over the 3 embedded docs");
    assert_eq!(hits[0]["_id"], "asics-sonicblast");
}

#[test]
fn sweep_removes_stale_leaked_tempdirs_only() {
    use crate::engine::sweep_stale_tempdirs_with;
    use std::fs;
    let dir = tempfile::tempdir().unwrap();
    let leaked = dir.path().join(".tmpAAAA11");
    let live = dir.path().join(".tmpBBBB22");
    fs::create_dir_all(&leaked).unwrap();
    fs::create_dir_all(&live).unwrap();
    fs::write(leaked.join("meta.json"), "{}").unwrap();
    fs::write(dir.path().join("vectors.f32"), "hdr").unwrap();
    // a large max age → nothing is stale: the race guard keeps fresh dirs
    sweep_stale_tempdirs_with(dir.path(), std::time::Duration::from_secs(3600));
    assert!(leaked.exists(), "dirs within max age must survive (race guard)");
    assert!(live.exists());
    // zero max age → everything .tmp* counts as stale and is swept
    sweep_stale_tempdirs_with(dir.path(), std::time::Duration::ZERO);
    assert!(!leaked.exists(), "stale leaked dir must be removed");
    assert!(!live.exists());
    assert!(
        dir.path().join("vectors.f32").exists(),
        "non-temp files must never be touched"
    );
}

/// Cross-crate drift guard (spec chat-instant-discovery-publish): the chat
/// service instant-publishes offers with a VERBATIM copy of the
/// SCRIPT_APPEND_OFFER literal. The pattern interpreter matches by
/// whitespace-normalized equality — if write.rs ever changes the pattern
/// without the chat copy following, the chat bridge would fail loudly with
/// 400s. This test fails at that exact moment, on the index side, by reading
/// the literal straight out of the chat source.
#[test]
fn chat_bridge_append_offer_literal_stays_in_pattern_catalog() {
    let chat_src = include_str!("../../truespotter-chat/src/discovery.rs");
    let marker = "const SCRIPT_APPEND_OFFER: &str = \"";
    let start = chat_src
        .find(marker)
        .expect("chat discovery.rs must carry SCRIPT_APPEND_OFFER");
    let lit_start = start + marker.len();
    let lit_end = chat_src[lit_start..]
        .find("\";")
        .expect("unterminated literal")
        + lit_start;
    let literal = &chat_src[lit_start..lit_end];

    let pattern = crate::write::match_pattern(&crate::dsn::normalize_ws(literal))
        .expect("the chat-side SCRIPT_APPEND_OFFER copy must stay in the pattern catalog");
    assert_eq!(pattern, crate::write::Pattern::AppendOffer);
}
