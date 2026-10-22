//! Typed deserialization of the OpenSearch Query-DSL subset our three Rust
//! services use (catalog-rs `SearchAsync`/`SemanticSearchAsync`, chat-rs
//! hybrid search, admin-rs product list — see STEP 0 of spec
//! rust-embedded-rag). Anything outside the subset is rejected with an
//! OpenSearch-shaped `parsing_exception` (see api.rs) so drift is loud,
//! not silently wrong.

use serde_json::Value;

/// The exact trust-score painless script catalog-rs sends (whitespace
/// normalized for comparison) — recognized and evaluated NATIVELY:
/// `boost = trust_score * ln1p(trust_count)`, `boost_mode: sum`.
pub const TRUST_SCRIPT: &str =
    "doc.containsKey('trust_score') && doc['trust_score'].size() != 0 ? Math.max(0.0, doc['trust_score'].value * Math.log1p(doc.containsKey('trust_count') && doc['trust_count'].size() != 0 ? doc['trust_count'].value : 0)) : 0";

pub fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Clone, PartialEq)]
pub struct DslError(pub String);

impl DslError {
    fn new(msg: impl Into<String>) -> Self {
        DslError(msg.into())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TermValue {
    Str(String),
    Bool(bool),
    Num(f64),
}

/// minimum_should_match forms: integer ("2"), percent ("75%") and
/// low-guard percent ("2<75%" — if there are fewer optional clauses than
/// the guard, all must match; otherwise floor(pct%)).
#[derive(Debug, Clone, PartialEq)]
pub enum Msm {
    Count(usize),
    Percent(u8),
    LowGuardPercent { low: usize, pct: u8 },
}

impl Msm {
    pub fn required(&self, clauses: usize) -> usize {
        match *self {
            Msm::Count(n) => n.min(clauses),
            Msm::Percent(p) => (clauses * p as usize / 100).min(clauses),
            Msm::LowGuardPercent { low, pct } => {
                if clauses < low {
                    clauses
                } else {
                    (clauses * pct as usize / 100).min(clauses)
                }
            }
        }
    }
}

fn parse_msm(v: &Value) -> Result<Msm, DslError> {
    match v {
        Value::Number(n) => n
            .as_u64()
            .map(|n| Msm::Count(n as usize))
            .ok_or_else(|| DslError::new("minimum_should_match must be a non-negative integer")),
        Value::String(s) => {
            let s = s.trim();
            if let Some((low, p)) = s.split_once('<') {
                let pct: u8 = p.strip_suffix('%').unwrap_or(p).parse().map_err(|_| {
                    DslError::new(format!("unsupported minimum_should_match '{s}'"))
                })?;
                let low: usize = low.parse().map_err(|_| {
                    DslError::new(format!("unsupported minimum_should_match '{s}'"))
                })?;
                return Ok(Msm::LowGuardPercent { low, pct });
            }
            if let Some(pct) = s.strip_suffix('%') {
                let pct: u8 = pct.parse().map_err(|_| {
                    DslError::new(format!("unsupported minimum_should_match '{s}'"))
                })?;
                Ok(Msm::Percent(pct))
            } else {
                s.parse::<usize>()
                    .map(Msm::Count)
                    .map_err(|_| DslError::new(format!("unsupported minimum_should_match '{s}'")))
            }
        }
        _ => Err(DslError::new("unsupported minimum_should_match value")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Operator {
    Or,
    And,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    MatchAll,
    Term {
        field: String,
        value: TermValue,
    },
    Terms {
        field: String,
        values: Vec<TermValue>,
    },
    Prefix {
        field: String,
        value: String,
    },
    Range {
        field: String,
        gte: Option<f64>,
        lte: Option<f64>,
    },
    Exists {
        field: String,
    },
    Bool(BoolQuery),
    /// `match` on an analyzed text field (plain or fuzzy-AUTO).
    Match {
        field: String,
        text: String,
        fuzziness: Option<String>,
        prefix_length: Option<usize>,
        boost: f64,
        operator: Operator,
    },
    MatchPhrase {
        field: String,
        text: String,
        boost: f64,
    },
    MatchPhrasePrefix {
        field: String,
        text: String,
        boost: f64,
    },
    MultiMatch {
        text: String,
        /// (field name, boost) — e.g. ("title", 3.0).
        fields: Vec<(String, f64)>,
        boost: f64,
        operator: Operator,
        minimum_should_match: Option<Msm>,
    },
    Knn {
        field: String,
        vector: Vec<f32>,
        k: usize,
    },
    FunctionScore {
        query: Box<Query>,
        script_score: Option<String>,
        random_seed: Option<i64>,
        boost_mode_sum: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BoolQuery {
    pub must: Vec<Query>,
    pub must_not: Vec<Query>,
    pub should: Vec<Query>,
    pub filter: Vec<Query>,
    pub minimum_should_match: Option<Msm>,
}

#[derive(Debug, Clone)]
pub enum Agg {
    Terms {
        field: String,
        size: usize,
        exclude: Vec<String>,
    },
    Stats {
        field: String,
    },
    Filter {
        filter: Box<Query>,
        aggs: Vec<(String, Agg)>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Order {
    Asc,
    Desc,
}

#[derive(Debug, Clone)]
pub struct SortSpec {
    pub key: String,
    pub order: Order,
    pub missing_last: bool,
}

#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub size: usize,
    pub from: usize,
    pub source_excludes: Vec<String>,
    pub query: Query,
    pub sort: Vec<SortSpec>,
    pub search_after: Option<Vec<SortValue>>,
    pub aggs: Vec<(String, Agg)>,
}

/// A parsed cursor/sort value — numbers stay f64, strings stay strings,
/// `null` marks a missing sort value (sorts last in both directions).
#[derive(Debug, Clone, PartialEq)]
pub enum SortValue {
    Num(f64),
    Str(String),
    Missing,
}

fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

fn parse_term_value(field: &str, v: &Value) -> Result<TermValue, DslError> {
    match v {
        Value::String(s) => Ok(TermValue::Str(s.clone())),
        Value::Bool(b) => Ok(TermValue::Bool(*b)),
        Value::Number(_) => as_f64(v)
            .map(TermValue::Num)
            .ok_or_else(|| DslError::new(format!("unsupported term value for field '{field}'"))),
        _ => Err(DslError::new(format!(
            "unsupported term value type for field '{field}'"
        ))),
    }
}

fn parse_boost(obj: &serde_json::Map<String, Value>) -> f64 {
    obj.get("boost")
        .and_then(|b| b.as_f64())
        .or_else(|| obj.get("boost").and_then(|b| b.as_i64().map(|i| i as f64)))
        .unwrap_or(1.0)
}

/// `{"field": {...}}` or `{"field": "text"}` inner-value unpacker for
/// match-family clauses.
fn match_params(
    field: &str,
    v: &Value,
) -> Result<(String, serde_json::Map<String, Value>), DslError> {
    match v {
        Value::String(s) => Ok((s.clone(), serde_json::Map::new())),
        Value::Object(o) => {
            let text = o
                .get("query")
                .and_then(|q| q.as_str())
                .ok_or_else(|| DslError::new(format!("match clause on '{field}' missing query")))?;
            Ok((text.to_string(), o.clone()))
        }
        _ => Err(DslError::new(format!(
            "unsupported match value on '{field}'"
        ))),
    }
}

fn parse_multi_match(obj: &serde_json::Map<String, Value>) -> Result<Query, DslError> {
    let text = obj
        .get("query")
        .and_then(|q| q.as_str())
        .ok_or_else(|| DslError::new("multi_match missing query"))?
        .to_string();
    let mut fields: Vec<(String, f64)> = Vec::new();
    let raw_fields = obj
        .get("fields")
        .and_then(|f| f.as_array())
        .ok_or_else(|| DslError::new("multi_match missing fields"))?;
    for f in raw_fields {
        let name = f
            .as_str()
            .ok_or_else(|| DslError::new("multi_match fields must be strings"))?;
        if let Some((base, boost)) = name.split_once('^') {
            fields.push((
                base.to_string(),
                boost
                    .parse()
                    .map_err(|_| DslError::new(format!("unsupported field boost '{name}'")))?,
            ));
        } else {
            fields.push((name.to_string(), 1.0));
        }
    }
    let mm_type = obj
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("best_fields");
    if mm_type != "best_fields" {
        return Err(DslError::new(format!(
            "unsupported multi_match type '{mm_type}' (only best_fields is implemented)"
        )));
    }
    let operator = match obj.get("operator").and_then(|o| o.as_str()).unwrap_or("or") {
        "or" => Operator::Or,
        "and" => Operator::And,
        other => {
            return Err(DslError::new(format!(
                "unsupported multi_match operator '{other}'"
            )))
        }
    };
    let minimum_should_match = match obj.get("minimum_should_match") {
        Some(v) => Some(parse_msm(v)?),
        None => None,
    };
    Ok(Query::MultiMatch {
        text,
        fields,
        boost: parse_boost(obj),
        operator,
        minimum_should_match,
    })
}

pub fn parse_query(v: &Value) -> Result<Query, DslError> {
    let obj = v
        .as_object()
        .ok_or_else(|| DslError::new("query clause must be an object"))?;
    if obj.len() != 1 {
        return Err(DslError::new(
            "query clause must contain exactly one query type",
        ));
    }
    let (name, body) = obj.iter().next().expect("len checked");
    match name.as_str() {
        "match_all" => Ok(Query::MatchAll),
        "term" => {
            let to = body
                .as_object()
                .filter(|o| o.len() == 1)
                .ok_or_else(|| DslError::new("term clause must contain exactly one field"))?;
            let (field, value) = to.iter().next().expect("len checked");
            Ok(Query::Term {
                field: field.clone(),
                value: parse_term_value(field, value)?,
            })
        }
        "terms" => {
            let to = body
                .as_object()
                .filter(|o| o.len() == 1)
                .ok_or_else(|| DslError::new("terms clause must contain exactly one field"))?;
            let (field, values) = to.iter().next().expect("len checked");
            let arr = values
                .as_array()
                .ok_or_else(|| DslError::new("terms clause value must be an array"))?;
            let parsed = arr
                .iter()
                .map(|v| parse_term_value(field, v))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Query::Terms {
                field: field.clone(),
                values: parsed,
            })
        }
        "prefix" => {
            let to = body
                .as_object()
                .filter(|o| o.len() == 1)
                .ok_or_else(|| DslError::new("prefix clause must contain exactly one field"))?;
            let (field, value) = to.iter().next().expect("len checked");
            let value = value
                .as_str()
                .ok_or_else(|| DslError::new("prefix clause value must be a string"))?
                .to_string();
            Ok(Query::Prefix {
                field: field.clone(),
                value,
            })
        }
        "range" => {
            let to = body
                .as_object()
                .filter(|o| o.len() == 1)
                .ok_or_else(|| DslError::new("range clause must contain exactly one field"))?;
            let (field, bounds) = to.iter().next().expect("len checked");
            let bounds = bounds
                .as_object()
                .ok_or_else(|| DslError::new("range bounds must be an object"))?;
            let gte = bounds.get("gte").and_then(as_f64);
            let lte = bounds.get("lte").and_then(as_f64);
            if bounds.contains_key("gt") || bounds.contains_key("lt") {
                return Err(DslError::new(
                    "range gt/lt bounds are not supported (only gte/lte)",
                ));
            }
            Ok(Query::Range {
                field: field.clone(),
                gte,
                lte,
            })
        }
        "exists" => {
            let field = body
                .get("field")
                .and_then(|f| f.as_str())
                .ok_or_else(|| DslError::new("exists clause missing field"))?;
            Ok(Query::Exists {
                field: field.to_string(),
            })
        }
        "bool" => {
            let b = body
                .as_object()
                .ok_or_else(|| DslError::new("bool clause must be an object"))?;
            let mut q = BoolQuery::default();
            for (k, v) in b {
                match k.as_str() {
                    "minimum_should_match" => {
                        q.minimum_should_match = Some(parse_msm(v)?);
                        continue;
                    }
                    "must" | "must_not" | "should" | "filter" => {}
                    other => {
                        return Err(DslError::new(format!("unsupported bool clause '{other}'")))
                    }
                }
                let list = v.as_array().cloned().unwrap_or_else(|| vec![v.clone()]);
                let parsed = list
                    .iter()
                    .map(parse_query)
                    .collect::<Result<Vec<_>, _>>()?;
                match k.as_str() {
                    "must" => q.must.extend(parsed),
                    "must_not" => q.must_not.extend(parsed),
                    "should" => q.should.extend(parsed),
                    "filter" => q.filter.extend(parsed),
                    _ => unreachable!("checked above"),
                }
            }
            Ok(Query::Bool(q))
        }
        "match" => {
            let to = body
                .as_object()
                .filter(|o| o.len() == 1)
                .ok_or_else(|| DslError::new("match clause must contain exactly one field"))?;
            let (field, value) = to.iter().next().expect("len checked");
            let (text, params) = match_params(field, value)?;
            let fuzziness = params
                .get("fuzziness")
                .and_then(|f| f.as_str())
                .map(String::from);
            if let Some(f) = &fuzziness {
                if f != "AUTO" {
                    return Err(DslError::new(format!(
                        "unsupported fuzziness '{f}' (only AUTO is implemented)"
                    )));
                }
            }
            let prefix_length = params
                .get("prefix_length")
                .and_then(|p| p.as_u64())
                .map(|p| p as usize);
            let operator = match params
                .get("operator")
                .and_then(|o| o.as_str())
                .unwrap_or("or")
            {
                "or" => Operator::Or,
                "and" => Operator::And,
                other => {
                    return Err(DslError::new(format!(
                        "unsupported match operator '{other}'"
                    )))
                }
            };
            Ok(Query::Match {
                field: field.clone(),
                text,
                fuzziness,
                prefix_length,
                boost: parse_boost(&params),
                operator,
            })
        }
        "match_phrase" => {
            let to = body.as_object().filter(|o| o.len() == 1).ok_or_else(|| {
                DslError::new("match_phrase clause must contain exactly one field")
            })?;
            let (field, value) = to.iter().next().expect("len checked");
            let (text, params) = match_params(field, value)?;
            Ok(Query::MatchPhrase {
                field: field.clone(),
                text,
                boost: parse_boost(&params),
            })
        }
        "match_phrase_prefix" => {
            let to = body.as_object().filter(|o| o.len() == 1).ok_or_else(|| {
                DslError::new("match_phrase_prefix clause must contain exactly one field")
            })?;
            let (field, value) = to.iter().next().expect("len checked");
            let (text, params) = match_params(field, value)?;
            Ok(Query::MatchPhrasePrefix {
                field: field.clone(),
                text,
                boost: parse_boost(&params),
            })
        }
        "multi_match" => {
            let o = body
                .as_object()
                .ok_or_else(|| DslError::new("multi_match clause must be an object"))?;
            parse_multi_match(o)
        }
        "knn" => {
            let to = body
                .as_object()
                .filter(|o| o.len() == 1)
                .ok_or_else(|| DslError::new("knn clause must contain exactly one field"))?;
            let (field, params) = to.iter().next().expect("len checked");
            let params = params
                .as_object()
                .ok_or_else(|| DslError::new("knn clause value must be an object"))?;
            let vector: Vec<f32> = params
                .get("vector")
                .and_then(|v| v.as_array())
                .ok_or_else(|| DslError::new("knn clause missing vector array"))?
                .iter()
                .map(|v| {
                    as_f64(v)
                        .map(|f| f as f32)
                        .ok_or_else(|| DslError::new("knn vector values must be numbers"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let k = params
                .get("k")
                .and_then(|k| k.as_u64())
                .ok_or_else(|| DslError::new("knn clause missing k"))? as usize;
            Ok(Query::Knn {
                field: field.clone(),
                vector,
                k,
            })
        }
        "function_score" => {
            let o = body
                .as_object()
                .ok_or_else(|| DslError::new("function_score clause must be an object"))?;
            let inner = o
                .get("query")
                .ok_or_else(|| DslError::new("function_score missing query"))?;
            let query = Box::new(parse_query(inner)?);
            let mut script_score = None;
            if let Some(ss) = o.get("script_score") {
                let source = ss
                    .get("script")
                    .and_then(|s| s.get("source"))
                    .and_then(|s| s.as_str())
                    .ok_or_else(|| DslError::new("script_score missing script.source"))?;
                let normalized = normalize_ws(source);
                if normalized != normalize_ws(TRUST_SCRIPT) {
                    return Err(DslError::new(
                        "unsupported script_score script — only the trust_score*log1p(trust_count) script is implemented",
                    ));
                }
                script_score = Some(normalized);
            }
            let mut random_seed = None;
            if let Some(fns) = o.get("functions").and_then(|f| f.as_array()) {
                if fns.len() > 1 {
                    return Err(DslError::new(
                        "function_score supports at most one scoring function",
                    ));
                }
                if let Some(f) = fns.first() {
                    let rs = f
                        .get("random_score")
                        .ok_or_else(|| {
                            DslError::new("unsupported function_score function (only random_score is implemented)")
                        })?
                        .as_object()
                        .ok_or_else(|| DslError::new("random_score must be an object"))?;
                    random_seed = Some(
                        rs.get("seed")
                            .and_then(|s| s.as_i64())
                            .ok_or_else(|| DslError::new("random_score missing seed"))?,
                    );
                }
            }
            let boost_mode_sum = match o.get("boost_mode").and_then(|b| b.as_str()) {
                Some("sum") => true,
                Some("multiply") | None => false,
                Some(other) => {
                    return Err(DslError::new(format!(
                        "unsupported boost_mode '{other}' (only sum/multiply are implemented)"
                    )))
                }
            };
            Ok(Query::FunctionScore {
                query,
                script_score,
                random_seed,
                boost_mode_sum,
            })
        }
        other => Err(DslError::new(format!("unsupported query clause '{other}'"))),
    }
}

fn parse_sort(body: Option<&Value>) -> Result<Vec<SortSpec>, DslError> {
    let Some(sort) = body else {
        return Ok(vec![SortSpec {
            key: "_score".into(),
            order: Order::Desc,
            missing_last: true,
        }]);
    };
    let arr = sort
        .as_array()
        .ok_or_else(|| DslError::new("sort must be an array"))?;
    let mut out = Vec::new();
    for item in arr {
        match item {
            Value::String(key) => out.push(SortSpec {
                key: key.clone(),
                order: Order::Desc,
                missing_last: true,
            }),
            Value::Object(o) if o.len() == 1 => {
                let (key, params) = o.iter().next().expect("len checked");
                let params = params
                    .as_object()
                    .ok_or_else(|| DslError::new("sort params must be an object"))?;
                let order = match params
                    .get("order")
                    .and_then(|o| o.as_str())
                    .unwrap_or("asc")
                {
                    "asc" => Order::Asc,
                    "desc" => Order::Desc,
                    other => {
                        return Err(DslError::new(format!("unsupported sort order '{other}'")))
                    }
                };
                let missing_last = match params.get("missing").and_then(|m| m.as_str()) {
                    Some("_last") => true,
                    Some("_first") => false,
                    Some(other) => {
                        return Err(DslError::new(format!("unsupported sort missing '{other}'")))
                    }
                    None => true,
                };
                out.push(SortSpec {
                    key: key.clone(),
                    order,
                    missing_last,
                });
            }
            _ => return Err(DslError::new("unsupported sort entry")),
        }
    }
    Ok(out)
}

fn parse_aggs(body: &Value) -> Result<Vec<(String, Agg)>, DslError> {
    let Some(obj) = body.as_object() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for (name, spec) in obj {
        let so = spec
            .as_object()
            .ok_or_else(|| DslError::new(format!("aggregation '{name}' must be an object")))?;
        let agg = if let Some(t) = so.get("terms") {
            let to = t
                .as_object()
                .ok_or_else(|| DslError::new("terms agg params must be an object"))?;
            Agg::Terms {
                field: to
                    .get("field")
                    .and_then(|f| f.as_str())
                    .ok_or_else(|| DslError::new("terms agg missing field"))?
                    .to_string(),
                size: to.get("size").and_then(|s| s.as_u64()).unwrap_or(10) as usize,
                exclude: to
                    .get("exclude")
                    .and_then(|e| e.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
            }
        } else if let Some(s) = so.get("stats") {
            Agg::Stats {
                field: s
                    .as_object()
                    .and_then(|o| o.get("field"))
                    .and_then(|f| f.as_str())
                    .ok_or_else(|| DslError::new("stats agg missing field"))?
                    .to_string(),
            }
        } else if let Some(f) = so.get("filter") {
            let filter = Box::new(parse_query(f)?);
            let mut sub = Vec::new();
            if let Some(a) = so.get("aggs") {
                sub = parse_aggs(a)?;
            }
            Agg::Filter { filter, aggs: sub }
        } else {
            return Err(DslError::new(format!(
                "unsupported aggregation type in '{name}' (terms/stats/filter are implemented)"
            )));
        };
        out.push((name.clone(), agg));
    }
    Ok(out)
}

fn parse_sort_values(v: &Value) -> Result<Option<Vec<SortValue>>, DslError> {
    let Some(arr) = v.as_array() else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for item in arr {
        match item {
            Value::Number(_) => out.push(SortValue::Num(
                as_f64(item).ok_or_else(|| DslError::new("bad search_after value"))?,
            )),
            Value::String(s) => out.push(SortValue::Str(s.clone())),
            Value::Bool(b) => out.push(SortValue::Num(if *b { 1.0 } else { 0.0 })),
            Value::Null => out.push(SortValue::Missing),
            _ => return Err(DslError::new("unsupported search_after value")),
        }
    }
    Ok(Some(out))
}

pub fn parse_search_body(body: &Value) -> Result<SearchRequest, DslError> {
    let obj = body
        .as_object()
        .ok_or_else(|| DslError::new("search body must be an object"))?;
    let size = obj.get("size").and_then(|s| s.as_u64()).unwrap_or(10) as usize;
    let from = obj.get("from").and_then(|f| f.as_u64()).unwrap_or(0) as usize;
    let source_excludes = obj
        .get("_source")
        .and_then(|s| s.get("excludes"))
        .and_then(|e| e.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let query = match obj.get("query") {
        Some(q) => parse_query(q)?,
        None => Query::MatchAll,
    };
    let sort = parse_sort(obj.get("sort"))?;
    let search_after = match obj.get("search_after") {
        Some(v) => parse_sort_values(v)?,
        None => None,
    };
    let aggs = match obj.get("aggs") {
        Some(a) => parse_aggs(a)?,
        None => Vec::new(),
    };
    Ok(SearchRequest {
        size,
        from,
        source_excludes,
        query,
        sort,
        search_after,
        aggs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_term_terms_range_exists() {
        let q = parse_query(&json!({ "term": { "slug.keyword": "foo" } })).unwrap();
        assert_eq!(
            q,
            Query::Term {
                field: "slug.keyword".into(),
                value: TermValue::Str("foo".into())
            }
        );
        let q = parse_query(&json!({ "term": { "available": true } })).unwrap();
        assert_eq!(
            q,
            Query::Term {
                field: "available".into(),
                value: TermValue::Bool(true)
            }
        );
        let q = parse_query(&json!({ "terms": { "slug.keyword": ["a", "b"] } })).unwrap();
        assert!(matches!(q, Query::Terms { .. }));
        let q = parse_query(&json!({
            "range": { "price": { "gte": 100.5, "lte": 500 } }
        }))
        .unwrap();
        assert_eq!(
            q,
            Query::Range {
                field: "price".into(),
                gte: Some(100.5),
                lte: Some(500.0)
            }
        );
        let q = parse_query(&json!({ "exists": { "field": "status" } })).unwrap();
        assert_eq!(
            q,
            Query::Exists {
                field: "status".into()
            }
        );
    }

    #[test]
    fn parses_bool_with_msm() {
        let q = parse_query(&json!({
            "bool": {
                "must": [{ "term": { "slug.keyword": "x" } }],
                "should": [
                    { "term": { "brand.keyword": "nike" } },
                    { "term": { "vendor.keyword": "nike" } }
                ],
                "minimum_should_match": 1,
                "filter": [{ "term": { "status.keyword": "approved" } }]
            }
        }))
        .unwrap();
        match q {
            Query::Bool(b) => {
                assert_eq!(b.must.len(), 1);
                assert_eq!(b.should.len(), 2);
                assert_eq!(b.filter.len(), 1);
                assert_eq!(b.minimum_should_match, Some(Msm::Count(1)));
            }
            _ => panic!("expected bool"),
        }
    }

    #[test]
    fn parses_multi_match_best_fields_with_boosts() {
        let q = parse_query(&json!({
            "multi_match": {
                "query": "buty biegania",
                "fields": ["title^3", "vendor^2", "description", "tags", "specs_text", "reviews_text"],
                "type": "best_fields",
                "operator": "or",
                "minimum_should_match": "2<75%",
                "lenient": true,
                "boost": 2
            }
        }))
        .unwrap();
        match q {
            Query::MultiMatch {
                fields,
                minimum_should_match,
                boost,
                operator,
                ..
            } => {
                assert_eq!(fields[0], ("title".into(), 3.0));
                assert_eq!(fields[1], ("vendor".into(), 2.0));
                assert_eq!(fields.len(), 6);
                assert_eq!(boost, 2.0);
                assert_eq!(operator, Operator::Or);
                assert_eq!(
                    minimum_should_match,
                    Some(Msm::LowGuardPercent { low: 2, pct: 75 })
                );
            }
            _ => panic!("expected multi_match"),
        }
    }

    #[test]
    fn msm_required_semantics() {
        // "2<75%": 1 clause → all; 2 clauses → 1; 4 clauses → 3; 8 → 6.
        let m = Msm::LowGuardPercent { low: 2, pct: 75 };
        assert_eq!(m.required(1), 1);
        assert_eq!(m.required(2), 1);
        assert_eq!(m.required(4), 3);
        assert_eq!(m.required(8), 6);
        assert_eq!(Msm::Count(1).required(5), 1);
        assert_eq!(Msm::Percent(75).required(4), 3);
    }

    #[test]
    fn parses_match_fuzzy_auto_and_phrase() {
        let q = parse_query(&json!({
            "match": { "title": { "query": "buty", "fuzziness": "AUTO", "prefix_length": 2 } }
        }))
        .unwrap();
        assert!(matches!(
            q,
            Query::Match {
                fuzziness: Some(_),
                prefix_length: Some(2),
                ..
            }
        ));
        let q = parse_query(&json!({ "match": { "brand": "nike" } })).unwrap();
        assert!(
            matches!(q, Query::Match { field, text, .. } if field == "brand" && text == "nike")
        );
        let q = parse_query(&json!({
            "match_phrase": { "title": { "query": "buty do biegania", "boost": 6 } }
        }))
        .unwrap();
        assert!(matches!(q, Query::MatchPhrase { boost: 6.0, .. }));
        let q = parse_query(&json!({
            "match_phrase_prefix": { "title": { "query": "buty do", "boost": 2 } }
        }))
        .unwrap();
        assert!(matches!(q, Query::MatchPhrasePrefix { boost: 2.0, .. }));
    }

    #[test]
    fn parses_knn() {
        let q = parse_query(&json!({
            "knn": { "embedding": { "vector": [0.1, 0.2, 0.3], "k": 50 } }
        }))
        .unwrap();
        assert_eq!(
            q,
            Query::Knn {
                field: "embedding".into(),
                vector: vec![0.1, 0.2, 0.3],
                k: 50
            }
        );
    }

    #[test]
    fn parses_function_score_trust_script() {
        let source = TRUST_SCRIPT;
        let body = json!({
            "function_score": {
                "query": { "bool": { "must": [{ "match_all": {} }] } },
                "script_score": { "script": { "source": source } },
                "boost_mode": "sum"
            }
        });
        let q = parse_query(&body).unwrap();
        match q {
            Query::FunctionScore {
                script_score: Some(_),
                boost_mode_sum: true,
                ..
            } => {}
            other => panic!("expected trust function_score, got {other:?}"),
        }
    }

    #[test]
    fn rejects_trust_script_with_wrong_script() {
        let source = "doc['foo'].value * 2";
        let body = json!({
            "function_score": {
                "query": { "match_all": {} },
                "script_score": { "script": { "source": source } }
            }
        });
        let e = parse_query(&body).unwrap_err();
        assert!(e.0.contains("trust_score"));
    }

    #[test]
    fn rejects_unknown_clauses_and_fuzziness() {
        let e = parse_query(&json!({ "wildcard": { "slug": "*x*" } })).unwrap_err();
        assert!(e.0.contains("unsupported query clause 'wildcard'"));
        let e = parse_query(&json!({
            "match": { "title": { "query": "x", "fuzziness": "2" } }
        }))
        .unwrap_err();
        assert!(e.0.contains("only AUTO"));
        let e = parse_query(&json!({ "function_score": {
            "query": { "match_all": {} },
            "script_score": { "script": { "source": "doc['foo'].value" } }
        }}))
        .unwrap_err();
        assert!(e.0.contains("trust_score"));
        let e = parse_query(&json!({ "bool": { "boost": 1 } })).unwrap_err();
        assert!(e.0.contains("unsupported bool clause 'boost'"));
    }

    #[test]
    fn parses_random_score_function_score() {
        let q = parse_query(&json!({
            "function_score": {
                "query": { "match_all": {} },
                "functions": [{ "random_score": { "seed": 42, "field": "_seq_no" } }]
            }
        }))
        .unwrap();
        match q {
            Query::FunctionScore {
                random_seed: Some(42),
                boost_mode_sum: false,
                ..
            } => {}
            other => panic!("expected random function_score, got {other:?}"),
        }
    }

    #[test]
    fn parses_full_search_body_shape() {
        let body = json!({
            "size": 24,
            "from": 0,
            "track_total_hits": true,
            "_source": { "excludes": ["embedding"] },
            "query": { "bool": { "must": [{ "match_all": {} }] } },
            "sort": [
                { "price": { "order": "asc", "missing": "_last" } },
                { "slug.keyword": { "order": "asc" } }
            ],
            "aggs": {
                "filtered": {
                    "filter": { "term": { "type.keyword": "Buty" } },
                    "aggs": {
                        "categories": { "terms": { "field": "type.keyword", "size": 20 } },
                        "price_stats": {
                            "filter": { "term": { "currency.keyword": "PLN" } },
                            "aggs": { "stats": { "stats": { "field": "price" } } }
                        }
                    }
                }
            }
        });
        let req = parse_search_body(&body).unwrap();
        assert_eq!(req.size, 24);
        assert_eq!(req.source_excludes, vec!["embedding"]);
        assert_eq!(req.sort.len(), 2);
        assert_eq!(req.sort[0].key, "price");
        assert_eq!(req.sort[0].order, Order::Asc);
        assert_eq!(req.sort[1].key, "slug.keyword");
        assert_eq!(req.aggs.len(), 1);
        match &req.aggs[0].1 {
            Agg::Filter { aggs, .. } => {
                assert_eq!(aggs[0].0, "categories");
                assert!(matches!(aggs[1].1, Agg::Filter { .. }));
            }
            _ => panic!("expected filter agg"),
        }
        assert_eq!(parse_search_body(&json!({})).unwrap().size, 10);
    }

    #[test]
    fn parses_search_after() {
        let body = json!({
            "query": { "match_all": {} },
            "sort": [{ "price": { "order": "asc", "missing": "_last" } }, { "slug.keyword": { "order": "asc" } }],
            "search_after": [91.76, "asics-1012b765-003"]
        });
        let req = parse_search_body(&body).unwrap();
        assert_eq!(
            req.search_after,
            Some(vec![
                SortValue::Num(91.76),
                SortValue::Str("asics-1012b765-003".into())
            ])
        );
    }
}
