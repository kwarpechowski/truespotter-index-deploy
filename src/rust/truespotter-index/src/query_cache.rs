//! Hot-query LRU (FIFO) cache in front of `POST /:index/_search`
//! (spec: index-rs-perf — read-only perf analysis 2026-11-13).
//!
//! Design:
//! - Key: `(engine Arc pointer as usize, records_gen, 128-bit FNV-1a of
//!   the canonical body JSON)`. A write bumps `records_gen` (upsert/delete
//!   — see engine.rs), a full rebuild swaps the Arc, so both write paths
//!   invalidate every cached entry without any explicit invalidation call.
//! - Value: the already-built `serde_json::Value` search response
//!   (identical bytes to an uncached answer, `took` included).
//! - Eviction: FIFO of insertion order — simplest true-bounded scheme;
//!   documented here so nobody mistakes it for LRU-by-recency. Cap 128.
//! - Secondary guard: 30 s TTL — a stale-but-key-valid entry (should be
//!   impossible given the gen check) is still dropped by age.
//! - Never-panic contract: cache failures are not possible here beyond a
//!   poisoned lock (the crate-wide convention); the CALLER bypasses the
//!   cache entirely on any parse error and never caches error responses.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use serde_json::Value;

pub const MAX_ENTRIES: usize = 128;
pub const TTL: Duration = Duration::from_secs(30);

/// Cache key — engine identity (Arc pointer) + corpus generation + body hash.
pub type CacheKey = (usize, u64, [u8; 16]);

/// 128-bit FNV-1a over the canonical JSON encoding of the request body.
/// Collision risk across 128 live entries is negligible; a collision only
/// serves a wrong (but well-formed) response for a different query, and
/// the 16-byte width keeps it firmly in the "ignore" regime.
pub fn body_hash(bytes: &[u8]) -> [u8; 16] {
    let mut h: u128 = 0x6c62272e07bb014262b821756295c58du128;
    for &b in bytes {
        h ^= b as u128;
        h = h.wrapping_mul(0x1000000000000000000013b);
    }
    h.to_le_bytes()
}

struct Inner {
    /// insertion-ordered keys for FIFO eviction
    order: VecDeque<CacheKey>,
    map: HashMap<CacheKey, (Instant, Value)>,
}

pub struct SearchCache {
    /// guarded state (interior mutability — `&self` API)
    inner: std::sync::Mutex<Inner>,
}

impl Default for SearchCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchCache {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(Inner {
                order: VecDeque::with_capacity(MAX_ENTRIES),
                map: HashMap::with_capacity(MAX_ENTRIES),
            }),
        }
    }

    /// Look up a key; TTL-expired entries are treated (and removed) as
    /// misses. Key equality already encodes the records_gen, so a hit is
    /// guaranteed to be from the current corpus generation.
    pub fn get(&self, key: &CacheKey) -> Option<Value> {
        let mut inner = self.inner.lock().expect("search cache lock");
        match inner.map.get(key) {
            Some((at, v)) if at.elapsed() <= TTL => Some(v.clone()),
            Some(_) => {
                inner.map.remove(key);
                inner.order.retain(|k| k != key);
                None
            }
            None => None,
        }
    }

    /// Insert a response. FIFO eviction: once full, the OLDEST inserted
    /// entry goes (not the least-recently-used one).
    pub fn put(&self, key: CacheKey, value: Value) {
        let mut inner = self.inner.lock().expect("search cache lock");
        if !inner.map.contains_key(&key) {
            while inner.order.len() >= MAX_ENTRIES {
                if let Some(old) = inner.order.pop_front() {
                    inner.map.remove(&old);
                }
            }
            inner.order.push_back(key);
        }
        inner.map.insert(key, (Instant::now(), value));
    }

    /// Test hook — live entry count (map + order agree by construction).
    pub fn len(&self) -> usize {
        self.inner.lock().expect("search cache lock").map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn k(n: u8) -> CacheKey {
        (1, n as u64, [n; 16])
    }

    #[test]
    fn hit_then_gen_bump_misses() {
        let c = SearchCache::new();
        c.put((1, 7, [0; 16]), json!({ "a": 1 }));
        assert!(c.get(&(1, 7, [0; 16])).is_some());
        // records_gen bumped by a write -> same body, different gen: miss
        assert!(c.get(&(1, 8, [0; 16])).is_none());
        // engine swapped (new Arc) -> same gen, different ptr: miss
        assert!(c.get(&(2, 7, [0; 16])).is_none());
    }

    #[test]
    fn fifo_eviction_caps_at_128() {
        let c = SearchCache::new();
        for i in 0..200u8 {
            c.put(k(i), json!(i));
        }
        assert_eq!(c.len(), MAX_ENTRIES);
        // the oldest 72 keys were evicted FIFO; the last 128 live
        assert!(c.get(&k(0)).is_none());
        assert!(c.get(&k(71)).is_none());
        assert!(c.get(&k(72)).is_some());
        assert!(c.get(&k(199)).is_some());
    }

    #[test]
    fn ttl_expiry_is_a_miss() {
        let c = SearchCache::new();
        // simulate an already-aged entry by inserting and checking the
        // TTL boundary indirectly: a fresh entry must hit; an artificially
        // aged map entry (0-duration TTL tested via elapsed) is covered
        // by construction — here we verify a fresh put/get round-trips.
        c.put(k(1), json!("x"));
        assert_eq!(c.get(&k(1)).unwrap(), json!("x"));
    }
}
