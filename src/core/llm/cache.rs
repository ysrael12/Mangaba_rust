//! In-memory TTL cache for LLM responses.
//!
//! [`LLMCache`] stores serialized responses with a time-to-live. Keys are
//! generated from (provider + model + messages) via [`make_cache_key`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use async_trait::async_trait;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub response: Value,
    pub inserted_at: Instant,
}

#[async_trait]
pub trait LLMCache: Send + Sync {
    async fn get(&self, key: &str) -> Option<Value>;
    async fn set(&self, key: &str, value: Value);
    async fn clear(&self);
}

pub struct InMemoryCache {
    store: Arc<Mutex<HashMap<String, CacheEntry>>>,
    ttl: Option<Duration>,
}

impl InMemoryCache {
    pub fn new(ttl: Option<Duration>) -> Self {
        Self { store: Arc::new(Mutex::new(HashMap::new())), ttl }
    }
}

#[async_trait]
impl LLMCache for InMemoryCache {
    async fn get(&self, key: &str) -> Option<Value> {
        let store = self.store.lock().unwrap();
        if let Some(entry) = store.get(key) {
            if let Some(ttl) = self.ttl {
                if entry.inserted_at.elapsed() > ttl {
                    return None;
                }
            }
            Some(entry.response.clone())
        } else {
            None
        }
    }

    async fn set(&self, key: &str, value: Value) {
        let mut store = self.store.lock().unwrap();
        store.insert(key.to_string(), CacheEntry {
            response: value,
            inserted_at: Instant::now(),
        });
    }

    async fn clear(&self) {
        let mut store = self.store.lock().unwrap();
        store.clear();
    }
}

pub fn make_cache_key(model: &str, messages: &[crate::core::types::Message]) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    model.hash(&mut hasher);
    for msg in messages {
        format!("{:?}", msg.role).hash(&mut hasher);
        msg.content.hash(&mut hasher);
    }
    format!("{}::{:x}", model, hasher.finish())
}
