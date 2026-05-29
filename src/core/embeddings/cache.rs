//! LRU cache decorator for any [`Embedding`] provider.
//!
//! [`CachedEmbedding`] wraps an inner embedding with two LRU caches (individual + batch).
//! Uses `lru::LruCache` with hash-based keys for O(1) lookup.

use anyhow::Result;
use async_trait::async_trait;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Mutex;

use crate::core::embeddings::Embedding;

// ---------------------------------------------------------------------------
// CachedEmbedding — LRU cache decorator over any Embedding provider
// ---------------------------------------------------------------------------
pub struct CachedEmbedding {
    inner: Box<dyn Embedding>,
    cache: Mutex<LruCache<String, Vec<f32>>>,
    batch_cache: Mutex<LruCache<String, Vec<Vec<f32>>>>,
}

impl CachedEmbedding {
    pub fn new(inner: Box<dyn Embedding>, capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::MAX);
        Self {
            inner,
            cache: Mutex::new(LruCache::new(cap)),
            batch_cache: Mutex::new(LruCache::new(NonZeroUsize::new(256).unwrap())),
        }
    }

    fn cache_key(text: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn batch_cache_key(texts: &[&str]) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for t in texts {
            t.hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }
}

#[async_trait]
impl Embedding for CachedEmbedding {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let key = Self::cache_key(text);
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(vec) = cache.get(&key) {
                return Ok(vec.clone());
            }
        }
        let result = self.inner.embed(text).await?;
        let mut cache = self.cache.lock().unwrap();
        cache.put(key, result.clone());
        Ok(result)
    }

    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.len() == 1 {
            return Ok(vec![self.embed(texts[0]).await?]);
        }

        let bkey = Self::batch_cache_key(texts);
        {
            let mut bcache = self.batch_cache.lock().unwrap();
            if let Some(vec) = bcache.get(&bkey) {
                return Ok(vec.clone());
            }
        }

        // check individual cache first
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached = Vec::new();
        let mut uncached_idx = Vec::new();

        for (i, t) in texts.iter().enumerate() {
            let key = Self::cache_key(t);
            let mut cache = self.cache.lock().unwrap();
            if let Some(vec) = cache.get(&key) {
                results.push((i, vec.clone()));
            } else {
                drop(cache);
                uncached.push(*t);
                uncached_idx.push(i);
            }
        }

        if !uncached.is_empty() {
            let fresh = self.inner.embed_many(&uncached).await?;
            let mut cache = self.cache.lock().unwrap();
            for (j, emb) in fresh.into_iter().enumerate() {
                let key = Self::cache_key(uncached[j]);
                cache.put(key, emb.clone());
                results.push((uncached_idx[j], emb));
            }
        }

        results.sort_by_key(|(i, _)| *i);
        let final_result: Vec<Vec<f32>> = results.into_iter().map(|(_, emb)| emb).collect();

        {
            let mut bcache = self.batch_cache.lock().unwrap();
            bcache.put(bkey, final_result.clone());
        }

        Ok(final_result)
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
}
