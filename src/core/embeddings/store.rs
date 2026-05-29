//! In-memory vector store with cosine-similarity search.
//!
//! [`InMemoryVectorStore`] stores [`VectorEntry`]s (id, text, embedding, metadata)
//! and supports add, add_batch, search (by query string), len, and clear.

use std::sync::Arc;
use anyhow::Result;
use tokio::sync::Mutex;
use uuid::Uuid;
use crate::core::embeddings::{Embedding, cosine_similarity};

#[derive(Debug, Clone)]
pub struct VectorEntry {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

pub struct InMemoryVectorStore {
    embedding: Arc<dyn Embedding>,
    entries: Arc<Mutex<Vec<VectorEntry>>>,
}

impl InMemoryVectorStore {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self {
            embedding,
            entries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn add(&self, text: &str, metadata: Option<serde_json::Value>) -> Result<()> {
        let emb = self.embedding.embed(text).await?;
        let mut entries = self.entries.lock().await;
        entries.push(VectorEntry {
            id: format!("vec_{}", Uuid::new_v4().simple()),
            text: text.to_string(),
            embedding: emb,
            metadata,
        });
        Ok(())
    }

    pub async fn add_batch(&self, texts: &[&str], metadata: Option<&[Option<serde_json::Value>]>) -> Result<()> {
        let embeddings = self.embedding.embed_many(texts).await?;
        let mut entries = self.entries.lock().await;
        for (i, text) in texts.iter().enumerate() {
            let meta = metadata.and_then(|m| m.get(i).cloned()).unwrap_or(None);
            entries.push(VectorEntry {
                id: format!("vec_{}", Uuid::new_v4().simple()),
                text: text.to_string(),
                embedding: embeddings[i].clone(),
                metadata: meta,
            });
        }
        Ok(())
    }

    pub async fn search(&self, query: &str, k: usize) -> Result<Vec<VectorEntry>> {
        let query_emb = self.embedding.embed(query).await?;
        let entries = self.entries.lock().await;

        let mut scored: Vec<(f32, usize)> = entries.iter().enumerate()
            .map(|(i, e)| (cosine_similarity(&query_emb, &e.embedding), i))
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored.iter()
            .take(k)
            .filter(|(score, _)| *score > 0.0)
            .map(|(_, i)| entries[*i].clone())
            .collect())
    }

    pub async fn len(&self) -> usize {
        self.entries.lock().await.len()
    }

    pub async fn clear(&self) {
        self.entries.lock().await.clear();
    }
}
