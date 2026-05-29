//! Embedding trait and implementations with vector storage.
//!
//! Defines [`Embedding`] with `embed` / `embed_many` / `dimensions`.
//! Providers: [`OpenAIEmbedding`], [`HuggingFaceEmbedding`] (Inference API),
//! [`NoOpEmbedding`] (zero vectors), [`CachedEmbedding`] (LRU cache decorator).
//!
//! [`InMemoryVectorStore`] provides cosine-similarity search backed by an in-memory vec.
//! The standalone [`cosine_similarity`] function works on raw `&[f32]` slices.

use anyhow::Result;
use async_trait::async_trait;

pub mod cache;
pub mod huggingface;
pub mod openai;
pub mod store;

pub use cache::CachedEmbedding;
pub use huggingface::HuggingFaceEmbedding;
pub use openai::OpenAIEmbedding;
pub use store::InMemoryVectorStore;

// ---------------------------------------------------------------------------
// Embedding trait
// ---------------------------------------------------------------------------
#[async_trait]
pub trait Embedding: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Embedding-free dummy — returns zero vector (no-op for when no provider)
// ---------------------------------------------------------------------------
pub struct NoOpEmbedding {
    dims: usize,
}

impl NoOpEmbedding {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

#[async_trait]
impl Embedding for NoOpEmbedding {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![0.0; self.dims])
    }

    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; self.dims]).collect())
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}
