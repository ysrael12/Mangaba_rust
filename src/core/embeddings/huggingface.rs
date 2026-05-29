//! HuggingFace Inference API embedding provider.
//!
//! [`HuggingFaceEmbedding`] calls the HF Inference API
//! (`https://api-inference.huggingface.co/models/{model}`). Conﬁgurable API
//! token and model name.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use crate::core::embeddings::Embedding;

pub struct HuggingFaceEmbedding {
    http: Client,
    api_token: String,
    model: String,
    base_url: String,
    dims: usize,
}

impl HuggingFaceEmbedding {
    pub fn new(api_token: &str, model: &str) -> Self {
        Self {
            http: Client::new(),
            api_token: api_token.to_string(),
            model: model.to_string(),
            base_url: "https://api-inference.huggingface.co/models".to_string(),
            dims: 0,
        }
    }

    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.trim_end_matches('/').to_string();
        self
    }

    fn parse_embedding(resp: Value) -> Result<Vec<f32>> {
        // single text → returns plain array: [0.1, 0.2, ...]
        if let Some(arr) = resp.as_array() {
            if arr.iter().all(|v| v.is_number()) {
                return arr.iter()
                    .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| anyhow!("Invalid float")))
                    .collect();
            }
        }
        // batched with one result: [[0.1, 0.2, ...]]
        if let Some(outer) = resp.as_array() {
            if let Some(inner) = outer.first().and_then(|v| v.as_array()) {
                return inner.iter()
                    .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| anyhow!("Invalid float")))
                    .collect();
            }
        }
        Err(anyhow!("Unexpected HuggingFace embedding response format"))
    }

    fn parse_embeddings(resp: Value) -> Result<Vec<Vec<f32>>> {
        let arr = resp.as_array()
            .ok_or_else(|| anyhow!("Expected array from HuggingFace batch embedding"))?;

        arr.iter()
            .map(|entry| {
                let inner = entry.as_array()
                    .ok_or_else(|| anyhow!("Expected array entry in batch"))?;
                inner.iter()
                    .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| anyhow!("Invalid float")))
                    .collect::<Result<Vec<f32>>>()
            })
            .collect()
    }
}

#[async_trait]
impl Embedding for HuggingFaceEmbedding {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/{}", self.base_url, self.model);
        let payload = json!({ "inputs": text });

        let resp = self.http
            .post(&url)
            .bearer_auth(&self.api_token)
            .json(&payload)
            .send()
            .await?
            .json::<Value>()
            .await?;

        Self::parse_embedding(resp)
    }

    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/{}", self.base_url, self.model);
        let payload = json!({ "inputs": texts });

        let resp = self.http
            .post(&url)
            .bearer_auth(&self.api_token)
            .json(&payload)
            .send()
            .await?
            .json::<Value>()
            .await?;

        Self::parse_embeddings(resp)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}
