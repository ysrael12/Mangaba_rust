//! OpenAI-compatible embedding provider.
//!
//! [`OpenAIEmbedding`] calls the embeddings API endpoint. Supports conﬁgurable
//! model (default `text-embedding-3-small`), base URL override, and API key.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use crate::core::embeddings::Embedding;

pub struct OpenAIEmbedding {
    http: Client,
    api_key: String,
    model: String,
    dims: usize,
}

impl OpenAIEmbedding {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            http: Client::new(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            dims: 0,
        }
    }

    fn parse_embedding(resp: Value) -> Result<Vec<f32>> {
        let data = resp["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| anyhow!("No embedding data in response"))?;
        data.iter()
            .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| anyhow!("Invalid embedding value")))
            .collect()
    }

    fn parse_embeddings(resp: Value) -> Result<Vec<Vec<f32>>> {
        let data = resp["data"]
            .as_array()
            .ok_or_else(|| anyhow!("No data in response"))?;

        let mut sorted: Vec<(usize, Vec<f32>)> = data.iter()
            .filter_map(|entry| {
                let idx = entry["index"].as_u64()? as usize;
                let emb: Vec<f32> = entry["embedding"]
                    .as_array()?
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                Some((idx, emb))
            })
            .collect();

        sorted.sort_by_key(|(idx, _)| *idx);
        Ok(sorted.into_iter().map(|(_, emb)| emb).collect())
    }
}

#[async_trait]
impl Embedding for OpenAIEmbedding {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let payload = json!({
            "model": self.model,
            "input": text,
        });
        let resp = self.http
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await?
            .json::<Value>()
            .await?;
        Self::parse_embedding(resp)
    }

    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let payload = json!({
            "model": self.model,
            "input": texts,
        });
        let resp = self.http
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
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
