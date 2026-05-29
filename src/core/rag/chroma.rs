//! HTTP client for remote ChromaDB vector database.
//!
//! [`ChromaDB`] connects to a running Chroma server via its REST API.
//! Supports collection CRUD, add/query/delete embeddings, and heartbeat.

use anyhow::{Result, anyhow};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// ChromaDB HTTP client
// ---------------------------------------------------------------------------
pub struct ChromaDB {
    http: Client,
    base_url: String,
    tenant: String,
    database: String,
}

#[derive(Debug, Deserialize)]
pub struct ChromaCollection {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ChromaQueryResult {
    pub ids: Vec<Vec<String>>,
    pub distances: Option<Vec<Vec<f64>>>,
    pub documents: Option<Vec<Vec<String>>>,
    pub metadatas: Option<Vec<Vec<Option<Value>>>>,
}

impl ChromaDB {
    pub fn new(base_url: &str) -> Self {
        Self {
            http: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            tenant: "default_tenant".to_string(),
            database: "default_database".to_string(),
        }
    }

    pub fn with_tenant(mut self, tenant: &str) -> Self {
        self.tenant = tenant.to_string();
        self
    }

    pub fn with_database(mut self, database: &str) -> Self {
        self.database = database.to_string();
        self
    }

    pub async fn heartbeat(&self) -> Result<i64> {
        let resp = self.http
            .get(format!("{}/api/v1/heartbeat", self.base_url))
            .send()
            .await?
            .json::<Value>()
            .await?;
        resp.as_object()
            .and_then(|m| m.values().next())
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("Invalid heartbeat response"))
    }

    pub async fn list_collections(&self) -> Result<Vec<ChromaCollection>> {
        let resp = self.http
            .get(format!("{}/api/v1/collections", self.base_url))
            .query(&[("tenant", &self.tenant), ("database", &self.database)])
            .send()
            .await?
            .json::<Value>()
            .await?;
        serde_json::from_value(resp).map_err(|e| anyhow!("Failed to parse collections: {e}"))
    }

    pub async fn get_collection(&self, name: &str) -> Result<ChromaCollection> {
        let resp = self.http
            .get(format!("{}/api/v1/collections/{name}", self.base_url))
            .query(&[("tenant", &self.tenant), ("database", &self.database)])
            .send()
            .await?
            .json::<Value>()
            .await?;
        serde_json::from_value(resp).map_err(|e| anyhow!("Failed to parse collection: {e}"))
    }

    pub async fn create_collection(&self, name: &str, metadata: Option<Value>) -> Result<ChromaCollection> {
        let mut payload = json!({
            "name": name,
            "tenant": self.tenant,
            "database": self.database,
        });
        if let Some(m) = metadata {
            payload["metadata"] = m;
        }
        let resp = self.http
            .post(format!("{}/api/v1/collections", self.base_url))
            .json(&payload)
            .send()
            .await?
            .json::<Value>()
            .await?;
        serde_json::from_value(resp).map_err(|e| anyhow!("Failed to create collection: {e}"))
    }

    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        self.http
            .delete(format!("{}/api/v1/collections/{name}", self.base_url))
            .query(&[("tenant", &self.tenant)])
            .send()
            .await?;
        Ok(())
    }

    pub async fn count(&self, collection_id: &str) -> Result<usize> {
        let resp = self.http
            .post(format!("{}/api/v1/collections/{collection_id}/count", self.base_url))
            .send()
            .await?
            .json::<Value>()
            .await?;
        resp.as_u64()
            .map(|n| n as usize)
            .ok_or_else(|| anyhow!("Invalid count response"))
    }

    pub async fn add(
        &self,
        collection_id: &str,
        ids: &[&str],
        embeddings: Option<&[Vec<f32>]>,
        metadatas: Option<&[Value]>,
        documents: Option<&[&str]>,
    ) -> Result<()> {
        let mut payload = json!({
            "ids": ids,
        });
        if let Some(emb) = embeddings {
            payload["embeddings"] = serde_json::to_value(emb)?;
        }
        if let Some(meta) = metadatas {
            payload["metadatas"] = serde_json::to_value(meta)?;
        }
        if let Some(docs) = documents {
            payload["documents"] = docs.into();
        }
        self.http
            .post(format!("{}/api/v1/collections/{collection_id}/add", self.base_url))
            .json(&payload)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn query(
        &self,
        collection_id: &str,
        query_embeddings: &[Vec<f32>],
        n_results: usize,
        r#where: Option<Value>,
    ) -> Result<ChromaQueryResult> {
        let mut payload = json!({
            "query_embeddings": query_embeddings,
            "n_results": n_results,
        });
        if let Some(w) = r#where {
            payload["where"] = w;
        }
        let resp = self.http
            .post(format!("{}/api/v1/collections/{collection_id}/query", self.base_url))
            .json(&payload)
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;
        serde_json::from_value(resp).map_err(|e| anyhow!("Failed to parse query result: {e}"))
    }

    pub async fn delete_where(&self, collection_id: &str, r#where: Value) -> Result<()> {
        self.http
            .post(format!("{}/api/v1/collections/{collection_id}/delete", self.base_url))
            .json(&json!({ "where": r#where }))
            .send()
            .await?;
        Ok(())
    }
}
