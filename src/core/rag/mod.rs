//! Retrieval-Augmented Generation engine.
//!
//! [`RAGEngine`] ingests documents (txt, md, csv, pdf) or raw text → chunks via
//! [`DocumentChunker`] → embeds via [`Embedding`]
//! → stores in [`LocalChroma`] (SQLite-backed vector store).
//! Query returns [`RAGResult`]s with text, source, metadata, and cosine distance.
//!
//! Also provides a [`ChromaDB`](chroma::ChromaDB) HTTP client for remote Chroma servers.

use anyhow::Result;
use std::sync::Arc;
use serde_json::Value;
use uuid::Uuid;

pub mod chroma;
pub mod document;
pub mod local;

use document::{Document, DocumentChunker, DocumentLoader};
use local::LocalChroma;
use crate::core::embeddings::Embedding;

// ---------------------------------------------------------------------------
// RAG config
// ---------------------------------------------------------------------------
pub struct RAGConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub default_k: usize,
}

impl Default for RAGConfig {
    fn default() -> Self {
        Self { chunk_size: 1000, chunk_overlap: 200, default_k: 5 }
    }
}

// ---------------------------------------------------------------------------
// RAG result
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct RAGResult {
    pub text: String,
    pub source: String,
    pub metadata: Value,
    pub distance: Option<f64>,
}

// ---------------------------------------------------------------------------
// RAGEngine
// ---------------------------------------------------------------------------
pub struct RAGEngine {
    pub store: LocalChroma,
    pub embedding: Arc<dyn Embedding>,
    pub config: RAGConfig,
    chunker: DocumentChunker,
}

impl RAGEngine {
    pub fn new(path: &str, embedding: Arc<dyn Embedding>, config: RAGConfig) -> Result<Self> {
        let store = LocalChroma::new(path)?;
        let chunker = DocumentChunker::new(config.chunk_size, config.chunk_overlap);
        Ok(Self { store, embedding, config, chunker })
    }

    pub fn in_memory(embedding: Arc<dyn Embedding>, config: RAGConfig) -> Result<Self> {
        let store = LocalChroma::in_memory()?;
        let chunker = DocumentChunker::new(config.chunk_size, config.chunk_overlap);
        Ok(Self { store, embedding, config, chunker })
    }

    pub fn collection_id(&self, name: &str) -> Result<String> {
        let col = self.store.get_collection(name)?;
        Ok(col.id)
    }

    pub fn create_collection(&self, name: &str) -> Result<String> {
        let col = self.store.create_collection(name, None)?;
        Ok(col.id)
    }

    // -- ingest file (txt, md, csv, pdf) ------------------------------------
    pub async fn ingest_file(&self, collection_id: &str, file_path: &str) -> Result<()> {
        let docs = DocumentLoader::load(file_path)?;
        let chunks = self.chunker.chunk(&docs);
        self.ingest_chunks(collection_id, &chunks).await
    }

    // -- ingest raw text ----------------------------------------------------
    pub async fn ingest_text(&self, collection_id: &str, text: &str, metadata: Option<Value>) -> Result<()> {
        let doc = Document::new(text, "raw_text");
        let doc = match metadata {
            Some(m) => Document { metadata: m, ..doc },
            None => doc,
        };
        let chunks = self.chunker.chunk(&[doc]);
        self.ingest_chunks(collection_id, &chunks).await
    }

    async fn ingest_chunks(&self, collection_id: &str, chunks: &[Document]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let texts: Vec<&str> = chunks.iter().map(|d| d.text.as_str()).collect();
        let embeddings = self.embedding.embed_many(&texts).await?;

        let ids: Vec<String> = (0..chunks.len())
            .map(|_| format!("rag_{}", Uuid::new_v4().simple()))
            .collect();
        let ids_ref: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();

        let metadatas: Vec<Value> = chunks.iter().map(|d| d.metadata.clone()).collect();

        let docs_ref: Vec<&str> = chunks.iter().map(|d| d.text.as_str()).collect();

        self.store.add(
            collection_id,
            &ids_ref,
            Some(&embeddings),
            Some(&metadatas),
            Some(&docs_ref),
        )
    }

    // -- query --------------------------------------------------------------
    pub async fn query(&self, collection_id: &str, query: &str, k: Option<usize>) -> Result<Vec<RAGResult>> {
        let n = k.unwrap_or(self.config.default_k);
        let query_emb = self.embedding.embed(query).await?;
        let result = self.store.query(collection_id, &[query_emb], n, None)?;

        let mut rag_results = Vec::new();

        for i in 0..result.ids.len() {
            let text = result.documents.get(i).cloned().flatten().unwrap_or_default();
            let distance = result.distances.get(i).copied();
            let metadata = result.metadatas.get(i).cloned().flatten().unwrap_or(Value::Object(Default::default()));
            let source = metadata["source"].as_str().unwrap_or("unknown").to_string();

            rag_results.push(RAGResult {
                text,
                source,
                metadata,
                distance,
            });
        }

        Ok(rag_results)
    }

    pub fn delete_collection(&self, name: &str) -> Result<()> {
        self.store.delete_collection(name)
    }
}
