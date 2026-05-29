//! Document representation, loading, and chunking.
//!
//! [`Document`] holds text, source, and metadata. [`DocumentLoader`] reads
//! txt, md, csv, and pdf ﬁles. [`DocumentChunker`] splits documents by
//! character count with optional overlap, or by a custom separator.

use anyhow::{Result, anyhow};
use std::path::Path;

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Document {
    pub text: String,
    pub source: String,
    pub metadata: serde_json::Value,
}

impl Document {
    pub fn new(text: &str, source: &str) -> Self {
        Self {
            text: text.to_string(),
            source: source.to_string(),
            metadata: serde_json::json!({}),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata[key] = value;
        self
    }
}

// ---------------------------------------------------------------------------
// Document loader
// ---------------------------------------------------------------------------
pub struct DocumentLoader;

impl DocumentLoader {
    fn extension(path: &str) -> Result<&str> {
        Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow!("Cannot determine file extension: {path}"))
    }

    pub fn load(path: &str) -> Result<Vec<Document>> {
        let ext = Self::extension(path)?.to_lowercase();
        match ext.as_str() {
            "txt" | "md" => Self::load_text(path),
            "csv" => Self::load_csv(path),
            "pdf" => Self::load_pdf(path),
            _ => Err(anyhow!("Unsupported file type: .{ext}")),
        }
    }

    pub fn load_text(path: &str) -> Result<Vec<Document>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read {path}: {e}"))?;
        Ok(vec![Document::new(&content, path)])
    }

    pub fn load_csv(path: &str) -> Result<Vec<Document>> {
        let mut reader = csv::Reader::from_path(path)
            .map_err(|e| anyhow!("Failed to open CSV {path}: {e}"))?;

        let headers: Vec<String> = reader.headers()
            .map_err(|e| anyhow!("Failed to read CSV headers: {e}"))?
            .iter()
            .map(|h| h.to_string())
            .collect();

        let mut docs = Vec::new();
        for (i, result) in reader.records().enumerate() {
            let record = result.map_err(|e| anyhow!("CSV row {i}: {e}"))?;
            let mut pairs: Vec<String> = Vec::new();
            for (h, v) in headers.iter().zip(record.iter()) {
                if !v.is_empty() {
                    pairs.push(format!("{h}: {v}"));
                }
            }
            let text = pairs.join("\n");
            let mut doc = Document::new(&text, path);
            doc.metadata["row"] = serde_json::json!(i);
            doc.metadata["source"] = serde_json::json!(path);
            docs.push(doc);
        }

        if docs.is_empty() {
            return Err(anyhow!("CSV file is empty or has no data rows: {path}"));
        }
        Ok(docs)
    }

    pub fn load_pdf(path: &str) -> Result<Vec<Document>> {
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow!("Failed to read PDF {path}: {e}"))?;
        let text = pdf_extract::extract_text_from_mem(&bytes)
            .map_err(|e| anyhow!("Failed to extract text from PDF {path}: {e}"))?;
        Ok(vec![Document::new(&text, path)])
    }
}

// ---------------------------------------------------------------------------
// Document chunker
// ---------------------------------------------------------------------------
pub struct DocumentChunker {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl Default for DocumentChunker {
    fn default() -> Self {
        Self { chunk_size: 1000, chunk_overlap: 200 }
    }
}

impl DocumentChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self { chunk_size, chunk_overlap }
    }

    pub fn chunk(&self, docs: &[Document]) -> Vec<Document> {
        let mut result = Vec::new();
        for doc in docs {
            let chunks = self.chunk_text(&doc.text);
            for (i, chunk) in chunks.into_iter().enumerate() {
                let mut meta = doc.metadata.clone();
                meta["chunk"] = serde_json::json!(i);
                meta["source"] = serde_json::json!(&doc.source);
                result.push(Document {
                    text: chunk,
                    source: doc.source.clone(),
                    metadata: meta,
                });
            }
        }
        result
    }

    pub fn chunk_text(&self, text: &str) -> Vec<String> {
        if text.len() <= self.chunk_size {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = std::cmp::min(start + self.chunk_size, text.len());
            let chunk = &text[start..end];
            chunks.push(chunk.to_string());

            if end >= text.len() {
                break;
            }

            let overlap_start = if end >= self.chunk_overlap {
                end - self.chunk_overlap
            } else {
                start + 1
            };
            start = overlap_start;
        }

        chunks
    }

    pub fn chunk_text_by_separator(&self, text: &str, separator: &str) -> Vec<String> {
        let segments: Vec<&str> = text.split(separator).collect();
        let mut chunks = Vec::new();
        let mut current = String::new();

        for segment in segments {
            if !current.is_empty() && current.len() + segment.len() + separator.len() > self.chunk_size {
                chunks.push(current.trim().to_string());
                current = String::new();
            }
            if !current.is_empty() {
                current.push_str(separator);
            }
            current.push_str(segment);
        }

        if !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
        }

        chunks
    }
}
