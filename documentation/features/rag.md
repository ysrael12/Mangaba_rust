# RAG (Retrieval-Augmented Generation)

Motor de busca e geração aumentada por recuperação de contexto.

## Visão Geral

O `RAGEngine` implementa o pipeline completo:

```
Arquivo/Texto → Chunker → Embeddings → Vector Store → Query → Resultados
```

## Componentes

### Document

```rust
pub struct Document {
    pub text: String,
    pub source: String,
    pub metadata: Value,
}
```

### DocumentLoader

Carrega documentos de arquivos:

```rust
use mangaba::core::rag::document::{DocumentLoader, Document};

// TXT / Markdown
let docs = DocumentLoader::load("file.txt")?;
let docs = DocumentLoader::load("README.md")?;

// CSV (cada linha vira um documento)
let docs = DocumentLoader::load("data.csv")?;

// PDF (extrai texto via pdf-extract)
let docs = DocumentLoader::load("report.pdf")?;

println!("{} documentos carregados", docs.len());
```

Formatos suportados: `txt`, `md`, `csv`, `pdf`.

### DocumentChunker

Divide documentos em chunks menores para embedding:

```rust
use mangaba::core::rag::document::DocumentChunker;

let chunker = DocumentChunker::new(500, 50); // size=500, overlap=50

// Chunk por caractere
let chunks: Vec<String> = chunker.chunk_text(long_text);

// Chunk por separator
let chunks: Vec<String> = chunker.chunk_text_by_separator(text, "\n\n");

// Chunk documentos completos
let doc_chunks: Vec<Document> = chunker.chunk(&loaded_docs);
for chunk in &doc_chunks {
    println!("Chunk {} (source: {})", chunk.metadata["chunk"], chunk.source);
}
```

### LocalChroma

Vector store local via SQLite (sem servidor):

```rust
use mangaba::core::rag::local::LocalChroma;

// Persistente em arquivo
let store = LocalChroma::new("chroma_data.db")?;

// Em memória
let store = LocalChroma::in_memory()?;

let collection = store.create_collection("my_docs", None)?;
store.add(&collection.id, &["id1"], Some(&[vec![0.1, 0.2, 0.3]]), None, Some(&["text content"]))?;

let results = store.query(&collection.id, &[vec![0.1, 0.2, 0.3]], 5, None)?;
```

### ChromaDB (Cliente HTTP)

Para servidores ChromaDB remotos:

```rust
use mangaba::core::rag::chroma::ChromaDB;

let client = ChromaDB::new("http://localhost:8000")?;
let heartbeat = client.heartbeat().await?;
let collections = client.list_collections().await?;
```

## RAGEngine

Motor completo que combina todos os componentes:

```rust
use mangaba::core::rag::RAGEngine;
use std::sync::Arc;

let engine = RAGEngine::new(
    "rag_data.db",                          // path do SQLite
    Arc::new(OpenAIEmbedding::new(&config)), // embedding provider
    RAGConfig {
        chunk_size: 1000,
        chunk_overlap: 200,
        default_k: 5,
    },
)?;

// Ou em memória (para testes):
let engine = RAGEngine::in_memory(embedding, RAGConfig::default())?;
```

### Criação de Coleção

```rust
let collection_id = engine.create_collection("my_knowledge_base")?;
```

### Ingestão

```rust
// De arquivo
engine.ingest_file(&collection_id, "documentos/report.pdf").await?;
engine.ingest_file(&collection_id, "documentos/notes.txt").await?;

// De texto cru
engine.ingest_text(&collection_id, "Conteúdo relevante aqui...", None).await?;

// Com metadados
engine.ingest_text(&collection_id, "Nota importante", Some(json!({
    "category": "science",
    "date": "2024-01-01",
}))).await?;
```

### Query

```rust
let results = engine.query(&collection_id, "quantum computing", Some(5)).await?;

for (i, r) in results.iter().enumerate() {
    println!("{}. [dist={:.4}] {}", i + 1, r.distance.unwrap_or(0.0), r.text.chars().take(100).collect::<String>());
    println!("   Fonte: {}", r.source);
}
```

### RAGResult

```rust
pub struct RAGResult {
    pub text: String,
    pub source: String,
    pub metadata: Value,
    pub distance: Option<f64>,
}
```

### Gerenciamento

```rust
let id = engine.collection_id("my_knowledge_base")?;
engine.delete_collection("old_collection")?;
```

## RAGConfig

```rust
pub struct RAGConfig {
    pub chunk_size: usize,      // tamanho do chunk (default: 1000)
    pub chunk_overlap: usize,   // overlap entre chunks (default: 200)
    pub default_k: usize,       // número de resultados por query (default: 5)
}
```

## Exemplo Completo

```rust
use mangaba::core::rag::RAGEngine;
use mangaba::core::rag::document::DocumentLoader;
use mangaba::core::embeddings::NoOpEmbedding;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let engine = RAGEngine::in_memory(
        Arc::new(NoOpEmbedding::new(384)),
        RAGConfig::default(),
    )?;

    let col = engine.create_collection("docs")?;

    // Ingest
    engine.ingest_text(&col, "Rust is a systems programming language", None).await?;
    engine.ingest_text(&col, "Python is a high-level interpreted language", None).await?;

    // Query
    let results = engine.query(&col, "programming languages", Some(5)).await?;
    for r in &results {
        println!("→ {}", r.text);
    }

    Ok(())
}
```
