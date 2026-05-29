# Embeddings

Sistema de embeddings para representação vetorial de texto.

## Embedding Trait

```rust
#[async_trait]
pub trait Embedding: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_many(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}
```

## Implementações

### NoOpEmbedding

Retorna vetores de zeros (útil para testes e desenvolvimento).

```rust
use mangaba::core::embeddings::NoOpEmbedding;

let emb = NoOpEmbedding::new(384); // vetor de 384 zeros
let vec = emb.embed("any text").await?;
assert_eq!(vec.len(), 384);
assert_eq!(vec, vec![0.0; 384]);
```

### OpenAIEmbedding

Usa a API de embeddings da OpenAI.

```rust
use mangaba::core::embeddings::openai::OpenAIEmbedding;

let emb = OpenAIEmbedding::new(LLMConfig {
    api_key: Some("sk-...".into()),
    model: "text-embedding-3-small".into(), // default
    base_url: Some("https://api.openai.com/v1".into()),
    ..Default::default()
})?;

let vec = emb.embed("Hello world").await?;
println!("Dimensões: {}, primeiros valores: {:?}", vec.len(), &vec[..3]);
```

### HuggingFaceEmbedding

Usa a Inference API do HuggingFace.

```rust
use mangaba::core::embeddings::huggingface::HuggingFaceEmbedding;

let emb = HuggingFaceEmbedding::new(
    "hf_your_token",
    "sentence-transformers/all-MiniLM-L6-v2",
)?;

let vec = emb.embed("text to embed").await?;
let batch = emb.embed_many(&["text1", "text2", "text3"]).await?;
```

### CachedEmbedding

Decorator que adiciona cache LRU a qualquer `Embedding`.

```rust
use mangaba::core::embeddings::CachedEmbedding;

let inner = Box::new(OpenAIEmbedding::new(&config)?);
let cached = CachedEmbedding::new(inner, 1000); // cache para 1000 textos

// Primeira chamada: embed real
let v1 = cached.embed("Rust programming").await?;

// Segunda chamada: cache hit (instantâneo)
let v2 = cached.embed("Rust programming").await?;
assert_eq!(v1, v2);
```

O cache usa dois níveis:
- **Cache individual**: LRU para chamadas `embed()` únicas
- **Cache de batch**: LRU para chamadas `embed_many()`, com verificação individual antes

## Cosine Similarity

```rust
use mangaba::core::embeddings::cosine_similarity;

let a = vec![1.0, 0.0, 0.0];
let b = vec![1.0, 0.0, 0.0];
let c = vec![0.0, 1.0, 0.0];

let sim = cosine_similarity(&a, &b); // ≈ 1.0
let sim2 = cosine_similarity(&a, &c); // ≈ 0.0

// Retorna 0.0 para vetores vazios ou de tamanhos diferentes
let empty: Vec<f32> = vec![];
assert_eq!(cosine_similarity(&empty, &vec![1.0]), 0.0);
```

## InMemoryVectorStore

Armazenamento e busca de vetores em memória.

```rust
use mangaba::core::embeddings::InMemoryVectorStore;
use std::sync::Arc;

let emb = Arc::new(OpenAIEmbedding::new(&config)?);
let store = InMemoryVectorStore::new(emb.clone());

// Adicionar documentos
store.add("Rust is fast", None).await?;
store.add("Python is flexible", Some(json!({"type": "interpreted"}))).await?;
store.add_batch(&["Go is concurrent", "C++ is powerful"], None).await?;

// Buscar por query
let results = store.search("fast programming language", 5).await?;
for entry in &results {
    println!("[score] {}", entry.text);
}

// Estatísticas
println!("Total de documentos: {}", store.len().await);
```

### VectorEntry

```rust
pub struct VectorEntry {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub metadata: Option<Value>,
}
```

## Uso com RAG

Os embeddings são o coração do pipeline RAG:

```
Document → Chunker → embed_many() → VectorStore → search(query) → Resultados
```

```rust
use std::sync::Arc;
use mangaba::core::embeddings::CachedEmbedding;

// Com cache para performance
let emb = Arc::new(CachedEmbedding::new(
    Box::new(OpenAIEmbedding::new(&config)?),
    5000,
));

let engine = RAGEngine::new("data.db", emb, RAGConfig::default())?;
```

## Dicas de Performance

- Use `CachedEmbedding` quando o mesmo texto puder ser embedado múltiplas vezes
- Use `embed_many()` em vez de `embed()` em loop para batches grandes
- Ajuste a capacidade do cache conforme o volume de dados únicos
- `NoOpEmbedding` é ideal para testes sem custo de API
