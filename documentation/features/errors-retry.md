# Tratamento de Erros & Retry

Sistema de erros e retry com backoff exponencial.

## MangabaError

Enum central de erros com 18 variantes usando `thiserror`.

```rust
#[derive(Debug, Error)]
pub enum MangabaError {
    // Configuração
    #[error("Configuration error: {0}")]
    Configuration(String),

    // LLM
    #[error("LLM error: {0}")]
    LLM(String),
    #[error("Authentication failed: {0}")]
    Authentication(String),
    #[error("Rate limit exceeded — retry after {retry_after}s: {detail}")]
    RateLimit { retry_after: u64, detail: String },
    #[error("Token limit exceeded: {0}")]
    TokenLimit(String),
    #[error("Content blocked by safety filter: {0}")]
    ContentFilter(String),

    // Tools
    #[error("Tool error `{tool}`: {message}")]
    Tool { tool: String, message: String },
    #[error("Tool `{0}` not found")]
    ToolNotFound(String),

    // Agent
    #[error("Agent error: {0}")]
    Agent(String),
    #[error("Max iterations ({0}) reached without final answer")]
    MaxIterations(usize),
    #[error("Delegation to `{0}` failed: {1}")]
    Delegation(String, String),

    // Task / Crew
    #[error("Task error: {0}")]
    Task(String),
    #[error("Crew error: {0}")]
    Crew(String),

    // Memory
    #[error("Memory error: {0}")]
    Memory(String),

    // Embeddings
    #[error("Embedding error: {0}")]
    Embedding(String),
    #[error("Vector store error: {0}")]
    VectorStore(String),

    // General
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("{0}")]
    Custom(String),
}
```

### Retryable Detection

```rust
impl MangabaError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, MangabaError::RateLimit { .. })
    }
}
```

Atualmente apenas `RateLimit` é considerado retryable.

### Conversão para anyhow

```rust
let err = MangabaError::LLM("API timeout".into());
let anyhow_err: anyhow::Error = err.to_anyhow();
```

## Retry System

### RetryConfig

```rust
use mangaba::core::retry::RetryConfig;

let config = RetryConfig {
    max_retries: 3,              // máximo de tentativas
    initial_backoff_ms: 1000,    // primeiro delay: 1s
    max_backoff_ms: 30_000,      // delay máximo: 30s
    jitter_factor: 0.25,         // 25% de jitter aleatório
};
```

### with_retry

Função genérica que executa uma closure com retry:

```rust
use mangaba::core::retry::with_retry;

let result = with_retry(&config, "source_id", "operation_name", || async {
    // Operação que pode falhar
    some_fallible_operation().await
}).await?;
```

### Algoritmo de Backoff

```
backoff = min(initial * 2^(attempt-1), max_backoff)
jitter = random(0, backoff * jitter_factor)
delay = min(backoff + jitter, max_backoff)
```

Exemplo com config default:
- Tentativa 1: ~1000ms (±250ms)
- Tentativa 2: ~2000ms (±500ms)
- Tentativa 3: ~4000ms (±1000ms)

### Eventos de Retry

Cada retry emite um evento `LLMRetry` via `EventBus`:

```rust
EventBus::emit(Event::new(
    EventType::LLMRetry,
    "source_id",
    json!({
        "operation": "llm.chat",
        "attempt": 2,
        "max_retries": 3,
        "error": "Rate limit exceeded",
    }),
));
```

## Integração com LLM

O `RetryLLMClient` wrappa qualquer `LLMClient` com retry + cache:

```rust
use mangaba::core::llm::RetryLLMClient;
use mangaba::core::llm::cache::InMemoryCache;
use mangaba::core::llm::token_counter::UsageTracker;
use std::sync::Arc;
use std::time::Duration;

let inner = create_llm_client(&config)?;

let client = RetryLLMClient::new(
    Box::new(inner),
    RetryConfig::default(),
    Some(Arc::new(InMemoryCache::new(Some(Duration::from_secs(300))))),
    Arc::new(UsageTracker::new()),
);

// Todas as chamadas chat() agora têm retry automático
let response = client.chat(&messages, &tools).await?;
```

## Tratamento de Erros no Agent

```rust
let mut agent = Agent::new(
    AgentConfig {
        max_retry_on_error: 3,
        ..Default::default()
    },
    tools,
    llm,
    None,
);

match agent.execute_task("Do something", None).await {
    Ok(result) => println!("Sucesso: {result}"),
    Err(e) => {
        // Tenta novamente após max_retry_on_error tentativas
        eprintln!("Falha após retries: {e}");
    }
}
```

O agent emite eventos `AgentError` quando todas as tentativas falham.

## Boas Práticas

1. **Use `anyhow` para erros de aplicação**: a crate usa `anyhow::Result`
   em todas as APIs públicas

2. **Converta `MangabaError` para `anyhow`** via `.to_anyhow()` ou
   `anyhow::Error::msg(e.to_string())`

3. **Configure `max_retries` adequadamente**: muitas tentativas podem
   mascarar problemas reais; poucas podem causar falhas em picos de taxa

4. **Monitore eventos de retry**: o `EventBus` permite logar e alertar
   sobre retries frequentes (possível indicação de rate limiting)
