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

Dentre as variantes de `MangabaError`, apenas `RateLimit` é retryable — erros de
autenticação, validação e parsing de JSON são **fatais** e não devem ser
retentados. Essa classificação é efetivamente consumida pelo `with_retry` (ver
"Classificação de Erros" abaixo), garantindo que o agente falhe rápido em erros
não-recuperáveis em vez de gastar todo o orçamento de retry com latência inútil.

### Mapeamento de erros HTTP → MangabaError

As chamadas de rede dos provedores LLM passam pelo helper interno
`send_and_parse`, que traduz o status HTTP para a taxonomia tipada **na origem**,
para que o `with_retry` decida corretamente:

| Condição                        | Variante                    | Retryable? |
|---------------------------------|-----------------------------|------------|
| `429 Too Many Requests`         | `RateLimit { retry_after }` | ✅ (respeita `Retry-After`) |
| `401` / `403`                   | `Authentication`            | ❌ |
| Outros `4xx` / `5xx`            | `LLM`                       | ❌ |
| `200` com payload `{"error":…}` | `LLM`                       | ❌ |
| Falha de transporte (`reqwest`) | erro `anyhow` não tipado    | ✅ (tratado como transiente) |

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

### Classificação de Erros

Antes de cada retry, `with_retry` classifica o erro retornado e decide o que
fazer — em vez de retentar tudo cegamente:

```rust
fn classify(err: &anyhow::Error) -> (bool /* retryable */, Option<u64> /* wait_ms */) {
    match err.downcast_ref::<MangabaError>() {
        // Rate limit: retenta, mas honra o atraso pedido pelo servidor.
        Some(MangabaError::RateLimit { retry_after, .. }) => (true, Some(retry_after * 1000)),
        // Demais MangabaError: só retenta se is_retryable() (auth/parse falham rápido).
        Some(other) => (other.is_retryable(), None),
        // Erros não tipados (rede/timeout do reqwest): tratados como transientes.
        None => (true, None),
    }
}
```

Consequências práticas:

- **Erro fatal** (auth, validação, parse) → retorna na **1ª tentativa**, sem
  desperdiçar o orçamento de retry.
- **`RateLimit`** → respeita o `Retry-After` do provedor em vez do backoff local,
  evitando agravar o rate limiting.
- **Erro transiente** (rede) → retentado com backoff exponencial + jitter.

### Algoritmo de Backoff

Para erros transientes (sem `retry_after` forçado):

```
backoff = min(initial * 2^(attempt-1), max_backoff)
jitter = random(0, backoff * jitter_factor)
delay = min(backoff + jitter, max_backoff)
```

Exemplo com config default:
- Tentativa 1: ~1000ms (±250ms)
- Tentativa 2: ~2000ms (±500ms)
- Tentativa 3: ~4000ms (±1000ms)

> Para `RateLimit`, o atraso é exatamente `retry_after * 1000ms` (o backoff
> exponencial é ignorado).

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
        // O erro preserva a CAUSA real encadeada via `.context(...)`.
        eprintln!("Falha: {e}");
        for cause in e.chain().skip(1) {
            eprintln!("  causado por: {cause}");
        }
    }
}
```

`execute_task` reexecuta o ReAct até `max_retry_on_error` vezes. Ao esgotar as
tentativas, **propaga o erro subjacente** (rede, rate limit, parse, etc.)
anexando o contexto `"Agent '<role>' failed after N attempt(s)"`, em vez de
mascará-lo com uma mensagem genérica. O agent também emite `AgentError` quando
todas as tentativas falham.

## Boas Práticas

1. **Use `anyhow` para erros de aplicação**: a crate usa `anyhow::Result` em
   todas as APIs públicas. Quando precisar reagir a um erro específico, recupere
   a variante tipada com `err.downcast_ref::<MangabaError>()` — é exatamente o
   que o `with_retry` faz.

2. **Converta `MangabaError` para `anyhow`** via `.into()` (a blanket `From` do
   `anyhow` aceita qualquer `std::error::Error`) ou `.to_anyhow()`. Preferir
   `.into()` preserva a variante para downcast posterior.

3. **Deixe os erros fatais falharem rápido**: não aumente `max_retries` para
   "forçar" sucesso — erros de auth/parse são não-retryable por design e
   retentá-los só adiciona latência.

4. **Configure `max_retries` adequadamente**: muitas tentativas podem mascarar
   problemas reais; poucas podem causar falhas em picos de taxa.

5. **Monitore eventos de retry**: o `EventBus` permite logar e alertar sobre
   retries frequentes (possível indicação de rate limiting).
