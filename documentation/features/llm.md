# LLM (Large Language Model)

Sistema de clientes LLM provider-agnóstico com suporte a tool calling, streaming,
cache e retry.

## LLMClient Trait

Todas as interações com LLMs passam pelo trait `LLMClient`:

```rust
#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn chat(&self, messages: &[Message], tools: &[&dyn BaseTool]) -> Result<LLMResponse>;
    async fn stream_chat(&self, messages: &[Message]) -> Result<StreamResult>;
}
```

- `chat()` — chamada completa com suporte a tools
- `stream_chat()` — streaming de tokens; default implementation wrappa `chat()`

### LLMResponse

```rust
pub struct LLMResponse {
    pub content: Option<String>,        // texto da resposta
    pub tool_calls: Vec<ToolCall>,      // chamadas de ferramenta
    pub usage: TokenUsage,              // tokens usados
    pub model: String,                  // modelo que respondeu
    pub finish_reason: FinishReason,    // Stop | ToolCalls | Length | Error
    pub raw: Option<Value>,             // resposta crua da API
}
```

## Provedores

### OpenAI (`OpenAIClient`)

URL default: `https://api.openai.com/v1`

```rust
let client = OpenAIClient::new(&LLMConfig {
    model: "gpt-4o-mini".into(),
    api_key: Some("sk-...".into()),
    ..Default::default()
})?;
```

Usa o formato OpenAI para messages, tools (function calling) e streaming SSE.

### DeepSeek (`DeepSeekClient`)

URL default: `https://api.deepseek.com/v1`

Totalmente compatível com OpenAI — usa a mesma macro `openai_chat_impl!`.

### Qwen (`QwenClient`)

URL default: `https://dashscope.aliyuncs.com/compatible-mode/v1`

API compatível com OpenAI.

### OpenRouter (`OpenRouterClient`)

URL default: `https://openrouter.ai/api/v1`

Requer `OpenRouterConfig` (não `LLMConfig`):

```rust
let client = OpenRouterClient::new(&OpenRouterConfig {
    model: vec!["openai/gpt-4o-mini".into()],
    site_name: "MyApp".into(),
    site_url: "https://example.com".into(),
    ..Default::default()
})?;
```

Envia headers `HTTP-Referer` para analytics do OpenRouter.

### Google Gemini (`GoogleClient`)

URL default: `https://generativelanguage.googleapis.com/v1beta`

Usa API `generateContent` com formato próprio de mensagens e tools (function declarations).
Chave enviada como query parameter `?key=`.

### Anthropic Claude (`ClaudeClient`)

URL default: `https://api.anthropic.com/v1`

Usa API `messages` com headers `x-api-key` e `anthropic-version`.
Formato próprio de tools (tool_use/tool_result).

### HuggingFace (`HuggingFaceClient`)

URL default: `https://api-inference.huggingface.co/models`

Constrói prompt textual a partir das mensagens, extrai tool calls via regex
(`<tool_call>{json}</tool_call>`).

## Streaming

`stream_chat()` retorna `StreamResult = Pin<Box<dyn Stream<Item = Result<String>> + Send>>`.

```rust
let mut stream = llm.stream_chat(&messages).await?;
while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(text) => print!("{text}"),
        Err(e) => eprintln!("Stream error: {e}"),
    }
}
```

**OpenAI / OpenRouter**: SSE real via `bytes_stream()` + `parse_sse_chunk()`.

**Outros provedores**: default implementation — resposta completa em um chunk.

## RetryLLMClient

Wrapper que adiciona retry com exponential backoff e cache:

```rust
use mangaba::core::llm::{RetryLLMClient, cache::InMemoryCache};
use mangaba::core::retry::RetryConfig;
use std::sync::Arc;

let retry_client = RetryLLMClient::new(
    Box::new(inner_client),
    RetryConfig {
        max_retries: 3,
        initial_backoff_ms: 1000,
        max_backoff_ms: 30_000,
        jitter_factor: 0.25,
    },
    Some(Arc::new(InMemoryCache::new(Some(Duration::from_secs(300))))),
    Arc::new(UsageTracker::new()),
);
```

### Cache

O `InMemoryCache` armazena respostas serializadas com TTL configurável:

```rust
let cache = InMemoryCache::new(Some(Duration::from_secs(300))); // 5 min TTL
cache.set("key", response_value).await;
let cached = cache.get("key").await;
```

Chaves geradas por `make_cache_key(model, messages)` via hash.

### Token Counter

```rust
use mangaba::core::llm::token_counter::{TokenCounter, UsageTracker};

// Estimativa heurística (~4 chars/token)
let tokens = TokenCounter::count_text("Hello world"); // 3

// Contagem de mensagens completas
let total = TokenCounter::count_messages(&messages);

// Rastreador acumulativo
let tracker = UsageTracker::new();
tracker.record(&response.usage);
let total = tracker.total();
```

## Factory

```rust
pub fn create_llm_client(config: &LLMConfig) -> Result<Box<dyn LLMClient + Send + Sync>>
```

Cria o cliente baseado no campo `config.provider`:
- `"openai"` → `OpenAIClient`
- `"google"` → `GoogleClient`
- `"deepseek"` → `DeepSeekClient`
- `"qwen"` → `QwenClient`
- `"claude"` → `ClaudeClient`
- `"huggingface"` → `HuggingFaceClient`
- `"openrouter"` → use `create_openrouter_client(&OpenRouterConfig)` separadamente

## Format Converters

Cada provedor tem seus próprios conversores de formato:

| Provedor | Messages | Tools | Response |
|----------|----------|-------|----------|
| OpenAI | `messages_to_openai()` | `tools_to_openai()` | `parse_openai_response()` |
| Google | `messages_to_google()` | `tools_to_google()` | `parse_google_response()` |
| Claude | `messages_to_claude()` | `tools_to_claude()` | `parse_claude_response()` |
| HuggingFace | `build_hf_prompt()` | (embutido no prompt) | `parse_hf_response()` |
