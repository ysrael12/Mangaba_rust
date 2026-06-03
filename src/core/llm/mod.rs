//! LLM client trait with 7 providers, streaming, retry, cache, and token counting.
//!
//! The [`LLMClient`] trait provides `chat()` (with tool support) and `stream_chat()`
//! (with a default implementation that wraps `chat()`).
//!
//! Providers:
//! - [`OpenAIClient`] / [`DeepSeekClient`] / [`QwenClient`] — OpenAI-compatible APIs
//! - [`OpenRouterClient`] — multi-model router
//! - [`GoogleClient`] — Gemini via `generativelanguage.googleapis.com`
//! - [`ClaudeClient`] — Anthropic Claude
//! - [`HuggingFaceClient`] — HF Inference API
//!
//! Additional wrappers: [`RetryLLMClient`] (retry), `cache::InMemoryCache` / `cache::LLMCache` (caching),
//! `token_counter::UsageTracker` (token counting).

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::{Client, RequestBuilder, StatusCode};
use serde_json::{json, Value};
use std::pin::Pin;
use std::sync::Arc;

use crate::core::types::{
    LLMConfig, OpenRouterConfig, LLMResponse, FinishReason,
    Message, Role, ToolCall, TokenUsage,
};
use crate::core::errors::MangabaError;
use crate::core::tools::BaseTool;
use crate::core::events::{EventBus, Event, EventType};
use crate::core::retry::{RetryConfig, with_retry};
use crate::core::llm::cache::{LLMCache, make_cache_key};
use crate::core::llm::token_counter::UsageTracker;

pub mod cache;
pub mod token_counter;

// ---------------------------------------------------------------------------
// StreamResult
// ---------------------------------------------------------------------------
pub type StreamResult = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

// ---------------------------------------------------------------------------
// LLMClient trait
// ---------------------------------------------------------------------------
#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn chat(&self, messages: &[Message], tools: &[&dyn BaseTool]) -> Result<LLMResponse>;

    async fn stream_chat(&self, messages: &[Message]) -> Result<StreamResult> {
        let resp = self.chat(messages, &[]).await?;
        let content = resp.content.unwrap_or_default();
        let stream = futures::stream::once(async move { Ok(content) });
        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// DummyLLM
// ---------------------------------------------------------------------------
pub struct DummyLLM;

#[async_trait]
impl LLMClient for DummyLLM {
    async fn chat(&self, _messages: &[Message], _tools: &[&dyn BaseTool]) -> Result<LLMResponse> {
        Ok(LLMResponse {
            content: Some(String::new()),
            tool_calls: vec![],
            usage: TokenUsage::default(),
            model: String::new(),
            finish_reason: FinishReason::Stop,
            raw: None,
        })
    }
}

// ===========================================================================
// HTTP helper — centralizes status classification + error-payload detection
// ===========================================================================
/// Send a request and parse the JSON body, mapping transport/HTTP failures onto
/// the [`MangabaError`] taxonomy so that [`with_retry`] can decide whether a
/// retry is worthwhile:
/// - `429` → [`MangabaError::RateLimit`] (honors `Retry-After`) — retryable,
/// - `401`/`403` → [`MangabaError::Authentication`] — NOT retryable,
/// - any other 4xx/5xx → [`MangabaError::LLM`],
/// - a `200` body that still carries an `"error"` object → [`MangabaError::LLM`].
async fn send_and_parse(req: RequestBuilder) -> Result<Value> {
    let resp = req.send().await?;
    let status = resp.status();

    if status == StatusCode::TOO_MANY_REQUESTS {
        let retry_after = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2);
        let detail = resp.text().await.unwrap_or_default();
        return Err(MangabaError::RateLimit { retry_after, detail }.into());
    }
    if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        let detail = resp.text().await.unwrap_or_default();
        return Err(MangabaError::Authentication(detail).into());
    }
    if status.is_server_error() {
        // 5xx (500/502/503/504, ...) are transient server-side failures. We
        // deliberately return an *untyped* error so `with_retry::classify`
        // treats it as retryable — a non-retryable MangabaError would make the
        // agent give up on a momentary "high demand" blip.
        let detail = resp.text().await.unwrap_or_default();
        return Err(anyhow!("HTTP {status} (transient): {detail}"));
    }
    if !status.is_success() {
        // Other 4xx are client errors (bad request, etc.) — retrying the same
        // payload won't help, so surface a fatal, typed error.
        let detail = resp.text().await.unwrap_or_default();
        return Err(MangabaError::LLM(format!("HTTP {status}: {detail}")).into());
    }

    let body: Value = resp.json().await?;
    // Most providers signal logical errors via an `"error"` object even on 200.
    if let Some(err) = body.get("error").filter(|e| !e.is_null()) {
        return Err(MangabaError::LLM(err.to_string()).into());
    }
    Ok(body)
}

// ===========================================================================
// OpenAI-format macro
// ===========================================================================
macro_rules! openai_chat_impl {
    ($struct:ident, $base_url:expr, $api_key_field:ident) => {
        #[async_trait]
        impl LLMClient for $struct {
            async fn chat(&self, messages: &[Message], tools: &[&dyn BaseTool]) -> Result<LLMResponse> {
                let url = format!("{}/chat/completions", self.$base_url);
                let mut payload = json!({
                    "model": self.model,
                    "messages": messages_to_openai(messages),
                });
                if !tools.is_empty() {
                    payload["tools"] = tools_to_openai(tools);
                }
                let resp = send_and_parse(
                    self.http.post(&url).bearer_auth(&self.api_key).json(&payload)
                ).await?;
                parse_openai_response(resp, &self.model)
            }

            async fn stream_chat(&self, messages: &[Message]) -> Result<StreamResult> {
                let url = format!("{}/chat/completions", self.$base_url);
                let payload = json!({
                    "model": self.model,
                    "messages": messages_to_openai(messages),
                    "stream": true,
                });
                let byte_stream = self.http
                    .post(&url)
                    .bearer_auth(&self.api_key)
                    .json(&payload)
                    .send()
                    .await?
                    .bytes_stream();
                Ok(Box::pin(openai_sse_stream(byte_stream)))
            }
        }
    };
}

/// Wrap a raw byte stream into a token stream, **buffering across chunks** so
/// that an SSE event split over multiple TCP packets is never dropped.
/// Events are delimited by a blank line (`\n\n`) per the SSE spec.
fn openai_sse_stream<S, B>(byte_stream: S) -> impl Stream<Item = Result<String>> + Send
where
    S: Stream<Item = reqwest::Result<B>> + Send + 'static,
    B: AsRef<[u8]>,
{
    byte_stream.scan(String::new(), |buf, chunk| {
        let result = match chunk {
            Ok(bytes) => {
                buf.push_str(&String::from_utf8_lossy(bytes.as_ref()));
                let mut emitted = String::new();
                while let Some(idx) = buf.find("\n\n") {
                    let block: String = buf.drain(..idx + 2).collect();
                    if let Some(tok) = parse_sse_event(&block) {
                        emitted.push_str(&tok);
                    }
                }
                Ok(emitted)
            }
            Err(e) => Err(anyhow!("Stream error: {e}")),
        };
        futures::future::ready(Some(result))
    })
}

/// Parse one *complete* SSE event block, concatenating all `delta.content`
/// fragments it contains. Returns `None` if the block yields no text.
fn parse_sse_event(block: &str) -> Option<String> {
    let mut out = String::new();
    for line in block.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(val) = serde_json::from_str::<Value>(data) {
                if let Some(delta) = val["choices"][0]["delta"]["content"].as_str() {
                    out.push_str(delta);
                }
            }
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

// ---------------------------------------------------------------------------
// OpenAI
// ---------------------------------------------------------------------------
pub struct OpenAIClient {
    http: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenAIClient {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        Ok(Self {
            http: Client::new(),
            api_key: config.api_key.clone().ok_or_else(|| anyhow!("OpenAI API key required"))?,
            model: config.model.clone(),
            base_url: config.base_url.clone().unwrap_or_else(|| "https://api.openai.com/v1".into()),
        })
    }
}

openai_chat_impl!(OpenAIClient, base_url, api_key);

// ---------------------------------------------------------------------------
// DeepSeek
// ---------------------------------------------------------------------------
pub struct DeepSeekClient {
    http: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl DeepSeekClient {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        Ok(Self {
            http: Client::new(),
            api_key: config.api_key.clone().ok_or_else(|| anyhow!("DeepSeek API key required"))?,
            model: config.model.clone(),
            base_url: config.base_url.clone().unwrap_or_else(|| "https://api.deepseek.com/v1".into()),
        })
    }
}

openai_chat_impl!(DeepSeekClient, base_url, api_key);

// ---------------------------------------------------------------------------
// Qwen
// ---------------------------------------------------------------------------
pub struct QwenClient {
    http: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl QwenClient {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        Ok(Self {
            http: Client::new(),
            api_key: config.api_key.clone().ok_or_else(|| anyhow!("Qwen API key required"))?,
            model: config.model.clone(),
            base_url: config.base_url.clone().unwrap_or_else(|| "https://dashscope.aliyuncs.com/compatible-mode/v1".into()),
        })
    }
}

openai_chat_impl!(QwenClient, base_url, api_key);

// ---------------------------------------------------------------------------
// OpenRouter
// ---------------------------------------------------------------------------
pub struct OpenRouterClient {
    http: Client,
    api_key: String,
    model: String,
    base_url: String,
    site_name: String,
    site_url: String,
    route: Option<String>,
}

impl OpenRouterClient {
    pub fn new(cfg: &OpenRouterConfig) -> Result<Self> {
        Ok(Self {
            http: Client::new(),
            api_key: cfg.base.api_key.clone().ok_or_else(|| anyhow!("OpenRouter API key required"))?,
            model: cfg.model.first().cloned().unwrap_or_else(|| "openai/gpt-4o-mini".into()),
            base_url: cfg.base.base_url.clone().unwrap_or_else(|| "https://openrouter.ai/api/v1".into()),
            site_name: cfg.site_name.clone(),
            site_url: cfg.site_url.clone(),
            route: cfg.route.clone(),
        })
    }
}

#[async_trait]
impl LLMClient for OpenRouterClient {
    async fn chat(&self, messages: &[Message], tools: &[&dyn BaseTool]) -> Result<LLMResponse> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut payload = json!({
            "model": self.model,
            "messages": messages_to_openai(messages),
        });
        if !tools.is_empty() {
            payload["tools"] = tools_to_openai(tools);
        }
        let req = self.http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&payload)
            .header("HTTP-Referer", &self.site_url);
        let resp = send_and_parse(req).await?;
        parse_openai_response(resp, &self.model)
    }

    async fn stream_chat(&self, messages: &[Message]) -> Result<StreamResult> {
        let url = format!("{}/chat/completions", self.base_url);
        let payload = json!({
            "model": self.model,
            "messages": messages_to_openai(messages),
            "stream": true,
        });
        let byte_stream = self.http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&payload)
            .header("HTTP-Referer", &self.site_url)
            .send()
            .await?
            .bytes_stream();
        Ok(Box::pin(openai_sse_stream(byte_stream)))
    }
}

// ---------------------------------------------------------------------------
// Google Gemini
// ---------------------------------------------------------------------------
pub struct GoogleClient {
    http: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl GoogleClient {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        Ok(Self {
            http: Client::new(),
            api_key: config.api_key.clone().ok_or_else(|| anyhow!("Google API key required"))?,
            model: config.model.clone(),
            base_url: config.base_url.clone().unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta".into()),
        })
    }
}

#[async_trait]
impl LLMClient for GoogleClient {
    async fn chat(&self, messages: &[Message], tools: &[&dyn BaseTool]) -> Result<LLMResponse> {
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, self.model, self.api_key
        );
        let mut payload = json!({ "contents": messages_to_google(messages) });
        if !tools.is_empty() {
            payload["tools"] = tools_to_google(tools);
        }
        let resp = send_and_parse(self.http.post(&url).json(&payload)).await?;
        parse_google_response(resp, &self.model)
    }
}

// ---------------------------------------------------------------------------
// Claude (Anthropic)
// ---------------------------------------------------------------------------
pub struct ClaudeClient {
    http: Client,
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: usize,
}

impl ClaudeClient {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        Ok(Self {
            http: Client::new(),
            api_key: config.api_key.clone().ok_or_else(|| anyhow!("Claude API key required"))?,
            model: config.model.clone(),
            base_url: config.base_url.clone().unwrap_or_else(|| "https://api.anthropic.com/v1".into()),
            max_tokens: config.max_tokens,
        })
    }
}

#[async_trait]
impl LLMClient for ClaudeClient {
    async fn chat(&self, messages: &[Message], tools: &[&dyn BaseTool]) -> Result<LLMResponse> {
        let url = format!("{}/messages", self.base_url);
        let mut payload = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages_to_claude(messages),
        });
        if !tools.is_empty() {
            payload["tools"] = tools_to_claude(tools);
        }
        let resp = send_and_parse(
            self.http
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&payload)
        ).await?;
        parse_claude_response(resp, &self.model)
    }
}

// ---------------------------------------------------------------------------
// HuggingFace
// ---------------------------------------------------------------------------
pub struct HuggingFaceClient {
    http: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl HuggingFaceClient {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        Ok(Self {
            http: Client::new(),
            api_key: config.api_key.clone().ok_or_else(|| anyhow!("HuggingFace API key required"))?,
            model: config.model.clone(),
            base_url: config.base_url.clone().unwrap_or_else(|| "https://api-inference.huggingface.co/models".into()),
        })
    }
}

#[async_trait]
impl LLMClient for HuggingFaceClient {
    async fn chat(&self, messages: &[Message], tools: &[&dyn BaseTool]) -> Result<LLMResponse> {
        let url = format!("{}/{}", self.base_url, self.model);
        let prompt = build_hf_prompt(messages, tools);
        let payload = json!({ "inputs": prompt, "parameters": json!({ "return_full_text": false }) });
        let resp = send_and_parse(
            self.http.post(&url).bearer_auth(&self.api_key).json(&payload)
        ).await?;
        parse_hf_response(resp, &self.model)
    }
}

// ===========================================================================
// RetryLLMClient
// ===========================================================================
pub struct RetryLLMClient {
    inner: Box<dyn LLMClient + Send + Sync>,
    retry_config: RetryConfig,
    cache: Option<Arc<dyn LLMCache + Send + Sync>>,
    usage: Arc<UsageTracker>,
}

impl RetryLLMClient {
    pub fn new(
        inner: Box<dyn LLMClient + Send + Sync>,
        retry_config: RetryConfig,
        cache: Option<Arc<dyn LLMCache + Send + Sync>>,
        usage: Arc<UsageTracker>,
    ) -> Self {
        Self { inner, retry_config, cache, usage }
    }
}

#[async_trait]
impl LLMClient for RetryLLMClient {
    async fn chat(&self, messages: &[Message], tools: &[&dyn BaseTool]) -> Result<LLMResponse> {
        let source_id = "retry_llm";

        if let Some(ref cache) = self.cache {
            let key = make_cache_key("retry", messages);
            if let Some(cached) = cache.get(&key).await {
                return serde_json::from_value(cached)
                    .map_err(|e| anyhow!("Cache deserialize error: {e}"));
            }
        }

        EventBus::emit(Event::new(EventType::LLMStart, source_id, json!({
            "message_count": messages.len(),
            "tool_count": tools.len(),
        })));

        let result = with_retry(&self.retry_config, source_id, "llm.chat", || async {
            self.inner.chat(messages, tools).await
        }).await;

        match result {
            Ok(response) => {
                self.usage.record(&response.usage);
                EventBus::emit(Event::new(EventType::LLMEnd, source_id, json!({
                    "finish_reason": format!("{:?}", response.finish_reason),
                })));
                if let Some(ref cache) = self.cache {
                    if let Ok(val) = serde_json::to_value(&response) {
                        let key = make_cache_key("retry", messages);
                        cache.set(&key, val).await;
                    }
                }
                Ok(response)
            }
            Err(e) => {
                EventBus::emit(Event::new(EventType::LLMError, source_id, json!({
                    "error": e.to_string(),
                })));
                Err(e)
            }
        }
    }
}

// ===========================================================================
// Factory
// ===========================================================================
pub fn create_llm_client(config: &LLMConfig) -> Result<Box<dyn LLMClient + Send + Sync>> {
    match config.provider.as_str() {
        "openai" => Ok(Box::new(OpenAIClient::new(config)?)),
        "google" => Ok(Box::new(GoogleClient::new(config)?)),
        "openrouter" => Err(anyhow!("OpenRouter requires OpenRouterConfig, use create_openrouter_client")),
        "qwen" => Ok(Box::new(QwenClient::new(config)?)),
        "deepseek" => Ok(Box::new(DeepSeekClient::new(config)?)),
        "claude" => Ok(Box::new(ClaudeClient::new(config)?)),
        "huggingface" => Ok(Box::new(HuggingFaceClient::new(config)?)),
        _ => Err(anyhow!("Unsupported LLM provider: {}", config.provider)),
    }
}

pub fn create_openrouter_client(cfg: &OpenRouterConfig) -> Result<Box<dyn LLMClient + Send + Sync>> {
    Ok(Box::new(OpenRouterClient::new(cfg)?))
}

// ===========================================================================
// Format converters
// ===========================================================================

fn messages_to_openai(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System => out.push(json!({"role": "system", "content": msg.content})),
            Role::User => out.push(json!({"role": "user", "content": msg.content})),
            Role::Assistant => {
                let mut m = json!({"role": "assistant"});
                if let Some(ref c) = msg.content { m["content"] = json!(c); }
                if let Some(ref calls) = msg.tool_calls {
                    m["tool_calls"] = json!(calls.iter().map(|tc| {
                        json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default(),
                            }
                        })
                    }).collect::<Vec<_>>());
                }
                out.push(m);
            }
            Role::Tool => {
                for tr in msg.tool_results.as_ref().unwrap_or(&vec![]) {
                    out.push(json!({
                        "role": "tool",
                        "tool_call_id": tr.call_id,
                        "content": tr.output,
                    }));
                }
            }
        }
    }
    out
}

fn parse_openai_response(resp: Value, model: &str) -> Result<LLMResponse> {
    // A well-formed response must carry at least one choice; otherwise the body
    // is malformed and we surface a typed error instead of an empty success.
    if resp["choices"].as_array().map_or(true, |c| c.is_empty()) {
        return Err(MangabaError::LLM(format!("malformed OpenAI response: {resp}")).into());
    }
    let choice = &resp["choices"][0];
    let message = &choice["message"];
    let content = message["content"].as_str().map(|s| s.to_string());
    let finish_reason = match choice["finish_reason"].as_str() {
        Some("stop") => FinishReason::Stop,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        _ => FinishReason::Stop,
    };
    let tool_calls = message["tool_calls"].as_array().map(|arr| {
        arr.iter().map(|tc| ToolCall {
            id: tc["id"].as_str().unwrap_or("").to_string(),
            tool_name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
            arguments: serde_json::from_str(tc["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or_default(),
        }).collect()
    }).unwrap_or_default();
    let usage = TokenUsage {
        prompt_tokens: resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize,
        completion_tokens: resp["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize,
        total_tokens: resp["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize,
    };
    Ok(LLMResponse { content, tool_calls, usage, model: model.to_string(), finish_reason, raw: Some(resp) })
}

fn tools_to_openai(tools: &[&dyn BaseTool]) -> Value {
    json!(tools.iter().map(|t| {
        json!({
            "type": "function",
            "function": t.get_function_schema(),
        })
    }).collect::<Vec<_>>())
}

// -- Google -----------------------------------------------------------------
fn messages_to_google(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for msg in messages {
        let role = match msg.role {
            Role::Assistant => "model",
            Role::Tool => "function",
            _ => "user",
        };
        let mut part = json!({"text": msg.content.as_deref().unwrap_or("")});
        if let Some(ref results) = msg.tool_results {
            if let Some(r) = results.first() {
                part = json!({"functionResponse": {
                    "name": r.tool_name,
                    "response": {"name": r.tool_name, "content": r.output},
                }});
            }
        }
        if let Some(ref calls) = msg.tool_calls {
            if let Some(tc) = calls.first() {
                out.push(json!({
                    "role": "model",
                    "parts": [{"functionCall": {"name": tc.tool_name, "args": tc.arguments}}],
                }));
                continue;
            }
        }
        out.push(json!({"role": role, "parts": [part]}));
    }
    out
}

fn parse_google_response(resp: Value, model: &str) -> Result<LLMResponse> {
    if resp["candidates"].as_array().map_or(true, |c| c.is_empty()) {
        return Err(MangabaError::LLM(format!("malformed Gemini response: {resp}")).into());
    }
    let candidate = &resp["candidates"][0];
    let parts = &candidate["content"]["parts"];
    let mut content = None;
    let mut tool_calls = Vec::new();

    for part in parts.as_array().unwrap_or(&vec![]) {
        if let Some(text) = part["text"].as_str() {
            content = Some(text.to_string());
        }
        if let Some(fc) = part["functionCall"].as_object() {
            tool_calls.push(ToolCall {
                id: "google_fc".to_string(),
                tool_name: fc["name"].as_str().unwrap_or("").to_string(),
                arguments: fc.get("args").cloned().and_then(|v| {
                    serde_json::from_value(v).ok()
                }).unwrap_or_default(),
            });
        }
    }

    let finish_reason = match candidate["finishReason"].as_str() {
        Some("STOP") => FinishReason::Stop,
        Some("TOOL_CALLS") | Some("FUNCTION_CALL") => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    };

    Ok(LLMResponse {
        content,
        tool_calls,
        usage: TokenUsage::default(),
        model: model.to_string(),
        finish_reason,
        raw: Some(resp),
    })
}

fn tools_to_google(tools: &[&dyn BaseTool]) -> Value {
    json!([{
        "functionDeclarations": tools.iter().map(|t| {
            let schema = t.get_function_schema();
            json!({
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            })
        }).collect::<Vec<_>>(),
    }])
}

// -- Claude ----------------------------------------------------------------
fn messages_to_claude(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for msg in messages {
        let role = match msg.role {
            Role::Assistant => "assistant",
            _ => "user",
        };
        let mut content_arr = Vec::new();
        if let Some(ref c) = msg.content {
            content_arr.push(json!({"type": "text", "text": c}));
        }
        if let Some(ref calls) = msg.tool_calls {
            for tc in calls {
                content_arr.push(json!({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.tool_name,
                    "input": tc.arguments,
                }));
            }
        }
        if let Some(ref results) = msg.tool_results {
            for tr in results {
                content_arr.push(json!({
                    "type": "tool_result",
                    "tool_use_id": tr.call_id,
                    "content": tr.output,
                }));
            }
        }
        out.push(json!({"role": role, "content": content_arr}));
    }
    out
}

fn parse_claude_response(resp: Value, model: &str) -> Result<LLMResponse> {
    if resp["content"].as_array().map_or(true, |c| c.is_empty()) {
        return Err(MangabaError::LLM(format!("malformed Claude response: {resp}")).into());
    }
    let mut content = None;
    let mut tool_calls = Vec::new();
    for block in resp["content"].as_array().unwrap_or(&vec![]) {
        match block["type"].as_str() {
            Some("text") => content = Some(block["text"].as_str().unwrap_or("").to_string()),
            Some("tool_use") => {
                tool_calls.push(ToolCall {
                    id: block["id"].as_str().unwrap_or("").to_string(),
                    tool_name: block["name"].as_str().unwrap_or("").to_string(),
                    arguments: block["input"].clone().as_object().map(|m| {
                        m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                    }).unwrap_or_default(),
                });
            }
            _ => {}
        }
    }
    let finish_reason = match resp["stop_reason"].as_str() {
        Some("end_turn") | Some("stop") => FinishReason::Stop,
        Some("tool_use") => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    };
    let usage = TokenUsage {
        prompt_tokens: resp["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize,
        completion_tokens: resp["usage"]["output_tokens"].as_u64().unwrap_or(0) as usize,
        total_tokens: 0,
    };
    Ok(LLMResponse { content, tool_calls, usage, model: model.to_string(), finish_reason, raw: Some(resp) })
}

fn tools_to_claude(tools: &[&dyn BaseTool]) -> Value {
    json!(tools.iter().map(|t| {
        let schema = t.get_function_schema();
        json!({
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["parameters"],
        })
    }).collect::<Vec<_>>())
}

// -- HuggingFace -----------------------------------------------------------
fn build_hf_prompt(messages: &[Message], tools: &[&dyn BaseTool]) -> String {
    let tool_block = if !tools.is_empty() {
        let descs: Vec<String> = tools.iter().map(|t| {
            let s = t.get_function_schema();
            format!(
                "  {name}: {desc}  Args: {args}",
                name = s["name"].as_str().unwrap_or("?"),
                desc = s["description"].as_str().unwrap_or(""),
                args = serde_json::to_string(&s["parameters"]).unwrap_or_default(),
            )
        }).collect();
        format!(
            "\n\nAvailable tools:\n{}\n\nWhen you want to call a tool, respond with ONLY valid JSON in this format:\n<tool_call>{{{{\"name\": \"tool_name\", \"arguments\": {{...}}}}}}</tool_call>",
            descs.join("\n")
        )
    } else {
        String::new()
    };

    let mut parts: Vec<String> = messages.iter().map(|msg| {
        let role = match msg.role {
            Role::System => "System",
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::Tool => "Tool",
        };
        let content = msg.content.as_deref().unwrap_or("");
        if let Some(ref results) = msg.tool_results {
            if let Some(r) = results.first() {
                return format!("Tool ({r}): {output}", r = r.tool_name, output = r.output.as_ref().map(|v| v.to_string()).unwrap_or_default());
            }
        }
        format!("{role}: {content}")
    }).collect();

    if let Some(first) = parts.first_mut() {
        first.push_str(&tool_block);
    }

    parts.push("Assistant:".into());
    parts.join("\n")
}

fn parse_hf_response(resp: Value, model: &str) -> Result<LLMResponse> {
    let text = resp[0]["generated_text"].as_str()
        .or_else(|| resp["generated_text"].as_str())
        .unwrap_or("");

    let tool_re = regex::Regex::new(r#"<tool_call>\s*(\{.*?\})\s*</tool_call>"#).ok();
    let mut tool_calls = Vec::new();

    if let Some(ref re) = tool_re {
        for cap in re.captures_iter(text) {
            if let Ok(val) = serde_json::from_str::<Value>(&cap[1]) {
                tool_calls.push(ToolCall {
                    id: "hf_tc".to_string(),
                    tool_name: val["name"].as_str().unwrap_or("").to_string(),
                    arguments: val.get("arguments").and_then(|v| {
                        serde_json::from_value(v.clone()).ok()
                    }).unwrap_or_default(),
                });
            }
        }
    }

    let finish_reason = if !tool_calls.is_empty() { FinishReason::ToolCalls } else { FinishReason::Stop };

    Ok(LLMResponse {
        content: Some(text.to_string()),
        tool_calls,
        usage: TokenUsage::default(),
        model: model.to_string(),
        finish_reason,
        raw: Some(resp),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_event_concatenates_deltas() {
        let block = "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n";
        assert_eq!(parse_sse_event(block).as_deref(), Some("Hel"));
        assert_eq!(parse_sse_event("data: [DONE]\n\n"), None);
        assert_eq!(parse_sse_event(": comment only\n\n"), None);
    }

    #[tokio::test]
    async fn sse_stream_buffers_events_split_across_chunks() {
        // A single SSE event is deliberately split across three byte chunks,
        // mimicking TCP fragmentation. The buffer must reassemble it.
        let chunks: Vec<reqwest::Result<Vec<u8>>> = vec![
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"".to_vec()),
            Ok(b"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"con".to_vec()),
            Ok(b"tent\":\" world\"}}]}\n\ndata: [DONE]\n\n".to_vec()),
        ];
        let src = futures::stream::iter(chunks);
        let collected: String = openai_sse_stream(src)
            .map(|r| r.unwrap())
            .collect::<Vec<_>>()
            .await
            .concat();
        assert_eq!(collected, "Hello world");
    }

    #[test]
    fn parse_openai_response_rejects_empty_choices() {
        let body = json!({ "choices": [] });
        assert!(parse_openai_response(body, "m").is_err());
    }

    #[test]
    fn parse_openai_response_accepts_valid_body() {
        let body = json!({
            "choices": [{ "message": { "content": "hi" }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        });
        let resp = parse_openai_response(body, "m").unwrap();
        assert_eq!(resp.text(), "hi");
        assert_eq!(resp.usage.total_tokens, 3);
    }
}
