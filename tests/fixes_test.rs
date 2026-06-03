//! Integration tests for the resilience / concurrency fixes.
//!
//! Covers, end-to-end through public APIs:
//! - `RetryLLMClient` recovering from transient errors and *not* burning the
//!   retry budget on fatal (auth) errors — wiring up the `MangabaError`
//!   taxonomy with the retry engine.
//! - `EventBus` reentrancy/poison safety under the public `subscribe`/`emit` API.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use mangaba::core::llm::{LLMClient, RetryLLMClient};
use mangaba::core::llm::token_counter::UsageTracker;
use mangaba::core::tools::BaseTool;
use mangaba::core::types::*;
use mangaba::core::retry::RetryConfig;
use mangaba::core::errors::MangabaError;
use mangaba::core::events::{EventBus, Event, EventType};

// ---------------------------------------------------------------------------
// A programmable LLM that fails a configurable number of times before
// succeeding, optionally with a fatal (non-retryable) error.
// ---------------------------------------------------------------------------
struct FlakyLLM {
    calls: Arc<AtomicUsize>,
    fail_times: usize,
    fatal: bool,
}

impl FlakyLLM {
    fn new(fail_times: usize, fatal: bool) -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (Self { calls: calls.clone(), fail_times, fatal }, calls)
    }
}

#[async_trait]
impl LLMClient for FlakyLLM {
    async fn chat(&self, _messages: &[Message], _tools: &[&dyn BaseTool]) -> Result<LLMResponse> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n < self.fail_times {
            if self.fatal {
                return Err(MangabaError::Authentication("invalid api key".into()).into());
            }
            return Err(anyhow::anyhow!("transient network error"));
        }
        Ok(LLMResponse {
            content: Some("recovered".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
            model: "flaky".into(),
            finish_reason: FinishReason::Stop,
            raw: None,
        })
    }
}

fn fast_retry() -> RetryConfig {
    RetryConfig { max_retries: 5, initial_backoff_ms: 1, max_backoff_ms: 4, jitter_factor: 0.0 }
}

#[tokio::test]
async fn retry_llm_recovers_from_transient_failures() {
    let (llm, calls) = FlakyLLM::new(2, false);
    let usage = Arc::new(UsageTracker::new());
    let client = RetryLLMClient::new(Box::new(llm), fast_retry(), None, usage.clone());

    let resp = client.chat(&[Message::user("hi")], &[]).await.unwrap();
    assert_eq!(resp.text(), "recovered");
    // 2 failures + 1 success.
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    // Usage is recorded only on the successful call.
    assert_eq!(usage.total().total_tokens, 2);
}

#[tokio::test]
async fn retry_llm_does_not_retry_fatal_errors() {
    let (llm, calls) = FlakyLLM::new(5, true); // would "fail" forever, but fatal
    let usage = Arc::new(UsageTracker::new());
    let client = RetryLLMClient::new(Box::new(llm), fast_retry(), None, usage);

    let result = client.chat(&[Message::user("hi")], &[]).await;
    assert!(result.is_err());
    // Authentication is non-retryable → exactly one attempt.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn event_bus_reentrant_emit_is_safe() {
    EventBus::clear();
    let outer = Arc::new(AtomicUsize::new(0));
    let inner = Arc::new(AtomicUsize::new(0));
    let o = outer.clone();
    let i = inner.clone();

    // Listener A re-emits a new event from inside emit() — must not deadlock.
    EventBus::subscribe(Box::new(move |ev: &Event| {
        if ev.source_id == "outer" {
            o.fetch_add(1, Ordering::SeqCst);
            EventBus::emit(Event::new(EventType::Custom("inner".into()), "inner", serde_json::json!({})));
        }
    }));
    let i2 = i.clone();
    EventBus::subscribe(Box::new(move |ev: &Event| {
        if ev.source_id == "inner" {
            i2.fetch_add(1, Ordering::SeqCst);
        }
    }));

    EventBus::emit(Event::new(EventType::Custom("outer".into()), "outer", serde_json::json!({})));

    assert_eq!(outer.load(Ordering::SeqCst), 1);
    assert_eq!(inner.load(Ordering::SeqCst), 1);
    EventBus::clear();
}
