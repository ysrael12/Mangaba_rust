//! Token counting and usage tracking for LLM calls.
//!
//! [`TokenCounter`] provides static estimation methods. [`UsageTracker`] wraps
//! an `Arc<Mutex<>>` around accumulated [`TokenUsage`]
//! and exposes `record()` / `total()` / `reset()`.

use std::sync::{Arc, Mutex};
use crate::core::types::TokenUsage;

pub struct TokenCounter;

impl TokenCounter {
    /// Heuristic: ~4 characters per token for general text
    pub fn count_text(text: &str) -> usize {
        let len = text.len();
        if len == 0 { return 0; }
        (len + 3) / 4
    }

    pub fn count_messages(messages: &[crate::core::types::Message]) -> usize {
        messages.iter().map(|m| {
            let content_tokens = m.content.as_deref().map(Self::count_text).unwrap_or(0);
            let tool_tokens = m.tool_calls.as_ref()
                .map(|calls| calls.len() * 10)
                .unwrap_or(0);
            let result_tokens = m.tool_results.as_ref()
                .map(|r| r.iter().map(|tr| {
                    tr.output.as_ref().map(|o| Self::count_text(&o.to_string())).unwrap_or(0)
                        + tr.error.as_ref().map(|e| Self::count_text(e)).unwrap_or(0)
                }).sum::<usize>())
                .unwrap_or(0);
            content_tokens + tool_tokens + result_tokens + 4 // overhead per message
        }).sum()
    }
}

pub struct UsageTracker {
    usage: Arc<Mutex<TokenUsage>>,
}

impl UsageTracker {
    pub fn new() -> Self {
        Self { usage: Arc::new(Mutex::new(TokenUsage::default())) }
    }

    pub fn record(&self, usage: &TokenUsage) {
        let mut u = self.usage.lock().unwrap();
        u.prompt_tokens += usage.prompt_tokens;
        u.completion_tokens += usage.completion_tokens;
        u.total_tokens += usage.total_tokens;
    }

    pub fn total(&self) -> TokenUsage {
        self.usage.lock().unwrap().clone()
    }

    pub fn reset(&self) {
        let mut u = self.usage.lock().unwrap();
        *u = TokenUsage::default();
    }
}
