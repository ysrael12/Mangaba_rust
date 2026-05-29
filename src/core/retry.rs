//! Exponential backoff retry with jitter for fallible operations.
//!
//! [`RetryConfig`] conﬁgures max retries, initial/max backoff, and jitter factor.
//! [`with_retry`] executes a closure, retrying on failure with increasing delays.

use std::time::Duration;
use rand::Rng;
use crate::core::events::{EventBus, Event, EventType};

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 1000,
            max_backoff_ms: 30_000,
            jitter_factor: 0.25,
        }
    }
}

pub async fn with_retry<T, F, Fut>(config: &RetryConfig, source_id: &str, operation_name: &str, f: F) -> anyhow::Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 1..=config.max_retries {
        match f().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                last_err = Some(e);
                if attempt == config.max_retries {
                    break;
                }
                EventBus::emit(Event::new(
                    EventType::LLMRetry,
                    source_id,
                    serde_json::json!({
                        "operation": operation_name,
                        "attempt": attempt,
                        "max_retries": config.max_retries,
                        "error": format!("{}", last_err.as_ref().unwrap()),
                    }),
                ));
                let backoff = compute_backoff(config, attempt as u64);
                tokio::time::sleep(Duration::from_millis(backoff)).await;
            }
        }
    }

    Err(anyhow::anyhow!("Operation '{}' failed after {} retries: {:?}",
        operation_name, config.max_retries, last_err))
}

fn compute_backoff(config: &RetryConfig, attempt: u64) -> u64 {
    let exponential = config.initial_backoff_ms * 2u64.saturating_pow(attempt as u32 - 1);
    let capped = exponential.min(config.max_backoff_ms);
    let jitter_range = (capped as f64 * config.jitter_factor) as u64;
    let mut rng = rand::thread_rng();
    let jitter = if jitter_range > 0 { rng.gen_range(0..=jitter_range) } else { 0 };
    capped.saturating_add(jitter).min(config.max_backoff_ms)
}
