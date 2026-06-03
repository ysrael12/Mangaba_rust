//! Exponential backoff retry with jitter for fallible operations.
//!
//! [`RetryConfig`] conﬁgures max retries, initial/max backoff, and jitter factor.
//! [`with_retry`] executes a closure, retrying on failure with increasing delays.

use std::time::Duration;
use rand::Rng;
use crate::core::events::{EventBus, Event, EventType};
use crate::core::errors::MangabaError;

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

/// Classify an error into `(should_retry, forced_wait_ms)`.
///
/// This is where the [`MangabaError`] taxonomy finally drives behavior:
/// - `RateLimit { retry_after }` → retry, but honor the server-provided delay
///   instead of our own backoff.
/// - any other [`MangabaError`] → retry only if `is_retryable()` (so auth
///   failures, validation errors and malformed-JSON errors fail fast).
/// - non-`MangabaError` (network/timeouts from `reqwest`, etc.) → treated as
///   transient and retried.
fn classify(err: &anyhow::Error) -> (bool, Option<u64>) {
    match err.downcast_ref::<MangabaError>() {
        Some(MangabaError::RateLimit { retry_after, .. }) => (true, Some(retry_after.saturating_mul(1000))),
        Some(other) => (other.is_retryable(), None),
        None => (true, None),
    }
}

pub async fn with_retry<T, F, Fut>(config: &RetryConfig, source_id: &str, operation_name: &str, f: F) -> anyhow::Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<T>>,
{
    let max_attempts = config.max_retries.max(1);

    for attempt in 1..=max_attempts {
        match f().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                let (retryable, forced_wait_ms) = classify(&e);

                // Fail fast on non-retryable errors (auth, validation, parse) —
                // burning the retry budget on them only adds latency.
                if !retryable || attempt >= max_attempts {
                    return Err(e);
                }

                EventBus::emit(Event::new(
                    EventType::LLMRetry,
                    source_id,
                    serde_json::json!({
                        "operation": operation_name,
                        "attempt": attempt,
                        "max_retries": max_attempts,
                        "error": format!("{}", e),
                    }),
                ));

                let backoff = forced_wait_ms.unwrap_or_else(|| compute_backoff(config, attempt as u64));
                tokio::time::sleep(Duration::from_millis(backoff)).await;
            }
        }
    }

    // Unreachable: the loop either returns Ok or returns the last Err above.
    Err(anyhow::anyhow!("Operation '{}' exhausted retries", operation_name))
}

fn compute_backoff(config: &RetryConfig, attempt: u64) -> u64 {
    let exponential = config.initial_backoff_ms * 2u64.saturating_pow(attempt as u32 - 1);
    let capped = exponential.min(config.max_backoff_ms);
    let jitter_range = (capped as f64 * config.jitter_factor) as u64;
    let mut rng = rand::thread_rng();
    let jitter = if jitter_range > 0 { rng.gen_range(0..=jitter_range) } else { 0 };
    capped.saturating_add(jitter).min(config.max_backoff_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn fast_config() -> RetryConfig {
        RetryConfig { max_retries: 5, initial_backoff_ms: 1, max_backoff_ms: 4, jitter_factor: 0.0 }
    }

    #[test]
    fn backoff_is_exponential_and_capped() {
        let cfg = RetryConfig { max_retries: 10, initial_backoff_ms: 100, max_backoff_ms: 500, jitter_factor: 0.0 };
        assert_eq!(compute_backoff(&cfg, 1), 100);
        assert_eq!(compute_backoff(&cfg, 2), 200);
        assert_eq!(compute_backoff(&cfg, 3), 400);
        assert_eq!(compute_backoff(&cfg, 4), 500); // capped
        assert_eq!(compute_backoff(&cfg, 50), 500); // saturating, still capped
    }

    #[test]
    fn classify_distinguishes_fatal_from_transient() {
        let auth: anyhow::Error = MangabaError::Authentication("bad key".into()).into();
        assert_eq!(classify(&auth), (false, None));

        let rate: anyhow::Error = MangabaError::RateLimit { retry_after: 3, detail: "slow down".into() }.into();
        assert_eq!(classify(&rate), (true, Some(3000)));

        let network = anyhow::anyhow!("connection reset");
        assert_eq!(classify(&network), (true, None));
    }

    #[tokio::test]
    async fn retries_transient_then_succeeds() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let cfg = fast_config();
        let result: anyhow::Result<u32> = with_retry(&cfg, "test", "op", || {
            let c = c.clone();
            async move {
                let n = c.fetch_add(1, Ordering::SeqCst);
                if n < 2 { Err(anyhow::anyhow!("transient network error")) } else { Ok(42) }
            }
        }).await;
        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn fatal_error_is_not_retried() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let cfg = fast_config();
        let result: anyhow::Result<u32> = with_retry(&cfg, "test", "op", || {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err(MangabaError::Authentication("invalid api key".into()).into())
            }
        }).await;
        assert!(result.is_err());
        // Auth errors must fail on the first attempt — no wasted retries.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn rate_limit_is_retried() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let cfg = fast_config();
        let result: anyhow::Result<u32> = with_retry(&cfg, "test", "op", || {
            let c = c.clone();
            async move {
                let n = c.fetch_add(1, Ordering::SeqCst);
                if n < 1 {
                    Err(MangabaError::RateLimit { retry_after: 0, detail: "429".into() }.into())
                } else {
                    Ok(7)
                }
            }
        }).await;
        assert_eq!(result.unwrap(), 7);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
