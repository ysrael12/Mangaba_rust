//! Central error enum with 18 variants and retryable detection.
//!
//! [`MangabaError`] covers conﬁguration, LLM, authentication, rate limiting,
//! tool errors, agent/crew/task failures, memory, embedding, and validation errors.
//! Use `is_retryable()` to check if the error warrants a retry (currently only
//! `RateLimit`). Convert to `anyhow::Error` via `.to_anyhow()`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MangabaError {
    // -- Configuration --
    #[error("Configuration error: {0}")]
    Configuration(String),

    // -- LLM --
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

    // -- Tools --
    #[error("Tool error `{tool}`: {message}")]
    Tool { tool: String, message: String },

    #[error("Tool `{0}` not found")]
    ToolNotFound(String),

    // -- Agent --
    #[error("Agent error: {0}")]
    Agent(String),

    #[error("Max iterations ({0}) reached without final answer")]
    MaxIterations(usize),

    #[error("Delegation to `{0}` failed: {1}")]
    Delegation(String, String),

    // -- Task / Crew --
    #[error("Task error: {0}")]
    Task(String),

    #[error("Crew error: {0}")]
    Crew(String),

    // -- Memory --
    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Vector store error: {0}")]
    VectorStore(String),

    // -- General --
    #[error("Validation error: {0}")]
    Validation(String),

    #[error("{0}")]
    Custom(String),
}

impl MangabaError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, MangabaError::RateLimit { .. })
    }
}

// Conversion helper — use .into() explicitly when needed
// anyhow already provides a blanket From<E: StdError>, so we cannot impl From<MangabaError> directly.
impl MangabaError {
    pub fn to_anyhow(self) -> anyhow::Error {
        anyhow::Error::msg(self.to_string())
    }
}
