//! # Mangaba AI
//!
//! A modular, provider-agnostic LLM agent framework in Rust.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`core::agent`] | Agent with memory, guardrails, tools, delegation, and ReAct execution |
//! | [`core::callbacks`] | Hook system for step/tool/LLM/task lifecycle events |
//! | [`core::config`] | Environment-based provider detection and [`LLMConfig`](core::types::LLMConfig) builder |
//! | [`core::crew`] | Multi-agent orchestration (sequential / hierarchical) |
//! | [`core::embeddings`] | Embedding trait + OpenAI, HuggingFace, NoOp, Cached LRU, vector store |
//! | [`core::errors`] | [`MangabaError`](core::errors::MangabaError) enum (thiserror, 18 variants, retryable detection) |
//! | [`core::events`] | Global EventBus for subscribe/emit decoupled communication |
//! | [`core::guardrails`] | Length, profanity, composite guardrails + [`GuardrailTool`](core::guardrails::GuardrailTool) |
//! | [`core::llm`] | [`LLMClient`](core::llm::LLMClient) trait + 7 providers + streaming + retry + cache + token counter |
//! | [`core::memory`] | Short-term, long-term (JSON-persisted), and entity memory stores |
//! | [`core::output_parsers`] | [`JSONOutputParser`](core::output_parsers::JSONOutputParser), [`NoOpOutputParser`](core::output_parsers::NoOpOutputParser) |
//! | [`core::pipeline`] | Stage, ParallelStage, ConditionalStage, Pipeline for workﬂow composition |
//! | [`core::planner`] | [`PlanStep`](core::planner::PlanStep), [`ExecutionPlan`](core::planner::ExecutionPlan), [`TaskPlanner`](core::planner::TaskPlanner) (LLM-generated plans) |
//! | [`core::prompt_templates`] | [`PromptTemplate`](core::prompt_templates::PromptTemplate), [`SystemPromptBuilder`](core::prompt_templates::SystemPromptBuilder) |
//! | [`core::protocols`] | A2A (agent-to-agent) + MCP (model context protocol) |
//! | [`core::rag`] | RAG engine: ingest files/text → chunk → embed → query |
//! | [`core::react`] | [`ReActEngine`](core::react::ReActEngine): Thought → Action → Observation loop |
//! | [`core::retry`] | Exponential backoff with jitter for LLM calls |
//! | [`core::task`] | Task with context chaining and output persistence |
//! | [`core::tools`] | [`BaseTool`](core::tools::BaseTool) trait + Calculator, File, Text, Web search tools |
//! | [`core::types`] | Core data types: [`LLMConfig`](core::types::LLMConfig), [`Message`](core::types::Message), [`ToolCall`](core::types::ToolCall), etc. |
pub mod core {
    pub mod agent;
    pub mod config;
    pub mod callbacks;
    pub mod crew;
    pub mod embeddings;
    pub mod errors;
    pub mod events;
    pub mod guardrails;
    pub mod llm;
    pub mod memory;
    pub mod output_parsers;
    pub mod pipeline;
    pub mod planner;
    pub mod prompt_templates;
    pub mod protocols;
    pub mod rag;
    pub mod react;
    pub mod retry;
    pub mod task;
    pub mod tools;
    pub mod types;
}
