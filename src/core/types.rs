//! Core data types shared across all modules.
//!
//! Includes conﬁguration structs ([`LLMConfig`], [`AgentConfig`], [`TaskConfig`],
//! [`MemoryConfig`], [`OpenRouterConfig`]), messaging ([`Message`], [`Role`]),
//! LLM responses ([`LLMResponse`], [`ToolCall`], [`ToolResult`], [`TokenUsage`]),
//! agent/task state ([`AgentState`], [`AgentStatus`], [`TaskState`], [`TaskStatus`],
//! [`ReActStep`]), and output ([`TaskOutput`]).
//!
//! All types implement `Serialize` + `Deserialize` for JSON interchange.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ProcessType {
    #[default]
    Sequential,
    Hierarchical,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum AgentStatus {
    #[default]
    Idle,
    Running,
    WaitingTool,
    Completed,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum TaskStatus {
    #[default]
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
    Error,
    ContentFilter,
}

// ---------------------------------------------------------------------------
// LLM configuration
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_model")]
    pub model: String,
    pub api_key: Option<String>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    pub base_url: Option<String>,
}

fn default_provider() -> String { "google".into() }
fn default_model() -> String { "gemini-2.5-flash".into() }
fn default_temperature() -> f32 { 0.7 }
fn default_max_tokens() -> usize { 1024 }
fn default_top_p() -> f32 { 1.0 }
fn default_timeout() -> u64 { 60 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterConfig {
    #[serde(default = "default_openrouter_provider")]
    pub provider: String,
    pub model: Vec<String>,
    pub site_name: String,
    pub site_url: String,
    pub route: Option<String>,
    #[serde(flatten)]
    pub base: LLMConfig,
}

fn default_openrouter_provider() -> String { "openrouter".into() }

// ---------------------------------------------------------------------------
// Token usage & tool structs
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default = "default_toolcall_id")]
    pub id: String,
    pub tool_name: String,
    #[serde(default)]
    pub arguments: HashMap<String, serde_json::Value>,
}

fn default_toolcall_id() -> String { format!("call_{}", Uuid::new_v4().simple()) }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    #[serde(default)]
    pub tool_name: String,
    #[serde(default)]
    pub output: Option<serde_json::Value>,
    pub error: Option<String>,
    #[serde(default = "default_success")]
    pub success: bool,
}

fn default_success() -> bool { true }

// ---------------------------------------------------------------------------
// Messaging
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_results: Option<Vec<ToolResult>>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(default = "now_iso")]
    pub timestamp: String,
}

fn now_iso() -> String { Utc::now().to_rfc3339() }

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: Some(content.into()), ..Default::default() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: Some(content.into()), ..Default::default() }
    }
    pub fn assistant(content: Option<String>, tool_calls: Option<Vec<ToolCall>>) -> Self {
        Self { role: Role::Assistant, content, tool_calls, ..Default::default() }
    }
    pub fn tool(results: Vec<ToolResult>) -> Self {
        Self { role: Role::Tool, tool_results: Some(results), ..Default::default() }
    }
}

impl Default for Message {
    fn default() -> Self {
        Self {
            role: Role::User,
            content: None,
            tool_calls: None,
            tool_results: None,
            metadata: HashMap::new(),
            timestamp: now_iso(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub usage: TokenUsage,
    pub model: String,
    pub finish_reason: FinishReason,
    #[serde(skip)]
    pub raw: Option<serde_json::Value>,
}

impl LLMResponse {
    pub fn text(&self) -> &str { self.content.as_deref().unwrap_or("") }
    pub fn has_tool_calls(&self) -> bool { !self.tool_calls.is_empty() }
}

// ---------------------------------------------------------------------------
// Config structs for Agent, Memory, Task
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryConfig {
    #[serde(default = "default_true")]
    pub short_term: bool,
    #[serde(default)]
    pub long_term: bool,
    #[serde(default)]
    pub entity: bool,
    #[serde(default = "default_max_items")]
    pub max_short_term_items: usize,
    pub storage_path: Option<String>,
}

fn default_true() -> bool { true }
fn default_max_items() -> usize { 50 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub role: String,
    pub goal: String,
    pub backstory: String,
    #[serde(default)]
    pub llm_config: Option<LLMConfig>,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub memory_config: MemoryConfig,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default = "default_max_retry")]
    pub max_retry_on_error: usize,
    #[serde(default)]
    pub verbose: bool,
    #[serde(default)]
    pub allow_delegation: bool,
    pub step_callback: Option<String>,
    #[serde(default)]
    pub guardrails: Vec<String>,
    pub output_parser: Option<String>,
}

fn default_max_iterations() -> usize { 15 }
fn default_max_retry() -> usize { 3 }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentState {
    pub agent_id: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub steps: Vec<ReActStep>,
    #[serde(default)]
    pub current_step: usize,
    #[serde(default)]
    pub iteration_count: usize,
    #[serde(default = "default_status")]
    pub status: AgentStatus,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

fn default_status() -> AgentStatus { AgentStatus::Idle }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReActStep {
    pub step_number: usize,
    pub thought: Option<String>,
    pub action: Option<ToolCall>,
    pub observation: Option<String>,
    #[serde(default = "now_iso")]
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    pub description: String,
    pub expected_output: String,
    pub agent_id: Option<String>,
    #[serde(default)]
    pub context_ids: Vec<String>,
    #[serde(default)]
    pub tools: Vec<String>,
    pub output_file: Option<String>,
    #[serde(default)]
    pub async_execution: bool,
    #[serde(default)]
    pub human_input: bool,
    #[serde(default)]
    pub guardrails: Vec<String>,
    pub output_parser: Option<String>,
    #[serde(default = "default_retry")]
    pub retry_on_failure: usize,
}

fn default_retry() -> usize { 0 }

// ---------------------------------------------------------------------------
// Task Output
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOutput {
    pub description: String,
    pub result: String,
    pub agent_id: String,
    #[serde(default = "default_success")]
    pub success: bool,
    pub timestamp: String,
}

impl TaskOutput {
    pub fn new(description: &str, result: &str, agent_id: &str, success: bool) -> Self {
        Self {
            description: description.to_string(),
            result: result.to_string(),
            agent_id: agent_id.to_string(),
            success,
            timestamp: now_iso(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TaskState {
    pub task_id: String,
    #[serde(default)]
    pub status: TaskStatus,
    pub output: Option<TaskOutput>,
    #[serde(default)]
    pub attempts: usize,
    pub agent_id: Option<String>,
}
