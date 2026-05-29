//! [`BaseTool`] trait and built-in tool implementations.
//!
//! The [`BaseTool`] trait requires `name()`, `description()`, and `run_impl()`.
//! Optional: `args_schema()`, `return_direct()`. The [`call()`](BaseTool::call) method wraps
//! `run_impl` with event emission (ToolStart / ToolEnd / ToolError).
//!
//! Built-in tools:
//! - [`CalculatorTool`] — evaluates arithmetic expressions
//! - [`FileReaderTool`] / [`FileWriterTool`] / [`DirectoryListTool`] — ﬁle system ops
//! - [`TextSplitterTool`] / [`WordCounterTool`] — text processing
//! - [`SerperSearchTool`] / [`DuckDuckGoSearchTool`] — web search
//! - [`EchoTool`] — echoes input (useful for testing)
//!
//! The [`BaseToolkit`] trait groups multiple tools into one object.

pub mod calculator;
pub mod file_tools;
pub mod text_tools;
pub mod web_search;

pub use calculator::CalculatorTool;
pub use file_tools::{FileReaderTool, FileWriterTool, DirectoryListTool};
pub use text_tools::{TextSplitterTool, WordCounterTool};
pub use web_search::{DuckDuckGoSearchTool, SerperSearchTool};

use async_trait::async_trait;
use anyhow::Result;
use serde_json::{json, Value};
use crate::core::types::ToolResult;
use crate::core::events::{EventBus, Event, EventType};

#[async_trait]
pub trait BaseTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn args_schema(&self) -> Option<Value> { None }
    fn return_direct(&self) -> bool { false }

    async fn run_impl(&self, args: Value) -> Result<ToolResult>;

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let tool_name = self.name().to_string();
        EventBus::emit(Event::new(
            EventType::ToolStart, &tool_name,
            json!({"tool": tool_name, "args": args}),
        ));
        match self.run_impl(args).await {
            Ok(result) => {
                let preview = result.output.clone().unwrap_or(json!(null));
                EventBus::emit(Event::new(
                    EventType::ToolEnd, &tool_name,
                    json!({"tool": tool_name, "result_preview": preview}),
                ));
                Ok(result)
            }
            Err(e) => {
                EventBus::emit(Event::new(
                    EventType::ToolError, &tool_name,
                    json!({"tool": tool_name, "error": format!("{:?}", e)}),
                ));
                Err(e)
            }
        }
    }

    fn get_function_schema(&self) -> Value {
        json!({
            "name": self.name(),
            "description": self.description(),
            "parameters": self.args_schema().unwrap_or(json!({
                "type": "object",
                "properties": {}
            })),
        })
    }
}

pub trait BaseToolkit: Send + Sync {
    fn get_tools(&self) -> Vec<Box<dyn BaseTool + Send + Sync>>;
}

pub struct EchoTool;

#[async_trait]
impl BaseTool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn description(&self) -> &str { "Echoes back the input arguments" }
    async fn run_impl(&self, args: Value) -> Result<ToolResult> {
        Ok(ToolResult {
            call_id: "echo".to_string(),
            tool_name: "echo".to_string(),
            output: Some(args),
            error: None,
            success: true,
        })
    }
}
