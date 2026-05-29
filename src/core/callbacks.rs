//! Hook system for step/tool/LLM/task lifecycle events.
//!
//! [`Callbacks`] aggregates closures (`on_step`, `on_tool_start`, `on_llm_end`, etc.)
//! that are called at key points during agent and ReAct execution.

use crate::core::types::{ReActStep, LLMResponse, ToolResult};

// ---------------------------------------------------------------------------
// Callback type aliases — sync closures
// ---------------------------------------------------------------------------
pub type StepCallback = Box<dyn Fn(&ReActStep, &str) + Send + Sync + 'static>;
pub type ToolStartCallback = Box<dyn Fn(&str, &serde_json::Value) + Send + Sync + 'static>;
pub type ToolEndCallback = Box<dyn Fn(&str, &ToolResult) + Send + Sync + 'static>;
pub type LLMStartCallback = Box<dyn Fn(usize, usize) + Send + Sync + 'static>;
pub type LLMEndCallback = Box<dyn Fn(&LLMResponse) + Send + Sync + 'static>;
pub type TaskStartCallback = Box<dyn Fn(&str) + Send + Sync + 'static>;
pub type TaskEndCallback = Box<dyn Fn(&str, &str) + Send + Sync + 'static>;

// ---------------------------------------------------------------------------
// Callbacks — aggregated set of hooks for Agent/Task/Crew
// ---------------------------------------------------------------------------
#[derive(Default)]
pub struct Callbacks {
    pub on_step: Vec<StepCallback>,
    pub on_tool_start: Vec<ToolStartCallback>,
    pub on_tool_end: Vec<ToolEndCallback>,
    pub on_llm_start: Vec<LLMStartCallback>,
    pub on_llm_end: Vec<LLMEndCallback>,
    pub on_task_start: Vec<TaskStartCallback>,
    pub on_task_end: Vec<TaskEndCallback>,
}

impl Callbacks {
    pub fn new() -> Self {
        Self::default()
    }

    // -- Step callbacks ----------------------------------------------------
    pub fn add_step<F>(&mut self, cb: F)
    where F: Fn(&ReActStep, &str) + Send + Sync + 'static {
        self.on_step.push(Box::new(cb));
    }

    pub fn call_step(&self, step: &ReActStep, agent_id: &str) {
        for cb in &self.on_step {
            cb(step, agent_id);
        }
    }

    // -- Tool callbacks ----------------------------------------------------
    pub fn add_tool_start<F>(&mut self, cb: F)
    where F: Fn(&str, &serde_json::Value) + Send + Sync + 'static {
        self.on_tool_start.push(Box::new(cb));
    }

    pub fn call_tool_start(&self, tool_name: &str, args: &serde_json::Value) {
        for cb in &self.on_tool_start {
            cb(tool_name, args);
        }
    }

    pub fn add_tool_end<F>(&mut self, cb: F)
    where F: Fn(&str, &ToolResult) + Send + Sync + 'static {
        self.on_tool_end.push(Box::new(cb));
    }

    pub fn call_tool_end(&self, tool_name: &str, result: &ToolResult) {
        for cb in &self.on_tool_end {
            cb(tool_name, result);
        }
    }

    // -- LLM callbacks -----------------------------------------------------
    pub fn add_llm_start<F>(&mut self, cb: F)
    where F: Fn(usize, usize) + Send + Sync + 'static {
        self.on_llm_start.push(Box::new(cb));
    }

    pub fn call_llm_start(&self, num_messages: usize, num_tools: usize) {
        for cb in &self.on_llm_start {
            cb(num_messages, num_tools);
        }
    }

    pub fn add_llm_end<F>(&mut self, cb: F)
    where F: Fn(&LLMResponse) + Send + Sync + 'static {
        self.on_llm_end.push(Box::new(cb));
    }

    pub fn call_llm_end(&self, response: &LLMResponse) {
        for cb in &self.on_llm_end {
            cb(response);
        }
    }

    // -- Task callbacks ----------------------------------------------------
    pub fn add_task_start<F>(&mut self, cb: F)
    where F: Fn(&str) + Send + Sync + 'static {
        self.on_task_start.push(Box::new(cb));
    }

    pub fn call_task_start(&self, description: &str) {
        for cb in &self.on_task_start {
            cb(description);
        }
    }

    pub fn add_task_end<F>(&mut self, cb: F)
    where F: Fn(&str, &str) + Send + Sync + 'static {
        self.on_task_end.push(Box::new(cb));
    }

    pub fn call_task_end(&self, description: &str, result: &str) {
        for cb in &self.on_task_end {
            cb(description, result);
        }
    }
}
