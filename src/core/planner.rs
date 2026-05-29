//! LLM-generated task planning and execution plans.
//!
//! [`TaskPlanner`] asks an LLM to decompose a task into [`PlanStep`]s
//! (each with a description, optional tool, expected result, and dependencies).
//! The result is an [`ExecutionPlan`] that can be executed step-by-step.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::core::llm::LLMClient;
use crate::core::types::Message;

// ---------------------------------------------------------------------------
// PlanStep
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub step_number: usize,
    pub description: String,
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(default)]
    pub expected_result: String,
    #[serde(default)]
    pub dependencies: Vec<usize>,
}

// ---------------------------------------------------------------------------
// ExecutionPlan
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub goal: String,
    pub steps: Vec<PlanStep>,
}

impl ExecutionPlan {
    pub fn total_steps(&self) -> usize {
        self.steps.len()
    }
}

// ---------------------------------------------------------------------------
// TaskPlanner
// ---------------------------------------------------------------------------
pub struct TaskPlanner {
    llm: Box<dyn LLMClient + Send + Sync>,
    tool_names: Vec<String>,
}

impl TaskPlanner {
    pub fn new(llm: Box<dyn LLMClient + Send + Sync>) -> Self {
        Self { llm, tool_names: vec![] }
    }

    pub fn with_tools(mut self, tools: &[&str]) -> Self {
        self.tool_names = tools.iter().map(|s| s.to_string()).collect();
        self
    }

    pub async fn plan(&self, task: &str) -> Result<ExecutionPlan> {
        let tools_str = if self.tool_names.is_empty() {
            "none".to_string()
        } else {
            self.tool_names.join(", ")
        };

        let prompt = format!(
            "You are a task planning assistant.\n\
             Decompose the following task into concrete, sequential steps.\n\
             If tools are available, indicate which tool to use for each step.\n\n\
             Respond ONLY with a JSON array of objects with keys: \
             \"step_number\", \"description\", \"tool\" (or null), \"expected_result\", \"dependencies\" (list of step numbers).\n\n\
             Available tools: {tools}\n\n\
             Task: {task}\n\n\
             JSON plan:",
            tools = tools_str,
            task = task,
        );

        let messages = vec![
            Message::system("You are a precise task planning assistant. Always respond with valid JSON."),
            Message::user(&prompt),
        ];

        let response = self.llm.chat(&messages, &[]).await?;
        let text = response.content.as_deref().unwrap_or("");

        let steps = Self::parse_steps(text)?;
        Ok(ExecutionPlan { goal: task.to_string(), steps })
    }

    fn parse_steps(raw: &str) -> Result<Vec<PlanStep>> {
        // Try to find a JSON array
        let start = raw.find('[').ok_or_else(|| anyhow!("No JSON array found in planner response"))?;
        let end = raw.rfind(']').ok_or_else(|| anyhow!("No closing bracket in planner response"))? + 1;

        let slice = &raw[start..end];
        let parsed: Vec<PlanStep> = serde_json::from_str(slice)
            .map_err(|e| anyhow!("Failed to parse plan steps: {e}"))?;

        Ok(parsed)
    }
}
