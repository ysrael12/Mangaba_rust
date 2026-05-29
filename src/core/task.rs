//! Task with context chaining and output persistence.
//!
//! A [`Task`] has a description, expected output, optional dedicated agent,
//! tools, and context tasks whose outputs are injected into the prompt.
//! Execution calls `agent.execute_task()`, saves output to a ﬁle if
//! `output_file` is set, and emits events at start/end.

use std::sync::Arc;
use anyhow::Result;
use tokio::sync::Mutex;
use uuid::Uuid;
use crate::core::types::*;
use crate::core::agent::Agent;
use crate::core::tools::BaseTool;
use crate::core::events::{EventBus, Event, EventType};

pub struct Task {
    pub config: TaskConfig,
    pub state: TaskState,
    pub agent: Option<Box<Agent>>,
    pub tools: Vec<Box<dyn BaseTool + Send + Sync>>,
    pub context_tasks: Vec<Arc<Mutex<Task>>>,
}

impl Task {
    pub fn new(
        config: TaskConfig,
        agent: Option<Box<Agent>>,
        tools: Vec<Box<dyn BaseTool + Send + Sync>>,
        context_tasks: Vec<Arc<Mutex<Task>>>,
    ) -> Self {
        let task_id = format!("task_{}", Uuid::new_v4().simple());
        Self {
            config,
            state: TaskState { task_id, ..Default::default() },
            agent,
            tools,
            context_tasks,
        }
    }

    pub async fn execute(&mut self) -> Result<TaskOutput> {
        EventBus::emit(Event::new(
            EventType::TaskStart, &self.task_id(),
            serde_json::json!({"description": self.config.description, "expected_output": self.config.expected_output}),
        ));

        self.state.status = TaskStatus::Running;
        self.state.attempts += 1;

        // Build context BEFORE borrowing agent
        let context = self.build_context().await;

        let agent = self.agent.as_mut().ok_or_else(|| anyhow::anyhow!("Task has no agent assigned"))?;
        let result = agent.execute_task(&self.config.description, Some(&context)).await?;

        // Write to output file if configured
        if let Some(ref path) = self.config.output_file {
            if let Err(e) = std::fs::write(path, &result) {
                log::warn!("Failed to write task output to {}: {}", path, e);
            }
        }

        let output = TaskOutput::new(
            &self.config.description,
            &result,
            &agent.state.agent_id,
            true,
        );

        self.state.output = Some(output.clone());
        self.state.status = TaskStatus::Completed;

        EventBus::emit(Event::new(
            EventType::TaskEnd, &self.task_id(),
            serde_json::json!({"result_preview": &result.chars().take(200).collect::<String>()}),
        ));

        Ok(output)
    }

    fn task_id(&self) -> String {
        self.state.task_id.clone()
    }

    async fn build_context(&self) -> String {
        let mut parts = Vec::new();

        // Gather outputs from context tasks
        for ctx_task in &self.context_tasks {
            let task = ctx_task.lock().await;
            if let Some(ref output) = task.state.output {
                if output.success {
                    parts.push(format!(
                        "Previous task '{}' result:\n{}",
                        output.description, output.result
                    ));
                }
            }
        }

        parts.join("\n\n")
    }
}
