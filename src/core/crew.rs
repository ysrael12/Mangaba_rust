//! Multi-agent orchestration with sequential and hierarchical execution.
//!
//! A [`Crew`] manages a set of [`Agent`]s and [`Task`]s.
//! - **Sequential**: tasks execute one after another; context flows forward via `previous_output`.
//! - **Hierarchical**: a manager LLM assigns each task to the most suitable agent.
//!
//! Agents are resolved by role name (or agent ID) from the crew's agent list.

use std::sync::Arc;
use anyhow::Result;
use tokio::sync::Mutex;
use crate::core::types::*;
use crate::core::agent::Agent;
use crate::core::task::Task;
use crate::core::llm::LLMClient;
use crate::core::events::{EventBus, Event, EventType};

pub struct Crew {
    pub agents: Vec<Box<Agent>>,
    pub tasks: Vec<Arc<Mutex<Task>>>,
    pub process: ProcessType,
    pub manager_llm: Option<Box<dyn LLMClient + Send + Sync>>,
    pub verbose: bool,
}

impl Crew {
    pub fn new(
        agents: Vec<Box<Agent>>,
        tasks: Vec<Arc<Mutex<Task>>>,
        process: ProcessType,
        manager_llm: Option<Box<dyn LLMClient + Send + Sync>>,
        verbose: bool,
    ) -> Self {
        Self { agents, tasks, process, manager_llm, verbose }
    }

    pub async fn kickoff(&mut self) -> Result<Vec<TaskOutput>> {
        EventBus::emit(Event::new(
            EventType::CrewStart, "crew",
            serde_json::json!({
                "num_agents": self.agents.len(),
                "num_tasks": self.tasks.len(),
                "process": format!("{:?}", self.process),
            }),
        ));

        let results = match self.process {
            ProcessType::Sequential => self.run_sequential().await,
            ProcessType::Hierarchical => self.run_hierarchical().await,
        };

        match &results {
            Ok(out) => {
                EventBus::emit(Event::new(
                    EventType::CrewEnd, "crew",
                    serde_json::json!({"num_results": out.len()}),
                ));
            }
            Err(e) => {
                EventBus::emit(Event::new(
                    EventType::CrewError, "crew",
                    serde_json::json!({"error": format!("{}", e)}),
                ));
            }
        }

        results
    }

    // -----------------------------------------------------------------------
    // Sequential: tasks one after another, context flows forward
    // -----------------------------------------------------------------------
    async fn run_sequential(&mut self) -> Result<Vec<TaskOutput>> {
        let mut results = Vec::new();
        let mut previous_output: Option<String> = None;

        for i in 0..self.tasks.len() {
            let (task_desc, agent_role) = {
                let task = self.tasks[i].lock().await;
                (task.config.description.clone(), task.config.agent_id.clone())
            };

            let context = previous_output.as_ref().map(|prev| {
                format!("Previous task result:\n{}", prev)
            });

            let (result, agent_id) = {
                let agent = self.resolve_agent(agent_role.as_deref())?;
                if agent.verbose {
                    log::info!("Agent '{}' executing: {}", agent.role, task_desc);
                }
                let result = agent.execute_task(&task_desc, context.as_deref()).await?;
                let agent_id = agent.state.agent_id.clone();
                (result, agent_id)
            };

            let mut task = self.tasks[i].lock().await;
            let output = TaskOutput::new(&task_desc, &result, &agent_id, true);
            task.state.output = Some(output.clone());
            task.state.status = TaskStatus::Completed;

            if let Some(ref path) = task.config.output_file {
                if let Err(e) = std::fs::write(path, &result) {
                    log::warn!("Failed to write task output to {}: {}", path, e);
                }
            }

            previous_output = Some(result);
            results.push(output);
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Hierarchical: manager agent assigns tasks to agents
    // -----------------------------------------------------------------------
    async fn run_hierarchical(&mut self) -> Result<Vec<TaskOutput>> {
        let manager_llm = self.manager_llm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Hierarchical process requires a manager_llm"))?;

        let mut results = Vec::new();

        for i in 0..self.tasks.len() {
            let task_desc = {
                let task = self.tasks[i].lock().await;
                task.config.description.clone()
            };

            let agent_descs: Vec<String> = self.agents.iter().map(|a| {
                let tools: Vec<String> = a.tools.iter().map(|t| t.name().to_string()).collect();
                format!("- {}: {} (tools: {})", a.role, a.goal, tools.join(", "))
            }).collect();

            let prompt = format!(
                "You are a crew manager. Your agents:\n{}\n\n\
                 Assign this task to the most suitable agent.\nTask: {}\n\n\
                 Reply with ONLY the agent's role name.",
                agent_descs.join("\n"), task_desc,
            );

            let response = manager_llm.chat(&[
                Message::system("You are a crew manager that assigns tasks to agents.".to_string()),
                Message::user(prompt),
            ], &[]).await?;

            let chosen_role = response.text().trim().to_string();
            let agent_idx = self.agents.iter().position(|a| a.role == chosen_role)
                .unwrap_or_else(|| {
                    log::warn!("Manager chose unknown agent '{}', using first", chosen_role);
                    0
                });

            let (result, agent_id) = {
                let agent = &mut self.agents[agent_idx];
                if agent.verbose {
                    log::info!("Manager assigned '{}' to '{}'", task_desc, agent.role);
                }
                let result = agent.execute_task(&task_desc, None).await?;
                let agent_id = agent.state.agent_id.clone();
                (result, agent_id)
            };

            let mut task = self.tasks[i].lock().await;
            let output = TaskOutput::new(&task_desc, &result, &agent_id, true);
            task.state.output = Some(output.clone());
            task.state.status = TaskStatus::Completed;

            results.push(output);
        }

        Ok(results)
    }

    fn resolve_agent(&mut self, agent_id_or_role: Option<&str>) -> Result<&mut Box<Agent>> {
        match agent_id_or_role {
            Some(id) => {
                let idx = self.agents.iter().position(|a| a.state.agent_id == id || a.role == id)
                    .ok_or_else(|| anyhow::anyhow!("No agent found with id/role '{}'", id))?;
                Ok(&mut self.agents[idx])
            }
            None => {
                self.agents.first_mut()
                    .ok_or_else(|| anyhow::anyhow!("No agents available in crew"))
            }
        }
    }
}
