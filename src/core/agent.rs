//! Agent with memory, guardrails, tools, delegation, and ReAct execution.
//!
//! An [`Agent`] wraps an [`LLMClient`] with a role, goal,
//! backstory, optional tools, guardrails, memory, and output parsers. Execution uses the
//! [`ReActEngine`] (Thought → Action → Observation loop)
//! with automatic retry on error.

use anyhow::Result;
use uuid::Uuid;
use crate::core::types::*;
use crate::core::llm::LLMClient;
use crate::core::memory::BaseMemory;
use crate::core::guardrails::{Guardrail, apply_guardrails};
use crate::core::output_parsers::OutputParser;
use crate::core::events::{EventBus, Event, EventType};
use crate::core::react::ReActEngine;
use crate::core::tools::BaseTool;
use crate::core::prompt_templates::SystemPromptBuilder;
use crate::core::callbacks::Callbacks;

pub struct Agent {
    pub guardrails: Vec<Box<dyn Guardrail + Send + Sync>>,
    pub output_parser: Option<Box<dyn OutputParser + Send + Sync>>,
    pub callbacks: Callbacks,
    pub role: String,
    pub goal: String,
    pub backstory: String,
    pub tools: Vec<Box<dyn BaseTool + Send + Sync>>,
    pub llm: Box<dyn LLMClient + Send + Sync>,
    pub memory: Option<Box<dyn BaseMemory + Send + Sync>>,
    pub config: AgentConfig,
    pub state: AgentState,
    pub peers: std::collections::HashMap<String, Agent>,
    pub max_iterations: usize,
    pub max_retry_on_error: usize,
    pub verbose: bool,
    pub allow_delegation: bool,
}

impl Agent {
    pub fn new(
        config: AgentConfig,
        tools: Vec<Box<dyn BaseTool + Send + Sync>>,
        llm: Box<dyn LLMClient + Send + Sync>,
        memory: Option<Box<dyn BaseMemory + Send + Sync>>,
    ) -> Self {
        let mut guardrails_vec: Vec<Box<dyn Guardrail + Send + Sync>> = Vec::new();
        for name in &config.guardrails {
            match name.as_str() {
                "no_op" => guardrails_vec.push(Box::new(crate::core::guardrails::NoOpGuardrail)),
                "length" => guardrails_vec.push(Box::new(crate::core::guardrails::LengthGuardrail { max_len: 1024, truncate: true })),
                "profanity" => guardrails_vec.push(Box::new(crate::core::guardrails::ProfanityGuardrail::new(
                    r"\b(badword|damn)\b", "****",
                ))),
                _ => guardrails_vec.push(Box::new(crate::core::guardrails::NoOpGuardrail)),
            }
        }

        let output_parser: Option<Box<dyn OutputParser + Send + Sync>> = match config.output_parser.as_deref() {
            Some("json") => Some(Box::new(crate::core::output_parsers::JSONOutputParser)),
            _ => None,
        };

        let agent_id = format!("agent_{}_{}", config.role.to_lowercase().replace(' ', "_"), Uuid::new_v4().simple());
        let state = AgentState {
            agent_id: agent_id.clone(),
            ..Default::default()
        };
        Self {
            max_iterations: config.max_iterations,
            max_retry_on_error: config.max_retry_on_error,
            verbose: config.verbose,
            allow_delegation: config.allow_delegation,
            role: config.role.clone(),
            goal: config.goal.clone(),
            backstory: config.backstory.clone(),
            guardrails: guardrails_vec,
            output_parser,
            callbacks: Callbacks::new(),
            config,
            tools,
            llm,
            memory,
            state,
            peers: std::collections::HashMap::new(),
        }
    }

    pub async fn execute_task(&mut self, task_description: &str, context: Option<&str>) -> Result<String> {
        EventBus::emit(Event::new(EventType::AgentStart, &self.state.agent_id, serde_json::json!({"task": task_description})));

        self.state.status = AgentStatus::Running;
        self.state.iteration_count = 0;

        let memory_context = if let Some(mem) = &self.memory {
            mem.get_relevant(task_description, 5).await
        } else {
            String::new()
        };

        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(task_description, context, &memory_context);

        self.callbacks.call_task_start(task_description);

        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 1..=self.max_retry_on_error {
            let mut messages = vec![
                Message::system(system_prompt.clone()),
                Message::user(user_prompt.clone()),
            ];

            let react = ReActEngine::new(
                self.llm.as_ref(), &self.tools, &self.callbacks,
                self.max_iterations, self.verbose,
            );
            match react.run(&mut messages).await {
                Ok((response, steps)) => {
                    self.state.messages = messages;
                    self.state.steps = steps;
                    self.state.status = AgentStatus::Completed;

                    // Step callbacks already fired inside ReActEngine
                    // Task-level callback
                    self.callbacks.call_task_end(task_description, response.text());

                    let mut result = apply_guardrails(&self.guardrails, &self.state.agent_id, response.text()).await;
                    if let Some(parser) = &self.output_parser {
                        result = parser.parse(&result)?;
                    }
                    if let Some(mem) = &mut self.memory {
                        for step in &self.state.steps {
                            let mut step_text = format!("Step {}: ", step.step_number);
                            if let Some(ref t) = step.thought {
                                step_text.push_str(&format!("Thought: {}. ", t));
                            }
                            if let Some(ref a) = step.action {
                                step_text.push_str(&format!("Action: {} with {:?}. ", a.tool_name, a.arguments));
                            }
                            if let Some(ref o) = step.observation {
                                step_text.push_str(&format!("Observation: {}", o));
                            }
                            mem.add(&step_text, None).await;
                        }
                        mem.add(
                            &format!("Task: {}\nResult: {}", task_description, result), None,
                        ).await;
                    }
                    let preview: String = result.chars().take(200).collect();
                    EventBus::emit(Event::new(EventType::AgentEnd, &self.state.agent_id, serde_json::json!({"result_preview": preview})));
                    return Ok(result);
                }
                Err(e) => {
                    last_err = Some(e);
                    if attempt < self.max_retry_on_error {
                        log::warn!("Agent retry {}/{}: {:?}", attempt, self.max_retry_on_error, last_err.as_ref().unwrap());
                        continue;
                    }
                }
            }
        }

        self.state.status = AgentStatus::Error;
        EventBus::emit(Event::new(EventType::AgentError, &self.state.agent_id, serde_json::json!({"error": format!("{:?}", last_err)})));
        Err(anyhow::anyhow!("Task failed after retries"))
    }

    pub fn add_peer(&mut self, peer: Agent) {
        let id = peer.state.agent_id.clone();
        self.peers.insert(id, peer);
    }

    pub async fn delegate(&mut self, peer_id: &str, task: &str, context: Option<&str>) -> Result<String> {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.execute_task(task, context).await
        } else {
            Err(anyhow::anyhow!("No peer with id {}", peer_id))
        }
    }

    fn build_system_prompt(&self) -> String {
        let mut builder = SystemPromptBuilder::new()
            .role(&self.role)
            .goal(&self.goal)
            .backstory(&self.backstory);

        if !self.tools.is_empty() {
            let tool_desc: Vec<String> = self.tools.iter()
                .map(|t| format!("- {}: {}", t.name(), t.description()))
                .collect();
            builder = builder.tools(&tool_desc);
        }

        if self.allow_delegation && !self.peers.is_empty() {
            let peers_desc = self.peers.values().map(|p| p.role.clone()).collect::<Vec<_>>().join(", ");
            builder = builder.section("You can delegate to these agents", &peers_desc);
        }

        builder.build()
    }

    fn build_user_prompt(&self, task: &str, context: Option<&str>, memory: &str) -> String {
        let mut parts = vec![];
        if let Some(ctx) = context { parts.push(format!("Context:\n{}", ctx)); }
        if !memory.is_empty() { parts.push(format!("Previous context:\n{}", memory)); }
        parts.push(format!("Task:\n{}", task));
        parts.push("Complete this task according to your role and goal.".to_string());
        parts.join("\n\n")
    }
}
