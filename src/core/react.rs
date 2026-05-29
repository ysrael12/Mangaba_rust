//! ReAct (Reasoning + Acting) engine for LLM agents.
//!
//! [`ReActEngine`] implements the Thought → Action → Observation loop:
//! 1. LLM generates a thought (and optionally tool calls)
//! 2. Tools are executed and results appended as observations
//! 3. Loop continues until the LLM produces a ﬁnal answer (no tool calls)
//!    or `max_iterations` is reached.
//!
//! Each iteration ﬁres callbacks and events for observability.

use anyhow::{Result, anyhow};
use crate::core::{llm::LLMClient, tools::BaseTool};
use crate::core::types::{LLMResponse, ReActStep, Message, ToolCall, ToolResult};
use crate::core::events::{EventBus, Event, EventType};
use crate::core::callbacks::Callbacks;

pub struct ReActEngine<'a> {
    pub llm: &'a dyn LLMClient,
    pub tools: &'a [Box<dyn BaseTool + Send + Sync>],
    pub callbacks: &'a Callbacks,
    pub max_iterations: usize,
    pub verbose: bool,
}

impl<'a> ReActEngine<'a> {
    pub fn new(
        llm: &'a dyn LLMClient,
        tools: &'a [Box<dyn BaseTool + Send + Sync>],
        callbacks: &'a Callbacks,
        max_iterations: usize,
        verbose: bool,
    ) -> Self {
        Self { llm, tools, callbacks, max_iterations, verbose }
    }

    pub async fn run(
        &self,
        messages: &mut Vec<Message>,
    ) -> Result<(LLMResponse, Vec<ReActStep>)> {
        let tool_refs: Vec<&dyn BaseTool> = self.tools.iter().map(|b| b.as_ref() as &dyn BaseTool).collect();
        let mut steps = Vec::new();

        for iteration in 0..self.max_iterations {
            if self.verbose {
                log::info!("ReAct iteration {}/{}", iteration + 1, self.max_iterations);
            }

            self.callbacks.call_llm_start(messages.len(), tool_refs.len());
            let response = self.llm.chat(messages, &tool_refs).await?;
            self.callbacks.call_llm_end(&response);

            if let Some(ref thought) = response.content {
                EventBus::emit(Event::new(
                    EventType::ReActThought, "react",
                    serde_json::json!({"step": iteration + 1, "thought": thought}),
                ));
            }

            let mut step = ReActStep {
                step_number: iteration + 1,
                thought: response.content.clone(),
                action: None,
                observation: None,
                timestamp: chrono::Utc::now().to_rfc3339(),
            };

            if response.has_tool_calls() {
                EventBus::emit(Event::new(
                    EventType::ReActAction, "react",
                    serde_json::json!({"step": iteration + 1, "tool_calls": response.tool_calls}),
                ));

                messages.push(Message::assistant(
                    response.content.clone(),
                    Some(response.tool_calls.clone()),
                ));

                for tool_call in &response.tool_calls {
                    self.callbacks.call_tool_start(&tool_call.tool_name, &serde_json::json!(tool_call.arguments));
                    let result = self.execute_tool(tool_call, iteration).await;
                    self.callbacks.call_tool_end(&tool_call.tool_name, &result);

                    step.action = Some(tool_call.clone());
                    step.observation = result.output.clone()
                        .map(|o| o.to_string())
                        .or_else(|| result.error.clone());

                    messages.push(Message::tool(vec![result]));

                    EventBus::emit(Event::new(
                        EventType::ReActObservation, "react",
                        serde_json::json!({"step": iteration + 1, "observation": step.observation}),
                    ));

                    if let Some(tool) = self.tools.iter().find(|t| t.name() == tool_call.tool_name) {
                        if tool.return_direct() {
                            self.callbacks.call_step(&step, "");
                            steps.push(step);
                            return Ok((response, steps));
                        }
                    }
                }
            } else {
                messages.push(Message::assistant(response.content.clone(), None));
                self.callbacks.call_step(&step, "");
                steps.push(step);
                return Ok((response, steps));
            }

            self.callbacks.call_step(&step, "");
            steps.push(step);
        }

        Err(anyhow!("ReAct loop exceeded max_iterations ({})", self.max_iterations))
    }

    async fn execute_tool(&self, tool_call: &ToolCall, _step: usize) -> ToolResult {
        let tool = match self.tools.iter().find(|t| t.name() == tool_call.tool_name) {
            Some(t) => t,
            None => {
                return ToolResult {
                    call_id: tool_call.id.clone(),
                    tool_name: tool_call.tool_name.clone(),
                    output: None,
                    error: Some(format!("Tool '{}' not found", tool_call.tool_name)),
                    success: false,
                };
            }
        };

        let args_value = serde_json::json!(tool_call.arguments);
        match tool.call(args_value).await {
            Ok(result) => result,
            Err(e) => ToolResult {
                call_id: tool_call.id.clone(),
                tool_name: tool_call.tool_name.clone(),
                output: None,
                error: Some(format!("{}", e)),
                success: false,
            },
        }
    }
}
