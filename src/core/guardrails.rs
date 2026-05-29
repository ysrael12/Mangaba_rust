//! Input/output guardrails for agent responses.
//!
//! [`Guardrail`] trait with:
//! - [`LengthGuardrail`] — truncates or validates by character length
//! - [`ProfanityGuardrail`] — regex-based replacement
//! - [`CompositeGuardrail`] — chains multiple guardrails
//! - [`GuardrailTool`] — wraps a guardrail as a [`BaseTool`] for the ReAct loop
//!
//! The [`apply_guardrails`] helper runs all guardrails with event emission.

use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;
use crate::core::events::{EventBus, Event, EventType};
use crate::core::tools::BaseTool;
use crate::core::types::ToolResult;

#[async_trait]
pub trait Guardrail: Send + Sync {
    fn validate(&self, text: &str) -> String;
}

pub struct NoOpGuardrail;

impl Guardrail for NoOpGuardrail {
    fn validate(&self, text: &str) -> String {
        text.to_string()
    }
}

pub struct LengthGuardrail {
    pub max_len: usize,
    pub truncate: bool,
}

impl Guardrail for LengthGuardrail {
    fn validate(&self, text: &str) -> String {
        if text.chars().count() <= self.max_len {
            text.to_string()
        } else if self.truncate {
            text.chars().take(self.max_len).collect()
        } else {
            text.chars().take(self.max_len).collect()
        }
    }
}

pub struct ProfanityGuardrail {
    regex: Regex,
    replacement: String,
}

impl ProfanityGuardrail {
    pub fn new(pattern: &str, replacement: &str) -> Self {
        Self {
            regex: Regex::new(pattern).unwrap_or_else(|_| Regex::new(r"(?!)").unwrap()),
            replacement: replacement.to_string(),
        }
    }
}

impl Guardrail for ProfanityGuardrail {
    fn validate(&self, text: &str) -> String {
        self.regex.replace_all(text, &self.replacement as &str).to_string()
    }
}

pub struct CompositeGuardrail {
    pub guardrails: Vec<Box<dyn Guardrail>>,
}

impl Guardrail for CompositeGuardrail {
    fn validate(&self, text: &str) -> String {
        let mut current = text.to_string();
        for g in &self.guardrails {
            current = g.validate(&current);
        }
        current
    }
}

// ---------------------------------------------------------------------------
// GuardrailTool — wraps a Guardrail as a BaseTool for use in ReAct loop
// ---------------------------------------------------------------------------
pub struct GuardrailTool {
    name: String,
    description: String,
    guardrail: Box<dyn Guardrail>,
}

impl GuardrailTool {
    pub fn new(name: &str, guardrail: Box<dyn Guardrail>) -> Self {
        Self {
            name: format!("guardrail_{}", name),
            description: format!("Validates text using the '{}' guardrail. Input: {{ \"text\": \"...\" }}", name),
            guardrail,
        }
    }
}

#[async_trait]
impl BaseTool for GuardrailTool {
    fn name(&self) -> &str { &self.name }
    fn description(&self) -> &str { &self.description }

    fn args_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to validate"}
            },
            "required": ["text"]
        }))
    }

    async fn run_impl(&self, args: Value) -> anyhow::Result<ToolResult> {
        let text = args["text"].as_str().unwrap_or("");
        let result = self.guardrail.validate(text);
        Ok(ToolResult {
            call_id: "guardrail".to_string(),
            tool_name: self.name.clone(),
            output: Some(serde_json::json!({"validated_text": result, "was_modified": result != text})),
            error: None,
            success: true,
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: run guardrails with event emission
// ---------------------------------------------------------------------------
pub async fn apply_guardrails(
    guardrails: &[Box<dyn Guardrail + Send + Sync>],
    source_id: &str,
    text: &str,
) -> String {
    let mut result = text.to_string();
    for g in guardrails {
        let before = result.clone();
        result = g.validate(&result);
        if result != before {
            EventBus::emit(Event::new(
                EventType::GuardrailPass, source_id,
                serde_json::json!({"modified": true, "preview": &result.chars().take(100).collect::<String>()}),
            ));
        } else {
            EventBus::emit(Event::new(
                EventType::GuardrailPass, source_id,
                serde_json::json!({"modified": false}),
            ));
        }
    }
    result
}
