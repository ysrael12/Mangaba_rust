//! Output parsers for structured extraction from LLM responses.
//!
//! [`OutputParser`] trait with [`JSONOutputParser`] (extracts JSON from code fences)
//! and [`NoOpOutputParser`] (passthrough).

use anyhow::{Result, anyhow};

pub trait OutputParser: Send + Sync {
    fn parse(&self, text: &str) -> Result<String>;
    fn get_format_instructions(&self) -> String;
}

pub struct NoOpOutputParser;

impl OutputParser for NoOpOutputParser {
    fn parse(&self, text: &str) -> Result<String> {
        Ok(text.to_string())
    }
    fn get_format_instructions(&self) -> String {
        String::new()
    }
}

pub struct JSONOutputParser;

impl OutputParser for JSONOutputParser {
    fn parse(&self, text: &str) -> Result<String> {
        let trimmed = text.trim();
        if let Some(block) = trimmed
            .strip_prefix("```json")
            .and_then(|s| s.find("```").map(|end| &s[..end]))
        {
            return Ok(block.trim().to_string());
        }
        if let Some(block) = trimmed
            .strip_prefix("```")
            .and_then(|s| s.find("```").map(|end| &s[..end]))
        {
            return Ok(block.trim().to_string());
        }
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return Ok(trimmed.to_string());
        }
        Err(anyhow!("No JSON content found in output"))
    }
    fn get_format_instructions(&self) -> String {
        "Responda apenas com JSON válido.".into()
    }
}
