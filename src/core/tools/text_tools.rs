use async_trait::async_trait;
use anyhow::{Result, anyhow};
use serde_json::{json, Value};
use crate::core::types::ToolResult;
use super::BaseTool;

fn recursive_split(text: &str, chunk_size: usize, overlap: usize, separators: &[&str]) -> Vec<String> {
    if text.len() <= chunk_size {
        let trimmed = text.trim();
        return if trimmed.is_empty() { vec![] } else { vec![trimmed.to_string()] };
    }

    let sep = separators.first().copied().unwrap_or("");
    let remaining_seps = if separators.len() > 1 { &separators[1..] } else { &[""] };

    let parts: Vec<&str> = if sep.is_empty() {
        vec![text]
    } else {
        text.split(sep).collect()
    };

    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();

    for part in parts {
        let candidate = if current.is_empty() {
            part.to_string()
        } else {
            format!("{}{}{}", current, sep, part)
        };

        if candidate.len() <= chunk_size {
            current = candidate;
        } else {
            if !current.is_empty() {
                if current.len() <= chunk_size {
                    chunks.push(current.trim().to_string());
                } else {
                    chunks.extend(recursive_split(&current, chunk_size, overlap, remaining_seps));
                }
            }
            current = part.to_string();
        }
    }

    if !current.trim().is_empty() {
        if current.len() <= chunk_size {
            chunks.push(current.trim().to_string());
        } else {
            chunks.extend(recursive_split(&current, chunk_size, overlap, remaining_seps));
        }
    }

    // Apply overlap
    if overlap > 0 && chunks.len() > 1 {
        let mut overlapped: Vec<String> = vec![chunks[0].clone()];
        for i in 1..chunks.len() {
            let prev = &chunks[i - 1];
            let overlap_text = if prev.len() > overlap {
                prev[prev.len().saturating_sub(overlap)..].to_string()
            } else {
                prev.clone()
            };
            overlapped.push(format!("{}{}", overlap_text, chunks[i]));
        }
        return overlapped;
    }

    chunks
}

pub struct TextSplitterTool;

#[async_trait]
impl BaseTool for TextSplitterTool {
    fn name(&self) -> &str { "text_splitter" }
    fn description(&self) -> &str { "Split long text into smaller chunks with optional overlap" }
    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to split into chunks"},
                "chunk_size": {"type": "integer", "description": "Maximum characters per chunk", "default": 1000},
                "chunk_overlap": {"type": "integer", "description": "Overlap between consecutive chunks", "default": 200}
            },
            "required": ["text"]
        }))
    }

    async fn run_impl(&self, args: Value) -> Result<ToolResult> {
        let text = args.get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'text' argument"))?;
        let chunk_size = args.get("chunk_size").and_then(|v| v.as_u64()).unwrap_or(1000) as usize;
        let chunk_overlap = args.get("chunk_overlap").and_then(|v| v.as_u64()).unwrap_or(200) as usize;

        let separators = &["\n\n", "\n", ". ", " ", ""];
        let chunks = recursive_split(text, chunk_size, chunk_overlap, separators);

        let result: String = chunks.iter().enumerate()
            .map(|(i, c)| format!("--- Chunk {}/{} ---\n{}", i + 1, chunks.len(), c))
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(ToolResult {
            call_id: "split".to_string(),
            tool_name: "text_splitter".to_string(),
            output: Some(json!({"chunks": chunks, "result": result})),
            error: None,
            success: true,
        })
    }
}

pub struct WordCounterTool;

#[async_trait]
impl BaseTool for WordCounterTool {
    fn name(&self) -> &str { "word_counter" }
    fn description(&self) -> &str { "Count words, sentences, and characters in text" }
    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to count words in"}
            },
            "required": ["text"]
        }))
    }

    async fn run_impl(&self, args: Value) -> Result<ToolResult> {
        let text = args.get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'text' argument"))?;

        let words = text.split_whitespace().count();
        let chars = text.chars().count();
        let sentences = text.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .count();
        let paragraphs = text.split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .count();

        let result = format!(
            "Words: {}\nCharacters: {}\nSentences: {}\nParagraphs: {}",
            words, chars, sentences, paragraphs
        );

        Ok(ToolResult {
            call_id: "wc".to_string(),
            tool_name: "word_counter".to_string(),
            output: Some(json!({
                "words": words,
                "characters": chars,
                "sentences": sentences,
                "paragraphs": paragraphs,
                "result": result
            })),
            error: None,
            success: true,
        })
    }
}
