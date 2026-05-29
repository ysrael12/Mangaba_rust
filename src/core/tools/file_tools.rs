use async_trait::async_trait;
use anyhow::{Result, anyhow};
use serde_json::{json, Value};
use std::path::Path;
use tokio::fs;
use crate::core::types::ToolResult;
use super::BaseTool;

pub struct FileReaderTool;

#[async_trait]
impl BaseTool for FileReaderTool {
    fn name(&self) -> &str { "file_reader" }
    fn description(&self) -> &str { "Read text files and return their contents" }
    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to read"},
                "encoding": {"type": "string", "description": "File encoding (default: utf-8)", "default": "utf-8"}
            },
            "required": ["file_path"]
        }))
    }

    async fn run_impl(&self, args: Value) -> Result<ToolResult> {
        let file_path = args.get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'file_path' argument"))?;

        if !Path::new(file_path).exists() {
            return Ok(ToolResult {
                call_id: "read".to_string(),
                tool_name: "file_reader".to_string(),
                output: Some(json!({"error": format!("File '{}' not found", file_path)})),
                error: Some(format!("File '{}' not found", file_path)),
                success: false,
            });
        }

        match fs::read_to_string(file_path).await {
            Ok(content) => Ok(ToolResult {
                call_id: "read".to_string(),
                tool_name: "file_reader".to_string(),
                output: Some(json!({"content": content})),
                error: None,
                success: true,
            }),
            Err(e) => Ok(ToolResult {
                call_id: "read".to_string(),
                tool_name: "file_reader".to_string(),
                output: Some(json!({"error": format!("Error reading file: {}", e)})),
                error: Some(format!("{}", e)),
                success: false,
            }),
        }
    }
}

pub struct FileWriterTool;

#[async_trait]
impl BaseTool for FileWriterTool {
    fn name(&self) -> &str { "file_writer" }
    fn description(&self) -> &str { "Write content to text files" }
    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write"},
                "mode": {"type": "string", "description": "Write mode: 'w' (overwrite) or 'a' (append)", "default": "w"}
            },
            "required": ["file_path", "content"]
        }))
    }

    async fn run_impl(&self, args: Value) -> Result<ToolResult> {
        let file_path = args.get("file_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'file_path' argument"))?;
        let content = args.get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'content' argument"))?;
        let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("w");

        let path = Path::new(file_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| anyhow!("Failed to create directory: {}", e))?;
        }

        let result = match mode {
            "a" => fs::write(file_path, content).await,
            _ => fs::write(file_path, content).await,
        };

        match result {
            Ok(_) => Ok(ToolResult {
                call_id: "write".to_string(),
                tool_name: "file_writer".to_string(),
                output: Some(json!({"message": format!("Successfully wrote to '{}'", file_path)})),
                error: None,
                success: true,
            }),
            Err(e) => Ok(ToolResult {
                call_id: "write".to_string(),
                tool_name: "file_writer".to_string(),
                output: Some(json!({"error": format!("Error writing file: {}", e)})),
                error: Some(format!("{}", e)),
                success: false,
            }),
        }
    }
}

pub struct DirectoryListTool;

#[async_trait]
impl BaseTool for DirectoryListTool {
    fn name(&self) -> &str { "directory_list" }
    fn description(&self) -> &str { "List files and directories in a given path" }
    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "directory_path": {"type": "string", "description": "Path to the directory to list"},
                "pattern": {"type": "string", "description": "Optional filter pattern (e.g. '*.txt')"}
            },
            "required": ["directory_path"]
        }))
    }

    async fn run_impl(&self, args: Value) -> Result<ToolResult> {
        let dir_path = args.get("directory_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing 'directory_path' argument"))?;
        let pattern = args.get("pattern").and_then(|v| v.as_str());

        let path = Path::new(dir_path);
        if !path.exists() {
            return Ok(ToolResult {
                call_id: "ls".to_string(),
                tool_name: "directory_list".to_string(),
                output: Some(json!({"error": format!("Directory '{}' not found", dir_path)})),
                error: Some(format!("Directory '{}' not found", dir_path)),
                success: false,
            });
        }
        if !path.is_dir() {
            return Ok(ToolResult {
                call_id: "ls".to_string(),
                tool_name: "directory_list".to_string(),
                output: Some(json!({"error": format!("'{}' is not a directory", dir_path)})),
                error: Some(format!("'{}' is not a directory", dir_path)),
                success: false,
            });
        }

        let mut entries = fs::read_dir(path).await.map_err(|e| anyhow!("Failed to read directory: {}", e))?;
        let mut dirs = Vec::new();
        let mut files = Vec::new();

        let mut items = Vec::new();
        loop {
            match entries.next_entry().await {
                Ok(Some(entry)) => items.push(entry),
                Ok(None) => break,
                Err(e) => return Err(anyhow!("Error reading directory: {}", e)),
            }
        }

        for entry in &items {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(pat) = pattern {
                if !glob_match(pat, &name) {
                    continue;
                }
            }
            match entry.file_type().await {
                Ok(ft) if ft.is_dir() => dirs.push(name),
                _ => files.push(name),
            }
        }

        dirs.sort();
        files.sort();

        let mut lines = Vec::new();
        if !dirs.is_empty() {
            lines.push("Directories:".to_string());
            for d in &dirs {
                lines.push(format!("  {}/", d));
            }
        }
        if !files.is_empty() {
            if !dirs.is_empty() { lines.push(String::new()); }
            lines.push("Files:".to_string());
            for f in &files {
                lines.push(format!("  {}", f));
            }
        }
        if lines.is_empty() {
            lines.push("Directory is empty or no files match the pattern".to_string());
        }

        Ok(ToolResult {
            call_id: "ls".to_string(),
            tool_name: "directory_list".to_string(),
            output: Some(json!({
                "directories": dirs,
                "files": files,
                "result": lines.join("\n")
            })),
            error: None,
            success: true,
        })
    }
}

fn glob_match(pattern: &str, name: &str) -> bool {
    if pattern == "*" { return true; }
    if let Some(ext) = pattern.strip_prefix("*.") {
        return name.ends_with(ext);
    }
    if let Some(prefix) = pattern.strip_suffix("*") {
        return name.starts_with(prefix);
    }
    pattern == name
}
