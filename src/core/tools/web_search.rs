use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::Value;


use crate::core::tools::BaseTool;
use crate::core::types::ToolResult;

// ---------------------------------------------------------------------------
// SerperSearchTool — uses serper.dev API
// ---------------------------------------------------------------------------
pub struct SerperSearchTool {
    http: reqwest::Client,
    api_key: String,
}

impl SerperSearchTool {
    pub fn new(api_key: &str) -> Self {
        Self { http: reqwest::Client::new(), api_key: api_key.to_string() }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("SERPER_API_KEY")
            .map_err(|_| anyhow!("SERPER_API_KEY env var not set"))?;
        Ok(Self::new(&key))
    }
}

#[async_trait]
impl BaseTool for SerperSearchTool {
    fn name(&self) -> &str { "serper_search" }
    fn description(&self) -> &str { "Search the web using the Serper API (Google results). Input: query string." }

    fn args_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }))
    }

    async fn run_impl(&self, args: Value) -> Result<ToolResult> {
        let query = args["query"].as_str()
            .ok_or_else(|| anyhow!("Missing 'query' argument"))?;

        let resp = self.http
            .post("https://google.serper.dev/search")
            .header("X-API-KEY", &self.api_key)
            .json(&serde_json::json!({"q": query}))
            .send()
            .await?
            .json::<Value>()
            .await?;

        let mut results = Vec::new();

        if let Some(organic) = resp["organic"].as_array() {
            for item in organic.iter().take(5) {
                let title = item["title"].as_str().unwrap_or("");
                let link = item["link"].as_str().unwrap_or("");
                let snippet = item["snippet"].as_str().unwrap_or("");
                results.push(format!("- [{title}]({link})\n  {snippet}"));
            }
        }

        if results.is_empty() {
            return Err(anyhow!("No search results found for: {query}"));
        }

        Ok(ToolResult {
            call_id: "serper_search".to_string(),
            tool_name: "serper_search".to_string(),
            output: Some(serde_json::json!({"results": results.join("\n")})),
            error: None,
            success: true,
        })
    }
}

// ---------------------------------------------------------------------------
// DuckDuckGoSearchTool — uses DuckDuckGo's instant answer API (no key)
// ---------------------------------------------------------------------------
pub struct DuckDuckGoSearchTool {
    http: reqwest::Client,
}

impl DuckDuckGoSearchTool {
    pub fn new() -> Self {
        Self { http: reqwest::Client::new() }
    }
}

impl Default for DuckDuckGoSearchTool {
    fn default() -> Self { Self::new() }
}

#[async_trait]
impl BaseTool for DuckDuckGoSearchTool {
    fn name(&self) -> &str { "duckduckgo_search" }
    fn description(&self) -> &str { "Search the web using DuckDuckGo (no API key required). Input: query string." }

    fn args_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }))
    }

    async fn run_impl(&self, args: Value) -> Result<ToolResult> {
        let query = args["query"].as_str()
            .ok_or_else(|| anyhow!("Missing 'query' argument"))?;

        let resp = self.http
            .get("https://api.duckduckgo.com/")
            .query(&[
                ("q", query),
                ("format", "json"),
                ("no_html", "1"),
                ("skip_disambig", "1"),
            ])
            .send()
            .await?
            .json::<Value>()
            .await?;

        let mut results = Vec::new();

        if let Some(answer) = resp["AbstractText"].as_str() {
            if !answer.is_empty() {
                results.push(format!("**Answer**: {answer}"));
                if let Some(src) = resp["AbstractSource"].as_str() {
                    results.push(format!("*Source*: {src}"));
                }
            }
        }

        if let Some(related) = resp["RelatedTopics"].as_array() {
            for item in related.iter().take(5) {
                let text = item["Text"].as_str().unwrap_or("");
                if !text.is_empty() {
                    results.push(format!("- {text}"));
                }
            }
        }

        if results.is_empty() {
            return Err(anyhow!("No DuckDuckGo results for: {query}"));
        }

        Ok(ToolResult {
            call_id: "duckduckgo_search".to_string(),
            tool_name: "duckduckgo_search".to_string(),
            output: Some(serde_json::json!({"results": results.join("\n")})),
            error: None,
            success: true,
        })
    }
}
