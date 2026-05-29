use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

use mangaba::core::react::ReActEngine;
use mangaba::core::llm::LLMClient;
use mangaba::core::tools::*;
use mangaba::core::types::*;
use mangaba::core::callbacks::Callbacks;

/// Mock LLM that can be programmed to return specific responses.
struct MockLLM {
    /// Queue of responses: each `chat()` call pops the next one.
    responses: Arc<Mutex<Vec<LLMResponse>>>,
}

impl MockLLM {
    fn new(responses: Vec<LLMResponse>) -> Self {
        Self { responses: Arc::new(Mutex::new(responses)) }
    }
}

#[async_trait]
impl LLMClient for MockLLM {
    async fn chat(&self, _messages: &[Message], _tools: &[&dyn BaseTool]) -> Result<LLMResponse> {
        let mut queue = self.responses.lock().await;
        Ok(queue.remove(0))
    }
}

fn tool_call_response(tool_name: &str, args: Value) -> LLMResponse {
    LLMResponse {
        content: None,
        tool_calls: vec![ToolCall {
            id: "call_1".into(),
            tool_name: tool_name.into(),
            arguments: args.as_object()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default(),
        }],
        usage: TokenUsage::default(),
        model: "mock".into(),
        finish_reason: FinishReason::ToolCalls,
        raw: None,
    }
}

fn text_response(text: &str) -> LLMResponse {
    LLMResponse {
        content: Some(text.into()),
        tool_calls: vec![],
        usage: TokenUsage::default(),
        model: "mock".into(),
        finish_reason: FinishReason::Stop,
        raw: None,
    }
}

#[tokio::test]
async fn test_react_immediate_answer() {
    let llm = MockLLM::new(vec![
        text_response("Hello, I am the assistant."),
    ]);
    let tools: Vec<Box<dyn BaseTool + Send + Sync>> = vec![];

    let cbs = Callbacks::new();
    let engine = ReActEngine::new(&llm, &tools, &cbs, 10, false);
    let mut messages = vec![
        Message::system("You are a helper."),
        Message::user("Say hello."),
    ];
    let (result, steps) = engine.run(&mut messages).await.unwrap();

    assert_eq!(result.text(), "Hello, I am the assistant.");
    assert_eq!(steps.len(), 1);
    assert_eq!(messages.len(), 3); // system + user + assistant
}

#[tokio::test]
async fn test_react_one_tool_call() {
    let calc = CalculatorTool;
    let llm = MockLLM::new(vec![
        tool_call_response("calculator", json!({"expression": "2 + 2"})),
        text_response("The result is 4."),
    ]);
    let tools: Vec<Box<dyn BaseTool + Send + Sync>> = vec![Box::new(calc)];

    let cbs = Callbacks::new();
    let engine = ReActEngine::new(&llm, &tools, &cbs, 10, false);
    let mut messages = vec![
        Message::system("You are a calculator."),
        Message::user("What is 2+2?"),
    ];
    let (result, steps) = engine.run(&mut messages).await.unwrap();

    assert_eq!(result.text(), "The result is 4.");
    assert_eq!(steps.len(), 2);
    assert!(messages.len() > 4); // system + user + assistant(tool) + tool(result) + assistant
}

#[tokio::test]
async fn test_react_max_iterations() {
    let calc = CalculatorTool;
    // Always returns tool_calls — loop should hit max_iterations
    let llm = MockLLM::new(vec![
        tool_call_response("calculator", json!({"expression": "1+1"})),
        tool_call_response("calculator", json!({"expression": "1+1"})),
    ]);
    let tools: Vec<Box<dyn BaseTool + Send + Sync>> = vec![Box::new(calc)];

    let cbs = Callbacks::new();
    let engine = ReActEngine::new(&llm, &tools, &cbs, 2, false);
    let mut messages = vec![
        Message::system("You are a calculator."),
        Message::user("Add numbers."),
    ];
    let result = engine.run(&mut messages).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("max_iterations"));
}

#[tokio::test]
async fn test_react_tool_not_found() {
    let llm = MockLLM::new(vec![
        tool_call_response("nonexistent_tool", json!({})),
        text_response("Tool failed."),
    ]);
    let tools: Vec<Box<dyn BaseTool + Send + Sync>> = vec![];

    let cbs = Callbacks::new();
    let engine = ReActEngine::new(&llm, &tools, &cbs, 10, false);
    let mut messages = vec![
        Message::system("You are a helper."),
        Message::user("Use the tool."),
    ];
    let (result, _steps) = engine.run(&mut messages).await.unwrap();

    // Should continue to next iteration after tool error
    assert_eq!(result.text(), "Tool failed.");
}
