# Mangaba AI

Port of Mangaba AI from Python to Rust — a modular, provider-agnostic LLM agent framework.

```
mangaba = { git = "..." }
```

## Architecture

```
mangaba::core
├── agent        Agent with memory, guardrails, tools, delegation
├── callbacks    Hook system for step/tool/LLM/task events
├── config       Env-based provider detection & LLMConfig builder
├── crew         Orchestrates multi-agent task execution (sequential / hierarchical)
├── embeddings   Embedding trait + OpenAI, HuggingFace, NoOp, Cached LRU, InMemoryVectorStore
├── errors       MangabaError enum (thiserror, 18 variants, is_retryable)
├── events       Global EventBus (subscribe/emit)
├── guardrails   Length, profanity, composite guardrails + GuardrailTool
├── llm          LLMClient trait + providers (OpenAI, DeepSeek, Qwen, OpenRouter,
│                Google, Claude, HuggingFace) + streaming + retry + cache + token counter
├── memory       Short-term, long-term (JSON-persisted), entity memory
├── output_parsers  JSONOutputParser, NoOpOutputParser
├── pipeline     Stage, ParallelStage, ConditionalStage, Pipeline
├── planner      PlanStep, ExecutionPlan, TaskPlanner (LLM-generated plans)
├── prompt_templates  PromptTemplate, SystemPromptBuilder
├── protocols    A2A (agent-to-agent) + MCP (model context protocol)
├── rag          RAGEngine: ingest files/text → chunk → embed → query
├── react        ReActEngine: Thought → Action → Observation loop
├── retry        Exponential backoff with jitter
├── task         Task with context chaining
├── tools        BaseTool trait + Calculator, FileReader/Writer, DirectoryList,
│                TextSplitter, WordCounter, SerperSearch, DuckDuckGoSearch, Echo
└── types        LLMConfig, Message, ToolCall, ToolResult, LLMResponse, TokenUsage,
                 AgentConfig, TaskConfig, etc.
```

## Quick Start

```rust
use mangaba::core::config::Config;
use mangaba::core::llm::create_llm;
use mangaba::core::agent::Agent;
use mangaba::core::task::Task;
use mangaba::core::crew::Crew;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = Config::load()?;
    let llm = create_llm(&cfg.to_llm_config())?;

    let agent = Agent::new(
        AgentConfig {
            role: "Researcher".into(),
            goal: "Find answers".into(),
            backstory: "Expert researcher".into(),
            ..Default::default()
        },
        vec![],
        llm,
        None,
    );

    let task = Task::new(
        TaskConfig {
            description: "Research Rust async patterns".into(),
            expected_output: "Summary".into(),
            agent_id: Some("Researcher".into()),
            ..Default::default()
        },
        None, vec![], vec![],
    );

    let mut crew = Crew::new(
        vec![Box::new(agent)],
        vec![Arc::new(Mutex::new(task))],
        ProcessType::Sequential, None, false,
    );

    let results = crew.kickoff().await?;
    println!("{}", results[0].result);
    Ok(())
}
```

## Features

### Provider-Agnostic LLM
Swap providers by changing env vars — `LLM_PROVIDER=openai`, `GOOGLE_API_KEY`, etc. All providers support tool calling.

### ReAct Agent Loop
The `ReActEngine` implements the Thought → Action → Observation cycle with automatic tool execution, step tracking, and event emission.

### Crew Orchestration
- **Sequential**: tasks execute one after another, context flows forward
- **Hierarchical**: a manager LLM assigns tasks to agents

### RAG Pipeline
```
File/Text → DocumentChunker → Embed → LocalChroma (SQLite) → Query
```

### Protocols
- **A2A**: agent-to-agent communication with bidirectional endpoints
- **MCP**: model context protocol with priority scoring and relevance filtering

### Streaming
`LLMClient::stream_chat()` returns `Pin<Box<dyn Stream<Item = Result<String>>>>`. Providers default to wrapping `chat()`; OpenAI/OpenRouter implement real SSE.

## Configuration

Set env vars (copy `.env.example` to `.env`):

```bash
LLM_PROVIDER=google
GOOGLE_API_KEY=...
MODEL_NAME=gemini-2.5-flash
MODEL_TEMPERATURE=0.7
```

Or use `Config::load()` / `create_llm_config()` programmatically.

## Testing

```bash
cargo test          # 49+ tests
cargo doc --no-deps # generate docs
```

## Status

Core ~90% ported from Python. See `AGENTS.md` for detailed progress.
