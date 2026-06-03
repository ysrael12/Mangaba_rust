//! Advanced multi-agent crew — a realistic "sales analysis" workflow.
//!
//! Demonstrates, in one runnable example:
//!   - A **custom `BaseTool`** (`StatisticsTool`) alongside the built-in calculator.
//!   - **Three specialized agents** chained in a sequential `Crew`, where each
//!     task automatically receives the previous task's output as context.
//!   - **Per-agent memory** (short-term) and **guardrails** (length).
//!   - **Observability**: a global `EventBus` listener that prints a compact
//!     trace, plus a local `Callbacks` hook on one agent.
//!
//! Run it (reads `.env` at the project root via dotenv):
//!   cargo run --example advanced_crew
//!
//! With no `LLM_PROVIDER`/key set it falls back to `DummyLLM` (empty responses),
//! so it always runs; with the Gemini key in `.env` you get real output.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use mangaba::core::agent::Agent;
use mangaba::core::config::create_llm_config;
use mangaba::core::crew::Crew;
use mangaba::core::events::{EventBus, Event, EventType};
use mangaba::core::llm::{create_llm_client, DummyLLM, LLMClient, RetryLLMClient};
use mangaba::core::llm::token_counter::UsageTracker;
use mangaba::core::memory::create_memory;
use mangaba::core::retry::RetryConfig;
use mangaba::core::task::Task;
use mangaba::core::tools::{BaseTool, CalculatorTool};
use mangaba::core::types::{
    AgentConfig, MemoryConfig, ProcessType, TaskConfig, ToolResult,
};

// ===========================================================================
// Custom tool: descriptive statistics over a list of numbers.
// ===========================================================================
struct StatisticsTool;

#[async_trait]
impl BaseTool for StatisticsTool {
    fn name(&self) -> &str {
        "statistics"
    }

    fn description(&self) -> &str {
        "Compute descriptive statistics (count, sum, mean, min, max) over a list \
         of numbers. Input: { \"numbers\": [1, 2, 3] }"
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": { "type": "number" },
                    "description": "The list of numeric values to summarize"
                }
            },
            "required": ["numbers"]
        }))
    }

    async fn run_impl(&self, args: Value) -> Result<ToolResult> {
        let numbers: Vec<f64> = args
            .get("numbers")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("Missing 'numbers' array argument"))?
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();

        if numbers.is_empty() {
            return Ok(ToolResult {
                call_id: "stats".into(),
                tool_name: "statistics".into(),
                output: None,
                error: Some("The 'numbers' list is empty".into()),
                success: false,
            });
        }

        let count = numbers.len();
        let sum: f64 = numbers.iter().sum();
        let mean = sum / count as f64;
        let min = numbers.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = numbers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Ok(ToolResult {
            call_id: "stats".into(),
            tool_name: "statistics".into(),
            output: Some(json!({
                "count": count,
                "sum": sum,
                "mean": mean,
                "min": min,
                "max": max,
            })),
            error: None,
            success: true,
        })
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Build an LLM client from the environment, falling back to `DummyLLM`, and
/// wrap it in `RetryLLMClient` so transient failures (HTTP 5xx, network blips)
/// are retried with exponential backoff before the agent ever sees them.
fn build_llm() -> Box<dyn LLMClient + Send + Sync> {
    let inner: Box<dyn LLMClient + Send + Sync> = if std::env::var("LLM_PROVIDER").is_ok() {
        let cfg = create_llm_config();
        match create_llm_client(&cfg) {
            Ok(client) => {
                println!("🤖 LLM: {} ({}) + retry", cfg.provider, cfg.model);
                client
            }
            Err(e) => {
                eprintln!("⚠️  LLM init failed ({e}); using DummyLLM");
                Box::new(DummyLLM)
            }
        }
    } else {
        println!("🤖 LLM: DummyLLM (set LLM_PROVIDER / .env for real responses)");
        Box::new(DummyLLM)
    };

    Box::new(RetryLLMClient::new(
        inner,
        RetryConfig {
            max_retries: 4,
            initial_backoff_ms: 500,
            max_backoff_ms: 8_000,
            jitter_factor: 0.3,
        },
        None,                          // no response cache for this demo
        Arc::new(UsageTracker::new()),
    ))
}

fn short_term_memory() -> Option<Box<dyn mangaba::core::memory::BaseMemory + Send + Sync>> {
    create_memory(&MemoryConfig {
        short_term: true,
        long_term: false,
        entity: false,
        max_short_term_items: 25,
        storage_path: None,
    })
}

/// AgentConfig spelled out (the struct doesn't derive `Default`).
fn agent_config(role: &str, goal: &str, backstory: &str, guardrails: Vec<String>) -> AgentConfig {
    AgentConfig {
        role: role.into(),
        goal: goal.into(),
        backstory: backstory.into(),
        llm_config: None,
        tools: vec![],
        memory_config: MemoryConfig {
            short_term: true,
            long_term: false,
            entity: false,
            max_short_term_items: 25,
            storage_path: None,
        },
        max_iterations: 8,
        max_retry_on_error: 2,
        verbose: false,
        allow_delegation: false,
        step_callback: None,
        guardrails,
        output_parser: None,
    }
}

fn task_config(description: &str, expected: &str, agent_role: &str) -> TaskConfig {
    TaskConfig {
        description: description.into(),
        expected_output: expected.into(),
        agent_id: Some(agent_role.into()),
        context_ids: vec![],
        tools: vec![],
        output_file: None,
        async_execution: false,
        human_input: false,
        guardrails: vec![],
        output_parser: None,
        retry_on_failure: 0,
    }
}

// ===========================================================================
// Main
// ===========================================================================
#[tokio::main]
async fn main() -> Result<()> {
    // Load `.env` from the project root and init logging.
    let _ = dotenvy::dotenv();
    let _ = env_logger::builder().filter_level(log::LevelFilter::Warn).try_init();

    // -- Observability: a global EventBus listener with a compact trace -----
    let tool_calls = Arc::new(AtomicUsize::new(0));
    let counter = tool_calls.clone();
    EventBus::subscribe(Box::new(move |ev: &Event| match ev.event_type {
        EventType::AgentStart => {
            if let Some(task) = ev.data.get("task").and_then(|v| v.as_str()) {
                println!("  ▶ agent '{}' started: {}", ev.source_id, truncate(task, 60));
            }
        }
        EventType::ToolStart => {
            counter.fetch_add(1, Ordering::SeqCst);
            if let Some(tool) = ev.data.get("tool").and_then(|v| v.as_str()) {
                println!("    🔧 tool: {tool}  args={}", ev.data.get("args").unwrap_or(&json!(null)));
            }
        }
        EventType::ReActThought => {
            if let Some(t) = ev.data.get("thought").and_then(|v| v.as_str()) {
                if !t.trim().is_empty() {
                    println!("    💭 {}", truncate(t, 80));
                }
            }
        }
        EventType::AgentError => {
            println!("    ❌ agent '{}' error: {}", ev.source_id, ev.data);
        }
        _ => {}
    }));

    // -- Agent 1: Data Analyst (custom tool + calculator) ------------------
    let analyst_tools: Vec<Box<dyn BaseTool + Send + Sync>> = vec![
        Box::new(StatisticsTool),
        Box::new(CalculatorTool),
    ];
    let analyst = Agent::new(
        agent_config(
            "Data Analyst",
            "Turn raw sales numbers into precise, tool-verified statistics.",
            "You are a meticulous quantitative analyst. You NEVER guess numbers — \
             you always call the `statistics` or `calculator` tool to compute them.",
            vec![], // no output guardrail — numbers must pass through verbatim
        ),
        analyst_tools,
        build_llm(),
        short_term_memory(),
    );

    // -- Agent 2: Business Strategist (reasoning only) ---------------------
    let strategist = Agent::new(
        agent_config(
            "Business Strategist",
            "Read the analyst's statistics and propose two concrete, actionable \
             recommendations.",
            "You are a sharp B2B strategist who reasons from data to decisions.",
            vec!["length".into()], // cap verbosity
        ),
        vec![],
        build_llm(),
        short_term_memory(),
    );

    // -- Agent 3: Report Writer (synthesis) --------------------------------
    let writer_cfg = agent_config(
        "Report Writer",
        "Write a crisp executive summary combining the analysis and the strategy.",
        "You write tight, executive-ready prose. No fluff.",
        vec!["length".into()],
    );
    let mut writer = Agent::new(writer_cfg, vec![], build_llm(), short_term_memory());
    // Local callback hook on just this agent.
    writer.callbacks.add_task_end(|desc, result| {
        println!(
            "  ✍️  writer finished '{}' → {} chars",
            truncate(desc, 40),
            result.len()
        );
    });

    // -- Tasks (sequential; each receives the previous output as context) --
    let sales = "[1200, 1850, 1430, 2100, 980, 2750, 1620]";
    let t1 = Task::new(
        task_config(
            &format!(
                "Here are this week's daily sales (USD): {sales}. \
                 Use the statistics tool to compute count, sum, mean, min and max, \
                 then report them clearly."
            ),
            "A short report listing count, sum, mean, min and max.",
            "Data Analyst",
        ),
        None,
        vec![],
        vec![],
    );
    let t2 = Task::new(
        task_config(
            "Based on the analyst's statistics in the context, give exactly two \
             concrete recommendations to grow next week's sales.",
            "Two numbered, actionable recommendations.",
            "Business Strategist",
        ),
        None,
        vec![],
        vec![],
    );
    let t3 = Task::new(
        task_config(
            "Write a 4-6 sentence executive summary that combines the statistics \
             and the two recommendations from the context.",
            "A concise executive summary paragraph.",
            "Report Writer",
        ),
        None,
        vec![],
        vec![],
    );

    // -- Crew: sequential pipeline -----------------------------------------
    let mut crew = Crew::new(
        vec![Box::new(analyst), Box::new(strategist), Box::new(writer)],
        vec![
            Arc::new(Mutex::new(t1)),
            Arc::new(Mutex::new(t2)),
            Arc::new(Mutex::new(t3)),
        ],
        ProcessType::Sequential,
        None,
        false,
    );

    println!("\n=== Kicking off Sales Analysis Crew ===\n");

    // Degrade gracefully: a free-tier quota/rate-limit error shouldn't crash the
    // demo with a stack trace. The typed MangabaError makes that easy to detect.
    match crew.kickoff().await {
        Ok(results) => {
            println!("\n=== Results ===");
            for (i, r) in results.iter().enumerate() {
                println!("\n── Step {} · agent={} ──", i + 1, r.agent_id);
                println!("{}", r.result.trim());
            }
            println!(
                "\n✅ Done. {} tool invocation(s) observed via EventBus.",
                tool_calls.load(Ordering::SeqCst)
            );
        }
        Err(e) => {
            let is_rate_limit = e
                .chain()
                .any(|c| c.to_string().contains("Rate limit") || c.to_string().contains("429"));
            if is_rate_limit {
                println!(
                    "\n⏳ The crew stopped on a provider rate limit / quota.\n\
                     This is expected on the Gemini free tier (≈20 requests/day per model).\n\
                     The fix layer worked: the 429 was classified as MangabaError::RateLimit\n\
                     and the real cause was propagated instead of being masked.\n\
                     Try again later, switch MODEL_NAME, or use a paid key."
                );
            } else {
                println!("\n❌ Crew failed:");
                for cause in e.chain() {
                    println!("   - {cause}");
                }
            }
            println!(
                "\n({} tool invocation(s) observed before stopping.)",
                tool_calls.load(Ordering::SeqCst)
            );
        }
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.trim().replace('\n', " ");
    if s.chars().count() <= max {
        s
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}
