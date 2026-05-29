//! Quick-start example for the Mangaba AI framework.
//!
//! Two modes:
//!   1. `cargo run --example quick_start` — uses DummyLLM (no API key needed)
//!   2. `LLM_PROVIDER=openai cargo run --example quick_start` — real LLM from env
//!
//! Copy `.env.example` to `.env` and fill in your keys for mode 2.

use anyhow::Result;
use mangaba::core::agent::Agent;
use mangaba::core::config::create_llm_config;
use mangaba::core::llm::{create_llm_client, DummyLLM};
use mangaba::core::memory::create_memory;
use mangaba::core::types::MemoryConfig;
use mangaba::core::tools::{CalculatorTool, EchoTool};
use mangaba::core::types::AgentConfig;

#[tokio::main]
async fn main() -> Result<()> {
    // -- LLM client -------------------------------------------------------
    // If a real provider is configured in the environment, use it.
    // Otherwise fall back to DummyLLM (returns empty responses, no key needed).
    let llm: Box<dyn mangaba::core::llm::LLMClient + Send + Sync> =
        if std::env::var("LLM_PROVIDER").is_ok() {
            let cfg = create_llm_config();
            println!("Using provider: {} (model: {})", cfg.provider, cfg.model);
            create_llm_client(&cfg)?
        } else {
            println!("No LLM_PROVIDER set — using DummyLLM (no API key required)");
            Box::new(DummyLLM)
        };

    // -- Tools ------------------------------------------------------------
    let tools: Vec<Box<dyn mangaba::core::tools::BaseTool + Send + Sync>> = vec![
        Box::new(CalculatorTool),
        Box::new(EchoTool),
    ];

    // -- Memory -----------------------------------------------------------
    let memory_config = MemoryConfig {
        short_term: true,
        long_term: false,
        entity: false,
        max_short_term_items: 20,
        storage_path: None,
    };
    let memory = create_memory(&memory_config);

    // -- Agent config -----------------------------------------------------
    let agent_config = AgentConfig {
        role: "Research Assistant".into(),
        goal: "Answer user questions accurately with the tools available".into(),
        backstory: "You are a helpful AI assistant who uses tools to find answers.".into(),
        llm_config: None,
        tools: vec![],
        memory_config,
        max_iterations: 10,
        max_retry_on_error: 2,
        verbose: true,
        allow_delegation: false,
        step_callback: None,
        guardrails: vec!["no_op".into()],
        output_parser: None,
    };

    // -- Agent ------------------------------------------------------------
    let mut agent = Agent::new(agent_config, tools, llm, memory);

    // -- Execute a task ---------------------------------------------------
    let result = agent
        .execute_task(
            "Calculate 42 * 7 using the calculator tool, then echo the result.",
            None,
        )
        .await?;

    println!("\n=== Final result ===\n{result}");

    // -- Show conversation history ----------------------------------------
    println!("\n=== Conversation history ===");
    for msg in &agent.state.messages {
        let role = format!("{:?}", msg.role).to_lowercase();
        let content = msg.content.as_deref().unwrap_or("");
        println!("  [{role}] {content}");
    }

    Ok(())
}
