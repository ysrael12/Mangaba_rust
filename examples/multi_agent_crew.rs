//! Multi-agent crew example.
//!
//! Creates two agents (Researcher + Writer) with a sequential crew.
//! Uses DummyLLM by default; set `LLM_PROVIDER=...` for real LLM responses.
//!
//! Usage:
//!   cargo run --example multi_agent_crew
//!   LLM_PROVIDER=openai cargo run --example multi_agent_crew

use std::sync::Arc;
use anyhow::Result;
use tokio::sync::Mutex;
use mangaba::core::agent::Agent;
use mangaba::core::config::create_llm_config;
use mangaba::core::crew::Crew;
use mangaba::core::llm::{create_llm_client, DummyLLM};
use mangaba::core::memory::create_memory;
use mangaba::core::task::Task;
use mangaba::core::tools::EchoTool;
use mangaba::core::types::{
    AgentConfig, TaskConfig, ProcessType, TaskOutput, MemoryConfig,
};

fn build_llm() -> Result<Box<dyn mangaba::core::llm::LLMClient + Send + Sync>> {
    if std::env::var("LLM_PROVIDER").is_ok() {
        let cfg = create_llm_config();
        println!("LLM: {} ({})", cfg.provider, cfg.model);
        create_llm_client(&cfg)
    } else {
        println!("LLM: DummyLLM (set LLM_PROVIDER for real responses)");
        Ok(Box::new(DummyLLM))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mem_cfg = MemoryConfig {
        short_term: true,
        long_term: false,
        entity: false,
        max_short_term_items: 10,
        storage_path: None,
    };

    // -- Agent 1: Researcher ----------------------------------------------
    let tools1: Vec<Box<dyn mangaba::core::tools::BaseTool + Send + Sync>> = vec![
        Box::new(EchoTool),
    ];
    let researcher = Agent::new(
        AgentConfig {
            role: "Researcher".into(),
            goal: "Find interesting facts and data.".into(),
            backstory: "Expert at gathering information and compiling data.".into(),
            llm_config: None,
            tools: vec![],
            memory_config: mem_cfg.clone(),
            max_iterations: 15,
            max_retry_on_error: 3,
            verbose: true,
            allow_delegation: false,
            step_callback: None,
            guardrails: vec![],
            output_parser: None,
        },
        tools1,
        build_llm()?,
        create_memory(&mem_cfg),
    );

    // -- Agent 2: Writer --------------------------------------------------
    let tools2: Vec<Box<dyn mangaba::core::tools::BaseTool + Send + Sync>> = vec![
        Box::new(EchoTool),
    ];
    let writer = Agent::new(
        AgentConfig {
            role: "Writer".into(),
            goal: "Turn research into clear, engaging text.".into(),
            backstory: "Skilled writer who produces polished output.".into(),
            llm_config: None,
            tools: vec![],
            memory_config: mem_cfg.clone(),
            max_iterations: 15,
            max_retry_on_error: 3,
            verbose: true,
            allow_delegation: false,
            step_callback: None,
            guardrails: vec![],
            output_parser: None,
        },
        tools2,
        build_llm()?,
        create_memory(&mem_cfg),
    );

    // -- Tasks: agents are resolved by role name from the crew --
    let task1 = Arc::new(Mutex::new(Task::new(
        TaskConfig {
            description: "Research the history of the Python programming language and list 3 key facts.".into(),
            expected_output: "A bullet list of 3 facts about Python's history.".into(),
            agent_id: Some("Researcher".into()),
            context_ids: vec![],
            tools: vec![],
            output_file: None,
            async_execution: false,
            human_input: false,
            guardrails: vec![],
            output_parser: None,
            retry_on_failure: 0,
        },
        None,
        vec![],
        vec![],
    )));

    let task2 = Arc::new(Mutex::new(Task::new(
        TaskConfig {
            description: "Write a short paragraph about Python's history using the research results.".into(),
            expected_output: "A concise paragraph about Python's history.".into(),
            agent_id: Some("Writer".into()),
            context_ids: vec![],
            tools: vec![],
            output_file: None,
            async_execution: false,
            human_input: false,
            guardrails: vec![],
            output_parser: None,
            retry_on_failure: 0,
        },
        None,
        vec![],
        vec![task1.clone()],
    )));

    // -- Crew with both agents -------------------------------------------
    let mut crew = Crew::new(
        vec![Box::new(researcher), Box::new(writer)],
        vec![task1, task2],
        ProcessType::Sequential,
        None,
        true,
    );

    let results: Vec<TaskOutput> = crew.kickoff().await?;

    println!("\n=== Crew results ===");
    for r in &results {
        println!("--- {} ---", r.description);
        println!("{}\n", r.result);
    }

    Ok(())
}
