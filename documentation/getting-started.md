# Getting Started

Guia rápido para começar a usar o Mangaba AI em Rust.

## Adicionando ao Projeto

```toml
[dependencies]
mangaba = { path = "../rust_mangaba" }
tokio = { version = "1.35", features = ["full"] }
anyhow = "1.0"
```

## Configuração Mínima

Crie um arquivo `.env` na raiz do seu projeto:

```bash
# Escolha o provedor: google, openai, anthropic, huggingface
LLM_PROVIDER=google

# Sua API key (a variável muda conforme o provedor)
GOOGLE_API_KEY=sua_chave_aqui

# Modelo (opcional — default por provedor)
MODEL_NAME=gemini-2.5-flash
```

Ou exporte as variáveis diretamente:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

## Quick Start — Agente Simples

```rust
use mangaba::core::config::Config;
use mangaba::core::llm::{create_llm_client, LLMClient};
use mangaba::core::types::{Message, LLMConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Carrega configuração do ambiente
    let cfg = Config::load()?;
    println!("{}", cfg); // Config(provider=google, model=gemini-2.5-flash, ...)

    // 2. Cria o cliente LLM
    let llm = create_llm_client(&cfg.to_llm_config())?;

    // 3. Envia uma mensagem
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the capital of France?"),
    ];

    let response = llm.chat(&messages, &[]).await?;
    println!("{}", response.text());

    Ok(())
}
```

## Quick Start — Agente com Ferramentas

```rust
use mangaba::core::config::Config;
use mangaba::core::llm::create_llm_client;
use mangaba::core::agent::Agent;
use mangaba::core::tools::CalculatorTool;
use mangaba::core::types::AgentConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = Config::load()?;
    let llm = create_llm_client(&cfg.to_llm_config())?;

    let mut agent = Agent::new(
        AgentConfig {
            role: "Math Assistant".into(),
            goal: "Solve math problems".into(),
            backstory: "Expert in mathematics".into(),
            max_iterations: 10,
            ..Default::default()
        },
        vec![Box::new(CalculatorTool)],  // ferramentas disponíveis
        llm,
        None,  // sem memória
    );

    let result = agent.execute_task("What is 2 + 3 * 4?", None).await?;
    println!("{result}");

    Ok(())
}
```

## Quick Start — Crew com Dois Agentes

```rust
use std::sync::Arc;
use tokio::sync::Mutex;
use mangaba::core::config::Config;
use mangaba::core::llm::create_llm_client;
use mangaba::core::agent::Agent;
use mangaba::core::task::Task;
use mangaba::core::crew::Crew;
use mangaba::core::types::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = Config::load()?;

    // Primeiro agente
    let llm1 = create_llm_client(&cfg.to_llm_config())?;
    let agent1 = Agent::new(
        AgentConfig {
            role: "Researcher".into(),
            goal: "Research topics".into(),
            backstory: "Expert researcher".into(),
            ..Default::default()
        },
        vec![],
        llm1,
        None,
    );

    // Segundo agente
    let llm2 = create_llm_client(&cfg.to_llm_config())?;
    let agent2 = Agent::new(
        AgentConfig {
            role: "Writer".into(),
            goal: "Write summaries".into(),
            backstory: "Expert writer".into(),
            ..Default::default()
        },
        vec![],
        llm2,
        None,
    );

    // Tarefas
    let task1 = Task::new(
        TaskConfig {
            description: "Research Rust async programming".into(),
            expected_output: "Research notes".into(),
            agent_id: Some("Researcher".into()),
            ..Default::default()
        },
        None, vec![], vec![],
    );

    let task2 = Task::new(
        TaskConfig {
            description: "Write a summary of the research".into(),
            expected_output: "Summary document".into(),
            agent_id: Some("Writer".into()),
            ..Default::default()
        },
        None, vec![], vec![],
    );

    // Crew sequencial
    let mut crew = Crew::new(
        vec![Box::new(agent1), Box::new(agent2)],
        vec![
            Arc::new(Mutex::new(task1)),
            Arc::new(Mutex::new(task2)),
        ],
        ProcessType::Sequential,
        None,  // manager_llm (necessário para Hierarchical)
        false, // verbose
    );

    let results = crew.kickoff().await?;
    for r in &results {
        println!("[{}] {}", r.agent_id, r.result.chars().take(100).collect::<String>());
    }

    Ok(())
}
```

## Testando

```bash
# Todos os testes
cargo test

# Testes de integração específicos
cargo test integration_test

# Ver documentação
cargo doc --no-deps --open
```

## Exemplos de Configuração por Provedor

### OpenAI

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4o-mini
```

### Anthropic Claude

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
MODEL_NAME=claude-3-haiku-20240307
```

### HuggingFace

```bash
LLM_PROVIDER=huggingface
HUGGINGFACE_API_KEY=hf_...
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
```
