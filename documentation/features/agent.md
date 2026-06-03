# Agent

Agente autônomo com ReAct loop, memória, guardrails, ferramentas e delegação.

## Visão Geral

O `Agent` é a unidade central de execução. Ele combina:

- **LLM**: cliente para geração de texto e chamada de ferramentas
- **Tools**: conjunto de ferramentas disponíveis
- **Memória**: contexto opcional de curta/longa duração
- **Guardrails**: validação e transformação de saída
- **Output Parser**: parsing estruturado (ex: JSON)
- **Callbacks**: hooks para eventos do ciclo de vida

## Criação

```rust
use mangaba::core::agent::Agent;
use mangaba::core::types::AgentConfig;

let agent = Agent::new(
    AgentConfig {
        role: "Research Assistant".into(),
        goal: "Find and summarize information".into(),
        backstory: "I am an expert researcher".into(),
        max_iterations: 15,
        max_retry_on_error: 3,
        verbose: true,
        allow_delegation: false,
        ..Default::default()
    },
    vec![Box::new(CalculatorTool), Box::new(SerperSearchTool::from_env()?)],
    llm_client,
    Some(Box::new(ShortTermMemory::new(50))),
);
```

### AgentConfig

| Campo | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `role` | String | — | Nome/função do agente |
| `goal` | String | — | Objetivo principal |
| `backstory` | String | — | Contexto/história do agente |
| `max_iterations` | usize | 15 | Máximo de passos ReAct |
| `max_retry_on_error` | usize | 3 | Tentativas em caso de erro |
| `verbose` | bool | false | Log detalhado |
| `allow_delegation` | bool | false | Permite delegar para peers |
| `guardrails` | Vec\<String\> | [] | Nomes dos guardrails (`"no_op"`, `"length"`, `"profanity"`) |
| `output_parser` | Option\<String\> | None | `"json"` para parser JSON |

### System Prompt Builder

O agente constrói seu system prompt automaticamente:

```rust
fn build_system_prompt(&self) -> String {
    // Inclui: role, goal, backstory, tools, peers
}
```

Exemplo de saída:
```
You are: Research Assistant

Your goal is: Find and summarize information

Background: I am an expert researcher

Available tools:
- calculator: Evaluate mathematical expressions
- serper_search: Search the web
```

## Execução de Tarefas

```rust
let result = agent.execute_task(
    "What is the population of Brazil?",
    Some("Use recent data if possible"),
).await?;
```

O fluxo de execução:

1. Dispara evento `AgentStart`
2. Busca contexto relevante da memória
3. Constrói system prompt + user prompt
4. Executa ReActEngine (Thought → Action → Observation)
5. Aplica guardrails na resposta
6. Executa output parser (se configurado)
7. Salva na memória
8. Retorna o resultado; em caso de erro, reexecuta o ReAct até
   `max_retry_on_error` vezes

Se todas as tentativas falharem, `execute_task` **propaga o erro subjacente**
(rede, rate limit, parse, etc.) com o contexto `"Agent '<role>' failed after N
attempt(s)"` — a causa real é preservada na cadeia de erros do `anyhow`, não
mascarada. Veja [errors-retry](errors-retry.md) para inspecionar a cadeia via
`err.chain()`.

## ReAct Engine

O `ReActEngine` implementa o loop Thought → Action → Observation:

```
Iteração 1:
  LLM → Thought + ToolCall
  Executa ferramenta → Observation
Iteração 2:
  LLM → Thought + ToolCall
  Executa ferramenta → Observation
...
Iteração N:
  LLM → Final Answer (sem tool calls) → FIM
```

```rust
use mangaba::core::react::ReActEngine;

let engine = ReActEngine::new(
    llm.as_ref(),
    &tools,
    &callbacks,
    max_iterations,
    verbose,
);

let (response, steps) = engine.run(&mut messages).await?;
```

### ReActStep

```rust
pub struct ReActStep {
    pub step_number: usize,
    pub thought: Option<String>,
    pub action: Option<ToolCall>,
    pub observation: Option<String>,
    pub timestamp: String,
}
```

## Delegação

Agentes podem delegar tarefas para peers:

```rust
agent.add_peer(another_agent);
let result = agent.delegate("agent_id", "task description", None).await?;
```

A delegação requer `allow_delegation: true` no config.

## Agente com Memória

```rust
use mangaba::core::memory::{ShortTermMemory, LongTermMemory, EntityMemory};

// Memória de curta duração (in-memory, FIFO)
let mem = ShortTermMemory::new(100);

// Memória de longa duração (persiste em JSON)
let mem = LongTermMemory::new(Some("memory.json".into()), 1000);

// Memória de entidades (key-value)
let mem = EntityMemory::new();
```

Cada step do ReAct é automaticamente registrado na memória, incluindo
thoughts, actions, observations e o resultado final da tarefa.

## Agente com Output Parser

```rust
// No AgentConfig
output_parser: Some("json".into()),

// O parser JSON extrai conteúdo de blocos ```json ... ```
```

## Eventos

O agente emite eventos em pontos chave:

| Evento | Momento |
|--------|---------|
| `AgentStart` | Início da execução |
| `AgentEnd` | Tarefa concluída |
| `AgentError` | Erro após todas as tentativas |
