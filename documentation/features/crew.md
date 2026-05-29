# Crew & Task

Orquestração multi-agente e sistema de tarefas.

## Crew

O `Crew` gerencia um conjunto de agentes e tarefas, coordenando a execução.

### Criação

```rust
use mangaba::core::crew::Crew;
use mangaba::core::types::ProcessType;

let crew = Crew::new(
    vec![Box::new(agent1), Box::new(agent2)],
    vec![
        Arc::new(Mutex::new(task1)),
        Arc::new(Mutex::new(task2)),
    ],
    ProcessType::Sequential,
    None,                 // manager_llm — necessário para Hierarchical
    false,                // verbose
);
```

### Processos

**Sequential**: tarefas executam uma após a outra. O output de cada tarefa
é passado como contexto para a próxima via `previous_output`.

```
Task 1 → output → Task 2 → output → Task 3 → resultado final
```

**Hierarchical**: um manager LLM designa cada tarefa ao agente mais adequado.

```
Manager LLM escolhe agente → Task 1 → resultado
Manager LLM escolhe agente → Task 2 → resultado
...
```

```rust
let manager_llm = create_llm_client(&config)?;

let mut crew = Crew::new(
    agents,
    tasks,
    ProcessType::Hierarchical,
    Some(manager_llm),  // obrigatório para hierárquico
    false,
);
```

### Execução

```rust
let results: Vec<TaskOutput> = crew.kickoff().await?;
for output in &results {
    println!("Task: {}", output.description);
    println!("Result: {}", output.result);
    println!("Success: {}", output.success);
}
```

### Resolução de Agentes

O crew busca agentes pelo `role` ou `agent_id`:

```rust
fn resolve_agent(&mut self, agent_id_or_role: Option<&str>) -> Result<&mut Box<Agent>>
```

- Se `agent_id` for `Some(id)`, busca na lista de agentes por ID ou role
- Se `None`, usa o primeiro agente disponível

## Task

A `Task` representa uma unidade de trabalho atribuída a um agente.

### Criação

```rust
use mangaba::core::task::Task;
use mangaba::core::types::TaskConfig;
use std::sync::Arc;
use tokio::sync::Mutex;

let task = Task::new(
    TaskConfig {
        description: "Research topic X".into(),
        expected_output: "A comprehensive summary".into(),
        agent_id: Some("Researcher".into()),
        context_ids: vec![],
        output_file: Some("output.txt".into()),
        async_execution: false,
        human_input: false,
        guardrails: vec![],
        output_parser: None,
        retry_on_failure: 2,
        ..Default::default()
    },
    Some(Box::new(agent)),  // agente dedicado (ou None se o crew resolve)
    vec![],                   // ferramentas extras
    vec![],                   // context_tasks (dependências)
);
```

### TaskConfig

| Campo | Tipo | Default | Descrição |
|-------|------|---------|-----------|
| `description` | String | — | Descrição da tarefa |
| `expected_output` | String | — | Descrição do resultado esperado |
| `agent_id` | Option\<String\> | None | Role ou ID do agente (para resolução via crew) |
| `context_ids` | Vec\<String\> | [] | IDs de tarefas de contexto |
| `output_file` | Option\<String\> | None | Arquivo para salvar o resultado |
| `async_execution` | bool | false | Se executa assíncrono |
| `human_input` | bool | false | Requer input humano |
| `retry_on_failure` | usize | 0 | Tentativas extras em caso de falha |

### Context Chaining

Tarefas podem depender de outputs de tarefas anteriores:

```rust
let task1 = Arc::new(Mutex::new(Task::new(..., agent1, ...)));

// Executa task1 primeiro
task1.lock().await.execute().await?;

// task2 usa o output de task1 como contexto
let task2 = Task::new(
    TaskConfig { description: "Summarize previous result".into(), ... },
    Some(Box::new(agent2)),
    vec![],
    vec![task1],  // context_tasks
);
task2.execute().await?;
```

### Execução Individual

```rust
let output: TaskOutput = task.execute().await?;
```

O método `execute()`:
1. Emite evento `TaskStart`
2. Constrói contexto a partir de `context_tasks`
3. Chama `agent.execute_task(description, context)`
4. Salva output em arquivo (se configurado)
5. Emite evento `TaskEnd`
6. Retorna `TaskOutput`

### TaskOutput

```rust
pub struct TaskOutput {
    pub description: String,
    pub result: String,
    pub agent_id: String,
    pub success: bool,
    pub timestamp: String,
}
```

## Eventos

| Evento | Origem |
|--------|--------|
| `CrewStart` / `CrewEnd` / `CrewError` | Crew |
| `TaskStart` / `TaskEnd` / `TaskError` | Task |
