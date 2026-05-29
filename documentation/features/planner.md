# Planner (Planejamento)

Decomposição de tarefas em planos executáveis via LLM.

## Visão Geral

O `TaskPlanner` usa um LLM para transformar uma descrição de tarefa em um
plano estruturado com passos ordenados, ferramentas sugeridas e dependências.

## PlanStep

```rust
pub struct PlanStep {
    pub step_number: usize,
    pub description: String,
    pub tool: Option<String>,
    pub expected_result: String,
    pub dependencies: Vec<usize>,
}
```

## ExecutionPlan

```rust
pub struct ExecutionPlan {
    pub goal: String,
    pub steps: Vec<PlanStep>,
}

impl ExecutionPlan {
    pub fn total_steps(&self) -> usize { self.steps.len() }
}
```

## TaskPlanner

### Criação

```rust
use mangaba::core::planner::TaskPlanner;

let planner = TaskPlanner::new(llm_client);
// ou com ferramentas disponíveis:
let planner = TaskPlanner::new(llm_client)
    .with_tools(&["search", "calculator", "file_writer"]);
```

### Planejamento

```rust
let plan = planner.plan("Create a research report on AI trends").await?;

println!("Goal: {}", plan.goal);
println!("Steps: {}", plan.total_steps());

for step in &plan.steps {
    println!("  {}. {} [tool: {:?}]", step.step_number, step.description, step.tool);
}
```

O planner envia um prompt para o LLM:

```
You are a task planning assistant.
Decompose the following task into concrete, sequential steps.
If tools are available, indicate which tool to use for each step.

Available tools: search, calculator, file_writer

Task: Create a research report on AI trends

JSON plan:
```

O LLM deve responder com um array JSON de `PlanStep`:

```json
[
    {
        "step_number": 1,
        "description": "Search for latest AI trends",
        "tool": "search",
        "expected_result": "List of current AI trends",
        "dependencies": []
    },
    {
        "step_number": 2,
        "description": "Analyze and summarize findings",
        "tool": null,
        "expected_result": "Summary of AI trends",
        "dependencies": [1]
    },
    {
        "step_number": 3,
        "description": "Write report to file",
        "tool": "file_writer",
        "expected_result": "Complete report saved",
        "dependencies": [2]
    }
]
```

### Parsing

O método `parse_steps()` extrai o array JSON da resposta do LLM:

```rust
fn parse_steps(raw: &str) -> Result<Vec<PlanStep>> {
    // Encontra [ ... ] no texto
    // Faz serde_json::from_str
}
```

## Ready Steps

O `ExecutionPlan` permite identificar passos que podem ser executados
(baseado em dependências resolvidas):

```rust
let plan = ExecutionPlan { goal: "...", steps: vec![...] };
let ready = plan.ready_steps(); // passos sem dependências ou com dependências resolvidas
```

## Exemplo Completo

```rust
use mangaba::core::planner::TaskPlanner;

let planner = TaskPlanner::new(llm)
    .with_tools(&["search", "calculator"]);

match planner.plan("Research quantum computing advances").await {
    Ok(plan) => {
        println!("Plano com {} passos:", plan.total_steps());
        for step in &plan.steps {
            let deps: Vec<String> = step.dependencies.iter()
                .map(|d| d.to_string()).collect();
            println!("  Passo {}: {} (deps: [{}])",
                step.step_number, step.description, deps.join(", "));
        }
    }
    Err(e) => eprintln!("Falha ao planejar: {e}"),
}
```
