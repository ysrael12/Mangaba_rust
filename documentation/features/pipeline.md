# Pipeline & Workflow

Composição de estágios de execução para workflows complexos.

## Visão Geral

O sistema de pipeline permite compor tarefas em estágios:

- **Stage**: execução sequencial de tarefas
- **ParallelStage**: execução concorrente via `tokio::spawn`
- **ConditionalStage**: escolha entre branches baseada em condição
- **Pipeline**: lista ordenada de estágios

## StageResult

```rust
pub struct StageResult {
    pub stage_name: String,
    pub outputs: Vec<TaskOutput>,
    pub duration: f64,
}
```

## PipelineResult

```rust
pub struct PipelineResult {
    pub stages: Vec<StageResult>,
    pub duration: f64,
}

impl PipelineResult {
    pub fn final_output(&self) -> String {
        // Último output bem-sucedido
    }
}
```

## Stage (Sequencial)

Executa tarefas uma após a outra.

```rust
use mangaba::core::pipeline::Stage;

let stage = Stage::new("research_phase", vec![task1, task2]);
let result: StageResult = stage.run(json!({})).await;

for output in &result.outputs {
    println!("[{}] {}", output.agent_id, output.description);
}
```

Se uma tarefa falha, o Stage continua com as demais (log de erro).

## ParallelStage

Executa tarefas concorrentemente.

```rust
use mangaba::core::pipeline::ParallelStage;

let stage = ParallelStage::new("parallel_phase", vec![task1, task2, task3]);
let result: StageResult = stage.run(json!({})).await;
// task1, task2, task3 executam em paralelo via tokio::spawn
```

## ConditionalStage

Escolhe entre branches baseado em uma condição.

```rust
use mangaba::core::pipeline::{ConditionalStage, Stage};

let true_branch = Stage::new("if_true", vec![task_a]);
let false_branch = Stage::new("if_false", vec![task_b]);

let stage = ConditionalStage::new(
    "check_condition",
    Box::new(|input: &Value| input.as_i64().unwrap_or(0) > 5),
    true_branch,
    Some(false_branch),  // None = não faz nada se condição for falsa
);

// Se input > 5, executa true_branch
let result = stage.run(json!(10)).await;

// Se input <= 5, executa false_branch (ou retorna vazio)
let result = stage.run(json!(2)).await;
```

## StageEnum

Enum que permite ao Pipeline conter diferentes tipos de estágio:

```rust
pub enum StageEnum {
    Sequential(Stage),
    Parallel(ParallelStage),
    Conditional(ConditionalStage),
}
```

## Pipeline

Orquestra uma sequência de estágios.

```rust
use mangaba::core::pipeline::{Pipeline, StageEnum, Stage};

let pipeline = Pipeline::new("full_workflow", vec![
    StageEnum::Sequential(Stage::new("research", vec![task1])),
    StageEnum::Parallel(ParallelStage::new("analysis", vec![task2, task3])),
    StageEnum::Sequential(Stage::new("report", vec![task4])),
]);

let result: PipelineResult = pipeline.run(None).await;
// pipeline.run(Some(json!({...}))) para passar input inicial

println!("Pipeline concluída em {:.2}s", result.duration);
println!("Resultado final: {}", result.final_output());
```

## Exemplo Completo

```rust
use std::sync::Arc;
use tokio::sync::Mutex;
use mangaba::core::pipeline::*;
use mangaba::core::task::Task;

let t1 = make_task("Research");
let t2 = make_task("Analyze data");
let t3 = make_task("Write report");

let pipeline = Pipeline::new("analysis_pipeline", vec![
    StageEnum::Sequential(Stage::new("research", vec![t1])),
    StageEnum::Parallel(ParallelStage::new("analysis", vec![t2, t3])),
]);

let result = pipeline.run(None).await;
assert_eq!(result.stages.len(), 2);
assert!(!result.final_output().is_empty());
```

## Eventos

O pipeline emite eventos `CrewStart` e `CrewEnd` (reusa o sistema de eventos
do módulo `events`).
