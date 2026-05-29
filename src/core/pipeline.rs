//! Workﬂow pipeline for composing task execution stages.
//!
//! - [`Stage`] — sequential task execution
//! - [`ParallelStage`] — tasks run concurrently via `tokio::spawn`
//! - [`ConditionalStage`] — chooses between branches based on a predicate
//! - [`Pipeline`] — ordered list of [`StageEnum`] variants
//!
//! Each stage returns a [`StageResult`]; the pipeline collects results into a
//! [`PipelineResult`] with total duration.

use std::sync::Arc;
use serde_json::Value;
use tokio::sync::Mutex;
use tokio::time::Instant;

use crate::core::events::{EventBus, Event, EventType};
use crate::core::task::Task;
use crate::core::types::TaskOutput;

// ---------------------------------------------------------------------------
// StageResult
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct StageResult {
    pub stage_name: String,
    pub outputs: Vec<TaskOutput>,
    pub duration: f64,
}

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------
pub struct Stage {
    pub name: String,
    pub tasks: Vec<Arc<Mutex<Task>>>,
}

impl Stage {
    pub fn new(name: &str, tasks: Vec<Arc<Mutex<Task>>>) -> Self {
        Self { name: name.to_string(), tasks }
    }

    pub async fn run(&self, _inputs: Value) -> StageResult {
        let start = Instant::now();
        let mut outputs = Vec::new();
        for task in &self.tasks {
            let mut task = task.lock().await;
            match task.execute().await {
                Ok(out) => outputs.push(out),
                Err(e) => {
                    log::error!("Task failed in stage '{}': {}", self.name, e);
                    outputs.push(TaskOutput::new(
                        &task.config.description,
                        &format!("Error: {e}"),
                        &task.state.agent_id.clone().unwrap_or_default(),
                        false,
                    ));
                }
            }
        }
        let duration = start.elapsed().as_secs_f64();
        StageResult { stage_name: self.name.clone(), outputs, duration }
    }
}

// ---------------------------------------------------------------------------
// ParallelStage
// ---------------------------------------------------------------------------
pub struct ParallelStage {
    pub name: String,
    pub tasks: Vec<Arc<Mutex<Task>>>,
}

impl ParallelStage {
    pub fn new(name: &str, tasks: Vec<Arc<Mutex<Task>>>) -> Self {
        Self { name: name.to_string(), tasks }
    }

    pub async fn run(&self, _inputs: Value) -> StageResult {
        let start = Instant::now();
        let mut handles = Vec::new();
        for task in &self.tasks {
            let task = Arc::clone(task);
            handles.push(tokio::spawn(async move {
                let mut t = task.lock().await;
                t.execute().await
            }));
        }
        let mut outputs = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(out)) => outputs.push(out),
                Ok(Err(e)) => {
                    log::error!("Parallel task error: {e}");
                    outputs.push(TaskOutput::new(
                        "parallel_task", &format!("Error: {e}"), "", false,
                    ));
                }
                Err(e) => {
                    log::error!("Parallel task join error: {e}");
                }
            }
        }
        let duration = start.elapsed().as_secs_f64();
        StageResult { stage_name: self.name.clone(), outputs, duration }
    }
}

// ---------------------------------------------------------------------------
// ConditionalStage
// ---------------------------------------------------------------------------
pub struct ConditionalStage {
    pub name: String,
    pub condition: Box<dyn Fn(&Value) -> bool + Send + Sync>,
    pub if_true: Stage,
    pub if_false: Option<Stage>,
}

impl ConditionalStage {
    pub fn new(
        name: &str,
        condition: Box<dyn Fn(&Value) -> bool + Send + Sync>,
        if_true: Stage,
        if_false: Option<Stage>,
    ) -> Self {
        Self { name: name.to_string(), condition, if_true, if_false }
    }

    pub async fn run(&self, inputs: Value) -> StageResult {
        if (self.condition)(&inputs) {
            self.if_true.run(inputs).await
        } else if let Some(ref alt) = self.if_false {
            alt.run(inputs).await
        } else {
            StageResult {
                stage_name: self.name.clone(),
                outputs: vec![],
                duration: 0.0,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineResult
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub stages: Vec<StageResult>,
    pub duration: f64,
}

impl PipelineResult {
    pub fn final_output(&self) -> String {
        for sr in self.stages.iter().rev() {
            if let Some(last) = sr.outputs.last() {
                if last.success {
                    return last.result.clone();
                }
            }
        }
        String::new()
    }
}

// ---------------------------------------------------------------------------
// StageEnum — allows Pipeline to hold Stage / Parallel / Conditional
// ---------------------------------------------------------------------------
pub enum StageEnum {
    Sequential(Stage),
    Parallel(ParallelStage),
    Conditional(ConditionalStage),
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------
pub struct Pipeline {
    pub name: String,
    pub stages: Vec<StageEnum>,
}

impl Pipeline {
    pub fn new(name: &str, stages: Vec<StageEnum>) -> Self {
        Self { name: name.to_string(), stages }
    }

    pub async fn run(&self, inputs: Option<Value>) -> PipelineResult {
        let inputs = inputs.unwrap_or(Value::Object(Default::default()));
        let start = Instant::now();
        let mut result = PipelineResult { stages: vec![], duration: 0.0 };

        EventBus::emit(Event::new(
            EventType::CrewStart, &self.name,
            serde_json::json!({"type": "pipeline"}),
        ));

        for stage in &self.stages {
            let sr = match stage {
                StageEnum::Sequential(s) => s.run(inputs.clone()).await,
                StageEnum::Parallel(p) => p.run(inputs.clone()).await,
                StageEnum::Conditional(c) => c.run(inputs.clone()).await,
            };
            result.stages.push(sr);
        }

        result.duration = start.elapsed().as_secs_f64();
        EventBus::emit(Event::new(
            EventType::CrewEnd, &self.name,
            serde_json::json!({"duration": result.duration}),
        ));
        result
    }
}
