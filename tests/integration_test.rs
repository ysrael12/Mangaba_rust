use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

use mangaba::core::llm::LLMClient;
use mangaba::core::tools::*;
use mangaba::core::types::*;
use mangaba::core::agent::Agent;
use mangaba::core::task::Task;
use mangaba::core::crew::Crew;
use mangaba::core::memory::{BaseMemory, ShortTermMemory, LongTermMemory, EntityMemory, create_memory};
use mangaba::core::guardrails::{Guardrail, LengthGuardrail, ProfanityGuardrail, CompositeGuardrail, GuardrailTool};
use mangaba::core::output_parsers::{OutputParser, JSONOutputParser, NoOpOutputParser};
use mangaba::core::pipeline::{Stage, ParallelStage, ConditionalStage, StageEnum, Pipeline};
use mangaba::core::planner::{PlanStep, ExecutionPlan, TaskPlanner};
use mangaba::core::embeddings::{cosine_similarity, InMemoryVectorStore, NoOpEmbedding};
use mangaba::core::rag::document::{Document, DocumentLoader, DocumentChunker};
use mangaba::core::events::{EventBus, Event, EventType};
use mangaba::core::prompt_templates::{PromptTemplate, SystemPromptBuilder};
use mangaba::core::errors::MangabaError;

// ---------------------------------------------------------------------------
// Mock LLM
// ---------------------------------------------------------------------------
struct MockLLM {
    responses: Arc<tokio::sync::Mutex<Vec<LLMResponse>>>,
}

impl MockLLM {
    fn new(responses: Vec<LLMResponse>) -> Self {
        Self { responses: Arc::new(tokio::sync::Mutex::new(responses)) }
    }
}

#[async_trait]
impl LLMClient for MockLLM {
    async fn chat(&self, _messages: &[Message], _tools: &[&dyn BaseTool]) -> Result<LLMResponse> {
        let mut queue = self.responses.lock().await;
        Ok(queue.remove(0))
    }
}

fn text_response(text: &str) -> LLMResponse {
    LLMResponse {
        content: Some(text.into()),
        tool_calls: vec![],
        usage: TokenUsage::default(),
        model: "mock".into(),
        finish_reason: FinishReason::Stop,
        raw: None,
    }
}

fn tool_call_response(tool_name: &str, args: Value) -> LLMResponse {
    LLMResponse {
        content: None,
        tool_calls: vec![ToolCall {
            id: "call_1".into(),
            tool_name: tool_name.into(),
            arguments: args.as_object()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default(),
        }],
        usage: TokenUsage::default(),
        model: "mock".into(),
        finish_reason: FinishReason::ToolCalls,
        raw: None,
    }
}

fn tmp_path(suffix: &str) -> String {
    std::env::temp_dir().join(format!("mangaba_test_{}_{}", std::process::id(), suffix))
        .to_str().unwrap().to_string()
}

fn make_agent_config(role: &str, goal: &str, backstory: &str) -> AgentConfig {
    AgentConfig {
        role: role.into(),
        goal: goal.into(),
        backstory: backstory.into(),
        llm_config: None,
        tools: vec![],
        memory_config: MemoryConfig {
            short_term: false,
            long_term: false,
            entity: false,
            max_short_term_items: 50,
            storage_path: None,
        },
        max_iterations: 5,
        max_retry_on_error: 1,
        verbose: false,
        allow_delegation: false,
        step_callback: None,
        guardrails: vec![],
        output_parser: None,
    }
}

fn make_task_config(description: &str, agent_id: Option<&str>) -> TaskConfig {
    TaskConfig {
        description: description.into(),
        expected_output: "Result".into(),
        agent_id: agent_id.map(|s| s.into()),
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

// ---------------------------------------------------------------------------
// Agent + Task + Crew integration tests
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_crew_sequential() {
    let llm1 = MockLLM::new(vec![text_response("Result of task 1")]);
    let llm2 = MockLLM::new(vec![text_response("Result of task 2")]);

    let agent1 = Agent::new(
        make_agent_config("Worker 1", "Execute task 1", "First worker"),
        vec![],
        Box::new(llm1),
        None,
    );

    let agent2 = Agent::new(
        make_agent_config("Worker 2", "Execute task 2", "Second worker"),
        vec![],
        Box::new(llm2),
        None,
    );

    let task1 = Task::new(make_task_config("Task 1", Some("Worker 1")), None, vec![], vec![]);
    let task2 = Task::new(make_task_config("Task 2", Some("Worker 2")), None, vec![], vec![]);

    let mut crew = Crew::new(
        vec![Box::new(agent1), Box::new(agent2)],
        vec![Arc::new(Mutex::new(task1)), Arc::new(Mutex::new(task2))],
        ProcessType::Sequential,
        None,
        false,
    );

    let results = crew.kickoff().await.unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].success);
    assert!(results[1].success);
    assert_eq!(results[0].result, "Result of task 1");
    assert_eq!(results[1].result, "Result of task 2");
}

#[tokio::test]
async fn test_agent_with_tools() {
    let llm = MockLLM::new(vec![
        tool_call_response("calculator", json!({"expression": "1 + 2"})),
        text_response("The answer is 3."),
    ]);

    let mut agent = Agent::new(
        make_agent_config("Calculator", "Solve math", "Math whiz"),
        vec![Box::new(CalculatorTool)],
        Box::new(llm),
        None,
    );

    let result = agent.execute_task("What is 1+2?", None).await.unwrap();
    assert_eq!(result, "The answer is 3.");
}

#[tokio::test]
async fn test_task_with_context() {
    let llm1 = MockLLM::new(vec![text_response("First result")]);

    let agent1 = Agent::new(
        make_agent_config("Agent 1", "Do first task", "First"),
        vec![],
        Box::new(llm1),
        None,
    );

    let task1 = Task::new(make_task_config("First task", None), Some(Box::new(agent1)), vec![], vec![]);
    let task1_arc = Arc::new(Mutex::new(task1));

    // Execute task1 first so it populates output
    {
        let mut t1 = task1_arc.lock().await;
        t1.execute().await.unwrap();
    }

    let llm2 = MockLLM::new(vec![text_response("Second result with context")]);
    let agent2 = Agent::new(
        make_agent_config("Agent 2", "Do second task", "Second"),
        vec![],
        Box::new(llm2),
        None,
    );

    let mut task2 = Task::new(
        make_task_config("Second task", None),
        Some(Box::new(agent2)),
        vec![],
        vec![task1_arc],
    );

    let output = task2.execute().await.unwrap();
    assert!(output.success);
    assert!(output.result.contains("Second result"));
}

// ---------------------------------------------------------------------------
// Memory tests
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_short_term_memory() {
    let mut mem = ShortTermMemory::new(3);
    mem.add("First entry", None).await;
    mem.add("Second entry", None).await;
    mem.add("Third entry", None).await;
    mem.add("Fourth entry", None).await; // should evict first

    let result = mem.get_relevant("entry", 10).await;
    assert!(!result.contains("First"));
    assert!(result.contains("Second"));
    assert!(result.contains("Fourth"));
}

#[tokio::test]
async fn test_long_term_memory() {
    let tmp = tmp_path("ltm");
    let _ = std::fs::remove_file(&tmp);
    let path = Some(tmp.clone());
    let mut mem = LongTermMemory::new(path.clone(), 100);

    mem.add("Persistent memory test", None).await;
    let result = mem.get_relevant("test", 10).await;
    assert!(result.contains("Persistent memory test"));

    // Reload from file
    let mem2 = LongTermMemory::new(path.clone(), 100);
    let result2 = mem2.get_relevant("test", 10).await;
    assert!(result2.contains("Persistent memory test"));
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn test_entity_memory() {
    let mut mem = EntityMemory::new();
    mem.add("Alice", Some(json!({"age": 30, "city": "NYC"}))).await;
    mem.add("Bob", Some(json!({"age": 25}))).await;

    let result = mem.get_relevant("Alice", 10).await;
    assert!(result.contains("Alice"));

    let result_all = mem.get_relevant("", 10).await;
    assert!(result_all.contains("Alice"));
    assert!(result_all.contains("Bob"));
}

#[tokio::test]
async fn test_create_memory() {
    let config = MemoryConfig {
        short_term: true,
        max_short_term_items: 10,
        long_term: false,
        entity: false,
        storage_path: None,
    };
    let mem = create_memory(&config);
    assert!(mem.is_some());
}

// ---------------------------------------------------------------------------
// Guardrails tests
// ---------------------------------------------------------------------------
#[test]
fn test_length_guardrail() {
    let g = LengthGuardrail { max_len: 5, truncate: true };
    assert_eq!(g.validate("hello world"), "hello");
    assert_eq!(g.validate("hi"), "hi");
}

#[test]
fn test_profanity_guardrail() {
    let g = ProfanityGuardrail::new(r"\b(darn|heck)\b", "****");
    assert_eq!(g.validate("what the heck"), "what the ****");
    assert_eq!(g.validate("clean text"), "clean text");
}

#[test]
fn test_composite_guardrail() {
    let g = CompositeGuardrail {
        guardrails: vec![
            Box::new(ProfanityGuardrail::new(r"\bdarn\b", "****")),
            Box::new(LengthGuardrail { max_len: 20, truncate: true }),
        ],
    };
    let result = g.validate("this is darn long text here");
    assert_eq!(result, "this is **** long te");
}

#[tokio::test]
async fn test_guardrail_tool() {
    let g = GuardrailTool::new("profanity", Box::new(ProfanityGuardrail::new(r"\bdarn\b", "****")));
    let result = g.call(json!({"text": "that darn cat"})).await.unwrap();
    assert!(result.success);
    let output = result.output.unwrap();
    assert_eq!(output["validated_text"], "that **** cat");
    assert!(output["was_modified"].as_bool().unwrap());
}

// ---------------------------------------------------------------------------
// Output parsers tests
// ---------------------------------------------------------------------------
#[test]
fn test_json_output_parser() {
    let parser = JSONOutputParser;
    let result = parser.parse("```json\n{\"key\": \"value\"}\n```").unwrap();
    assert_eq!(result, "{\"key\": \"value\"}");

    let result2 = parser.parse("just text");
    assert!(result2.is_err());
}

#[test]
fn test_noop_output_parser() {
    let parser = NoOpOutputParser;
    let result = parser.parse("any text").unwrap();
    assert_eq!(result, "any text");
}

// ---------------------------------------------------------------------------
// Pipeline tests (pipeline operates on Task vecs)
// ---------------------------------------------------------------------------
fn make_test_task(description: &str) -> Arc<Mutex<Task>> {
    let llm = MockLLM::new(vec![text_response(&format!("Result: {description}"))]);
    let agent = Agent::new(
        make_agent_config("Worker", "Execute", "Bot"),
        vec![],
        Box::new(llm),
        None,
    );
    let task = Task::new(make_task_config(description, None), Some(Box::new(agent)), vec![], vec![]);
    Arc::new(Mutex::new(task))
}

#[tokio::test]
async fn test_stage() {
    let task = make_test_task("Stage task");
    let stage = Stage::new("stage1", vec![task]);
    let result = stage.run(json!({})).await;
    assert_eq!(result.stage_name, "stage1");
    assert_eq!(result.outputs.len(), 1);
    assert!(result.outputs[0].success);
}

#[tokio::test]
async fn test_parallel_stage() {
    let t1 = make_test_task("Parallel task 1");
    let t2 = make_test_task("Parallel task 2");
    let stage = ParallelStage::new("parallel1", vec![t1, t2]);
    let result = stage.run(json!({})).await;
    assert_eq!(result.outputs.len(), 2);
}

#[tokio::test]
async fn test_conditional_stage() {
    let true_task = make_test_task("True branch");
    let true_stage = Stage::new("if_true", vec![true_task]);
    let cond = ConditionalStage::new(
        "cond1",
        Box::new(|input: &Value| input.as_i64().unwrap_or(0) > 5),
        true_stage,
        None,
    );

    let result = cond.run(json!(10)).await;
    assert_eq!(result.outputs.len(), 1);

    let result2 = cond.run(json!(2)).await;
    assert_eq!(result2.outputs.len(), 0); // no false branch
}

#[tokio::test]
async fn test_pipeline() {
    let t1 = make_test_task("Pipeline task 1");
    let t2 = make_test_task("Pipeline task 2");
    let s1 = Stage::new("first", vec![t1]);
    let s2 = Stage::new("second", vec![t2]);

    let pipeline = Pipeline::new("pipe1", vec![
        StageEnum::Sequential(s1),
        StageEnum::Sequential(s2),
    ]);

    let result = pipeline.run(None).await;
    assert_eq!(result.stages.len(), 2);
    assert!(result.final_output().contains("Result: Pipeline task 2"));
}

// ---------------------------------------------------------------------------
// Planner tests
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_task_planner_plan() {
    let llm = MockLLM::new(vec![
        text_response(r#"[{"step_number":1,"description":"Step one","tool":null,"expected_result":"Done","dependencies":[]}]"#),
    ]);
    let planner = TaskPlanner::new(Box::new(llm));
    let plan = planner.plan("Do something").await.unwrap();
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].description, "Step one");
}

#[test]
fn test_execution_plan() {
    let plan = ExecutionPlan {
        goal: "Test".into(),
        steps: vec![
            PlanStep { step_number: 1, description: "First".into(), tool: None, expected_result: "".into(), dependencies: vec![] },
            PlanStep { step_number: 2, description: "Second".into(), tool: None, expected_result: "".into(), dependencies: vec![1] },
        ],
    };
    assert_eq!(plan.total_steps(), 2);
}

// ---------------------------------------------------------------------------
// Embeddings tests
// ---------------------------------------------------------------------------
#[test]
fn test_cosine_similarity() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

    let c = vec![0.0, 1.0, 0.0];
    assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);

    let empty: Vec<f32> = vec![];
    assert!((cosine_similarity(&empty, &vec![1.0]) - 0.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_vector_store() {
    let emb = Arc::new(NoOpEmbedding::new(3));
    let store = InMemoryVectorStore::new(emb.clone());

    store.add("doc1", None).await.unwrap();
    store.add("doc2", None).await.unwrap();

    // NoOpEmbedding returns all-zero vectors → all cosine similarities are 0.0 → filtered out
    let results = store.search("text1", 2).await.unwrap();
    assert_eq!(results.len(), 0);
    assert_eq!(store.len().await, 2);
}

// ---------------------------------------------------------------------------
// RAG / Document tests
// ---------------------------------------------------------------------------
#[test]
fn test_document_chunker_chars() {
    let chunker = DocumentChunker::new(10, 0);
    let text = "A. B. C. D. E. F. G. H. I. J. K.";
    let chunks = chunker.chunk_text(text);
    assert!(chunks.len() > 1);
}

#[test]
fn test_document_loader_txt() {
    let path = tmp_path("doc_txt.txt");
    std::fs::write(&path, b"Hello world").unwrap();
    let docs = DocumentLoader::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].text, "Hello world");
}

#[test]
fn test_document_loader_csv() {
    let path = tmp_path("doc_csv.csv");
    std::fs::write(&path, b"name,age\nAlice,30\nBob,25").unwrap();
    let docs = DocumentLoader::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(docs.len(), 2);
}

#[test]
fn test_document_loader_md() {
    let path = tmp_path("doc_md.md");
    std::fs::write(&path, b"# Title\nContent").unwrap();
    let docs = DocumentLoader::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(docs.len(), 1);
}

#[test]
fn test_document_metadata() {
    let doc = Document::new("content", "test");
    assert_eq!(doc.text, "content");
    assert_eq!(doc.source, "test");
}

// ---------------------------------------------------------------------------
// Events tests
// ---------------------------------------------------------------------------
#[test]
fn test_event_bus() {
    let last_event = Arc::new(std::sync::Mutex::new(None::<EventType>));
    let last_clone = last_event.clone();

    EventBus::subscribe(Box::new(move |event: &Event| {
        *last_clone.lock().unwrap() = Some(event.event_type.clone());
    }));

    EventBus::emit(Event::new(EventType::Custom("ping".into()), "test", json!({})));
    let captured = last_event.lock().unwrap().take();
    assert!(captured.is_some());
}

// ---------------------------------------------------------------------------
// Prompt templates tests
// ---------------------------------------------------------------------------
#[test]
fn test_prompt_template() {
    let tmpl = PromptTemplate::new("Hello {name}, you are {role}!");
    assert_eq!(tmpl.variables.len(), 2);

    let mut values = std::collections::HashMap::new();
    values.insert("name", "Alice");
    values.insert("role", "admin");
    let result = tmpl.render(&values);
    assert_eq!(result, "Hello Alice, you are admin!");
}

#[test]
fn test_system_prompt_builder() {
    let prompt = SystemPromptBuilder::new()
        .role("Bot")
        .goal("Help")
        .backstory("Created to assist")
        .build();

    assert!(prompt.contains("Bot"));
    assert!(prompt.contains("Help"));
}

// ---------------------------------------------------------------------------
// Errors tests
// ---------------------------------------------------------------------------
#[test]
fn test_mangaba_error() {
    let err = MangabaError::RateLimit { retry_after: 30, detail: "too many requests".into() };
    let msg = err.to_string();
    assert!(msg.contains("30"));
    assert!(err.is_retryable());

    let err2 = MangabaError::LLM("API error".into());
    assert!(!err2.is_retryable());

    let anyhow_err = err.to_anyhow();
    assert!(anyhow_err.to_string().contains("Rate limit"));
}

// ---------------------------------------------------------------------------
// ReAct with guardrails test
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_react_with_guardrails() {
    use mangaba::core::guardrails::apply_guardrails;
    let guardrails: Vec<Box<dyn Guardrail + Send + Sync>> = vec![
        Box::new(LengthGuardrail { max_len: 6, truncate: true }),
    ];

    let result = apply_guardrails(&guardrails, "test", "Hello there!").await;
    assert_eq!(result, "Hello ");
}

// ---------------------------------------------------------------------------
// Tool edge cases
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_calculator_invalid_expression() {
    let tool = CalculatorTool;
    let result = tool.call(json!({"expression": "not math"})).await.unwrap();
    assert!(!result.success);
}

#[tokio::test]
async fn test_file_reader_success() {
    let path = tmp_path("reader");
    std::fs::write(&path, b"file content").unwrap();
    let tool = FileReaderTool;
    let result = tool.call(json!({"file_path": &path})).await.unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(result.success);
    assert_eq!(result.output.unwrap()["content"], "file content");
}

#[tokio::test]
async fn test_file_writer() {
    let path = tmp_path("writer");
    let tool = FileWriterTool;
    let result = tool.call(json!({"file_path": &path, "content": "written content"})).await.unwrap();
    assert!(result.success);
    let readback = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(readback, "written content");
}

#[tokio::test]
async fn test_text_splitter_with_overlap() {
    let tool = TextSplitterTool;
    let text = "A. B. C. D. E. F. G. H. I. J. K.";
    let result = tool.call(json!({"text": text, "chunk_size": 15, "chunk_overlap": 5})).await.unwrap();
    assert!(result.success);
    let binding = result.output.unwrap();
    let chunks = binding["chunks"].as_array().unwrap();
    assert!(chunks.len() > 1);
}


