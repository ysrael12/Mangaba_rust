//! Global [`EventBus`] for decoupled subscribe/emit communication.
//!
//! Listeners are registered via `EventBus::subscribe()` and fire synchronously
//! on every `EventBus::emit()`. Events carry a [`EventType`], source ID, and
//! arbitrary JSON data. Used internally by all modules (agent, crew, ReAct,
//! tools, LLM, memory, guardrails).

use serde_json::Value;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventType {
    AgentStart,
    AgentEnd,
    AgentError,
    LLMStart,
    LLMEnd,
    LLMError,
    LLMRetry,
    LLMStreamChunk,
    ToolStart,
    ToolEnd,
    ToolError,
    ReActStep,
    ReActThought,
    ReActAction,
    ReActObservation,
    TaskStart,
    TaskEnd,
    TaskError,
    CrewStart,
    CrewEnd,
    CrewError,
    MemoryAdd,
    MemorySearch,
    GuardrailPass,
    GuardrailFail,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct Event {
    pub event_type: EventType,
    pub source_id: String,
    pub data: Value,
}

impl Event {
    pub fn new(event_type: EventType, source_id: &str, data: Value) -> Self {
        Self { event_type, source_id: source_id.to_string(), data }
    }
}

pub type Listener = Box<dyn Fn(&Event) + Send + Sync + 'static>;

static EVENT_BUS: Lazy<Arc<Mutex<Vec<Listener>>>> = Lazy::new(|| Arc::new(Mutex::new(Vec::new())));

pub struct EventBus;

impl EventBus {
    pub fn emit(event: Event) {
        let listeners = EVENT_BUS.lock().unwrap();
        for listener in listeners.iter() {
            listener(&event);
        }
    }

    pub fn subscribe(listener: Listener) {
        EVENT_BUS.lock().unwrap().push(listener);
    }
}
