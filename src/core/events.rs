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

/// Internally listeners are stored behind `Arc` so `emit` can clone the handle
/// list out of the lock and invoke them *without* holding the global mutex.
type ListenerHandle = Arc<dyn Fn(&Event) + Send + Sync + 'static>;

static EVENT_BUS: Lazy<Mutex<Vec<ListenerHandle>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Recover the guard even if a previous listener panicked and poisoned the lock.
/// A poisoned `EventBus` must never take down the whole framework.
fn lock_bus() -> std::sync::MutexGuard<'static, Vec<ListenerHandle>> {
    EVENT_BUS.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub struct EventBus;

impl EventBus {
    /// Emit an event to all listeners.
    ///
    /// The listener handles are cloned out of the lock first, so the global
    /// mutex is released *before* any listener runs. This makes `emit`:
    /// - **reentrant-safe**: a listener may call `emit`/`subscribe` again,
    /// - **panic-safe**: a panicking listener poisons nothing fatal,
    /// - **non-blocking** for other emitters while a slow listener runs.
    pub fn emit(event: Event) {
        let listeners: Vec<ListenerHandle> = {
            let guard = lock_bus();
            guard.clone()
        };
        for listener in &listeners {
            listener(&event);
        }
    }

    pub fn subscribe(listener: Listener) {
        lock_bus().push(Arc::from(listener));
    }

    /// Remove all listeners. Primarily useful to isolate tests from each other,
    /// since the bus is a process-global singleton.
    pub fn clear() {
        lock_bus().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn emit_does_not_deadlock_on_reentrant_subscribe() {
        EventBus::clear();
        let hits = Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        // A listener that re-enters the bus by subscribing again must not deadlock.
        EventBus::subscribe(Box::new(move |_ev| {
            h.fetch_add(1, Ordering::SeqCst);
            EventBus::subscribe(Box::new(|_| {}));
        }));
        EventBus::emit(Event::new(EventType::Custom("x".into()), "t", serde_json::json!({})));
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        EventBus::clear();
    }

    #[test]
    fn poisoned_lock_is_recovered() {
        EventBus::clear();
        // Force a panic inside the lock to poison it.
        let _ = std::panic::catch_unwind(|| {
            let _g = lock_bus();
            panic!("poison the bus");
        });
        // Subsequent operations must still work rather than panicking on unwrap.
        EventBus::subscribe(Box::new(|_| {}));
        EventBus::emit(Event::new(EventType::Custom("y".into()), "t", serde_json::json!({})));
        EventBus::clear();
    }
}
