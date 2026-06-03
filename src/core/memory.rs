//! Memory stores for agent context.
//!
//! [`BaseMemory`] trait with `add` / `get_relevant` for keyword-based retrieval.
//! Implementations:
//! - [`ShortTermMemory`] — in-memory ring buffer (max_items)
//! - [`LongTermMemory`] — persists entries to a JSON file
//! - [`EntityMemory`] — key-value store for entity facts
//!
//! Use [`create_memory`] to build from a [`MemoryConfig`].

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use crate::core::events::{EventBus, Event, EventType};
use crate::core::types::MemoryConfig;

#[async_trait]
pub trait BaseMemory {
    async fn add(&mut self, entry: &str, metadata: Option<serde_json::Value>);
    async fn get_relevant(&self, query: &str, max_results: usize) -> String;
}

pub fn create_memory(config: &MemoryConfig) -> Option<Box<dyn BaseMemory + Send + Sync>> {
    match (config.short_term, config.long_term, config.entity) {
        (true, _, _) => Some(Box::new(ShortTermMemory::new(config.max_short_term_items))),
        (_, true, _) => Some(Box::new(LongTermMemory::new(config.storage_path.clone(), config.max_short_term_items))),
        (_, _, true) => Some(Box::new(EntityMemory::new())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn is_relevant(entry: &str, query: &str) -> bool {
    if query.is_empty() { return true; }
    let q = query.to_lowercase();
    let e = entry.to_lowercase();
    q.split_whitespace().any(|word| e.contains(word))
}

fn emit_search(source: &str, query: &str, count: usize) {
    EventBus::emit(Event::new(EventType::MemorySearch, source,
        serde_json::json!({"query": query, "results": count})));
}

fn emit_add(source: &str, entry: &str) {
    EventBus::emit(Event::new(EventType::MemoryAdd, source,
        serde_json::json!({"entry_preview": &entry.chars().take(100).collect::<String>()})));
}

// ---------------------------------------------------------------------------
// ShortTermMemory
// ---------------------------------------------------------------------------
pub struct ShortTermMemory {
    items: Vec<STEntry>,
    max_items: usize,
}

#[allow(dead_code)]
struct STEntry {
    text: String,
    metadata: Option<serde_json::Value>,
}

impl ShortTermMemory {
    pub fn new(max_items: usize) -> Self {
        Self { items: Vec::new(), max_items }
    }
}

#[async_trait]
impl BaseMemory for ShortTermMemory {
    async fn add(&mut self, entry: &str, metadata: Option<serde_json::Value>) {
        if self.items.len() >= self.max_items { self.items.remove(0); }
        self.items.push(STEntry { text: entry.to_string(), metadata });
        emit_add("short_term", entry);
    }

    async fn get_relevant(&self, query: &str, max_results: usize) -> String {
        let mut out = String::new();
        let mut count = 0;
        for entry in self.items.iter().rev() {
            if count >= max_results { break; }
            if is_relevant(&entry.text, query) {
                out.push_str(&format!("{}: {}\n", count + 1, entry.text));
                count += 1;
            }
        }
        emit_search("short_term", query, count);
        out
    }
}

// ---------------------------------------------------------------------------
// LongTermMemory — persists to JSON file
// ---------------------------------------------------------------------------
pub struct LongTermMemory {
    items: Vec<LTEntry>,
    max_items: usize,
    storage_path: Option<PathBuf>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct LTEntry {
    text: String,
    metadata: Option<serde_json::Value>,
}

impl LongTermMemory {
    pub fn new(storage_path: Option<String>, max_items: usize) -> Self {
        let path = storage_path.map(PathBuf::from);
        let items = path.as_ref().and_then(|p| Self::load(p).ok()).unwrap_or_default();
        Self { items, max_items, storage_path: path }
    }

    fn load(path: &PathBuf) -> std::io::Result<Vec<LTEntry>> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Persist to disk without blocking the async executor thread.
    async fn save(&self) {
        if let Some(ref path) = self.storage_path {
            if let Ok(data) = serde_json::to_string(&self.items) {
                let _ = tokio::fs::write(path, &data).await;
            }
        }
    }
}

#[async_trait]
impl BaseMemory for LongTermMemory {
    async fn add(&mut self, entry: &str, metadata: Option<serde_json::Value>) {
        if self.items.len() >= self.max_items { self.items.remove(0); }
        self.items.push(LTEntry { text: entry.to_string(), metadata });
        self.save().await;
        emit_add("long_term", entry);
    }

    async fn get_relevant(&self, query: &str, max_results: usize) -> String {
        let mut out = String::new();
        let mut count = 0;
        for entry in self.items.iter().rev() {
            if count >= max_results { break; }
            if is_relevant(&entry.text, query) {
                out.push_str(&format!("{}: {}\n", count + 1, entry.text));
                count += 1;
            }
        }
        emit_search("long_term", query, count);
        out
    }
}

// ---------------------------------------------------------------------------
// EntityMemory — key-value store for entity facts
// ---------------------------------------------------------------------------
pub struct EntityMemory {
    entities: HashMap<String, Vec<HashMap<String, serde_json::Value>>>,
}

impl EntityMemory {
    pub fn new() -> Self {
        Self { entities: HashMap::new() }
    }

    fn fmt_entity(name: &str, facts: &[HashMap<String, serde_json::Value>]) -> String {
        let mut lines = vec![format!("Entity: {}", name)];
        for fact in facts {
            for (k, v) in fact {
                lines.push(format!("  {}: {}", k, v));
            }
        }
        lines.join("\n")
    }

    fn entity_match(name: &str, facts: &[HashMap<String, serde_json::Value>], query: &str) -> bool {
        if query.is_empty() { return true; }
        let q = query.to_lowercase();
        if name.to_lowercase().contains(&q) { return true; }
        for fact in facts {
            for (k, v) in fact {
                if k.to_lowercase().contains(&q) { return true; }
                if let Some(s) = v.as_str() {
                    if s.to_lowercase().contains(&q) { return true; }
                }
            }
        }
        false
    }
}

#[async_trait]
impl BaseMemory for EntityMemory {
    async fn add(&mut self, entry: &str, metadata: Option<serde_json::Value>) {
        let attrs = match metadata {
            Some(val) => {
                if let Some(obj) = val.as_object() {
                    obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                } else {
                    vec![("value".to_string(), val)].into_iter().collect()
                }
            }
            None => HashMap::new(),
        };
        self.entities.entry(entry.to_string()).or_default().push(attrs);
        emit_add("entity", entry);
    }

    async fn get_relevant(&self, query: &str, max_results: usize) -> String {
        let mut out = String::new();
        let mut count = 0;
        let mut names: Vec<&String> = self.entities.keys().collect();
        names.sort();
        for name in names {
            if count >= max_results { break; }
            if let Some(facts) = self.entities.get(name) {
                if Self::entity_match(name, facts, query) {
                    out.push_str(&Self::fmt_entity(name, facts));
                    out.push('\n');
                    count += 1;
                }
            }
        }
        emit_search("entity", query, count);
        out
    }
}
