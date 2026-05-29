//! Model Context Protocol — hierarchical context management.
//!
//! [`MCPContext`] holds a context entry with tags, priority, and source.
//! [`MCPSession`] manages a collection of contexts with addition, update,
//! and relevance-based retrieval. [`MCPProtocol`] wraps sessions with
//! scoring and keyword/tag ﬁltering via [`get_relevant_contexts`](MCPProtocol::get_relevant_contexts).

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// ContextType
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextType {
    Conversation,
    Task,
    Knowledge,
    Memory,
    System,
    UserProfile,
}

// ---------------------------------------------------------------------------
// ContextPriority
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ContextPriority {
    #[serde(rename = "1")]
    Low = 1,
    #[serde(rename = "2")]
    Medium = 2,
    #[serde(rename = "3")]
    High = 3,
    #[serde(rename = "4")]
    Critical = 4,
}

impl Default for ContextPriority {
    fn default() -> Self {
        ContextPriority::Medium
    }
}

// ---------------------------------------------------------------------------
// MCPContext
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPContext {
    pub id: String,
    pub context_type: ContextType,
    pub content: Value,
    pub priority: ContextPriority,
    pub created_at: String,
    pub updated_at: String,
    pub expires_at: Option<String>,
    pub tags: Vec<String>,
    pub metadata: Value,
    pub parent_id: Option<String>,
    pub children_ids: Vec<String>,
}

impl MCPContext {
    pub fn create(
        context_type: ContextType,
        content: Value,
        priority: ContextPriority,
        expires_in_hours: Option<i64>,
        tags: Vec<String>,
        parent_id: Option<&str>,
    ) -> Self {
        let now = Utc::now();
        let expires_at = expires_in_hours.map(|h| (now + Duration::hours(h)).to_rfc3339());
        Self {
            id: Uuid::new_v4().to_string(),
            context_type,
            content,
            priority,
            created_at: now.to_rfc3339(),
            updated_at: now.to_rfc3339(),
            expires_at,
            tags,
            metadata: serde_json::json!({}),
            parent_id: parent_id.map(String::from),
            children_ids: Vec::new(),
        }
    }

    pub fn update_content(&mut self, new_content: Value) {
        if let (Some(obj), Some(new_obj)) = (self.content.as_object_mut(), new_content.as_object()) {
            for (k, v) in new_obj {
                obj.insert(k.clone(), v.clone());
            }
        } else {
            self.content = new_content;
        }
        self.updated_at = Utc::now().to_rfc3339();
    }

    pub fn add_tag(&mut self, tag: &str) {
        if !self.tags.contains(&tag.to_string()) {
            self.tags.push(tag.to_string());
            self.updated_at = Utc::now().to_rfc3339();
        }
    }

    pub fn is_expired(&self) -> bool {
        self.expires_at.as_ref().map_or(false, |exp| {
            DateTime::parse_from_rfc3339(exp)
                .map(|dt| dt.to_utc() < Utc::now())
                .unwrap_or(false)
        })
    }

    pub fn content_hash(&self) -> String {
        let content_str = serde_json::to_string(&self.content).unwrap_or_default();
        format!("{:x}", content_digest(&content_str))
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }
}

fn content_digest(input: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    input.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// MCPSession
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPSession {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub updated_at: String,
    pub context_ids: Vec<String>,
    pub metadata: Value,
}

impl MCPSession {
    pub fn create(name: &str) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            created_at: now.clone(),
            updated_at: now,
            context_ids: Vec::new(),
            metadata: serde_json::json!({}),
        }
    }
}

// ---------------------------------------------------------------------------
// MCPProtocol
// ---------------------------------------------------------------------------
pub struct MCPProtocol {
    contexts: Mutex<HashMap<String, MCPContext>>,
    sessions: Mutex<HashMap<String, MCPSession>>,
    context_index: Mutex<HashMap<String, Vec<String>>>,
    max_contexts: usize,
}

impl Default for MCPProtocol {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl MCPProtocol {
    pub fn new(max_contexts: usize) -> Self {
        Self {
            contexts: Mutex::new(HashMap::new()),
            sessions: Mutex::new(HashMap::new()),
            context_index: Mutex::new(HashMap::new()),
            max_contexts,
        }
    }

    pub fn add_context(&self, context: MCPContext, session_id: Option<&str>) -> String {
        let mut contexts = self.contexts.lock().unwrap();
        let mut sessions = self.sessions.lock().unwrap();
        let mut index = self.context_index.lock().unwrap();

        self.cleanup_expired_inner(&mut contexts, &mut sessions, &mut index);

        if contexts.len() >= self.max_contexts {
            Self::remove_oldest_inner(&mut contexts, &mut sessions, &mut index, 100);
        }

        let ctx_id = context.id.clone();

        for tag in &context.tags {
            index.entry(tag.clone()).or_default().push(ctx_id.clone());
        }

        if let Some(sid) = session_id {
            if let Some(session) = sessions.get_mut(sid) {
                session.context_ids.push(ctx_id.clone());
                session.updated_at = Utc::now().to_rfc3339();
            }
        }

        contexts.insert(ctx_id.clone(), context);
        ctx_id
    }

    pub fn get_context(&self, context_id: &str) -> Option<MCPContext> {
        let mut contexts = self.contexts.lock().unwrap();
        let sessions = self.sessions.lock().unwrap();
        let mut index = self.context_index.lock().unwrap();

        let expired = contexts.get(context_id).map(|c| c.is_expired()).unwrap_or(false);
        if expired {
            Self::remove_inner(context_id, &mut contexts, &sessions, &mut index);
            return None;
        }

        contexts.get(context_id).cloned()
    }

    pub fn update_context(&self, context_id: &str, new_content: Value) -> bool {
        let mut contexts = self.contexts.lock().unwrap();
        if let Some(ctx) = contexts.get_mut(context_id) {
            if ctx.is_expired() {
                return false;
            }
            ctx.update_content(new_content);
            true
        } else {
            false
        }
    }

    pub fn remove_context(&self, context_id: &str) -> bool {
        let mut contexts = self.contexts.lock().unwrap();
        let sessions = self.sessions.lock().unwrap();
        let mut index = self.context_index.lock().unwrap();
        Self::remove_inner(context_id, &mut contexts, &sessions, &mut index)
    }

    fn remove_inner(
        context_id: &str,
        contexts: &mut HashMap<String, MCPContext>,
        _sessions: &HashMap<String, MCPSession>,
        index: &mut HashMap<String, Vec<String>>,
    ) -> bool {
        if let Some(ctx) = contexts.remove(context_id) {
            for tag in &ctx.tags {
                if let Some(ids) = index.get_mut(tag) {
                    ids.retain(|id| id != context_id);
                    if ids.is_empty() {
                        index.remove(tag);
                    }
                }
            }
            true
        } else {
            false
        }
    }

    pub fn find_contexts_by_tag(&self, tag: &str) -> Vec<MCPContext> {
        let index = self.context_index.lock().unwrap();
        let contexts = self.contexts.lock().unwrap();
        index.get(tag).map_or_else(Vec::new, |ids| {
            ids.iter()
                .filter_map(|id| {
                    let ctx = contexts.get(id)?;
                    if ctx.is_expired() { None } else { Some(ctx.clone()) }
                })
                .collect()
        })
    }

    pub fn find_contexts_by_type(&self, context_type: ContextType) -> Vec<MCPContext> {
        let contexts = self.contexts.lock().unwrap();
        contexts.values()
            .filter(|ctx| ctx.context_type == context_type && !ctx.is_expired())
            .cloned()
            .collect()
    }

    pub fn find_contexts_by_priority(&self, min_priority: ContextPriority) -> Vec<MCPContext> {
        let contexts = self.contexts.lock().unwrap();
        contexts.values()
            .filter(|ctx| ctx.priority as u8 >= min_priority as u8 && !ctx.is_expired())
            .cloned()
            .collect()
    }

    pub fn create_session(&self, name: &str) -> String {
        let mut sessions = self.sessions.lock().unwrap();
        let session = MCPSession::create(name);
        let id = session.id.clone();
        sessions.insert(id.clone(), session);
        id
    }

    pub fn get_session_contexts(&self, session_id: &str) -> Vec<MCPContext> {
        let sessions = self.sessions.lock().unwrap();
        let contexts = self.contexts.lock().unwrap();
        sessions.get(session_id).map_or_else(Vec::new, |session| {
            session.context_ids.iter()
                .filter_map(|id| {
                    let ctx = contexts.get(id)?;
                    if ctx.is_expired() { None } else { Some(ctx.clone()) }
                })
                .collect()
        })
    }

    pub fn get_relevant_contexts(&self, query: &str, max_results: usize) -> Vec<MCPContext> {
        let lower = query.to_lowercase();
        let query_words: Vec<&str> = lower.split_whitespace().collect();
        let contexts = self.contexts.lock().unwrap();

        let mut scored: Vec<(i32, MCPContext)> = contexts.values()
            .filter(|ctx| !ctx.is_expired())
            .map(|ctx| {
                let mut score: i32 = 0;
                let content_str = serde_json::to_string(&ctx.content)
                    .unwrap_or_default()
                    .to_lowercase();

                for word in &query_words {
                    let count = content_str.matches(word).count() as i32;
                    score += count;
                }

                for tag in &ctx.tags {
                    if query_words.iter().any(|w| tag.to_lowercase().contains(w)) {
                        score += 2;
                    }
                }

                score += ctx.priority as i32;

                (score, ctx.clone())
            })
            .filter(|(score, _)| *score > 0)
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.truncate(max_results);
        scored.into_iter().map(|(_, ctx)| ctx).collect()
    }

    pub fn context_count(&self) -> usize {
        let contexts = self.contexts.lock().unwrap();
        contexts.values().filter(|c| !c.is_expired()).count()
    }

    pub fn session_count(&self) -> usize {
        self.sessions.lock().unwrap().len()
    }

    fn cleanup_expired_inner(
        &self,
        contexts: &mut HashMap<String, MCPContext>,
        sessions: &HashMap<String, MCPSession>,
        index: &mut HashMap<String, Vec<String>>,
    ) {
        let expired: Vec<String> = contexts.iter()
            .filter(|(_, ctx)| ctx.is_expired())
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            Self::remove_inner(&id, contexts, sessions, index);
        }
    }

    pub fn cleanup_expired(&self) {
        let mut contexts = self.contexts.lock().unwrap();
        let sessions = self.sessions.lock().unwrap();
        let mut index = self.context_index.lock().unwrap();
        let expired: Vec<String> = contexts.iter()
            .filter(|(_, ctx)| ctx.is_expired())
            .map(|(id, _)| id.clone())
            .collect();
        for id in expired {
            Self::remove_inner(&id, &mut contexts, &sessions, &mut index);
        }
    }

    fn remove_oldest_inner(
        contexts: &mut HashMap<String, MCPContext>,
        _sessions: &HashMap<String, MCPSession>,
        index: &mut HashMap<String, Vec<String>>,
        count: usize,
    ) {
        let mut sorted: Vec<(String, String)> = contexts.iter()
            .map(|(id, ctx)| (id.clone(), ctx.created_at.clone()))
            .collect();
        sorted.sort_by(|a, b| a.1.cmp(&b.1));
        for (id, _) in sorted.iter().take(count) {
            if let Some(ctx) = contexts.remove(id) {
                for tag in &ctx.tags {
                    if let Some(ids) = index.get_mut(tag) {
                        ids.retain(|i| i != id);
                        if ids.is_empty() {
                            index.remove(tag);
                        }
                    }
                }
            }
        }
    }

    pub fn get_context_summary(&self) -> Value {
        let contexts = self.contexts.lock().unwrap();
        let sessions = self.sessions.lock().unwrap();
        let index = self.context_index.lock().unwrap();

        let mut type_counts: HashMap<String, usize> = HashMap::new();
        let mut priority_counts: HashMap<String, usize> = HashMap::new();

        for ctx in contexts.values() {
            if !ctx.is_expired() {
                *type_counts.entry(format!("{:?}", ctx.context_type)).or_default() += 1;
                *priority_counts.entry(format!("{:?}", ctx.priority)).or_default() += 1;
            }
        }

        serde_json::json!({
            "total_contexts": contexts.values().filter(|c| !c.is_expired()).count(),
            "total_sessions": sessions.len(),
            "contexts_by_type": type_counts,
            "contexts_by_priority": priority_counts,
            "total_tags": index.len(),
        })
    }
}
