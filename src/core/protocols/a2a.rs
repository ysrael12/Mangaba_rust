//! Agent-to-Agent protocol for bidirectional messaging.
//!
//! [`A2AProtocol`] manages message history and delegates sending to an
//! [`A2AEndpoint`] trait object. [`A2AAgent`] wraps the protocol with
//! connect/disconnect lifecycle. Messages are typed ([`MessageType`]) and
//! carry structured [`A2AMessage`] content.

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// MessageType
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    Request,
    Response,
    Broadcast,
    Notification,
    Error,
}

// ---------------------------------------------------------------------------
// A2AMessage
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AMessage {
    pub id: String,
    pub sender_id: String,
    pub receiver_id: Option<String>,
    #[serde(rename = "message_type")]
    pub message_type: MessageType,
    pub content: Value,
    pub timestamp: String,
    pub correlation_id: Option<String>,
    pub metadata: Option<Value>,
}

impl A2AMessage {
    pub fn create(
        sender_id: &str,
        message_type: MessageType,
        content: Value,
        receiver_id: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            sender_id: sender_id.to_string(),
            receiver_id: receiver_id.map(String::from),
            message_type,
            content,
            timestamp: Utc::now().to_rfc3339(),
            correlation_id: correlation_id.map(String::from),
            metadata: Some(serde_json::json!({})),
        }
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

// ---------------------------------------------------------------------------
// A2AEndpoint — trait for receiving messages (breaks circular ref)
// ---------------------------------------------------------------------------
pub trait A2AEndpoint: Send + Sync {
    fn agent_id(&self) -> &str;
    fn receive_message(&self, msg: A2AMessage);
}

// ---------------------------------------------------------------------------
// A2AProtocol
// ---------------------------------------------------------------------------
pub struct A2AProtocol {
    pub agent_id: String,
    handlers: Mutex<HashMap<MessageType, Vec<Arc<dyn Fn(A2AMessage) + Send + Sync>>>>,
    connected_agents: Mutex<HashMap<String, Arc<dyn A2AEndpoint>>>,
    message_history: Mutex<Vec<A2AMessage>>,
}

impl A2AProtocol {
    pub fn new(agent_id: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            handlers: Mutex::new(HashMap::new()),
            connected_agents: Mutex::new(HashMap::new()),
            message_history: Mutex::new(Vec::new()),
        }
    }

    pub fn register_handler<F>(&self, message_type: MessageType, handler: F)
    where
        F: Fn(A2AMessage) + Send + Sync + 'static,
    {
        let mut handlers = self.handlers.lock().unwrap();
        handlers.entry(message_type).or_default().push(Arc::new(handler));
    }

    pub fn connect_agent(&self, agent: Arc<dyn A2AEndpoint>) {
        let mut agents = self.connected_agents.lock().unwrap();
        agents.insert(agent.agent_id().to_string(), agent);
    }

    pub fn disconnect_agent(&self, agent_id: &str) {
        let mut agents = self.connected_agents.lock().unwrap();
        agents.remove(agent_id);
    }

    pub fn connected_agent_ids(&self) -> Vec<String> {
        let agents = self.connected_agents.lock().unwrap();
        agents.keys().cloned().collect()
    }

    pub fn send_message(&self, message: A2AMessage) -> bool {
        let connected = self.connected_agents.lock().unwrap();

        let delivered = if let Some(ref receiver) = message.receiver_id {
            connected.get(receiver).map(|agent| {
                agent.receive_message(message.clone());
                true
            }).unwrap_or(false)
        } else if message.message_type == MessageType::Broadcast {
            for agent in connected.values() {
                agent.receive_message(message.clone());
            }
            true
        } else {
            false
        };

        if delivered {
            let mut history = self.message_history.lock().unwrap();
            history.push(message);
        }
        delivered
    }

    pub fn receive_message(&self, message: A2AMessage) {
        {
            let mut history = self.message_history.lock().unwrap();
            history.push(message.clone());
        }

        let handlers = self.handlers.lock().unwrap();
        if let Some(ms_handlers) = handlers.get(&message.message_type) {
            for handler in ms_handlers {
                handler(message.clone());
            }
        }
    }

    pub fn create_request(&self, receiver_id: &str, action: &str, params: Value) -> A2AMessage {
        A2AMessage::create(
            &self.agent_id,
            MessageType::Request,
            serde_json::json!({ "action": action, "params": params }),
            Some(receiver_id),
            None,
        )
    }

    pub fn create_response(&self, original: &A2AMessage, result: Value, success: bool) -> A2AMessage {
        A2AMessage::create(
            &self.agent_id,
            MessageType::Response,
            serde_json::json!({ "result": result, "success": success }),
            Some(&original.sender_id),
            Some(&original.id),
        )
    }

    pub fn broadcast(&self, content: Value, target_tags: Option<Vec<String>>) -> A2AMessage {
        let mut message = A2AMessage::create(
            &self.agent_id,
            MessageType::Broadcast,
            content,
            None,
            None,
        );
        if let Some(tags) = target_tags {
            let meta = message.metadata.get_or_insert(serde_json::json!({}));
            meta["target_tags"] = serde_json::to_value(tags).unwrap_or_default();
        }
        self.send_message(message.clone());
        message
    }

    pub fn message_history(&self) -> Vec<A2AMessage> {
        self.message_history.lock().unwrap().clone()
    }

    pub fn is_connected(&self, agent_id: &str) -> bool {
        self.connected_agents.lock().unwrap().contains_key(agent_id)
    }
}

// ---------------------------------------------------------------------------
// A2AAgent
// ---------------------------------------------------------------------------
pub struct A2AAgent {
    pub agent_id: String,
    pub a2a_protocol: Arc<A2AProtocol>,
}

impl A2AAgent {
    pub fn new(agent_id: &str) -> Self {
        let protocol = Arc::new(A2AProtocol::new(agent_id));
        let agent = Self {
            agent_id: agent_id.to_string(),
            a2a_protocol: protocol,
        };
        agent.setup_default_handlers();
        agent
    }

    fn setup_default_handlers(&self) {
        let aid = self.agent_id.clone();
        let proto = self.a2a_protocol.clone();
        self.a2a_protocol.register_handler(MessageType::Request, move |msg| {
            let action = msg.content.get("action").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let params = msg.content.get("params").cloned().unwrap_or(serde_json::json!({}));
            let result = serde_json::json!({
                "message": format!("Ação '{}' processada pelo agente {}", action, aid),
                "params": params,
            });
            let response = proto.create_response(&msg, result, true);
            proto.send_message(response);
        });

        self.a2a_protocol.register_handler(MessageType::Response, move |msg| {
            log::info!("Resposta recebida de {}: {:?}", msg.sender_id, msg.content);
        });

        self.a2a_protocol.register_handler(MessageType::Notification, move |msg| {
            log::info!("Notificação de {}: {:?}", msg.sender_id, msg.content);
        });
    }

    pub fn receive_message(&self, msg: A2AMessage) {
        self.a2a_protocol.receive_message(msg);
    }

    pub fn connect_to(&self, other: Arc<A2AAgent>) {
        self.a2a_protocol.connect_agent(other.clone());
        other.a2a_protocol.connect_agent(Arc::new(AgentEndpoint {
            agent_id: self.agent_id.clone(),
            protocol: self.a2a_protocol.clone(),
        }));
    }

    pub fn send_request(&self, receiver_id: &str, action: &str, params: Value) -> bool {
        let msg = self.a2a_protocol.create_request(receiver_id, action, params);
        self.a2a_protocol.send_message(msg)
    }

    pub fn notify_all(&self, content: Value) {
        self.a2a_protocol.broadcast(content, None);
    }
}

impl A2AEndpoint for A2AAgent {
    fn agent_id(&self) -> &str {
        &self.agent_id
    }

    fn receive_message(&self, msg: A2AMessage) {
        self.a2a_protocol.receive_message(msg);
    }
}

// Internal endpoint wrapper for bidirectional connect
struct AgentEndpoint {
    agent_id: String,
    protocol: Arc<A2AProtocol>,
}

impl A2AEndpoint for AgentEndpoint {
    fn agent_id(&self) -> &str {
        &self.agent_id
    }

    fn receive_message(&self, msg: A2AMessage) {
        self.protocol.receive_message(msg);
    }
}
