//! Communication protocols for agent-to-agent and context management.
//!
//! - **A2A** — Agent-to-Agent: bidirectional messaging with typed messages
//!   (`system`, `user`, `assistant`, `tool`, `error`). [`A2AAgent`] sends/receives
//!   via an [`A2AEndpoint`] trait object.
//! - **MCP** — Model Context Protocol: hierarchical contexts with priority scoring,
//!   keyword/tag-based relevance ﬁltering, and session management.

pub mod a2a;
pub mod mcp;

pub use a2a::{A2AEndpoint, A2AAgent, A2AMessage, A2AProtocol, MessageType};
pub use mcp::{ContextPriority, ContextType, MCPContext, MCPProtocol, MCPSession};
