# Protocolos (A2A & MCP)

Protocolos de comunicação entre agentes e gerenciamento de contexto.

## A2A — Agent-to-Agent

Protocolo bidirecional para comunicação entre agentes.

### Conceitos

- **A2AEndpoint**: trait que define como enviar/receber mensagens
- **A2AProtocol**: gerencia histórico e delega envio ao endpoint
- **A2AAgent**: wrappa protocolo com lifecycle (connect/disconnect)
- **A2AMessage**: estrutura de mensagem tipada

### MessageType

```rust
pub enum MessageType {
    System,
    User,
    Assistant,
    Tool,
    Error,
}
```

### A2AMessage

```rust
pub struct A2AMessage {
    pub id: String,
    pub msg_type: MessageType,
    pub sender: String,
    pub recipient: String,
    pub content: Value,
    pub timestamp: String,
    pub metadata: HashMap<String, Value>,
}
```

### A2AEndpoint Trait

```rust
#[async_trait]
pub trait A2AEndpoint: Send + Sync {
    async fn send(&self, message: &A2AMessage) -> Result<A2AMessage>;
}
```

### A2AProtocol

```rust
use mangaba::core::protocols::a2a::{A2AProtocol, A2AMessage, MessageType};

let endpoint = Box::new(MyEndpoint::new());
let mut protocol = A2AProtocol::new("agent_1", endpoint);

// Enviar mensagem
let response = protocol.send(
    A2AMessage::new(
        MessageType::User,
        "agent_1",
        "agent_2",
        json!({"text": "What is the capital of France?"}),
    ),
).await?;

// Histórico
println!("Total de mensagens: {}", protocol.history().len());
```

### A2AAgent

```rust
use mangaba::core::protocols::a2a::A2AAgent;

let mut agent = A2AAgent::new("agent_1", Box::new(MyEndpoint::new()));
agent.connect().await?;

// Comunicar
let response = agent.send_message(MessageType::User, "agent_2", json!({"text": "Hello"})).await?;

agent.disconnect().await?;
```

## MCP — Model Context Protocol

Protocolo para gerenciamento hierárquico de contexto.

### Conceitos

- **MCPContext**: entrada de contexto com tags, prioridade e fonte
- **MCPSession**: gerencia coleção de contextos
- **MCPProtocol**: wrappa sessão com scoring e filtragem

### ContextType e ContextPriority

```rust
pub enum ContextType {
    System,
    Conversation,
    Task,
    Knowledge,
    External,
}

pub enum ContextPriority {
    Low,
    Medium,
    High,
    Critical,
}
```

### MCPContext

```rust
pub struct MCPContext {
    pub id: String,
    pub content: String,
    pub context_type: ContextType,
    pub priority: ContextPriority,
    pub tags: Vec<String>,
    pub source: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub metadata: Value,
}
```

### MCPSession

```rust
use mangaba::core::protocols::mcp::{MCPSession, MCPContext, ContextType, ContextPriority};

let mut session = MCPSession::new();

// Adicionar contexto
session.add(MCPContext::new(
    "ctx_1",
    "User prefers Rust over Python",
    ContextType::Conversation,
    ContextPriority::High,
    vec!["rust", "preference"],
    "conversation",
));

// Buscar por tags
let results = session.get_by_tags(&["rust"]);
println!("{} contextos encontrados", results.len());

// Buscar por tipo
let conv = session.get_by_type(ContextType::Conversation);

// Remover expirados
session.cleanup_expired();
```

### MCPProtocol

Wrappa a sessão com funcionalidades avançadas:

```rust
use mangaba::core::protocols::mcp::MCPProtocol;

let mut protocol = MCPProtocol::new(1000); // max 1000 contextos

// Adicionar contexto
protocol.add_context(
    "ctx_1",
    "Important document content",
    vec!["document", "important"],
    ContextType::Knowledge,
    ContextPriority::High,
    "document_loader",
);

// Buscar contextos relevantes (text scoring + tags + prioridade)
let relevant = protocol.get_relevant_contexts(
    "Tell me about the document",
    5,  // max resultados
);

for ctx in &relevant {
    println!("[{}] {}", ctx.id, ctx.content.chars().take(100).collect::<String>());
}

// Estatísticas
let stats = protocol.stats();
println!("Total: {}, por tipo: {:?}", stats.0, stats.1);
```

## Eventos

Ambos protocolos integram-se ao sistema de eventos (`EventBus`) para logging
e monitoramento das operações.
