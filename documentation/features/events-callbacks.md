# Eventos & Callbacks

Sistemas de observabilidade para monitorar a execução dos agentes.

## EventBus

Sistema global publish-subscribe para eventos do framework.

### Estrutura

```rust
pub struct Event {
    pub event_type: EventType,
    pub source_id: String,
    pub data: Value,
}
```

### EventType

```rust
pub enum EventType {
    // Agent
    AgentStart, AgentEnd, AgentError,
    // LLM
    LLMStart, LLMEnd, LLMError, LLMRetry, LLMStreamChunk,
    // Tools
    ToolStart, ToolEnd, ToolError,
    // ReAct
    ReActStep, ReActThought, ReActAction, ReActObservation,
    // Task
    TaskStart, TaskEnd, TaskError,
    // Crew
    CrewStart, CrewEnd, CrewError,
    // Memory
    MemoryAdd, MemorySearch,
    // Guardrails
    GuardrailPass, GuardrailFail,
    // Custom
    Custom(String),
}
```

### Uso

```rust
use mangaba::core::events::{EventBus, Event, EventType};
use std::sync::{Arc, Mutex};

// Inscrever para receber eventos
EventBus::subscribe(Box::new(|event: &Event| {
    println!("[{}] {:?}: {:?}",
        event.source_id, event.event_type, event.data);
}));

// Emitir eventos (qualquer lugar do código)
EventBus::emit(Event::new(
    EventType::Custom("my_event".into()),
    "my_component",
    json!({"key": "value"}),
));
```

### Limpeza de Listeners

O `EventBus` é global e listeners acumulam entre execuções. Em testes,
é importante considerar que listeners de testes anteriores podem
ainda estar registrados.

## Callbacks

Sistema de hooks síncronos para pontos específicos do ciclo de vida.

### Tipos de Callback

```rust
pub type StepCallback = Box<dyn Fn(&ReActStep, &str) + Send + Sync + 'static>;
pub type ToolStartCallback = Box<dyn Fn(&str, &Value) + Send + Sync + 'static>;
pub type ToolEndCallback = Box<dyn Fn(&str, &ToolResult) + Send + Sync + 'static>;
pub type LLMStartCallback = Box<dyn Fn(usize, usize) + Send + Sync + 'static>;
pub type LLMEndCallback = Box<dyn Fn(&LLMResponse) + Send + Sync + 'static>;
pub type TaskStartCallback = Box<dyn Fn(&str) + Send + Sync + 'static>;
pub type TaskEndCallback = Box<dyn Fn(&str, &str) + Send + Sync + 'static>;
```

### Callbacks

```rust
use mangaba::core::callbacks::Callbacks;

let mut cbs = Callbacks::new();

// Callback quando um step do ReAct é concluído
cbs.add_step(|step, agent_id| {
    println!("Step {} pelo agente {}: {:?}",
        step.step_number, agent_id, step.thought);
});

// Callback quando uma tool é chamada
cbs.add_tool_start(|tool_name, args| {
    println!("Tool {tool_name} chamada com {args}");
});

// Callback quando a LLM é chamada
cbs.add_llm_start(|num_messages, num_tools| {
    println!("LLM chamada com {num_messages} mensagens e {num_tools} tools");
});

// Callback quando uma tarefa termina
cbs.add_task_end(|description, result| {
    println!("Tarefa '{description}' concluída: {} chars", result.len());
});
```

### Integração com Agent

```rust
let mut agent = Agent::new(config, tools, llm, None);
agent.callbacks.add_step(|step, _| {
    println!("Thought: {:?}", step.thought);
    println!("Action: {:?}", step.action);
});
```

Os callbacks são disparados durante:
- `execute_task()` — task_start / task_end
- `ReActEngine::run()` — step, tool_start, tool_end, llm_start, llm_end

## Diferenças entre EventBus e Callbacks

| Aspecto | EventBus | Callbacks |
|---------|----------|-----------|
| **Escopo** | Global | Local (por instância) |
| **Comunicação** | Desacoplada | Direta |
| **Tipo** | Assíncrono (sem retorno) | Síncrono (sem retorno) |
| **Filtro** | Por tipo de evento | Por ponto do ciclo |
| **Uso principal** | Logging, monitoria, métricas | Debug, hooks de UI |

## Exemplo: Logging Completo

```rust
// EventBus para logging global
EventBus::subscribe(Box::new(|event: &Event| {
    match event.event_type {
        EventType::LLMStart => log::info!("LLM chamado"),
        EventType::ToolStart => log::info!("Tool iniciada"),
        EventType::ReActThought => {
            if let Some(text) = event.data["thought"].as_str() {
                log::debug!("Thought: {text}");
            }
        }
        EventType::AgentEnd => log::info!("Agente concluído"),
        _ => {}
    }
}));

// Callbacks para monitoramento local
let mut cbs = Callbacks::new();
cbs.add_step(|step, agent_id| {
    println!("[{}] Passo {}: {} -> {:?}",
        agent_id, step.step_number,
        step.thought.as_deref().unwrap_or("..."),
        step.action.as_ref().map(|a| &a.tool_name));
});
```
