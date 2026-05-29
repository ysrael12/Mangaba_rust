# Mangaba AI — Documentação

Framework modular e agnóstico de provedor para construção de agentes LLM em Rust.

## Visão Geral

Mangaba AI é um port do framework Python homônimo para Rust. Ele fornece todos os blocos
necessários para criar sistemas multi-agente com suporte a ferramentas (tool calling),
memória, guardrails, pipelines, RAG, planejamento e protocolos de comunicação.

```
mangaba::core
├── agent        → Agente com ReAct loop, memória, guardrails, ferramentas, delegação
├── callbacks    → Sistema de hooks para eventos (step, tool, LLM, task)
├── config       → Detecção de provedor via env vars, builder de LLMConfig
├── crew         → Orquestração multi-agente (sequencial / hierárquico)
├── embeddings   → Trait + providers (OpenAI, HuggingFace, NoOp, Cached LRU)
├── errors       → Enum MangabaError (18 variantes, detecção de retry)
├── events       → EventBus global (subscribe/emit)
├── guardrails   → Length, profanity, composite + GuardrailTool
├── llm          → LLMClient trait + 7 providers + streaming + retry + cache
├── memory       → Curta duração, longa duração (JSON), memória de entidades
├── output_parsers → JSONOutputParser, NoOpOutputParser
├── pipeline     → Stage, ParallelStage, ConditionalStage, Pipeline
├── planner      → PlanStep, ExecutionPlan, TaskPlanner (planos via LLM)
├── prompt_templates → PromptTemplate, SystemPromptBuilder
├── protocols    → A2A (agent-to-agent) + MCP (model context protocol)
├── rag          → RAGEngine: ingest → chunk → embed → query
├── react        → ReActEngine: Thought → Action → Observation loop
├── retry        → Exponential backoff com jitter
├── task         → Task com encadeamento de contexto
├── tools        → BaseTool trait + Calculator, File, Text, Web search
└── types        → LLMConfig, Message, ToolCall, ToolResult, etc.
```

## Features Principais

| Feature | Descrição |
|---------|-----------|
| **Provider-agnóstico** | Troque de provedor via env var. Todos suportam tool calling. |
| **ReAct Loop** | Thought → Action → Observation loop com execução automática de tools |
| **Crew Multi-Agente** | Sequencial ou hierárquico com manager LLM |
| **Streaming** | SSE real (OpenAI/OpenRouter) ou fallback síncrono |
| **RAG** | Ingestão de arquivos → chunk → embed → query via SQLite |
| **Memória** | Curta, longa (JSON persistente), e memória de entidades |
| **Guardrails** | Validação de comprimento, profanidade, composição |
| **Pipeline** | Estágios sequenciais, paralelos e condicionais |
| **Planejamento** | LLM decompõe tarefas em planos executáveis |
| **Protocolos** | A2A (agente↔agente) + MCP (contexto hierárquico) |
| **Eventos** | EventBus global para logging e monitoramento |
| **Cache** | Cache TTL para respostas LLM + cache LRU para embeddings |

## Estrutura de Documentação

| Documento | Conteúdo |
|-----------|----------|
| [getting-started.md](getting-started.md) | Setup, configuração e quick start |
| [configuration.md](configuration.md) | Variáveis de ambiente, provedores, Config |
| [features/llm.md](features/llm.md) | LLMClient, provedores, streaming, retry, cache |
| [features/agent.md](features/agent.md) | Agent, ReAct, delegação |
| [features/crew.md](features/crew.md) | Crew, Task, orquestração |
| [features/tools.md](features/tools.md) | BaseTool, ferramentas built-in |
| [features/memory.md](features/memory.md) | Memória curta, longa, entidades |
| [features/guardrails.md](features/guardrails.md) | Guardrails de entrada/saída |
| [features/pipeline.md](features/pipeline.md) | Pipeline, Stage, Parallel, Conditional |
| [features/planner.md](features/planner.md) | Planejamento de tarefas via LLM |
| [features/rag.md](features/rag.md) | RAGEngine, DocumentLoader, ChromaDB |
| [features/embeddings.md](features/embeddings.md) | Embeddings, vector store, cache |
| [features/protocols.md](features/protocols.md) | A2A, MCP |
| [features/events-callbacks.md](features/events-callbacks.md) | EventBus, Callbacks |
| [features/errors-retry.md](features/errors-retry.md) | Tratamento de erros, retry |
| [features/output-parsers.md](features/output-parsers.md) | JSONOutputParser, NoOpOutputParser |
| [features/prompt-templates.md](features/prompt-templates.md) | PromptTemplate, SystemPromptBuilder |
