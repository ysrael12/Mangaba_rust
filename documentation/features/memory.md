# Memory (Memória)

Sistemas de memória para agentes manterem contexto entre execuções.

## BaseMemory Trait

```rust
#[async_trait]
pub trait BaseMemory {
    async fn add(&mut self, entry: &str, metadata: Option<Value>);
    async fn get_relevant(&self, query: &str, max_results: usize) -> String;
}
```

- `add()` — armazena uma entrada com metadados opcionais
- `get_relevant()` — busca entradas relevantes por palavra-chave

## ShortTermMemory

Memória em buffer circular (FIFO) em memória RAM.

```rust
use mangaba::core::memory::ShortTermMemory;

let mut mem = ShortTermMemory::new(50); // max 50 entradas

mem.add("User asked about Rust", None).await;
mem.add("Provided code example", None).await;

let context = mem.get_relevant("Rust", 5).await;
println!("{context}");
```

Quando o limite é atingido, a entrada mais antiga é removida.

## LongTermMemory

Memória persistente em arquivo JSON.

```rust
use mangaba::core::memory::LongTermMemory;

let mut mem = LongTermMemory::new(
    Some("memory_data.json".into()),  // caminho opcional
    1000,                              // max entradas
);

mem.add("Important fact", Some(json!({"category": "science"}))).await;

// Na próxima execução, recarrega do arquivo
let mem2 = LongTermMemory::new(Some("memory_data.json".into()), 1000);
let context = mem2.get_relevant("fact", 10).await;
```

A busca por relevância usa correspondência de palavras-chave (case-insensitive).

## EntityMemory

Memória estruturada para entidades com atributos.

```rust
use mangaba::core::memory::EntityMemory;

let mut mem = EntityMemory::new();

// Adiciona entidades com metadados
mem.add("Alice", Some(json!({"age": 30, "city": "NYC", "role": "engineer"}))).await;
mem.add("Bob", Some(json!({"age": 25, "city": "SFO"}))).await;

// Busca por entidade específica
let result = mem.get_relevant("Alice", 10).await;
// → Entity: Alice
//     age: 30
//     city: NYC
//     role: engineer

// Busca por atributo
let result = mem.get_relevant("engineer", 10).await;
// → Entity: Alice (porque contém "engineer" como valor)
```

## Factory

Cria a memória apropriada baseada em configuração:

```rust
use mangaba::core::memory::create_memory;
use mangaba::core::types::MemoryConfig;

let config = MemoryConfig {
    short_term: true,
    long_term: false,
    entity: false,
    max_short_term_items: 100,
    storage_path: None,
};

let mem = create_memory(&config);
// → Some(Box<ShortTermMemory>)
```

Prioridade: short_term > long_term > entity (primeiro `true` vence).

## Integração com Agent

A memória é automaticamente integrada ao agente:

```rust
let agent = Agent::new(config, tools, llm, Some(Box::new(mem)));

// Durante execute_task():
// 1. Busca contexto relevante via mem.get_relevant(task_description, 5)
// 2. Após execução, salva cada step (thought, action, observation)
// 3. Salva o resultado final: "Task: ...\nResult: ..."
```

## Relevância por Palavra-Chave

Todas as memórias usam o mesmo algoritmo de relevância:

```rust
fn is_relevant(entry: &str, query: &str) -> bool {
    // Retorna true se alguma palavra do query aparece no entry
    // Case-insensitive, separa por whitespace
}
```

## Eventos

| Evento | Momento |
|--------|---------|
| `MemoryAdd` | Quando uma entrada é adicionada |
| `MemorySearch` | Quando uma busca é realizada |
