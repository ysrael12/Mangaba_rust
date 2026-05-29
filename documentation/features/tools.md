# Tools (Ferramentas)

Sistema de ferramentas que agentes LLM podem chamar durante o ReAct loop.

## BaseTool Trait

Para criar uma ferramenta, implemente `BaseTool`:

```rust
#[async_trait]
pub trait BaseTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn args_schema(&self) -> Option<Value> { None }
    fn return_direct(&self) -> bool { false }
    async fn run_impl(&self, args: Value) -> Result<ToolResult>;
}
```

- `name()` — identificador único usado pelo LLM para chamar a ferramenta
- `description()` — descrição para o LLM entender quando usar
- `args_schema()` — schema JSON opcional para validação dos argumentos
- `return_direct()` — se `true`, o resultado é retornado imediatamente sem continuar o ReAct
- `run_impl()` — lógica principal da ferramenta

### ToolResult

```rust
pub struct ToolResult {
    pub call_id: String,
    pub tool_name: String,
    pub output: Option<Value>,
    pub error: Option<String>,
    pub success: bool,
}
```

### Método `call()`

O método `call()` (da trait) wrappa `run_impl()` com emissão de eventos:

```rust
async fn call(&self, args: Value) -> Result<ToolResult> {
    // Emite ToolStart
    let result = self.run_impl(args).await;
    // Emite ToolEnd ou ToolError
    result
}
```

### Function Schema

`get_function_schema()` gera o schema no formato esperado pelos LLMs:

```json
{
    "name": "calculator",
    "description": "Evaluate a mathematical expression",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        },
        "required": ["expression"]
    }
}
```

## Ferramentas Built-in

### CalculatorTool

Avalia expressões matemáticas com parser próprio (sem `eval`).

```rust
let tool = CalculatorTool;
let result = tool.call(json!({"expression": "2 + 3 * 4"})).await?;
// → ToolResult { output: {"result": 14.0}, success: true }
```

Suporta: `+`, `-`, `*`, `/`, `%`, `^`, parênteses, números decimais.

### FileReaderTool

Lê arquivos de texto do sistema de arquivos.

```rust
let tool = FileReaderTool;
let result = tool.call(json!({"file_path": "/path/to/file.txt"})).await?;
// → ToolResult { output: {"content": "file contents..."}, success: true }
```

### FileWriterTool

Escreve conteúdo em arquivos (cria diretórios intermediários automaticamente).

```rust
let tool = FileWriterTool;
let result = tool.call(json!({
    "file_path": "/path/to/output.txt",
    "content": "Hello, world!",
})).await?;
```

### DirectoryListTool

Lista arquivos e diretórios com filtro opcional.

```rust
let tool = DirectoryListTool;
let result = tool.call(json!({
    "directory_path": "/tmp",
    "pattern": "*.txt",
})).await?;
// → ToolResult { output: {"directories": [...], "files": [...]} }
```

### TextSplitterTool

Divide texto em chunks com tamanho e overlap configuráveis.

```rust
let tool = TextSplitterTool;
let result = tool.call(json!({
    "text": "long text here...",
    "chunk_size": 500,
    "chunk_overlap": 50,
})).await?;
// → ToolResult { output: {"chunks": ["...", "..."]} }
```

### WordCounterTool

Conta palavras, sentenças, parágrafos e caracteres.

```rust
let tool = WordCounterTool;
let result = tool.call(json!({"text": "Hello world. How are you?"})).await?;
// → ToolResult { output: {"words": 5, "sentences": 2, ...} }
```

### SerperSearchTool

Busca na web via API do Serper.dev (requer `SERPER_API_KEY`).

```rust
let tool = SerperSearchTool::from_env()?;  // ou SerperSearchTool::new("api_key")
let result = tool.call(json!({"query": "latest AI news"})).await?;
// → ToolResult { output: {"results": "- [Title](url)\n  snippet..."} }
```

### DuckDuckGoSearchTool

Busca na web via DuckDuckGo Instant Answer API (sem chave).

```rust
let tool = DuckDuckGoSearchTool;
let result = tool.call(json!({"query": "Rust programming"})).await?;
// → ToolResult { output: {"results": "**Answer**: ...\n- Related topic..."} }
```

### EchoTool

Ecoa os argumentos de volta (útil para testes).

```rust
let tool = EchoTool;
let result = tool.call(json!({"hello": "world"})).await?;
// → ToolResult { output: {"hello": "world"}, success: true }
```

## Criando uma Ferramenta Customizada

```rust
use async_trait::async_trait;
use anyhow::{Result, anyhow};
use serde_json::{json, Value};
use mangaba::core::tools::BaseTool;
use mangaba::core::types::ToolResult;

pub struct WeatherTool {
    api_key: String,
}

#[async_trait]
impl BaseTool for WeatherTool {
    fn name(&self) -> &str { "get_weather" }
    fn description(&self) -> &str { "Get current weather for a city" }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        }))
    }

    async fn run_impl(&self, args: Value) -> Result<ToolResult> {
        let city = args["city"].as_str()
            .ok_or_else(|| anyhow!("Missing 'city'"))?;

        // Lógica de API aqui...
        let temp = "25°C";

        Ok(ToolResult {
            call_id: "weather".to_string(),
            tool_name: "get_weather".to_string(),
            output: Some(json!({"city": city, "temperature": temp})),
            error: None,
            success: true,
        })
    }
}
```

## BaseToolkit

Agrupa múltiplas ferramentas em um objeto:

```rust
pub trait BaseToolkit: Send + Sync {
    fn get_tools(&self) -> Vec<Box<dyn BaseTool + Send + Sync>>;
}
```

## Eventos

| Evento | Momento |
|--------|---------|
| `ToolStart` | Antes de executar a ferramenta |
| `ToolEnd` | Após execução bem-sucedida |
| `ToolError` | Quando `run_impl()` retorna erro |
