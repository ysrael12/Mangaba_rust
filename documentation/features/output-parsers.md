# Output Parsers

Parsers para extração estruturada de respostas LLM.

## OutputParser Trait

```rust
pub trait OutputParser: Send + Sync {
    fn parse(&self, text: &str) -> Result<String>;
    fn get_format_instructions(&self) -> String;
}
```

## NoOpOutputParser

Passagem direta — não modifica o texto.

```rust
use mangaba::core::output_parsers::NoOpOutputParser;

let parser = NoOpOutputParser;
let result = parser.parse("any text").unwrap();
assert_eq!(result, "any text");
```

## JSONOutputParser

Extrai JSON de blocos de código ou texto puro.

```rust
use mangaba::core::output_parsers::JSONOutputParser;

let parser = JSONOutputParser;

// De bloco ```json
let json = parser.parse("```json\n{\"key\": \"value\"}\n```").unwrap();
assert_eq!(json, "{\"key\": \"value\"}");

// De bloco ``` genérico
let json = parser.parse("```{\"key\": \"value\"}```").unwrap();
assert_eq!(json, "{\"key\": \"value\"}");

// De texto que começa com { ou [
let json = parser.parse("{\"key\": \"value\"}").unwrap();
assert_eq!(json, "{\"key\": \"value\"}");

// Texto inválido retorna erro
assert!(parser.parse("just text").is_err());
```

### Integração com Agent

```rust
AgentConfig {
    output_parser: Some("json".into()),
    // O parser JSON será aplicado após guardrails
}
```
