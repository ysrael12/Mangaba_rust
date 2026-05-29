# Guardrails

Validação e transformação de entrada/saída dos agentes.

## Guardrail Trait

```rust
pub trait Guardrail: Send + Sync {
    fn validate(&self, text: &str) -> String;
}
```

Recebe texto, retorna texto modificado ou original.

## Implementações

### NoOpGuardrail

Pass-through — não altera o texto.

```rust
let g = NoOpGuardrail;
assert_eq!(g.validate("anything"), "anything");
```

### LengthGuardrail

Trunca ou valida por comprimento de caracteres.

```rust
let g = LengthGuardrail { max_len: 10, truncate: true };
assert_eq!(g.validate("Hello world"), "Hello worl");

let g = LengthGuardrail { max_len: 5, truncate: false };
assert_eq!(g.validate("Hello world"), "Hello"); // também trunca quando não truncate (bug conhecido)
```

### ProfanityGuardrail

Substitui palavras ofensivas via regex.

```rust
let g = ProfanityGuardrail::new(
    r"\b(darn|heck)\b",   // padrão regex
    "****",                // substituição
);
assert_eq!(g.validate("what the heck"), "what the ****");
assert_eq!(g.validate("clean text"), "clean text");
```

### CompositeGuardrail

Encadeia múltiplos guardrails em sequência.

```rust
let g = CompositeGuardrail {
    guardrails: vec![
        Box::new(ProfanityGuardrail::new(r"\bdarn\b", "****")),
        Box::new(LengthGuardrail { max_len: 20, truncate: true }),
    ],
};
let result = g.validate("this is darn long text here");
// → "this is **** long te"
```

## GuardrailTool

Wrappa um guardrail como uma ferramenta LLM, permitindo uso no ReAct loop.

```rust
use mangaba::core::guardrails::GuardrailTool;

let tool = GuardrailTool::new(
    "profanity_filter",
    Box::new(ProfanityGuardrail::new(r"\bdarn\b", "****")),
);

let result = tool.call(json!({"text": "that darn cat"})).await.unwrap();
assert!(result.success);
assert_eq!(result.output.unwrap()["validated_text"], "that **** cat");
```

Schema de args:
```json
{
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "Text to validate"}
    },
    "required": ["text"]
}
```

## apply_guardrails Helper

Função auxiliar que aplica uma lista de guardrails com emissão de eventos:

```rust
use mangaba::core::guardrails::apply_guardrails;

let guardrails: Vec<Box<dyn Guardrail + Send + Sync>> = vec![
    Box::new(LengthGuardrail { max_len: 100, truncate: true }),
];

let result = apply_guardrails(&guardrails, "agent_id", "Long text here...").await;
```

Cada guardrail que modifica o texto dispara um evento `GuardrailPass` com
`modified: true` e um preview do resultado.

## Integração com Agent

```rust
AgentConfig {
    guardrails: vec!["length".into(), "profanity".into()],
    // ...
}
```

O agent cria automaticamente os guardrails baseado nos nomes:
- `"no_op"` → `NoOpGuardrail`
- `"length"` → `LengthGuardrail { max_len: 1024, truncate: true }`
- `"profanity"` → `ProfanityGuardrail` com padrão `\b(badword|damn)\b`
- Qualquer outro → `NoOpGuardrail`

São aplicados após a execução do ReAct, antes do output parser.
