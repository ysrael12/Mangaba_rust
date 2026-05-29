# Prompt Templates

Sistema de templates para construção de prompts.

## PromptTemplate

Template simples com variáveis `{placeholder}`.

```rust
use mangaba::core::prompt_templates::PromptTemplate;
use std::collections::HashMap;

// Cria template — detecta variáveis automaticamente
let tmpl = PromptTemplate::new("Hello {name}, you are {role}!");
assert_eq!(tmpl.variables.len(), 2);
assert!(tmpl.variables.contains(&"name".to_string()));
assert!(tmpl.variables.contains(&"role".to_string()));

// Renderiza com valores
let mut values = HashMap::new();
values.insert("name", "Alice");
values.insert("role", "admin");
let result = tmpl.render(&values);
assert_eq!(result, "Hello Alice, you are admin!");
```

## SystemPromptBuilder

Construtor fluente para system prompts de agentes.

```rust
use mangaba::core::prompt_templates::SystemPromptBuilder;

let prompt = SystemPromptBuilder::new()
    .role("Research Assistant")
    .goal("Find and summarize information")
    .backstory("I am an AI assistant specialized in research")
    .tools(&[
        "- calculator: Evaluate math expressions".into(),
        "- search: Search the web".into(),
    ])
    .section("Instructions", "Always cite your sources")
    .build();

println!("{prompt}");
```

Saída gerada:
```
You are: Research Assistant

Your goal is: Find and summarize information

Background: I am an AI assistant specialized in research

Available tools:
- calculator: Evaluate math expressions
- search: Search the web

Instructions:
Always cite your sources
```

### Métodos

| Método | Descrição |
|--------|-----------|
| `new()` | Cria builder vazio |
| `role(r)` | Adiciona "You are: {r}" |
| `goal(g)` | Adiciona "Your goal is: {g}" |
| `backstory(b)` | Adiciona "Background: {b}" |
| `tools(t)` | Adiciona "Available tools:\n{...}" (só se não vazio) |
| `section(title, content)` | Adiciona "{title}:\n{content}" |
| `build()` | Junta com "\n\n" |
