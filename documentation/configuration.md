# Configuração

Sistema de configuração unificado via variáveis de ambiente.

## Provedores Suportados

| Provedor | Provider ID | API Key Env Var | Modelo Default |
|----------|-------------|-----------------|----------------|
| Google Gemini | `google` | `GOOGLE_API_KEY` ou `GEMINI_API_KEY` | `gemini-2.5-flash` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic Claude | `anthropic` | `ANTHROPIC_API_KEY` | `claude-3-haiku-20240307` |
| HuggingFace | `huggingface` | `HUGGINGFACE_API_KEY`, `HF_TOKEN`, etc. | `mistralai/Mistral-7B-Instruct-v0.2` |

## Variáveis de Ambiente

### Provedor e Modelo

| Variável | Descrição | Default |
|----------|-----------|---------|
| `LLM_PROVIDER` | Provedor principal | `google` |
| `AI_PROVIDER` | Alternativa para provider | — |
| `PROVIDER` | Alternativa para provider | — |
| `MODEL_NAME` | Nome do modelo | Default por provedor |
| `MODEL` | Alternativa para modelo | — |

### API Keys

Cada provedor tem suas próprias variáveis de API key:

**Google:**
- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`

**OpenAI:**
- `OPENAI_API_KEY`

**Anthropic:**
- `ANTHROPIC_API_KEY`

**HuggingFace:**
- `HUGGINGFACE_API_KEY`
- `HUGGINGFACE_TOKEN`
- `HF_TOKEN`
- `HUGGINGFACEHUB_API_TOKEN`

**Fallback genérico:**
- `API_KEY` (usado se a chave específica do provedor não for encontrada)

### Hiperparâmetros

| Variável | Descrição | Default |
|----------|-----------|---------|
| `MODEL_TEMPERATURE` | Temperatura do modelo (0.0–1.0) | `0.7` |
| `TEMPERATURE` | Alternativa | — |
| `MAX_OUTPUT_TOKENS` | Máximo de tokens na resposta | `1024` |
| `MAX_TOKENS` | Alternativa | — |

### Rede e Timeout

| Variável | Descrição | Default |
|----------|-----------|---------|
| `BASE_URL` | URL base da API (para proxies/compatibilidade) | — |
| `API_TIMEOUT` | Timeout em segundos | `60` |

### Logging

| Variável | Descrição | Default |
|----------|-----------|---------|
| `LOG_LEVEL` | Nível de log (DEBUG, INFO, WARN, ERROR) | `INFO` |

## Normalização de Aliases

O sistema aceita vários nomes alternativos para cada provedor:

```
gemini, google-ai, googleai  →  google
gpt, chatgpt                 →  openai
claude                       →  anthropic
hf, hugging-face             →  huggingface
```

## Uso Programático

```rust
use mangaba::core::config::{Config, create_llm_config, detect_provider};

// Carrega do ambiente (.env + env vars)
let cfg = Config::load()?;
println!("{}", cfg);
// → Config(provider=google, model=gemini-2.5-flash, log_level=INFO)

// Converte para LLMConfig
let llm_cfg = cfg.to_llm_config();

// Função simplificada com fallback para defaults
let llm_cfg = create_llm_config();

// Detecta apenas o provedor
let provider = detect_provider();
assert_eq!(provider, "google");
```

## Arquivo .env

Copie o `.env.example` para `.env` na raiz do projeto:

```bash
cp .env.example .env
```

## Exemplo Completo de Configuração

```rust
use mangaba::core::config::Config;
use mangaba::core::llm::create_llm_client;

fn main() -> anyhow::Result<()> {
    // Config::load() chama dotenvy::dotenv() automaticamente
    // (apenas na primeira vez — usa OnceLock)
    let cfg = Config::load()?;

    // Acesso direto aos campos
    println!("Provider: {}", cfg.provider);
    println!("Model: {}", cfg.model);
    println!("API Key: {}...", &cfg.api_key[..4]);

    // Cria cliente LLM
    let llm = create_llm_client(&cfg.to_llm_config())?;

    Ok(())
}
```

## Tratamento de Erros

`Config::load()` retorna `Result<Self, String>`:

```rust
match Config::load() {
    Ok(cfg) => { /* usar cfg */ }
    Err(msg) => {
        eprintln!("Erro de configuração: {msg}");
        eprintln!("Certifique-se de que .env existe e tem as chaves corretas.");
    }
}
```

Erros comuns:
- **Provedor não suportado**: nome digitado errado ou alias não reconhecido
- **API key não encontrada**: variável de ambiente do provedor não está definida
