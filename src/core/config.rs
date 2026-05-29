    //! Environment-based provider detection and [`LLMConfig`] builder.
//!
//! Mirrors the Python `Config` class: loads `.env` via `dotenvy`, detects the LLM provider
//! from env vars, normalizes aliases (`gemini` → `google`, `gpt` → `openai`), and looks up
//! provider-specific API keys with sensible defaults for model, temperature, and max tokens.

use std::sync::OnceLock;

use crate::core::types::LLMConfig;

static LOAD_ENV: OnceLock<()> = OnceLock::new();

fn ensure_env_loaded() {
    LOAD_ENV.get_or_init(|| {
        let _ = dotenvy::dotenv();
    });
}

const SUPPORTED_PROVIDERS: &[&str] = &["google", "openai", "anthropic", "huggingface"];

fn normalize_provider(raw: &str) -> Option<String> {
    let p = raw.to_lowercase().replace('_', "-");
    let normalized = match p.as_str() {
        "gemini" | "google-ai" | "googleai" => "google",
        "gpt" | "chatgpt" => "openai",
        "claude" => "anthropic",
        "hf" | "hugging-face" => "huggingface",
        other => other,
    };
    if SUPPORTED_PROVIDERS.contains(&normalized) {
        Some(normalized.to_string())
    } else {
        None
    }
}

fn provider_env_keys() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("google", vec!["GOOGLE_API_KEY", "GEMINI_API_KEY"]),
        ("openai", vec!["OPENAI_API_KEY"]),
        ("anthropic", vec!["ANTHROPIC_API_KEY"]),
        ("huggingface", vec!["HUGGINGFACE_API_KEY", "HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]),
    ]
}

fn default_model(provider: &str) -> &str {
    match provider {
        "google" => "gemini-2.5-flash",
        "openai" => "gpt-4o-mini",
        "anthropic" => "claude-3-haiku-20240307",
        "huggingface" => "mistralai/Mistral-7B-Instruct-v0.2",
        _ => "gemini-2.5-flash",
    }
}

fn default_temperature(provider: &str) -> f32 {
    let _ = provider;
    0.7
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

pub struct Config {
    pub provider: String,
    pub model: String,
    pub api_key: String,
    pub temperature: f32,
    pub max_tokens: usize,
    pub base_url: Option<String>,
    pub log_level: String,
}

impl Config {
    pub fn load() -> Result<Self, String> {
        ensure_env_loaded();

        let provider = {
            let raw = env("LLM_PROVIDER")
                .or_else(|| env("AI_PROVIDER"))
                .or_else(|| env("PROVIDER"))
                .unwrap_or_else(|| "google".to_string());
            normalize_provider(&raw).ok_or_else(|| {
                format!(
                    "Unsupported provider '{raw}'. Use one of: {}",
                    SUPPORTED_PROVIDERS.join(", ")
                )
            })?
        };

        let model = env("MODEL_NAME")
            .or_else(|| env("MODEL"))
            .unwrap_or_else(|| default_model(&provider).to_string());

        let key_vars = match provider_env_keys().into_iter().find(|(p, _)| p == &provider) {
            Some((_, keys)) => keys,
            None => vec![],
        };

        let api_key = key_vars.iter()
            .find_map(|k| env(k))
            .or_else(|| env("API_KEY"))
            .ok_or_else(|| format!(
                "No API key found for provider '{provider}'. Set one of: {}",
                key_vars.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(", ")
            ))?;

        let temperature = env("MODEL_TEMPERATURE")
            .or_else(|| env("TEMPERATURE"))
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or_else(|| default_temperature(&provider));

        let max_tokens = env("MAX_OUTPUT_TOKENS")
            .or_else(|| env("MAX_TOKENS"))
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024);

        let base_url = env("BASE_URL");
        let log_level = env("LOG_LEVEL").unwrap_or_else(|| "INFO".to_string());

        Ok(Self { provider, model, api_key, temperature, max_tokens, base_url, log_level })
    }

    pub fn to_llm_config(&self) -> LLMConfig {
        LLMConfig {
            provider: self.provider.clone(),
            model: self.model.clone(),
            api_key: Some(self.api_key.clone()),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: 1.0,
            stop_sequences: None,
            timeout: 60,
            base_url: self.base_url.clone(),
        }
    }
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Config(provider={}, model={}, log_level={})", self.provider, self.model, self.log_level)
    }
}

pub fn create_llm_config() -> LLMConfig {
    match Config::load() {
        Ok(cfg) => cfg.to_llm_config(),
        Err(e) => {
            log::warn!("Config load failed (using defaults): {e}");
            LLMConfig {
                provider: "google".into(),
                model: "gemini-2.5-flash".into(),
                api_key: None,
                temperature: 0.7,
                max_tokens: 1024,
                top_p: 1.0,
                stop_sequences: None,
                timeout: 60,
                base_url: None,
            }
        }
    }
}

/// Provider detection helper — returns the provider from env or a default
pub fn detect_provider() -> String {
    ensure_env_loaded();
    let raw = env("LLM_PROVIDER")
        .or_else(|| env("AI_PROVIDER"))
        .or_else(|| env("PROVIDER"))
        .unwrap_or_else(|| "google".to_string());
    normalize_provider(&raw).unwrap_or_else(|| "google".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_provider() {
        assert_eq!(normalize_provider("google"), Some("google".into()));
        assert_eq!(normalize_provider("gemini"), Some("google".into()));
        assert_eq!(normalize_provider("google-ai"), Some("google".into()));
        assert_eq!(normalize_provider("gpt"), Some("openai".into()));
        assert_eq!(normalize_provider("chatgpt"), Some("openai".into()));
        assert_eq!(normalize_provider("claude"), Some("anthropic".into()));
        assert_eq!(normalize_provider("hf"), Some("huggingface".into()));
        assert_eq!(normalize_provider("hugging-face"), Some("huggingface".into()));
        assert_eq!(normalize_provider("unknown"), None);
    }

    #[test]
    fn test_default_model() {
        assert_eq!(default_model("google"), "gemini-2.5-flash");
        assert_eq!(default_model("openai"), "gpt-4o-mini");
        assert_eq!(default_model("anthropic"), "claude-3-haiku-20240307");
        assert_eq!(default_model("huggingface"), "mistralai/Mistral-7B-Instruct-v0.2");
    }

    #[test]
    fn test_to_llm_config() {
        let cfg = Config {
            provider: "openai".into(),
            model: "gpt-4".into(),
            api_key: "sk-test".into(),
            temperature: 0.5,
            max_tokens: 512,
            base_url: Some("https://api.openai.com/v1".into()),
            log_level: "DEBUG".into(),
        };
        let llm = cfg.to_llm_config();
        assert_eq!(llm.provider, "openai");
        assert_eq!(llm.model, "gpt-4");
        assert_eq!(llm.api_key, Some("sk-test".into()));
        assert_eq!(llm.temperature, 0.5);
        assert_eq!(llm.max_tokens, 512);
        assert_eq!(llm.base_url, Some("https://api.openai.com/v1".into()));
    }
}
