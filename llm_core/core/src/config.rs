use crate::error::LLMCoreError;
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::env;

pub mod toolkit;
pub mod storage;
pub use toolkit::get_rust_tool_library;

// --- Data Structures for models.json ---

/// An enum to represent the different reasoning capabilities of a model.
#[derive(Deserialize, Debug, Clone, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ReasoningCapability {
    Always,
    Toggle,
    #[default]
    PromptInducible,
}

/// Holds the specific details for an individual AI model.
#[derive(Deserialize, Debug, Clone)]
pub struct ModelDetails {
    pub model_tag: String,
    pub input_price: f32,
    #[serde(default)]
    pub output_price: f32,
    #[serde(default)]
    pub token_window: u32,
    #[serde(rename = "reasoning", default)]
    pub reasoning_capability: ReasoningCapability,
    #[serde(default)]
    pub dimensions: usize,
}

/// Holds the configuration for a specific provider, including API keys and models.
#[derive(Deserialize, Debug, Clone)]
pub struct ProviderConfig {
    pub api_key: String,
    pub base_url: String,
    pub models: HashMap<String, ModelDetails>,
    #[serde(default)]
    pub embedders: HashMap<String, ModelDetails>,
}

// --- Helper Function ---

/// Gets a variable from the environment, loading from a .env file first.
/// The `key_ref` is expected to be in the format "env:VAR_NAME".
pub fn get_env_var(key_ref: &str) -> Result<String, LLMCoreError> {
    dotenvy::dotenv().ok(); // Load .env file, ignore errors if it doesn't exist.
    if let Some(var_name) = key_ref.strip_prefix("env:") {
        env::var(var_name).map_err(|_| {
            LLMCoreError::ConfigError(format!(
                "Environment variable '{}' not found. Please set it in your .env file.",
                var_name
            ))
        })
    } else {
        // If it doesn't start with "env:", assume it's a literal value.
        Ok(key_ref.to_string())
    }
}


// --- ModelLibrary for loading and accessing model data ---

pub struct ModelLibrary {
    pub providers: HashMap<String, ProviderConfig>,
}

impl ModelLibrary {
    fn new() -> Result<Self, LLMCoreError> {
        // The `models.json` is included at compile time, making the library self-contained.
        let json_str = include_str!("config/models.json");

        let providers: HashMap<String, ProviderConfig> = serde_json::from_str(json_str)
            .map_err(|e| LLMCoreError::ConfigError(format!("Failed to parse models.json: {}", e)))?;

        Ok(ModelLibrary { providers })
    }

    // science: This lookup function efficiently finds model details by iterating through providers.
    // It now also returns the provider's friendly name (e.g., "OpenAI").
    pub fn find_model(&self, model_name: &str) -> Option<(&str, &ProviderConfig, &ModelDetails)> {
        for (provider_name, provider_data) in &self.providers {
            // First, search in the standard chat models.
            if let Some(model_details) = provider_data.models.get(model_name) {
                return Some((provider_name, provider_data, model_details));
            }
        }
        None
    }

    pub fn find_embedder(
            &self,
            embedder_name: &str,
        ) -> Option<(&str, &ProviderConfig, &ModelDetails)> {
        for (provider_name, provider_data) in &self.providers {
            if let Some(embedder_details) = provider_data.embedders.get(embedder_name) {
                return Some((provider_name, provider_data, embedder_details));
            }
        }
        None
    }
}

// --- Lazy Static Initializer ---

/// A global, lazily-initialized instance of the ModelLibrary.
///
/// This ensures the `models.json` file is read and parsed only once,
/// the first time it is accessed.
// --- Sorter Default Paths ---
pub static MANIFEST_DIR: Lazy<String> = Lazy::new(|| {
    env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR environment variable not set.")
});

pub static MODEL_LIBRARY: Lazy<ModelLibrary> =
    Lazy::new(|| ModelLibrary::new().expect("Failed to load model library from models.json"));

pub static DEFAULT_SORTER_INPUT_DIR: Lazy<PathBuf> = Lazy::new(|| {
    let mut path = PathBuf::from(MANIFEST_DIR.as_str());
    // This path is relative to the llm-core project root.
    path.push("src/data/prompt/sorter/data-to-sort");
    path
});

pub static DEFAULT_APP_DIR: Lazy<PathBuf> = Lazy::new(|| {
    home::home_dir()
        .map(|mut path| {
            path.push(".llm-core");
            path
        })
        .unwrap_or_else(|| PathBuf::from(".llm-core"))
});

pub static DEFAULT_SORTER_OUTPUT_DIR: Lazy<PathBuf> = Lazy::new(|| {
    let mut path = DEFAULT_APP_DIR.clone();
    path.push("output");
    path
});

pub static USAGE_DATA_DIR: Lazy<PathBuf> = Lazy::new(|| {
    let mut path = DEFAULT_APP_DIR.clone();
    path.push("usage");
    path
});
