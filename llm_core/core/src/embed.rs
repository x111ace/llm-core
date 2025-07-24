use std::sync::Arc;

use crate::client::{self, Jitter, RetryPolicy};
use crate::config;
use crate::error::LLMCoreError;
use crate::providers::{
    openai::{OpenAIAdapter, OpenAIParser},
    unsupported::{UnsupportedAdapter, UnsupportedParser},
    ProviderAdapter, ResponseParser,
};

/// The primary engine for generating text embeddings.
pub struct Embedder {
    api_key: String,
    base_url: String,
    model_tag: String,
    pub dimensions: usize,
    provider_adapter: Arc<dyn ProviderAdapter>,
    response_parser: Arc<dyn ResponseParser>,
    retry_policy: RetryPolicy,
    debug: bool,
}

impl Embedder {
    /// Creates a new `Embedder` instance for a specific embedding model.
    pub fn new(model_name: &str, debug: Option<bool>) -> Result<Self, LLMCoreError> {
        let (provider_name, provider_data, model_details) =
            config::MODEL_LIBRARY.find_embedder(model_name).ok_or_else(|| {
                LLMCoreError::ConfigError(format!("Embedder '{}' not found in config", model_name))
            })?;

        let api_key = config::get_env_var(&provider_data.api_key)?;
        let base_url = config::get_env_var(&provider_data.base_url)?;

        let provider_adapter: Arc<dyn ProviderAdapter> = match provider_name {
            "OpenAI" => Arc::new(OpenAIAdapter),
            _ => Arc::new(UnsupportedAdapter {
                provider_name: provider_name.to_string(),
            }),
        };

        let response_parser: Arc<dyn ResponseParser> = match provider_name {
            "OpenAI" => Arc::new(OpenAIParser),
            _ => Arc::new(UnsupportedParser {
                provider_name: provider_name.to_string(),
            }),
        };

        if !provider_adapter.supports_embeddings(&model_details.model_tag) {
            return Err(LLMCoreError::ConfigError(format!(
                "Model '{}' via provider '{}' does not support embeddings.",
                model_name, provider_name
            )));
        }

        Ok(Self {
            api_key,
            base_url,
            model_tag: model_details.model_tag.clone(),
            dimensions: model_details.dimensions,
            provider_adapter,
            response_parser,
            retry_policy: RetryPolicy {
                max_retries: 3,
                base_delay_ms: 200,
                jitter: Jitter::Full,
            },
            debug: debug.unwrap_or(false),
        })
    }

    /// Generates embeddings for a list of texts.
    pub async fn get_embeddings(
        &self,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, LLMCoreError> {
        let payload = self
            .provider_adapter
            .prepare_embedding_request(&self.model_tag, texts);
        let headers = self.provider_adapter.get_request_headers(&self.api_key);

        let provider_name = self.provider_adapter.get_provider_name();
        let provider_config = config::MODEL_LIBRARY
            .providers
            .get(provider_name)
            .unwrap();

        let url = self.provider_adapter.get_embedding_url(
            &self.base_url,
            &self.model_tag,
            provider_config,
        );

        if self.debug {
            println!("[EMBEDDER DEBUG] URL: {}", url);
            println!("[EMBEDDER DEBUG] Payload: {}", payload);
        }

        let response_text = client::execute_single_call(url, headers, payload, &self.retry_policy)
            .await?;

        if self.debug {
            println!("[EMBEDDER DEBUG] Raw Response: {}", response_text);
        }

        self.response_parser.parse_embedding_response(&response_text)
    }
}
