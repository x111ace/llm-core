use crate::datam::{Message, ResponsePayload};
use crate::tools::ToolDefinition;
use crate::lucky::SimpleSchema;
use crate::error::LLMCoreError;
use crate::config::ProviderConfig;

use serde_json::{json, Value as JsonValue};
use reqwest::header;

/// A trait for provider-specific payload adjustments and request building.
///
/// Each provider (OpenAI, Google, etc.) will have its own implementation of this
/// trait to handle its unique API format.
pub trait ProviderAdapter: Send + Sync {
    /// Returns the friendly name of the provider (e.g., "OpenAI").
    fn get_provider_name(&self) -> &str;

    /// Prepares the base request payload specific to the provider's API.
    fn prepare_request_payload(
        &self,
        model_tag: &str,
        messages: Vec<Message>,
        temperature: f32,
        schema: Option<SimpleSchema>,
        tools: Option<&Vec<ToolDefinition>>,
        thinking_mode: bool,
        debug: bool,
    ) -> JsonValue;

    /// Prepares the payload for an embedding request.
    fn prepare_embedding_request(&self, _model_tag: &str, _texts: Vec<String>) -> JsonValue {
        // Default implementation for providers that don't support embeddings.
        json!({ "error": "Embeddings not supported by this provider." })
    }

    /// Prepares the payload for an image generation request.
    fn prepare_image_request_payload(&self, _prompt: &str, _model_tag: &str) -> JsonValue {
        // Default implementation returns an empty JSON object.
        // Providers that don't support image generation will effectively do nothing.
        json!({ "error": "Image generation not supported by this provider." })
    }

    /// Returns the full, provider-specific request URL.
    fn get_request_url(&self, base_url: &str, model_tag: &str, api_key: &str) -> String;

    /// Returns the full URL for an embedding request.
    fn get_embedding_url(
            &self,
            base_url: &str,
            _model_tag: &str,
            _provider_config: &ProviderConfig,
        ) -> String {
        // A sensible default that works for OpenAI-like APIs.
        format!("{}/embeddings", base_url.trim_end_matches('/'))
    }

    /// Returns the full URL for an image generation request.
    /// The default implementation can be a generic endpoint, but providers should override it.
    fn get_image_request_url(&self, base_url: &str, _model_tag: &str, api_key: &str) -> String {
        format!(
            "{}/images/generations?api_key={}",
            base_url.trim_end_matches('/'),
            api_key
        )
    }

    /// Returns the provider-specific request headers, including authentication.
    fn get_request_headers(&self, api_key: &str) -> header::HeaderMap;

    /// Returns `true` if the provider supports native JSON schema enforcement for a given model.
    fn supports_native_schema(&self, model_tag: &str) -> bool;

    /// Returns `true` if the provider supports native tool calling for a given model.
    fn supports_tools(&self, model_tag: &str) -> bool;

    /// Returns `true` if the provider supports embeddings for a given model.
    fn supports_embeddings(&self, _model_tag: &str) -> bool {
        false // Default to false for safety.
    }
}

/// A trait for provider-specific response parsing.
///
/// This allows each provider to handle its unique response format and
/// convert it into our standardized `ResponsePayload`.
pub trait ResponseParser: Send + Sync {
    /// Parses a raw JSON response string into a standardized `ResponsePayload`.
    fn parse_response(
            &self,
            raw_response_text: &str,
            model_name: &str,
            input_price: f32,
            output_price: f32,
        ) -> Result<ResponsePayload, LLMCoreError>;

    /// Parses the response from an image generation call into a tuple of (text, image_data).
    fn parse_image_response(
            &self,
            _raw_response_text: &str,
        ) -> Result<(Option<String>, Option<String>), LLMCoreError> {
        // Default implementation returns an error.
        Err(LLMCoreError::ImageGenerationError(
            "Image generation not supported by this provider's parser.".to_string(),
        ))
    }

    /// Parses the response from an embedding call into a list of vectors.
    fn parse_embedding_response(
            &self,
            _raw_response_text: &str,
        ) -> Result<Vec<Vec<f32>>, LLMCoreError> {
        Err(LLMCoreError::ResponseParseError(
            "Embedding parsing not supported by this provider.".to_string(),
        ))
    }
}

// We will declare the specific provider modules here as we create them.
pub mod gemini;
pub mod grok;
pub mod anthropic;
pub mod openai;
pub mod mercury;
pub mod ollama;
pub mod openrouter;
pub mod unsupported;
