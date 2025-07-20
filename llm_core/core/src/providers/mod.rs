use crate::datam::{Message, ResponsePayload};
use crate::tools::ToolDefinition;
use crate::lucky::SimpleSchema;
use crate::error::LLMCoreError;

use serde_json::Value as JsonValue;
use reqwest::header;

/// A trait for provider-specific payload adjustments and request building.
///
/// Each provider (OpenAI, Google, etc.) will have its own implementation of this
/// trait to handle its unique API format.
pub trait ProviderAdapter: Send + Sync {
    /// Prepares the base request payload specific to the provider's API.
    fn prepare_request_payload(
        &self,
        model_tag: &str,
        messages: Vec<Message>,
        temperature: f32,
        schema: Option<SimpleSchema>,
        tools: Option<&Vec<ToolDefinition>>,
    ) -> JsonValue;

    /// Returns the full, provider-specific request URL.
    fn get_request_url(&self, base_url: &str, model_tag: &str, api_key: &str) -> String;

    /// Returns the provider-specific request headers, including authentication.
    fn get_request_headers(&self, api_key: &str) -> header::HeaderMap;

    /// Returns `true` if the provider supports native JSON schema enforcement for a given model.
    fn supports_native_schema(&self, model_tag: &str) -> bool;

    /// Returns `true` if the provider supports native tool calling for a given model.
    fn supports_tools(&self, model_tag: &str) -> bool;
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
