// src/graph/nodes/providers/unsupported.rs

use super::{ProviderAdapter, ResponseParser};
use crate::datam::{Message, ResponsePayload};
use crate::tools::ToolDefinition;
use crate::lucky::SimpleSchema;
use crate::error::LLMCoreError;

use serde_json::{json, Value as JsonValue};
use reqwest::header;

/// An adapter for unsupported providers, providing graceful fallbacks and warnings.
pub struct UnsupportedAdapter {
    pub provider_name: String,
}

impl ProviderAdapter for UnsupportedAdapter {
    fn prepare_request_payload(
        &self,
        _model_tag: &str,
        messages: Vec<Message>,
        temperature: f32,
        _schema: Option<SimpleSchema>,
        tools: Option<&Vec<ToolDefinition>>,
        _thinking_mode: bool,
        _debug: bool,
    ) -> JsonValue {
        if tools.is_some() {
            eprintln!("[WARNING] Tool calling is not supported for provider: {}. Ignoring tool definitions.", self.provider_name);
        }
        eprintln!("[WARNING] Preparing a generic payload for unsupported provider: {}. This might not work.", self.provider_name);
        json!({
            "messages": messages,
            "temperature": temperature,
        })
    }

    fn get_request_url(&self, base_url: &str, _model_tag: &str, _api_key: &str) -> String {
        eprintln!("[WARNING] Using generic request URL for unsupported provider: {}.", self.provider_name);
        format!("{}/chat/completions", base_url)
    }

    fn get_request_headers(&self, api_key: &str) -> header::HeaderMap {
        eprintln!("[WARNING] Using generic headers for unsupported provider: {}.", self.provider_name);
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap(),
        );
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        headers
    }

    fn supports_native_schema(&self, _model_tag: &str) -> bool {
        false
    }

    fn supports_tools(&self, _model_tag: &str) -> bool {
        false
    }
}

/// A parser for unsupported providers, which will always return an error.
pub struct UnsupportedParser {
    pub provider_name: String,
}

impl ResponseParser for UnsupportedParser {
    fn parse_response(
            &self,
            raw_response_text: &str,
            _model_name: &str,
            _input_price: f32,
            _output_price: f32,
        ) -> Result<ResponsePayload, LLMCoreError> {
        Err(LLMCoreError::ConfigError(format!(
            "Provider '{}' is not supported. Raw response: {}",
            self.provider_name, raw_response_text
        )))
    }
} 