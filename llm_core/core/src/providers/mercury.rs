use crate::datam::{Message, ResponsePayload};
use crate::tools::ToolDefinition;
use crate::lucky::SimpleSchema;
use crate::error::LLMCoreError;

use super::{ProviderAdapter, ResponseParser};
use serde_json::{json, Value as JsonValue};
use std::collections::HashSet;
use reqwest::header;
use regex::Regex;

/// Adapter for the Inception Labs Mercury API.
pub struct MercuryAdapter;
/// Parser for the Mercury API response.
pub struct MercuryParser;

// Helper function to identify Mercury models that support native tools.
fn tool_supporting_models() -> HashSet<&'static str> {
    ["mercury-coder"].iter().cloned().collect()
}

impl ProviderAdapter for MercuryAdapter {
    fn prepare_request_payload(
        &self,
        model_tag: &str,
        messages: Vec<Message>,
        temperature: f32,
        _schema: Option<SimpleSchema>,
        tools: Option<&Vec<ToolDefinition>>,
        _thinking_mode: bool,
        debug: bool,
    ) -> JsonValue {
        // Mercury expects the 'arguments' in a tool_call message to be a string,
        // not a JSON object. We need to manually convert it for the synthesis turn.
        let processed_messages: Vec<JsonValue> = messages
            .clone() // Clone messages to check for the role later.
            .into_iter()
            .map(|mut msg| {
                if let Some(tool_calls) = &mut msg.tool_calls {
                    for call in tool_calls {
                        // If arguments is not already a string, stringify it.
                        if !call.function.arguments.is_string() {
                            call.function.arguments = json!(call.function.arguments.to_string());
                        }
                    }
                }
                // The Mercury API does not support the `name` field in tool messages.
                // We must remove it before serializing.
                if msg.role == "tool" {
                    msg.name = None;
                }
                serde_json::to_value(msg).unwrap()
            })
            .collect();

        let mut payload = json!({
            "model": model_tag,
            "messages": processed_messages,
            "temperature": temperature,
        });

        if let Some(tools_vec) = tools {
            if !tools_vec.is_empty() {
                payload["tools"] = json!(tools_vec);
                payload["tool_choice"] = json!("auto");
            }
        }

        // Add a debug print to inspect the final payload before sending.
        if debug {
            println!("[MERCURY ADAPTER DEBUG] Prepared Payload for Synthesis:\n{}\n", serde_json::to_string_pretty(&payload).unwrap());
        }

        payload
    }

    fn get_request_url(&self, base_url: &str, _model_tag: &str, _api_key: &str) -> String {
        format!("{}/chat/completions", base_url.trim_end_matches('/'))
    }

    fn get_request_headers(&self, api_key: &str) -> header::HeaderMap {
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
        // Forcing fallback to Lucky mode as Mercury's native schema support is unreliable.
        // It sometimes returns a tool call instead of the direct JSON, or ignores the request entirely.
        false
    }

    fn supports_tools(&self, model_tag: &str) -> bool {
        // Check if the specific model tag is in our set of known tool-supporting models.
        tool_supporting_models().contains(model_tag)
    }
}

impl ResponseParser for MercuryParser {
    fn parse_response(
            &self,
            raw_response_text: &str,
            _model_name: &str,
            input_price: f32,
            output_price: f32,
        ) -> Result<ResponsePayload, LLMCoreError> {
        let mut payload: ResponsePayload = serde_json::from_str(raw_response_text).map_err(|e| {
            LLMCoreError::ResponseParseError(format!(
                "Failed to parse Mercury response: {}. Raw text: {}",
                e, raw_response_text
            ))
        })?;

        if let Some(choice) = payload.choices.get_mut(0) {
            if let Some(content) = &mut choice.message.content {
                let think_re = Regex::new(r"(?is)<think>(.*)</think>").unwrap();
                if let Some(captures) = think_re.captures(content) {
                    if let Some(thought) = captures.get(1) {
                        choice.message.reasoning_content = Some(thought.as_str().trim().to_string());
                    }
                    *content = think_re.replace(content, "").trim().to_string();
                }
            }
        }

        if let Some(usage) = &mut payload.usage {
            usage.calculate_cost(input_price, output_price);
        }

        Ok(payload)
    }
} 