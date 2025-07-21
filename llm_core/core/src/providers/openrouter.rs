use crate::datam::{Message, ResponsePayload};
use crate::tools::ToolDefinition;
use crate::lucky::SimpleSchema;
use crate::error::LLMCoreError;

use super::{ProviderAdapter, ResponseParser};
use serde_json::{json, Value as JsonValue};
use reqwest::header;
use regex::Regex;

/// Adapter for the OpenRouter API.
pub struct OpenRouterAdapter;

/// Parser for the OpenRouter API response.
pub struct OpenRouterParser;

impl ProviderAdapter for OpenRouterAdapter {
    fn prepare_request_payload(
        &self,
        model_tag: &str,
        messages: Vec<Message>,
        temperature: f32,
        schema: Option<SimpleSchema>,
        tools: Option<&Vec<ToolDefinition>>,
        _thinking_mode: bool,
        _debug: bool,
    ) -> JsonValue {
        let mut payload = json!({
            "model": model_tag,
            "messages": messages,
            "temperature": temperature,
        });

        if let Some(tools) = tools {
            payload["tools"] = json!(tools);
            payload["tool_choice"] = json!("auto");
        } else if let Some(schema) = schema {
            let mut properties = json!({});
            let mut required: Vec<String> = Vec::new();

            if let Some(props_obj) = properties.as_object_mut() {
                for prop in schema.properties {
                    let mut prop_val = json!({
                        "type": prop.property_type,
                        "description": prop.description,
                    });
                    if let Some(items) = prop.items {
                        prop_val["items"] = json!({ "type": items.item_type });
                    }
                    props_obj.insert(prop.name.clone(), prop_val);
                    required.push(prop.name);
                }
            }

            let modified_schema = json!({
                "type": "object",
                "properties": properties,
                "required": required,
            });

            let json_schema_payload = json!({
                "name": schema.name,
                "schema": modified_schema
            });

            payload["response_format"] = json!({
                "type": "json_schema",
                "json_schema": json_schema_payload
            });
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
        headers.insert(
            "HTTP-Referer",
            header::HeaderValue::from_static("https://github.com/your-repo/llm-core"),
        );
        headers.insert(
            "X-Title",
            header::HeaderValue::from_static("LLM Core"),
        );
        headers
    }

    fn supports_native_schema(&self, _model_tag: &str) -> bool {
        true
    }

    fn supports_tools(&self, _model_tag: &str) -> bool {
        true
    }
}

impl ResponseParser for OpenRouterParser {
    fn parse_response(
            &self,
            raw_response_text: &str,
            _model_name: &str,
            input_price: f32,
            output_price: f32,
        ) -> Result<ResponsePayload, LLMCoreError> {
        let raw_json: JsonValue = serde_json::from_str(raw_response_text)
            .map_err(|e| LLMCoreError::ResponseParseError(format!("Failed to parse OpenRouter response into JSON: {}", e)))?;

        let mut payload: ResponsePayload = serde_json::from_value(raw_json.clone())
            .map_err(|e| LLMCoreError::ResponseParseError(format!("Failed to parse OpenRouter response into standard payload: {}", e)))?;

        if let Some(usage) = &mut payload.usage {
            usage.calculate_cost(input_price, output_price);
        }

        if let Some(raw_choices) = raw_json.get("choices").and_then(|c| c.as_array()) {
            for (i, choice) in payload.choices.iter_mut().enumerate() {
                // First, handle prompt-induced <think> tags.
                if let Some(content) = &mut choice.message.content {
                    let think_re = Regex::new(r"(?is)<think>(.*)</think>").unwrap();
                    if let Some(captures) = think_re.captures(content) {
                        if let Some(thought) = captures.get(1) {
                            choice.message.reasoning_content = Some(thought.as_str().trim().to_string());
                        }
                        *content = think_re.replace(content, "").trim().to_string();
                    }
                }
                
                // Then, check for native reasoning content if none was found via tags.
                if choice.message.reasoning_content.is_none() {
                    if let Some(raw_choice) = raw_choices.get(i) {
                        if let Some(message) = raw_choice.get("message") {
                            let reasoning_content = message
                                .get("reasoning")
                                .or_else(|| message.get("reasoning_content"))
                                .and_then(|r| r.as_str().map(|s| s.trim().to_string()))
                                .filter(|s| !s.is_empty());

                            if reasoning_content.is_some() {
                                choice.message.reasoning_content = reasoning_content;
                            }
                        }
                    }
                }
            }
        }

        Ok(payload)
    }
} 