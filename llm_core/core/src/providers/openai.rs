use crate::datam::{Message, ResponsePayload};
use crate::tools::ToolDefinition;
use crate::lucky::SimpleSchema;
use crate::error::LLMCoreError;
use crate::config::ProviderConfig;

use super::{ProviderAdapter, ResponseParser};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use reqwest::header;
use regex::Regex;

/// Adapter for the OpenAI API.
pub struct OpenAIAdapter;

/// Parser for the OpenAI API response.
pub struct OpenAIParser;

impl ProviderAdapter for OpenAIAdapter {
    fn get_provider_name(&self) -> &str {
        "OpenAI"
    }

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
        // OpenAI expects tool_call arguments to be a string. We must re-serialize
        // our internal JSON object representation before sending it back.
        let processed_messages: Vec<JsonValue> = messages
            .into_iter()
            .map(|mut msg| {
                if msg.role == "assistant" {
                    if let Some(tool_calls) = &mut msg.tool_calls {
                        for call in tool_calls {
                            if !call.function.arguments.is_string() {
                                call.function.arguments = json!(call.function.arguments.to_string());
                            }
                        }
                    }
                }
                serde_json::to_value(msg).unwrap()
            })
            .collect();

        let mut payload = json!({
            "model": model_tag,
            "messages": processed_messages,
            "temperature": temperature,
        });

        // Priority: Tools > Schema.
        if let Some(tools) = tools {
            payload["tools"] = json!(tools);
            payload["tool_choice"] = json!("auto");
        } else if let Some(schema) = schema {
            // Convert the SimpleSchema into a ToolDefinition for OpenAI
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();

            for prop in schema.properties {
                let mut prop_val = json!({
                    "type": prop.property_type,
                    "description": prop.description,
                });
                if let Some(items) = prop.items {
                    prop_val["items"] = json!({ "type": items.item_type });
                }
                properties.insert(prop.name.clone(), prop_val);
                required.push(serde_json::Value::String(prop.name));
            }

            let function_tool = json!({
                "type": "function",
                "function": {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                }
            });

            // Add the tool to the payload and force the model to use it
            payload["tools"] = json!([function_tool]);
            payload["tool_choice"] = json!({"type": "function", "function": {"name": schema.name}});
        }

        payload
    }

    /// Prepares the payload for an embedding request.
    fn prepare_embedding_request(&self, model_tag: &str, texts: Vec<String>) -> JsonValue {
        json!({
            "model": model_tag,
            "input": texts,
            "encoding_format": "float"
        })
    }

    /// Prepares the payload for an image generation request.
    fn prepare_image_request_payload(&self, _prompt: &str, _model_tag: &str) -> JsonValue {
        json!({ "error": "Image generation not supported by OpenAI." })
    }

    fn get_request_url(&self, base_url: &str, _model_tag: &str, _api_key: &str) -> String {
        format!("{}/chat/completions", base_url)
    }

    /// Gets the URL for embedding requests.
    fn get_embedding_url(
            &self,
            base_url: &str,
            _model_tag: &str,
            _provider_config: &ProviderConfig,
        ) -> String {
        format!("{}/embeddings", base_url)
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
        true
    }

    fn supports_tools(&self, _model_tag: &str) -> bool {
        true
    }

    /// Checks if the model supports embeddings.
    fn supports_embeddings(&self, _model_tag: &str) -> bool {
        // OpenAI supports embeddings for models like text-embedding-3-small and text-embedding-3-large.
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    pub tool_type: String,
    pub function: OpenAIFunctionCall,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: JsonValue,
}

// We need a separate struct for Deserialization to handle the string-to-JsonValue conversion.
#[derive(Debug, Clone, Deserialize)]
struct OpenAIFunctionCallForParsing {
    name: String,
    arguments: String,
}

impl<'de> Deserialize<'de> for OpenAIFunctionCall {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let parsed: OpenAIFunctionCallForParsing = Deserialize::deserialize(deserializer)?;
        let args_json: JsonValue = serde_json::from_str(&parsed.arguments)
            .map_err(serde::de::Error::custom)?;
        Ok(OpenAIFunctionCall {
            name: parsed.name,
            arguments: args_json,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEmbeddingResponse {
    pub object: String,
    pub data: Vec<OpenAIEmbeddingData>,
    pub model: String,
    pub usage: OpenAIEmbeddingUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEmbeddingData {
    pub object: String,
    pub index: u32,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIEmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

impl ResponseParser for OpenAIParser {
    fn parse_response(
        &self,
        raw_response_text: &str,
        _model_name: &str,
        input_price: f32,
        output_price: f32,
    ) -> Result<ResponsePayload, LLMCoreError> {
        let mut payload: ResponsePayload = serde_json::from_str(raw_response_text)?;

        if let Some(choice) = payload.choices.get_mut(0) {
            // NEW: Handle prompt-induced reasoning by parsing <think> tags.
            if let Some(content) = &mut choice.message.content {
                let think_re = Regex::new(r"(?is)<think>(.*)</think>").unwrap();
                if let Some(captures) = think_re.captures(content) {
                    if let Some(thought) = captures.get(1) {
                        choice.message.reasoning_content = Some(thought.as_str().trim().to_string());
                    }
                    *content = think_re.replace(content, "").trim().to_string();
                }
            }

            // OpenAI returns tool arguments as a stringified JSON. We must parse it.
            if let Some(tool_calls) = &mut choice.message.tool_calls {
                for call in tool_calls {
                    if let Some(args_str) = call.function.arguments.as_str() {
                        match serde_json::from_str(args_str) {
                            Ok(parsed_args) => {
                                call.function.arguments = parsed_args;
                            }
                            Err(_) => {
                                // If parsing fails (e.g., empty string), default to an empty JSON object.
                                call.function.arguments = json!({});
                            }
                        }
                    }
                }
            }

            // Also enforce the rule that assistant messages with tool_calls have non-null content.
            if choice.message.role == "assistant" && choice.message.tool_calls.is_some() {
                if choice.message.content.is_none() {
                    choice.message.content = Some("".to_string());
                }
            }
        }

        if let Some(usage) = &mut payload.usage {
            usage.calculate_cost(input_price, output_price);
        }

        Ok(payload)
    }

    /// Parses the response from an embedding call into a list of vectors.
    fn parse_embedding_response(
        &self,
        raw_response_text: &str,
    ) -> Result<Vec<Vec<f32>>, LLMCoreError> {
        let response: OpenAIEmbeddingResponse = serde_json::from_str(raw_response_text)?;
        let embeddings = response
            .data
            .into_iter()
            .map(|data| data.embedding)
            .collect();
        Ok(embeddings)
    }
}
