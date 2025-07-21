use super::{ProviderAdapter, ResponseParser};
use crate::datam::{Choice, Message, ResponsePayload};
use crate::error::LLMCoreError;
use crate::lucky::SimpleSchema;
use crate::tools::{ToolCall, ToolDefinition};
use reqwest::header;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use regex::Regex;

pub struct AnthropicAdapter;
pub struct AnthropicParser;

// --- Request Structs ---
#[derive(Serialize)]
struct AnthropicRequestPayload {
    model: String,
    messages: Vec<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: u32,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<JsonValue>,
}

// A new struct for Anthropic's tool format, which omits `tool_type`.
#[derive(Serialize, Clone)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: JsonValue,
}

// --- Response Structs ---
#[derive(Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContentBlock>,
    #[serde(rename = "stop_reason")]
    _stop_reason: String,
    usage: AnthropicUsage,
}

#[derive(Deserialize, Debug)]
struct TextBlock {
    #[serde(rename = "type")]
    _type: String,
    text: String,
}

#[derive(Deserialize, Debug)]
struct ToolUseBlock {
    #[serde(rename = "type")]
    _type: String,
    id: String,
    name: String,
    input: JsonValue,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum AnthropicContentBlock {
    Text(TextBlock),
    ToolUse(ToolUseBlock),
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl ProviderAdapter for AnthropicAdapter {
    fn prepare_request_payload(
        &self,
        model_tag: &str,
        mut messages: Vec<Message>,
        temperature: f32,
        schema: Option<SimpleSchema>,
        tools: Option<&Vec<ToolDefinition>>,
        _thinking_mode: bool,
        _debug: bool,
    ) -> JsonValue {
        // Anthropic uses a top-level `system` prompt.
        let system_prompt = messages
            .iter()
            .find(|m| m.role == "system")
            .and_then(|m| m.content.clone());
        messages.retain(|m| m.role != "system");

        let mut final_messages: Vec<JsonValue> = Vec::new();
        for msg in messages {
            if msg.role == "tool" {
                // Convert our standard tool result message into Anthropic's format.
                final_messages.push(json!({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content.unwrap_or_default()
                        }
                    ]
                }));
            } else if msg.role == "assistant" {
                // For assistant messages, we need to structure the content into an array of blocks,
                // especially when tool calls are present.
                let mut content_blocks: Vec<JsonValue> = Vec::new();
                if let Some(text_content) = &msg.content {
                    content_blocks.push(json!({"type": "text", "text": text_content}));
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    for call in tool_calls {
                        content_blocks.push(json!({
                            "type": "tool_use",
                            "id": &call.id,
                            "name": &call.function.name,
                            "input": &call.function.arguments
                        }));
                    }
                }
                final_messages.push(json!({
                    "role": "assistant",
                    "content": content_blocks
                }));
            } else {
                final_messages.push(serde_json::to_value(msg).unwrap());
            }
        }

        let mut final_tools: Option<Vec<AnthropicTool>> = None;
        if let Some(t) = tools {
            final_tools = Some(
                t.iter()
                    .map(|tool_def| {
                        let mut params = tool_def.function.parameters.clone();
                        // Anthropic requires a specific JSON schema version. We'll inject it
                        // if the parameters look like a valid schema object.
                        if let Some(obj) = params.as_object_mut() {
                            if obj.contains_key("type") && obj.contains_key("properties") {
                                obj.insert("$schema".to_string(), "http://json-schema.org/draft-2020-12/schema".into());
                            }
                        }
                        AnthropicTool {
                            name: tool_def.function.name.clone(),
                            description: tool_def.function.description.clone(),
                            input_schema: params,
                        }
                    })
                    .collect(),
            );
        }

        let mut tool_choice = None;

        if let Some(s) = schema {
            let schema_as_tool = AnthropicTool {
                name: s.name,
                description: s.description,
                input_schema: json!({
                    "$schema": "http://json-schema.org/draft-2020-12/schema",
                    "type": "object",
                    "properties": s.properties.iter().map(|p| {
                        let mut prop_json = json!({
                            "type": p.property_type,
                            "description": p.description,
                        });
                        if let Some(items) = &p.items {
                            prop_json["items"] = json!({"type": items.item_type});
                        }
                        (p.name.clone(), prop_json)
                    }).collect::<serde_json::Map<String, JsonValue>>(),
                    "required": s.properties.iter().map(|p| p.name.clone()).collect::<Vec<String>>()
                }),
            };
            final_tools = Some(vec![schema_as_tool.clone()]);
            // Force the model to use our schema tool.
            tool_choice = Some(json!({"type": "tool", "name": schema_as_tool.name}));
        }

        let payload = AnthropicRequestPayload {
            model: model_tag.to_string(),
            messages: final_messages,
            system: system_prompt,
            max_tokens: 4096, // A sensible default
            temperature,
            tools: final_tools,
            tool_choice,
        };

        serde_json::to_value(payload).unwrap()
    }

    fn get_request_url(&self, base_url: &str, _model_tag: &str, _api_key: &str) -> String {
        format!("{}/messages", base_url.trim_end_matches('/'))
    }

    fn get_request_headers(&self, api_key: &str) -> header::HeaderMap {
        let mut headers = header::HeaderMap::new();
        headers.insert("x-api-key", header::HeaderValue::from_str(api_key).unwrap());
        headers.insert("anthropic-version", header::HeaderValue::from_static("2023-06-01"));
        headers.insert(header::CONTENT_TYPE, header::HeaderValue::from_static("application/json"));
        headers
    }

    fn supports_native_schema(&self, _model_tag: &str) -> bool {
        true // We can enforce schemas via their tool-use functionality.
    }

    fn supports_tools(&self, _model_tag: &str) -> bool {
        true // All modern Claude models support tools.
    }
}

impl ResponseParser for AnthropicParser {
    fn parse_response(
            &self,
            raw_response_text: &str,
            _model_name: &str,
            input_price: f32,
            output_price: f32,
        ) -> Result<ResponsePayload, LLMCoreError> {
        let response: AnthropicResponse = serde_json::from_str(raw_response_text).map_err(|e| {
            LLMCoreError::ResponseParseError(format!(
                "Failed to parse Anthropic response: {}. Raw: {}",
                e, raw_response_text
            ))
        })?;

        let mut final_content = String::new();
        let mut tool_calls = Vec::new();

        for block in response.content {
            match block {
                AnthropicContentBlock::Text(text_block) => final_content.push_str(&text_block.text),
                AnthropicContentBlock::ToolUse(tool_block) => {
                    tool_calls.push(ToolCall {
                        id: tool_block.id,
                        tool_type: "function".to_string(),
                        function: crate::tools::FunctionCall {
                            name: tool_block.name,
                            arguments: tool_block.input, // Anthropic uses 'input' for arguments
                        },
                    });
                }
            }
        }

        let mut reasoning_content: Option<String> = None;
        let think_re = Regex::new(r"(?is)<think>(.*)</think>").unwrap();
        if let Some(captures) = think_re.captures(&final_content) {
            if let Some(thought) = captures.get(1) {
                reasoning_content = Some(thought.as_str().trim().to_string());
            }
            final_content = think_re.replace(&final_content, "").trim().to_string();
        }

        // The Anthropic API sometimes returns an empty content string for tool_use stops.
        // We'll treat an empty string as `None` for consistency with other providers.
        let final_content_option = if final_content.is_empty() {
            None
        } else {
            Some(final_content)
        };

        let final_message = Message {
            role: "assistant".to_string(),
            content: final_content_option,
            tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
            reasoning_content,
            ..Default::default()
        };

        Ok(ResponsePayload {
            id: response.id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            model: response.model,
            choices: vec![Choice {
                message: final_message,
            }],
            usage: {
                let mut usage = crate::datam::Usage {
                    prompt_tokens: response.usage.input_tokens,
                    completion_tokens: response.usage.output_tokens,
                    total_tokens: response.usage.input_tokens + response.usage.output_tokens,
                    ..Default::default()
                };
                usage.calculate_cost(input_price, output_price);
                Some(usage)
            },
        })
    }
}
