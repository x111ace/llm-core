use super::{ProviderAdapter, ResponseParser};
use crate::datam::{Message, ResponsePayload};
use crate::tools::ToolDefinition;
use crate::lucky::SimpleSchema;
use crate::error::LLMCoreError;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use std::collections::HashSet;
use reqwest::header;
use regex::Regex;

/// Adapter for the xAI Grok API.
pub struct GrokAdapter;

/// Parser for the xAI Grok API response.
pub struct GrokParser;

fn get_reasoner_models() -> HashSet<&'static str> {
    ["grok-3-mini"].iter().cloned().collect()
}

#[derive(Serialize)]
struct GrokRequestPayload {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<JsonValue>,
}

impl ProviderAdapter for GrokAdapter {
    fn get_provider_name(&self) -> &str {
        "xAI"
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
        let reasoner_models = get_reasoner_models();
        let reasoning_effort = if reasoner_models.contains(model_tag) {
            Some("high".to_string())
        } else {
            None
        };

        let mut payload = GrokRequestPayload {
            model: model_tag.to_string(),
            messages,
            temperature,
            reasoning_effort,
            tool_choice: None,
            tools: None,
            json_schema: None,
        };

        // Grok can't handle tools and schema simultaneously. Prioritize tools.
        if let Some(tools) = tools {
            payload.tools = Some(tools.to_vec());
            payload.tool_choice = Some("auto".to_string());
        } else if let Some(schema) = schema {
            // According to the documentation, when using json_schema, we must
            // also specify a tool_choice with the function name.
            payload.tool_choice = Some(json!({
                "type": "function",
                "function": {"name": schema.name}
            }).to_string());
            
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

            // Create a single "tool" that wraps the JSON schema
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
            
            payload.tools = Some(vec![serde_json::from_value(function_tool).unwrap()]);
        }

        serde_json::to_value(payload).unwrap()
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
        true
    }

    fn supports_tools(&self, _model_tag: &str) -> bool {
        true
    }
}

#[derive(Deserialize, Clone)]
struct GrokMessage {
    role: String,
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<GrokToolCall>>,
}

#[derive(Deserialize, Clone)]
struct GrokChoice {
    message: GrokMessage,
}

#[derive(Deserialize)]
struct GrokResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Option<Vec<Option<GrokChoice>>>,
    usage: Option<GrokUsage>,
}

#[derive(Deserialize, Clone)]
struct GrokToolCall {
    function: GrokFunctionCall,
}

#[derive(Deserialize, Clone)]
struct GrokFunctionCall {
    name: String,
    arguments: JsonValue, // Grok provides a JSON object directly
}

#[derive(Deserialize, Clone)]
struct GrokUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl ResponseParser for GrokParser {
    fn parse_response(
            &self,
            raw_response_text: &str,
            _model_name: &str,
            input_price: f32,
            output_price: f32,
        ) -> Result<ResponsePayload, LLMCoreError> {
        let grok_response: GrokResponse = serde_json::from_str(raw_response_text)?;

        let mut choices = Vec::new();
        if let Some(candidates) = grok_response.choices {
            for grok_choice_opt in candidates {
                if let Some(grok_choice) = grok_choice_opt {
                    let mut message_content = grok_choice.message.content;
                    let mut reasoning_content = grok_choice.message.reasoning_content;

                    // Handle prompt-induced reasoning tags if native reasoning is absent.
                    if reasoning_content.is_none() {
                        if let Some(content) = &mut message_content {
                            let think_re = Regex::new(r"(?is)<think>(.*)</think>").unwrap();
                            if let Some(captures) = think_re.captures(content) {
                                if let Some(thought) = captures.get(1) {
                                    reasoning_content = Some(thought.as_str().trim().to_string());
                                }
                                *content = think_re.replace(content, "").trim().to_string();
                            }
                        }
                    }
                    
                    let message = Message {
                        role: grok_choice.message.role.clone(), // Use the role from the response
                        content: message_content,
                        reasoning_content,
                        tool_calls: grok_choice
                            .message
                            .tool_calls
                            .map(|calls| {
                                calls
                                    .into_iter()
                                    .map(|call| crate::tools::ToolCall {
                                        id: format!("grok-tool-{}", uuid::Uuid::new_v4()),
                                        tool_type: "function".to_string(),
                                        function: crate::tools::FunctionCall {
                                            name: call.function.name,
                                            arguments: call.function.arguments,
                                        },
                                    })
                                    .collect()
                            }),
                        ..Default::default()
                    };

                    choices.push(crate::datam::Choice { message });
                }
            }
        }

        Ok(ResponsePayload {
            id: grok_response.id,
            object: grok_response.object,
            created: grok_response.created,
            model: grok_response.model,
            choices,
            usage: grok_response.usage.map(|u| {
                let mut usage = crate::datam::Usage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                    ..Default::default()
                };
                usage.calculate_cost(input_price, output_price);
                usage
            }),
        })
    }
} 