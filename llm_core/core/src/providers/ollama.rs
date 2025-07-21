use crate::datam::{Choice, Message, ResponsePayload};
use crate::tools::ToolDefinition;
use crate::lucky::SimpleSchema;
use crate::error::LLMCoreError;

use super::{ProviderAdapter, ResponseParser};
use serde_json::{json, Value as JsonValue};
use serde::{Deserialize, Serialize};
use std::collections::HashSet; // Added for HashSet
use reqwest::header;
use regex::Regex;

/// Adapter for the Ollama API.
pub struct OllamaAdapter;

/// Parser for the Ollama API response.
pub struct OllamaParser;

// Helper to identify models that have custom tool-calling formats like Granite
fn granite_tool_supported_models() -> HashSet<&'static str> {
    ["granite3.3:2b"].iter().cloned().collect()
}

// Helper to identify models that have a 'think' option (like Qwen)
fn standard_ollama_think_supported_models() -> HashSet<&'static str> {
    ["qwen3:0.6b", "deepseek-r1:free"].iter().cloned().collect()
}

// Helper to identify models that support standard Ollama tool calls (like Qwen)
fn standard_ollama_tool_supported_models() -> HashSet<&'static str> {
    ["qwen3:0.6b", "llama3.2:1b"].iter().cloned().collect()
}

#[derive(Serialize)]
struct OllamaRequestOptions {
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>, // Added think option
}

#[derive(Serialize)]
struct OllamaRequestPayload {
    model: String,
    // Use JsonValue for messages to allow arbitrary roles and nesting for custom formats
    messages: Vec<JsonValue>, 
    stream: bool,
    options: OllamaRequestOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>, // Standard tools field (used by some models, but not Granite this way)
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>, // Add this to force tool use
}

impl ProviderAdapter for OllamaAdapter {
    fn prepare_request_payload(
            &self,
            model_tag: &str,
            messages: Vec<Message>, // Incoming messages are still standard
            temperature: f32,
            schema: Option<SimpleSchema>,
            tools: Option<&Vec<ToolDefinition>>,
            thinking_mode: bool,
            _debug: bool,
        ) -> JsonValue {
        let request_options = OllamaRequestOptions {
            temperature,
            think: if standard_ollama_think_supported_models().contains(model_tag) {
                Some(thinking_mode)
            } else {
                None
            },
        };

        // --- Granite-specific Tool Calling ---
        if granite_tool_supported_models().contains(model_tag.to_lowercase().as_str()) {
            // Helper function to convert a standard JSON Schema for parameters into Granite's simple format.
            let simplify_params = |params: &JsonValue| -> JsonValue {
                if let Some(props) = params.get("properties").and_then(|p| p.as_object()) {
                    let new_args: serde_json::Map<String, JsonValue> = props
                        .iter()
                        .map(|(key, value)| {
                            let desc = value.get("description").cloned().unwrap_or(json!(""));
                            (key.clone(), json!({ "description": desc }))
                        })
                        .collect();
                    return json!(new_args);
                }
                // Fallback for simple/empty objects.
                json!({})
            };
            
            // Separate system message from the rest of the conversation.
            let mut system_content = String::new();
            let mut other_messages = Vec::new();
            for msg in messages {
                if msg.role == "system" {
                    system_content = msg.content.unwrap_or_default();
                } else {
                    other_messages.push(serde_json::to_value(msg).unwrap());
                }
            }

            // Construct the tool/schema definition string in Granite's simplified format.
            let tools_json_string = if let Some(schema) = schema {
                let simplified_arguments: serde_json::Map<String, JsonValue> = schema.properties.iter()
                    .map(|p| (p.name.clone(), json!({"description": p.description})))
                    .collect();

                let tool_schema = json!({
                    "name": schema.name,
                    "description": schema.description,
                    "arguments": simplified_arguments,
                });
                Some(serde_json::to_string(&vec![tool_schema]).unwrap())
            } else if let Some(tools_vec) = tools {
                let granite_tools: Vec<JsonValue> = tools_vec.iter().map(|tool_def| {
                    json!({
                        "name": tool_def.function.name,
                        "description": tool_def.function.description,
                        "arguments": simplify_params(&tool_def.function.parameters)
                    })
                }).collect();
                Some(serde_json::to_string(&granite_tools).unwrap())
            } else {
                None
            };
            
            let mut final_messages = Vec::new();

            // 1. Build the special Granite system prompt.
            let granite_system_prompt = format!(
                "You have access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.{}",
                if system_content.is_empty() { "".to_string() } else { format!("\n\n{}", system_content) }
            );
            final_messages.push(json!({"role": "system", "content": granite_system_prompt}));

            // 2. Add the `available_tools` message if tools/schema are provided.
            if let Some(tools_str) = tools_json_string {
                final_messages.push(json!({
                    "role": "available_tools",
                    "content": tools_str
                }));
            }

            // 3. Add the rest of the conversation history.
            final_messages.extend(other_messages);

            return json!({
                "model": model_tag,
                "messages": final_messages,
                "stream": false,
                "options": request_options,
            });
        }

        // --- Standard Ollama Tool Calls & JSON Mode ---
        let mut processed_messages: Vec<JsonValue> = messages.into_iter().map(serde_json::to_value).filter_map(Result::ok).collect();
        let mut final_tools_vec = tools.cloned();
        let mut format_option: Option<String> = None;
        let mut tool_choice_option: Option<String> = None;

        // Models like Qwen support the standard 'tools' array in the payload.
        if standard_ollama_tool_supported_models().contains(model_tag) {
            // If we are about to use tools, we must ensure the system prompt guides the model correctly.
            if schema.is_some() || tools.is_some() {
                let tool_system_prompt = "You are a helpful assistant with access to tools. Use them when appropriate to answer the user's request.";
                
                if let Some(system_message) = processed_messages.iter_mut().find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system")) {
                    // Prepend the instruction to the existing system message to ensure it's seen first.
                    if let Some(content) = system_message.get("content").and_then(|c| c.as_str()) {
                        let new_content = format!("{}\n\n---\n\n{}", tool_system_prompt, content);
                        system_message["content"] = json!(new_content);
                    }
                } else {
                    // If no system message exists, insert one.
                    processed_messages.insert(0, json!({
                        "role": "system",
                        "content": tool_system_prompt
                    }));
                }
            }
            
            if let Some(schema_def) = schema {
                // If a schema is provided, convert it into the 'ToolDefinition' format
                // that these models expect.
                let tool_def = ToolDefinition {
                    tool_type: "function".to_string(),
                    function: crate::tools::FunctionDefinition {
                        name: schema_def.name,
                        description: schema_def.description,
                        parameters: json!({
                            "type": "object",
                            "properties": schema_def.properties.iter().map(|p| {
                                (p.name.clone(), json!({
                                    "type": p.property_type,
                                    "description": p.description,
                                    "items": p.items.as_ref().map(|i| json!({"type": i.item_type}))
                                }))
                            }).collect::<serde_json::Map<String, JsonValue>>(),
                            "required": schema_def.properties.iter().map(|p| p.name.clone()).collect::<Vec<String>>()
                        }),
                    },
                };
                final_tools_vec = Some(vec![tool_def]);
                // Force the model to use the provided tool.
                tool_choice_option = Some("required".to_string());
            } else if tools.is_some() {
                // If tools are provided directly, we can also require their use.
                tool_choice_option = Some("required".to_string());
            }
            // For any model that uses the `tools` array, `format` must be None to avoid conflicts.
            format_option = None;
        } else if schema.is_some() {
            // For models that don't support the 'tools' array, we fall back to
            // Ollama's basic `format: "json"` mode if a schema is requested.
            // This is less strict but provides basic JSON output.
            format_option = Some("json".to_string());
        }

        let payload = OllamaRequestPayload {
            model: model_tag.to_string(),
            messages: processed_messages,
            stream: false,
            options: request_options,
            format: format_option,
            tools: final_tools_vec,
            tool_choice: tool_choice_option,
        };
        serde_json::to_value(payload).unwrap()
    }

    fn get_request_url(&self, base_url: &str, _model_tag: &str, _api_key: &str) -> String {
        format!("{}/api/chat", base_url.trim_end_matches('/'))
    }

    fn get_request_headers(&self, _api_key: &str) -> header::HeaderMap {
        header::HeaderMap::new()
    }

    fn supports_native_schema(&self, model_tag: &str) -> bool {
        // Granite's native schema support is unreliable. It can return empty tool calls
        // or malformed JSON. We force it to use the Lucky fallback for consistency.
        // Standard models like Qwen do have reliable support.
        let lower_model_tag = model_tag.to_lowercase();
        standard_ollama_tool_supported_models().contains(lower_model_tag.as_str())
    }

    fn supports_tools(&self, model_tag: &str) -> bool {
        let lower_model_tag = model_tag.to_lowercase();
        granite_tool_supported_models().contains(lower_model_tag.as_str())
            || standard_ollama_tool_supported_models().contains(lower_model_tag.as_str())
    }
}

#[derive(Deserialize)]
struct OllamaResponse {
    model: String,
    created_at: String,
    message: OllamaMessage, // Use our new intermediate struct
    #[serde(rename = "done")]
    _done: bool,
    #[serde(default)]
    prompt_eval_count: u32,
    #[serde(default)]
    eval_count: u32,
}

// Intermediate struct to match raw Ollama response
#[derive(Deserialize)]
struct OllamaMessage {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Deserialize)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Deserialize)]
struct OllamaFunctionCall {
    name: String,
    arguments: JsonValue, // Ollama provides a JSON object
}


impl ResponseParser for OllamaParser {
    fn parse_response(
            &self,
            raw_response_text: &str,
            _model_name: &str,
            input_price: f32,
            output_price: f32,
        ) -> Result<ResponsePayload, LLMCoreError> {
        // --- Standard Ollama JSON Response Handling ---
        // Always parse the standard response structure first.
        let mut ollama_response: OllamaResponse =
            serde_json::from_str(raw_response_text).map_err(|e| {
                LLMCoreError::ResponseParseError(format!(
                    "Failed to parse Ollama response: {}. Raw text: {}",
                    e, raw_response_text
                ))
            })?;

        // --- Granite Tool Call Normalization ---
        // Granite returns tool calls inside the 'content' field, prefixed with a special tag.
        // We need to parse this and move it to the 'tool_calls' field for consistency.
        if let Some(content) = &ollama_response.message.content {
            let trimmed_content = content.trim();
            if trimmed_content.starts_with("<|tool_call|>") {
                if let Some(json_start_index) = trimmed_content.find('[') {
                    let json_part = &trimmed_content[json_start_index..];
                    
                    let tool_calls_result: Result<Vec<OllamaToolCall>, _> = serde_json::from_str(json_part);

                    if let Ok(parsed_calls) = tool_calls_result {
                        // Move the parsed calls to the dedicated field.
                        ollama_response.message.tool_calls = Some(parsed_calls);
                        // Clear the content field as it has been processed.
                        ollama_response.message.content = None;
                    }
                }
            }
        }
        
        let mut reasoning_content: Option<String> = None;

        // NEW (Corrected): Extract <think> blocks and place them in `reasoning_content`.
        if let Some(content) = &mut ollama_response.message.content {
            // Use regex to be more robust.
            let think_re = Regex::new(r"(?is)<think>(.*)</think>").unwrap();
            if let Some(captures) = think_re.captures(content) {
                if let Some(thought) = captures.get(1) {
                    reasoning_content = Some(thought.as_str().trim().to_string());
                }
                // Remove the entire think block from the original content, leaving the rest.
                *content = think_re.replace(content, "").trim().to_string();
            }
        }
        
        // FINAL NORMALIZATION: If the response contains tool calls, any accompanying `content`
        // is often model chatter or garbage. We normalize it by setting it to `None` to completely
        // remove the field, creating a clean history for the synthesis step.
        if ollama_response.message.tool_calls.is_some() {
            ollama_response.message.content = None;
        }
        
        let created_timestamp =
            chrono::DateTime::parse_from_rfc3339(&ollama_response.created_at)
                .map(|dt| dt.timestamp() as u64)
                .unwrap_or_else(|_| {
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                });

        let final_message = Message {
            role: ollama_response.message.role,
            content: ollama_response.message.content,
            tool_calls: ollama_response.message.tool_calls.map(|calls| {
                calls
                    .into_iter()
                    .map(|call| crate::tools::ToolCall {
                        id: format!("ollama-tool-{}", uuid::Uuid::new_v4()),
                        tool_type: "function".to_string(),
                        function: crate::tools::FunctionCall {
                            name: call.function.name,
                            arguments: call.function.arguments,
                        },
                    })
                    .collect()
            }),
            reasoning_content,
            ..Default::default()
        };

        Ok(ResponsePayload {
            id: format!("ollama-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: created_timestamp,
            model: ollama_response.model,
            choices: vec![Choice {
                message: final_message,
            }],
            usage: {
                let mut usage = crate::datam::Usage {
                    prompt_tokens: ollama_response.prompt_eval_count,
                    completion_tokens: ollama_response.eval_count,
                    total_tokens: ollama_response.prompt_eval_count + ollama_response.eval_count,
                    ..Default::default()
                };
                usage.calculate_cost(input_price, output_price);
                Some(usage)
            },
        })
    }
} 