use crate::datam::{Choice, Message, ResponsePayload};
use crate::tools::{FunctionCall, ToolCall, ToolDefinition};
use crate::lucky::SimpleSchema;
use crate::error::LLMCoreError;

use super::{ProviderAdapter, ResponseParser};
use serde_json::{json, Value as JsonValue};
use serde::{Deserialize, Serialize};
use regex::Regex;

// --- Structs for Gemini API ---

/// Adapter for the Google Gemini API.
pub struct GoogleAdapter;

/// Parser for the Google Gemini API response.
pub struct GoogleParser;

#[derive(Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>, // Use the updated struct
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
}

#[derive(Serialize)]
struct GeminiFunctionResponse {
    name: String,
    response: JsonValue,
}


impl ProviderAdapter for GoogleAdapter {
    fn prepare_request_payload(
            &self,
            _model_tag: &str,
            messages: Vec<Message>,
            temperature: f32,
            schema: Option<SimpleSchema>,
            tools: Option<&Vec<ToolDefinition>>,
            _thinking_mode: bool,
            debug: bool,
        ) -> JsonValue {
        let mut system_prompt = String::new();
        let mut regular_messages = Vec::new();

        for message in messages {
            if message.role == "system" {
                if let Some(content) = message.content {
                    system_prompt.push_str(&content);
                    system_prompt.push('\n');
                }
            } else {
                regular_messages.push(message);
            }
        }
        
        let mut contents: Vec<GeminiContent> = Vec::new();
        for msg in regular_messages {
            let role = match msg.role.as_str() {
                "user" => "user",
                "assistant" => "model",
                "tool" => "function",
                _ => "user",
            };

            let mut parts = Vec::new();

            if let Some(tool_calls) = &msg.tool_calls {
                for tool_call in tool_calls {
                    let args: JsonValue = serde_json::from_str(&tool_call.function.arguments.to_string())
                        .unwrap_or(json!({}));
                    parts.push(GeminiPart {
                        text: None,
                        function_call: Some(GeminiFunctionCall {
                            name: tool_call.function.name.clone(),
                            args,
                        }),
                        function_response: None,
                    });
                }
            } else if let Some(text) = &msg.content {
                 if role == "function" {
                    // Try to parse the tool result string as JSON, which Gemini expects.
                    // If it fails, fall back to wrapping the raw string.
                    let response_json = serde_json::from_str(text).unwrap_or(json!({ "content": text }));
                     parts.push(GeminiPart {
                        text: None,
                        function_call: None,
                        function_response: Some(GeminiFunctionResponse {
                            name: msg.name.as_ref().expect("Tool name is required for function response").clone(),
                            response: response_json,
                        }),
                    });
                } else if !text.is_empty() {
                    parts.push(GeminiPart { text: Some(text.clone()), function_call: None, function_response: None });
                }
            }

            if !parts.is_empty() {
                contents.push(GeminiContent { role: role.to_string(), parts });
            }
        }

        let mut base_payload = json!({
            "contents": contents,
        });
        
        if !system_prompt.is_empty() {
            base_payload["systemInstruction"] = json!({
                "parts": [{ "text": system_prompt.trim() }]
            });
        }

        let mut generation_config = json!({ "temperature": temperature });

        if let Some(tools) = tools {
            let function_declarations: Vec<_> =
                tools.iter().map(|t| &t.function).collect();
            base_payload["tools"] = json!([
                { "function_declarations": function_declarations }
            ]);
        } else if schema.is_some() {
            generation_config["response_mime_type"] = json!("application/json");
        }
        
        if !generation_config.as_object().unwrap().is_empty() {
            base_payload["generationConfig"] = generation_config;
        }

        if debug {
            println!("\n[GEMINI ADAPTER DEBUG] Prepared Payload for Synthesis:\n{}\n", serde_json::to_string_pretty(&base_payload).unwrap());
        }

        base_payload
    }

    fn prepare_image_request_payload(&self, prompt: &str, _model_tag: &str) -> JsonValue {
        json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            }
        })
    }

    fn get_request_url(&self, base_url: &str, model_tag: &str, api_key: &str) -> String {
        format!(
            "{}/{}:generateContent?key={}",
            base_url.trim_end_matches('/'),
            model_tag,
            api_key
        )
    }

    fn get_image_request_url(&self, base_url: &str, model_tag: &str, api_key: &str) -> String {
        // Gemini uses the same `generateContent` endpoint for both text and images.
        self.get_request_url(base_url, model_tag, api_key)
    }

    fn get_request_headers(&self, _api_key: &str) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
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

// --- Response Structs ---

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: u32,
    total_token_count: u32,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContentResponse,
}

#[derive(Deserialize)]
struct GeminiContentResponse {
    parts: Vec<GeminiPartResponse>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPartResponse {
    text: Option<String>,
    #[serde(default)]
    function_call: Option<GeminiFunctionCall>,
}

// NEW STRUCT for parsing image data from the response
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImageGenPartResponse {
    text: Option<String>,
    inline_data: Option<GeminiInlineData>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    _mime_type: String,
    data: String,
}

#[derive(Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: JsonValue,
}


impl ResponseParser for GoogleParser {
    fn parse_response(
            &self,
            raw_response_text: &str,
            _model_name: &str,
            input_price: f32,
            output_price: f32,
        ) -> Result<ResponsePayload, LLMCoreError> {
        let gemini_response: GeminiResponse = serde_json::from_str(raw_response_text)?;

        let first_candidate = gemini_response
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| {
                LLMCoreError::ResponseParseError(
                    "Gemini response did not contain any candidates".to_string(),
                )
            })?;

        let mut content: Option<String> = None;
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut tool_name_for_message: Option<String> = None;

        for part in first_candidate.content.parts {
            if let Some(text) = part.text {
                content = Some(text);
            }
            if let Some(fc) = part.function_call {
                tool_name_for_message = Some(fc.name.clone());
                tool_calls.push(ToolCall {
                    id: format!("gemini-tool-{}", uuid::Uuid::new_v4()),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: fc.name,
                        arguments: fc.args, // Use the JsonValue directly
                    },
                });
            }
        }

        let mut reasoning_content: Option<String> = None;
        if let Some(c) = &mut content {
            let think_re = Regex::new(r"(?is)<think>(.*)</think>").unwrap();
            if let Some(captures) = think_re.captures(c) {
                if let Some(thought) = captures.get(1) {
                    reasoning_content = Some(thought.as_str().trim().to_string());
                }
                *c = think_re.replace(c, "").trim().to_string();
            }
        }

        Ok(ResponsePayload {
            id: format!("gemini-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: "gemini-synthetic".to_string(),
            choices: vec![Choice {
                message: Message {
                    role: "assistant".to_string(),
                    content,
                    name: tool_name_for_message,
                    tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                    reasoning_content,
                    ..Default::default()
                },
            }],
            usage: gemini_response.usage_metadata.map(|meta| {
                let mut usage = crate::datam::Usage {
                    prompt_tokens: meta.prompt_token_count,
                    completion_tokens: meta.candidates_token_count,
                    total_tokens: meta.total_token_count,
                    ..Default::default()
                };
                usage.calculate_cost(input_price, output_price);
                usage
            }),
        })
    }

    fn parse_image_response(
            &self,
            raw_response_text: &str,
        ) -> Result<(Option<String>, Option<String>), LLMCoreError> {
        let response_json: JsonValue = serde_json::from_str(raw_response_text)?;

        let parts: Vec<ImageGenPartResponse> = response_json
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| serde_json::from_value(p.clone()).ok())
            .ok_or_else(|| {
                LLMCoreError::ResponseParseError("Could not find 'parts' in Gemini image response".to_string())
            })?;

        let mut text_content = None;
        let mut image_data = None;

        for part in parts {
            if let Some(text) = part.text {
                text_content = Some(text);
            }
            if let Some(inline_data) = part.inline_data {
                image_data = Some(inline_data.data);
            }
        }

        Ok((text_content, image_data))
    }
} 