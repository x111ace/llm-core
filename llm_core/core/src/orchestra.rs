use pyo3::{prelude::*, types::PyDict};

use crate::bindings;
use crate::config;
use crate::client::{self, Jitter, RetryPolicy};
use crate::datam::{
    format_system_message, format_tool_message, format_user_message, Message, ResponsePayload,
};
use crate::tools::{Tool, ToolDefinition, ToolLibrary};
use crate::lucky::{self, SimpleSchema};
use crate::error::LLMCoreError;
use crate::providers::{
    gemini::{GoogleAdapter, GoogleParser},
    grok::{GrokAdapter, GrokParser},
    anthropic::{AnthropicAdapter, AnthropicParser},
    openai::{OpenAIAdapter, OpenAIParser},
    mercury::{MercuryAdapter, MercuryParser},
    ollama::{OllamaAdapter, OllamaParser},
    openrouter::{OpenRouterAdapter, OpenRouterParser},
    unsupported::{UnsupportedAdapter, UnsupportedParser},
    ProviderAdapter, ResponseParser,
};

use serde_json::Value as JsonValue;
use serde_json::{json};
use std::sync::Arc;

// NEW: Internal enums to represent the chosen strategy for a given Orchestra instance.
// This simplifies the logic inside call_ai by having the strategy pre-determined.

#[derive(Clone)]
enum InternalToolStrategy {
    /// Use the provider's native `tools` payload. The `ToolLibrary` is stored here.
    Payload(Arc<ToolLibrary>),
    /// Use the `Lucky` mode to generate a JSON object representing a tool call.
    /// Stores the `ToolLibrary` and the specific `JsonValue` format for the Lucky prompt.
    Lucky(Arc<ToolLibrary>, JsonValue),
    /// No tools are being used.
    None,
}

#[derive(Clone)]
enum InternalStructuredStrategy {
    /// Use the provider's native schema enforcement mode.
    Schema(SimpleSchema),
    /// Use the `Lucky` mode to generate a JSON object matching a specific format.
    Lucky(JsonValue),
    /// No specific structure is enforced.
    None,
}

/// A new public struct to hold the results of an image generation call.
#[pyclass]
#[derive(Debug, Clone)]
pub struct ImageGenerationResult {
    #[pyo3(get)]
    pub text_response: Option<String>,
    #[pyo3(get)]
    pub image_data_b64: Option<String>,
}

/// The main orchestrator for making LLM calls.
/// This struct holds the configuration for a specific model and provider.
#[derive(Clone)]
pub struct Orchestra {
    api_key: String,
    base_url: String,
    model_tag: String,
    pub user_facing_model_name: String,
    input_price: f32,
    output_price: f32,
    provider_adapter: Arc<dyn ProviderAdapter>,
    response_parser: Arc<dyn ResponseParser>,
    temperature: f32,
    retry_policy: RetryPolicy,
    debug: bool,
    
    // NEW: Internal strategy fields, determined at initialization.
    tool_strategy: InternalToolStrategy,
    structured_strategy: InternalStructuredStrategy,
}

impl Orchestra {
    /// Creates a new `Orchestra` instance.
    ///
    /// This function looks up the model configuration from `models.json`,
    /// resolves the correct provider, and intelligently determines the best strategy
    /// (e.g., native tools vs. `Lucky` mode fallback) based on the provider's capabilities.
    pub fn new(
            model_name: &str,
            temperature: Option<f32>,
            tools: Option<ToolLibrary>,
            schema: Option<SimpleSchema>,
            debug: Option<bool>,
        ) -> Result<Self, LLMCoreError> {
        // --- Configuration Validation ---
        if tools.is_some() && schema.is_some() {
            return Err(LLMCoreError::ConfigError(
                "Unsupported configuration: Cannot provide both a tool library and a schema simultaneously. To enforce a structured output, please define it as a single tool in the tool library.".to_string()
            ));
        }

        let debug_mode = debug.unwrap_or(false);

        let (provider_name, provider_data, model_details) = config::MODEL_LIBRARY
            .find_model(model_name)
            .ok_or_else(|| {
                LLMCoreError::ConfigError(format!("Model '{}' not found in `models.json`", model_name))
            })?;

        let provider_adapter: Arc<dyn ProviderAdapter> = match provider_name {
            "OpenAI" => Arc::new(OpenAIAdapter),
            "Google" => Arc::new(GoogleAdapter),
            "xAI" => Arc::new(GrokAdapter),
            "Inception Labs" => Arc::new(MercuryAdapter),
            "OpenRouter" => Arc::new(OpenRouterAdapter),
            "Ollama" => Arc::new(OllamaAdapter),
            "Anthropic" => Arc::new(AnthropicAdapter),
            _ => Arc::new(UnsupportedAdapter { provider_name: provider_name.to_string() }),
        };
        let response_parser: Arc<dyn ResponseParser> = match provider_name {
            "OpenAI" => Arc::new(OpenAIParser),
            "Inception Labs" => Arc::new(MercuryParser),
            "OpenRouter" => Arc::new(OpenRouterParser),
            "Google" => Arc::new(GoogleParser),
            "xAI" => Arc::new(GrokParser),
            "Ollama" => Arc::new(OllamaParser),
            "Anthropic" => Arc::new(AnthropicParser),
            _ => Arc::new(UnsupportedParser { provider_name: provider_name.to_string() }),
        };

        // --- Determine Strategy based on Provider Capabilities ---
        
        let tool_strategy = if let Some(tool_lib) = tools {
            let arc_tool_lib = Arc::new(tool_lib);
            if provider_adapter.supports_tools(&model_details.model_tag) {
                if debug_mode {
                    println!("[Orchestra] Model supports native tools. Using Payload strategy.");
                }
                InternalToolStrategy::Payload(arc_tool_lib)
            } else {
                if debug_mode {
                    println!("[ORCHESTRA DEBUG] Model does not support native tools. Falling back to Lucky strategy for tool calls.");
                }
                let lucky_tool_schema = json!({
                    "tool_name": "<type:str>",
                    "arguments": "<type:object>"
                });
                InternalToolStrategy::Lucky(arc_tool_lib, lucky_tool_schema)
            }
        } else {
            InternalToolStrategy::None
        };
        
        let structured_strategy = if let Some(s) = schema {
            if provider_adapter.supports_native_schema(&model_details.model_tag) {
                if debug_mode {
                    println!("[ORCHESTRA DEBUG] Model supports native schema. Using Schema strategy.");
                }
                InternalStructuredStrategy::Schema(s)
            } else {
                if debug_mode {
                    println!("[ORCHESTRA DEBUG] Model does not support native schema. Falling back to Lucky strategy for structured response.");
                }
                let lucky_json_map: serde_json::Map<String, JsonValue> = s.properties.iter().map(|prop| {
                    let type_hint = match prop.property_type.as_str() {
                        "array" => json!(["<type:string>"]), // Simple default for array items
                        _ => json!(format!("<type:{}>", prop.property_type)),
                    };
                    (prop.name.clone(), type_hint)
                }).collect();
                let lucky_format = JsonValue::Object(lucky_json_map);
                InternalStructuredStrategy::Lucky(lucky_format)
            }
        } else {
            InternalStructuredStrategy::None
        };

        let api_key = config::get_env_var(&provider_data.api_key)?;
        let base_url = config::get_env_var(&provider_data.base_url)?;

        Ok(Self {
            api_key,
            base_url,
            model_tag: model_details.model_tag.clone(),
            user_facing_model_name: model_name.to_string(),
            input_price: model_details.input_price,
            output_price: model_details.output_price,
            provider_adapter,
            response_parser,
            temperature: temperature.unwrap_or(0.7),
            retry_policy: RetryPolicy {
                max_retries: 3,
                base_delay_ms: 200,
                jitter: Jitter::Full,
            },
            tool_strategy,
            structured_strategy,
            debug: debug_mode,
        })
    }

    /// Generates an image based on a prompt using a specified image model.
    ///
    /// This function is separate from the main chat flow and uses the new
    /// `prepare_image_request_payload` and `parse_image_response` methods
    /// on the provider adapters.
    pub async fn generate_image(
            &self,
            prompt: &str,
            image_model_name: &str, // e.g., "GEMINI 2.0 FLASH IMAGE GEN"
        ) -> Result<ImageGenerationResult, LLMCoreError> {
        let (_provider_name, _provider_data, model_details) =
            config::MODEL_LIBRARY.find_model(image_model_name).ok_or_else(|| {
                LLMCoreError::ConfigError(format!(
                    "Image model '{}' not found in `models.json`",
                    image_model_name
                ))
            })?;
        
        // We can reuse the existing provider adapter and parser from the Orchestra instance
        // as long as the image model belongs to the same provider. This is a reasonable
        // assumption for now.

        let url = self.provider_adapter.get_image_request_url(
            &self.base_url,
            &model_details.model_tag,
            &self.api_key,
        );
        let headers = self.provider_adapter.get_request_headers(&self.api_key);
        let payload = self.provider_adapter
            .prepare_image_request_payload(prompt, &model_details.model_tag);

        if self.debug {
            println!("[ORCHESTRA DEBUG] Image generation payload: {:?}", payload);
        }

        let response_text =
            client::execute_single_call(url, headers, payload, &self.retry_policy).await?;
        
        if self.debug {
            println!("[ORCHESTRA DEBUG] Raw image response: {}", response_text);
        }

        let (text_response, image_data_b64) = self
            .response_parser
            .parse_image_response(&response_text)?;
            
        Ok(ImageGenerationResult {
            text_response,
            image_data_b64,
        })
    }

    pub fn model_tag(&self) -> &str {
        &self.model_tag
    }

    /// Executes the first API call in a potential multi-step conversation.
    /// It prepares the prompt according to the determined strategy (Native vs. Lucky)
    /// and parses the initial response.
    async fn execute_initial_turn(
            &self,
            messages: Vec<Message>,
        ) -> Result<(ResponsePayload, Vec<Message>), LLMCoreError> {
        let mut final_messages = messages.clone();
        let mut schema_for_provider: Option<SimpleSchema> = None;
        let mut tools_for_provider: Option<Vec<ToolDefinition>> = None;

        // --- Prepare prompts and payload based on determined strategy ---

        let is_synthesis_turn = messages.last().map_or(false, |m| m.role == "tool");

        // Handle structured response strategy
        if let InternalStructuredStrategy::Lucky(output_format) = &self.structured_strategy {
            let (system, user) = self.get_prompts_from_messages(&final_messages);
            let (lucky_system, lucky_user) = lucky::prepare_lucky_prompt(system, user, output_format, "###", None, is_synthesis_turn);
            final_messages = vec![format_system_message(lucky_system), format_user_message(lucky_user)];
        } else if let InternalStructuredStrategy::Schema(s) = &self.structured_strategy {
            schema_for_provider = Some(s.clone());
        }
        
        // Handle tool strategy
        if let InternalToolStrategy::Lucky(tool_lib, output_format) = &self.tool_strategy {
            let tool_defs = tool_lib.values().map(|t| t.definition().clone()).collect::<Vec<_>>();
            let (system, user) = self.get_prompts_from_messages(&final_messages);
            let (lucky_system, lucky_user) = lucky::prepare_lucky_prompt(system, user, output_format, "###", Some(&tool_defs), is_synthesis_turn);
            final_messages = vec![format_system_message(lucky_system), format_user_message(lucky_user)];
        } else if let InternalToolStrategy::Payload(tool_lib) = &self.tool_strategy {
             if !is_synthesis_turn {
                tools_for_provider = Some(tool_lib.values().map(|t| t.definition().clone()).collect());
             }
        }
        
        // --- Execute API Call ---
        let url = self.provider_adapter.get_request_url(&self.base_url, &self.model_tag, &self.api_key);
        let headers = self.provider_adapter.get_request_headers(&self.api_key);
        let payload = self.provider_adapter.prepare_request_payload(
            &self.model_tag,
            final_messages.clone(),
            self.temperature,
            schema_for_provider,
            tools_for_provider.as_ref(),
        );

        let response_text = client::execute_single_call(url, headers, payload, &self.retry_policy).await?;
        if self.debug {
            println!("[ORCHESTRA DEBUG] Raw response from model: {}", response_text);
        }

        let initial_payload = self.response_parser.parse_response(
            &response_text,
            &self.user_facing_model_name,
            self.input_price,
            self.output_price,
        )?;

        // --- Normalize response for different provider behaviors ---
        // OpenAI-style providers return schema results in `tool_calls`. We normalize this
        // by moving the result into the `content` field to provide a consistent output.
        let mut processed_payload = initial_payload;
        if let InternalStructuredStrategy::Schema(schema) = &self.structured_strategy {
            if let Some(choice) = processed_payload.choices.get_mut(0) {
                let mut extracted_args: Option<JsonValue> = None;

                // First, check tool_calls for a matching schema name (for OpenAI, Grok, etc.).
                if let Some(tool_calls) = &choice.message.tool_calls {
                    if tool_calls.len() == 1 && tool_calls[0].function.name == schema.name {
                        extracted_args = Some(tool_calls[0].function.arguments.clone());
                    }
                }

                // If not found, check if the content itself is a tool-call-like JSON (for Ollama, Mercury, etc.).
                if extracted_args.is_none() {
                    if let Some(content) = &choice.message.content {
                        if let Ok(json_content) = serde_json::from_str::<JsonValue>(content) {
                            // Relaxed check: If it has 'name' and 'arguments', we assume it's the schema response.
                            // This handles cases where models (like Qwen) hallucinate the wrong function name.
                            if let (Some(_name), Some(args)) = (json_content.get("name"), json_content.get("arguments")) {
                                extracted_args = Some(args.clone());
                            }
                        }
                    }
                }
                
                // If we extracted arguments from either source, normalize the content.
                if let Some(arguments_json) = extracted_args {
                    let content_str = if let Some(s) = arguments_json.as_str() {
                        s.to_string()
                    } else {
                        arguments_json.to_string()
                    };
                    choice.message.content = Some(content_str);
                    // This was for schema enforcement, not execution, so clear tool_calls.
                    choice.message.tool_calls = None;
                }
            }
        }

        // --- Post-process if Lucky strategy was used ---
        let final_payload = match (&self.structured_strategy, &self.tool_strategy) {
            (InternalStructuredStrategy::Lucky(fmt), _) | (_, InternalToolStrategy::Lucky(_, fmt)) => {
                let content = processed_payload
                    .choices
                    .get(0)
                    .and_then(|c| c.message.content.as_ref())
                    .ok_or(LLMCoreError::ResponseParseError(
                        "No content for Lucky parsing".to_string(),
                    ))?;
                let lucky_json = lucky::parse_lucky_response(content, fmt, "###")?;
                let mut new_payload = processed_payload;
                if let Some(choice) = new_payload.choices.get_mut(0) {
                    choice.message.content = Some(serde_json::to_string(&lucky_json)?);
                }
                new_payload
            }
            _ => processed_payload,
        };

        Ok((final_payload, messages))
    }

    /// Handles the multi-step tool execution cycle if the initial response contained a tool call.
    async fn handle_tool_cycle(
            &self,
            initial_payload: ResponsePayload,
            mut messages: Vec<Message>,
        ) -> Result<ResponsePayload, LLMCoreError> {
        let tool_library_arc = match &self.tool_strategy {
            InternalToolStrategy::Payload(lib) | InternalToolStrategy::Lucky(lib, _) => Some(lib),
            InternalToolStrategy::None => None,
        };

        let has_tool_calls = initial_payload.choices.get(0).map_or(false, |c| {
            // Standard providers populate `tool_calls`.
            let has_native_call = c.message.tool_calls.is_some();
            // Granite returns a special tag in `content`.
            let has_granite_call = c.message.content.as_deref().map_or(false, |s| s.trim().starts_with("<|tool_call|>"));
            // Our `Lucky` fallback puts the tool call JSON in `content`.
            let has_lucky_call = matches!(self.tool_strategy, InternalToolStrategy::Lucky(_, _)) && c.message.content.is_some();
            
            has_native_call || has_granite_call || has_lucky_call
        });
        
        if !has_tool_calls || tool_library_arc.is_none() {
            return Ok(initial_payload);
        }
        
        let tool_library = tool_library_arc.unwrap();
        let mut assistant_message = initial_payload.choices.get(0).unwrap().message.clone();
        
        // --- Extract and Execute Tool Calls ---
        let has_non_empty_tool_calls = assistant_message.tool_calls.as_ref().map_or(false, |calls| !calls.is_empty());

        if has_non_empty_tool_calls {
            // --- Standard Native Tool Call Path ---
            messages.push(assistant_message.clone()); // Add the full assistant message to history.
            let tool_calls = assistant_message.tool_calls.take().unwrap(); // Now take the tool calls to run them.
            for call in tool_calls {
                let result = self
                    .execute_tool(
                        Arc::clone(tool_library),
                        &call.function.name,
                        call.function.arguments,
                    )
                    .await;
                messages.push(format_tool_message(result, call.id, call.function.name));
            }
        } else if let Some(content) = assistant_message.content.take() {
            let content_trimmed = content.trim();
            if content_trimmed.starts_with("<|tool_call|>") {
                // --- Granite Native Tool Call Path ---
                messages.push(assistant_message); // Add assistant's turn to history
                if let Some(json_start) = content_trimmed.find('[') {
                    let json_part = &content_trimmed[json_start..];
                    if let Ok(calls) = serde_json::from_str::<Vec<crate::tools::FunctionCall>>(json_part)
                    {
                        for call in calls {
                            let tool_id = format!("granite-tool-{}", uuid::Uuid::new_v4());
                            let result = self
                                .execute_tool(Arc::clone(tool_library), &call.name, call.arguments)
                                .await;
                            messages.push(format_tool_message(result, tool_id, call.name));
                        }
                    }
                }
            } else if matches!(self.tool_strategy, InternalToolStrategy::Lucky(_, _)) {
                // --- Lucky Fallback Tool Call Path ---
                let tool_data: JsonValue = serde_json::from_str(&content)?;
                let name = tool_data["tool_name"]
                    .as_str()
                    .ok_or(LLMCoreError::ResponseParseError(
                        "`tool_name` not found in Lucky tool call".to_string(),
                    ))?
                    .to_string();
                let mut args = tool_data["arguments"].clone();

                // Normalize arguments. If the model returns a non-object (like an empty string),
                // default it to an empty JSON object to ensure compatibility with downstream APIs.
                if !args.is_object() {
                    args = json!({});
                }
                
                let tool_id = format!("lucky-tool-{}", uuid::Uuid::new_v4());
                
                let result = self
                    .execute_tool(Arc::clone(tool_library), &name, args.clone())
                    .await;
                
                // Add assistant's "thought" (the tool call) and the result to history
                messages.push(Message { role: "assistant".to_string(), tool_calls: Some(vec![crate::tools::ToolCall { id: tool_id.clone(), tool_type: "function".to_string(), function: crate::tools::FunctionCall { name: name.clone(), arguments: args }}]), ..Default::default()});
                messages.push(format_tool_message(result, tool_id, name));
            }
        }
        
        // --- Make Synthesis Call ---
        if self.debug {
            println!("[ORCHESTRA DEBUG] Synthesizing tool results...");
        }

        // Add a final system message to guide the synthesis turn.
        // This is a strong prompt engineering technique that places the final instruction
        // at the end of the context, which is often more effective for some models.
        // let synthesis_system_prompt = "You have just received the result from a tool. Your task is to respond to the user's original query in a natural, conversational way based on the tool's output.";
        // messages.push(format_system_message(synthesis_system_prompt.to_string()));
        
        let synthesis_messages_for_debug = messages.clone(); // Clone for debugging.
        let url = self.provider_adapter.get_request_url(&self.base_url, &self.model_tag, &self.api_key);
        let headers = self.provider_adapter.get_request_headers(&self.api_key);
        let payload = self.provider_adapter.prepare_request_payload(&self.model_tag, messages, self.temperature, None, None);
        let final_text = client::execute_single_call(url, headers, payload, &self.retry_policy).await?;
        
        let final_payload = self.response_parser.parse_response(
            &final_text,
            &self.user_facing_model_name,
            self.input_price,
            self.output_price,
        )?;

        if self.debug {
            println!("[ORCHESTRA DEBUG] Final message history sent for synthesis:\n{:#?}", synthesis_messages_for_debug);
        }

        Ok(final_payload)
    }

    /// Executes a single tool function and returns the result as a string.
    /// This is now async and uses `spawn_blocking` to avoid stalling the runtime.
    async fn execute_tool(&self, library: Arc<ToolLibrary>, name: &str, args: JsonValue) -> String {
        let tool_name = name.to_string();
        let debug_mode = self.debug;
    
        // Use spawn_blocking to run the synchronous tool code on a dedicated thread.
        let result = tokio::task::spawn_blocking(move || {
            if debug_mode {
                println!("[ORCHESTRA DEBUG] Executing tool: {}", &tool_name);
            }
            match library.get(&tool_name) {
                Some(tool) => match tool {
                    Tool::Rust { function, .. } => match function(args) {
                        Ok(res) => serde_json::to_string(&res).unwrap_or_else(|e| e.to_string()),
                        Err(e) => e,
                    },
                    Tool::Python { function, .. } => Python::with_gil(|py| {
                        let py_args = match bindings::python_b::json_to_pyobject(py, &args) {
                            Ok(a) => a,
                            Err(e) => return format!("Failed to convert arguments to Python: {}", e),
                        };
    
                        let py_kwargs: &Bound<PyDict> =
                            match py_args.downcast_bound(py) {
                                Ok(kwargs) => kwargs,
                                Err(_) => {
                                    return "Tool arguments must be a JSON object (dict in Python)"
                                        .to_string()
                                }
                            };
    
                        match function.call_bound(py, (), Some(py_kwargs)) {
                            Ok(result) => {
                               match bindings::python_b::pyobject_to_json(py, &result) {
                                    Ok(json_val) => serde_json::to_string(&json_val).unwrap_or_else(|e| format!("Failed to serialize tool result: {}", e)),
                                    Err(e) => format!("Failed to convert Python tool result to JSON: {}", e),
                                }
                            }
                            Err(e) => format!("Python tool execution failed: {}", e),
                        }
                    }),
                },
                None => format!("Tool '{}' not found in library.", &tool_name),
            }
        }).await;
    
        match result {
            Ok(s) => s,
            Err(e) => format!("Tool panicked during execution: {}", e),
        }
    }

    /// Helper to extract system and user prompts from message history.
    fn get_prompts_from_messages<'a>(&self, messages: &'a [Message]) -> (&'a str, &'a str) {
        let system = messages.iter().find(|m| m.role == "system").and_then(|m| m.content.as_deref()).unwrap_or("");
        let user = messages.iter().rfind(|m| m.role == "user").and_then(|m| m.content.as_deref()).unwrap_or("");
        (system, user)
    }

    /// The main entry point for making a single, conversational turn to the LLM.
    /// This function orchestrates the entire process, including prompt preparation,
    /// making the API call, and handling multi-step tool execution.
    pub async fn call_ai(&self, messages: Vec<Message>) -> Result<ResponsePayload, LLMCoreError> {
        let (initial_payload, updated_messages) = self.execute_initial_turn(messages).await?;
        self.handle_tool_cycle(initial_payload, updated_messages).await
    }

    /// Executes a swarm of concurrent API calls.
    /// Note: Swarm calls do not support multi-step tool execution.
    pub async fn swarm_call(
            &self,
            system_prompt: &str,
            prompts: Vec<String>,
            swarm_size: usize,
        ) -> Vec<Result<ResponsePayload, LLMCoreError>> {
        let mut all_payloads = Vec::new();

        for user_prompt in &prompts {
            let (final_system_prompt, final_user_prompt, schema_for_provider) =
                match &self.structured_strategy {
                    InternalStructuredStrategy::Lucky(output_format) => {
                        let (s, u) = lucky::prepare_lucky_prompt(system_prompt, user_prompt, output_format, "###", None, false);
                        (s, u, None)
                    }
                    InternalStructuredStrategy::Schema(s) => (system_prompt.to_string(), user_prompt.to_string(), Some(s.clone())),
                    InternalStructuredStrategy::None => (system_prompt.to_string(), user_prompt.to_string(), None),
                };

            let messages = vec![
                format_system_message(final_system_prompt),
                format_user_message(final_user_prompt),
            ];

            let payload = self.provider_adapter.prepare_request_payload(
                &self.model_tag, messages, self.temperature, schema_for_provider, None,
            );
            all_payloads.push(payload);
        }

        let url = self.provider_adapter.get_request_url(&self.base_url, &self.model_tag, &self.api_key);
        let headers = self.provider_adapter.get_request_headers(&self.api_key);

        let raw_responses =
            client::execute_swarm_call(url, headers, all_payloads, swarm_size, self.retry_policy)
                .await;

        raw_responses
            .into_iter()
            .map(|res_result| {
                res_result.and_then(|text| {
                    let initial_payload = self.response_parser.parse_response(
                        &text,
                        &self.user_facing_model_name,
                        self.input_price,
                        self.output_price,
                    )?;
                    match &self.structured_strategy {
                        InternalStructuredStrategy::Lucky(fmt) => {
                            let content = initial_payload.choices.get(0).and_then(|c| c.message.content.as_ref()).ok_or_else(|| LLMCoreError::ResponseParseError("No content for Lucky parsing".to_string()))?;
                            let lucky_json = lucky::parse_lucky_response(content, fmt, "###")?;
                            let mut new_payload = initial_payload;
                            if let Some(choice) = new_payload.choices.get_mut(0) {
                                choice.message.content = Some(serde_json::to_string(&lucky_json)?);
                            }
                            Ok(new_payload)
                        }
                        _ => Ok(initial_payload),
                    }
                })
            })
            .collect()
    }
} 