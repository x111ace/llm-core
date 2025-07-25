use crate::tools::{FunctionDefinition, Tool, ToolDefinition, ToolLibrary};
use crate::config::{get_env_var, MODEL_LIBRARY};

// Add back the necessary imports for a self-contained, blocking HTTP call.
use base64::engine::{general_purpose::STANDARD as BASE64, Engine as _};
use serde_json::{json, Value as JsonValue};
use reqwest::blocking::Client;
use std::path::Path;
use std::io::Write;
use std::fs::File;
use uuid::Uuid;

// --- Imports for the new Sorter Tool ---
use crate::sorter::sort_data_items_tool;

// --- Imports for the new KnowledgeBase Tools ---
use crate::retrieval::{
    knowledge_base_get_full_document, knowledge_base_list_sources, knowledge_base_search,
};

/// Returns a `ToolLibrary` containing all the natively implemented Rust tools.
pub fn get_rust_tool_library() -> ToolLibrary {
    let mut tool_library = ToolLibrary::new();

    // --- get_current_time tool ---

    fn get_current_time(_args: JsonValue) -> Result<JsonValue, String> {
        use chrono::Local;
        let now = Local::now();
        let formatted_time = now.format("%-I:%M %p on %A, %-m/%-d/%Y").to_string();
        Ok(json!({ "time": formatted_time }))
    }

    // --- generate_image tool (Refactored to be self-contained and synchronous) ---
    
    fn generate_image(args: JsonValue) -> Result<JsonValue, String> {
        let prompt = args["prompt"]
            .as_str()
            .ok_or("Missing 'prompt' in arguments")?
            .to_string();

        let output_path = args["output_path"].as_str();

        // --- Logic is now self-contained within the tool ---
        let (_provider_name, provider_data, model_details) =
            MODEL_LIBRARY.find_model("GEMINI 2.0 FLASH IMAGE GEN")
            .ok_or_else(|| "Model 'GEMINI 2.0 FLASH IMAGE GEN' not found".to_string())?;

        let api_key = get_env_var(&provider_data.api_key).map_err(|e| e.to_string())?;
        let base_url = get_env_var(&provider_data.base_url).map_err(|e| e.to_string())?;

        let url = format!(
            "{}/{}:generateContent?key={}",
            base_url.trim_end_matches('/'),
            &model_details.model_tag,
            api_key
        );

        let payload = json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            }
        });

        let client = Client::new();
        let res = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .map_err(|e| format!("Request error: {}", e))?;

        if !res.status().is_success() {
            return Err(format!(
                "API error (status {}): {}",
                res.status(),
                res.text().unwrap_or_else(|_| "Could not read error body".to_string())
            ));
        }

        let response_json: JsonValue = res.json().map_err(|e| format!("JSON parse error: {}", e))?;

        // --- Simplified parsing logic, moved from the provider ---
        let parts = response_json
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
            .ok_or("No 'parts' found in response".to_string())?;

        let mut text_content = None;
        let mut image_data = None;

        for part in parts {
            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                text_content = Some(text.to_string());
            }
            if let Some(inline_data) = part.get("inlineData") {
                if let Some(data) = inline_data.get("data").and_then(|d| d.as_str()) {
                    image_data = Some(data.to_string());
                }
            }
        }
        
        if let Some(image_b64) = image_data {
            let bytes = BASE64
                .decode(&image_b64)
                .map_err(|e| format!("Base64 decode error: {}", e))?;

            let path_str = match output_path {
                Some(p) => p.to_string(),
                None => format!("gemini_image_{}.png", Uuid::new_v4().to_string()[0..8].to_string()),
            };
            
            let path = Path::new(&path_str);

            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create directories: {}", e))?;
            }

            let mut file =
                File::create(&path).map_err(|e| format!("File create error: {}", e))?;
            file.write_all(&bytes)
                .map_err(|e| format!("File write error: {}", e))?;

            Ok(json!({
                "text_response": text_content,
                "image_path": path_str,
            }))
        } else {
            Err("No image data received from the API.".to_string())
        }
    }

    // --- KnowledgeBase Tools ---

    tool_library.insert(
        "get_current_time".to_string(),
        Tool::Rust {
            definition: ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "get_current_time".to_string(),
                    description: "Get the current time.".to_string(),
                    parameters: json!({"type": "object", "properties": {}}),
                },
            },
            function: get_current_time,
        },
    );

    tool_library.insert(
        "generate_image".to_string(),
        Tool::Rust {
            definition: ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "generate_image".to_string(),
                    description: "Generate an image from a text prompt using the Gemini API and save it to a file.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The text prompt for generating the image."
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Optional. The full path, including filename and extension (e.g., 'images/my_lion.png'), where the image should be saved. If not provided, a unique filename will be generated in the current directory."
                            }
                        },
                        "required": ["prompt"]
                    }),
                },
            },
            function: generate_image,
        },
    );

    tool_library.insert(
        "sort_data_items".to_string(),
        Tool::Rust {
            definition: ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "sort_data_items".to_string(),
                    description: "Sorts a list of data items into categories based on provided guidelines. Saves the output to a JSON file.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "input_path": {
                                "type": "string",
                                "description": "Optional. The path to a file or folder containing a JSON list of strings to be sorted. One of 'input_path' or 'items_list' is required."
                            },
                            "items_list": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Optional. A direct list of strings to be sorted. One of 'input_path' or 'items_list' is required."
                            },
                             "output_path": {
                                "type": "string",
                                "description": "Optional. The path to a directory where the sorted output file should be saved. Defaults to 'tests/output'."
                            },
                            "data_item_name": {
                                "type": "string",
                                "description": "A brief, descriptive name for the type of items being sorted (e.g., 'Customer Feedback', 'Product Titles'). Defaults to 'Data Item'."
                            },
                            "data_profile_description": {
                                "type": "string",
                                "description": "A detailed description of the data's characteristics or the user's interests to guide the sorting AI. Required if no categories are provided."
                            },
                            "item_sorting_guidelines": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "A list of rules or guidelines for the AI to follow when sorting each item."
                            },
                            "provided_categories": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Optional. A predefined list of categories to sort items into. If not provided, the tool will attempt to generate categories automatically."
                            },
                            "model_name": {
                                "type": "string",
                                "description": "Optional. The name of the AI model to use for sorting (e.g., 'GPT 4o MINI'). Defaults to a capable model."
                            },
                            "swarm_size": {
                                "type": "number",
                                "description": "Optional. The number of concurrent requests to make to the AI. Defaults to 5."
                            }
                        },
                        "required": []
                    }),
                },
            },
            function: sort_data_items_tool,
        },
    );

    // --- KnowledgeBase Tools ---

    tool_library.insert(
        "knowledge_base_search".to_string(),
        Tool::Rust {
            definition: ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "knowledge_base_search".to_string(),
                    description: "Searches the knowledge base for documents relevant to a query.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The natural language query to search for."
                            },
                            "limit": {
                                "type": "number",
                                "description": "Optional. The maximum number of results to return. Defaults to 5."
                            }
                        },
                        "required": ["query"]
                    }),
                },
            },
            function: knowledge_base_search,
        },
    );

    tool_library.insert(
        "knowledge_base_list_sources".to_string(),
        Tool::Rust {
            definition: ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "knowledge_base_list_sources".to_string(),
                    description: "Lists all the unique document sources (URLs) available in the knowledge base.".to_string(),
                    parameters: json!({"type": "object", "properties": {}}),
                },
            },
            function: knowledge_base_list_sources,
        },
    );

    tool_library.insert(
        "knowledge_base_get_full_document".to_string(),
        Tool::Rust {
            definition: ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDefinition {
                    name: "knowledge_base_get_full_document".to_string(),
                    description: "Retrieves the full, combined content of a specific document from the knowledge base.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The exact URL of the document source to retrieve."
                            }
                        },
                        "required": ["url"]
                    }),
                },
            },
            function: knowledge_base_get_full_document,
        },
    );

    // --- Add other Rust tools here in the future ---

    tool_library
} 

