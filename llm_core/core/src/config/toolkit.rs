use crate::tools::{FunctionDefinition, Tool, ToolDefinition, ToolLibrary};
use serde_json::{json, Value as JsonValue};

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

    // --- Add other Rust tools here in the future ---

    tool_library
} 