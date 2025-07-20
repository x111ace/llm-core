use pyo3::prelude::*;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Represents a tool call requested by the model in its response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String, // e.g., "function"
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: JsonValue, // Changed from String to JsonValue
}

/// Defines the structure for a tool that can be provided to an AI model.
/// This is the "schema" for a single function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String, // Should always be "function" for now
    pub function: FunctionDefinition,
}

/// The definition of the function, including its parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: JsonValue, // JSON Schema object
}

/// A self-contained, executable tool including its definition and function.
/// This is what the user will create and provide to the library.
pub enum Tool {
    Rust {
        definition: ToolDefinition,
        // The function takes JSON arguments and returns a JSON result or an error string.
        function: fn(JsonValue) -> Result<JsonValue, String>,
    },
    Python {
        definition: ToolDefinition,
        function: PyObject, // This will hold the Python callable
    },
}

impl Tool {
    /// A helper method to get the definition from any tool variant.
    pub fn definition(&self) -> &ToolDefinition {
        match self {
            Tool::Rust { definition, .. } => definition,
            Tool::Python { definition, .. } => definition,
        }
    }
}

/// A collection of executable tools, searchable by name, to be passed to the Orchestra.
pub type ToolLibrary = HashMap<String, Tool>;
