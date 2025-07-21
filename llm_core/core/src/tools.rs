use pyo3::prelude::*;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Represents a tool call requested by the model in its response.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolCall {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    #[serde(rename = "type")]
    pub tool_type: String, // e.g., "function"
    #[pyo3(get, set)]
    pub function: FunctionCall,
}

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FunctionCall {
    #[pyo3(get, set)]
    pub name: String,
    // This field cannot be exposed directly. We use custom getter/setter methods.
    pub arguments: JsonValue,
}

#[pymethods]
impl FunctionCall {
    #[getter]
    fn get_arguments(&self, py: Python) -> PyResult<PyObject> {
        serde_pyobject::to_pyobject(py, &self.arguments)
            .map(|bound| bound.into())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[setter]
    fn set_arguments(&mut self, py: Python, value: PyObject) -> PyResult<()> {
        let bound_value = value.bind(py);
        self.arguments = serde_pyobject::from_pyobject(bound_value.clone())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }
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
