use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::config;
use crate::lucky::{SchemaItems, SchemaProperty, SimpleSchema};
use crate::sorter::{Sorter, SortingInstructions};
use crate::tools::{FunctionDefinition, Tool, ToolDefinition, ToolLibrary};
use crate::usage::log_usage_turn;
use crate::orchestra::Orchestra;
use crate::convo::Chat;
use serde_json::json;

use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

#[pyclass(name = "ToolDefinition")]
#[derive(Clone)]
pub struct PyToolDefinition {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    description: String,
    #[pyo3(get, set)]
    parameters: PySimpleSchema,
}

#[pymethods]
impl PyToolDefinition {
    #[new]
    fn new(name: String, description: String, parameters: PySimpleSchema) -> Self {
        Self {
            name,
            description,
            parameters,
        }
    }
}

#[pyclass(name = "Tool")]
#[derive(FromPyObject)]
pub struct PyTool {
    #[pyo3(get, set)]
    definition: PyToolDefinition,
    #[pyo3(get, set)]
    function: PyObject,
}

#[pymethods]
impl PyTool {
    #[new]
    fn new(definition: PyToolDefinition, function: PyObject) -> Self {
        Self {
            definition,
            function,
        }
    }
}

#[pyclass(name = "SortingInstructions")]
#[derive(Clone)]
pub struct PySortingInstructions {
    #[pyo3(get, set)]
    data_item_name: String,
    #[pyo3(get, set)]
    data_profile_description: String,
    #[pyo3(get, set)]
    item_sorting_guidelines: Vec<String>,
    #[pyo3(get, set)]
    provided_categories: Vec<String>,
}

#[pymethods]
impl PySortingInstructions {
    #[new]
    #[pyo3(signature = (data_item_name, data_profile_description, item_sorting_guidelines, provided_categories = Vec::new()))]
    fn new(
        data_item_name: &str,
        data_profile_description: &str,
        item_sorting_guidelines: Vec<String>,
        provided_categories: Vec<String>,
    ) -> Self {
        PySortingInstructions {
            data_item_name: data_item_name.to_string(),
            data_profile_description: data_profile_description.to_string(),
            item_sorting_guidelines,
            provided_categories,
        }
    }
}

/// A high-level standalone function for sorting a file of data items.
///
/// This is a main entry point for Python developers. It encapsulates the
/// underlying Rust Sorter and Orchestra objects to provide a simple,
/// one-shot sorting utility.
#[pyfunction]
#[pyo3(signature = (model_name, instructions, *, input_path = None, items_list = None, output_path = None, swarm_size = 1, debug_out = false))]
pub fn run_sorter(
        model_name: &str,
        instructions: PySortingInstructions,
        input_path: Option<&str>,
        items_list: Option<Vec<String>>,
        output_path: Option<String>,
        swarm_size: usize,
        debug_out: bool,
    ) -> PyResult<PyObject> {
    if input_path.is_some() && items_list.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Please provide either 'input_path' or 'items_list', but not both.",
        ));
    }
    if input_path.is_none() && items_list.is_none() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Please provide either 'input_path' or 'items_list'.",
        ));
    }

    // Create a dedicated runtime for this async task.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    
    // Generate a unique ID for this sorting job for logging purposes.
    let job_id = Uuid::new_v4();
    
    let result = rt.block_on(async {
        // 1. Create the Orchestra engine.
        let orchestra = Orchestra::new(model_name, None, None, None, Some(debug_out))?;

        // 2. Convert PySortingInstructions to the Rust struct.
        let rust_instructions = SortingInstructions {
            data_item_name: instructions.data_item_name,
            data_profile_description: instructions.data_profile_description,
            item_sorting_guidelines: instructions.item_sorting_guidelines,
            provided_categories: instructions.provided_categories,
        };

        // 3. Run the sorting task.
        Sorter::run_sorting_task(
            Arc::new(orchestra),
            input_path.map(PathBuf::from),
            items_list,
            output_path.map(PathBuf::from),
            rust_instructions,
            swarm_size,
            debug_out,
        )
        .await
    });

    match result {
        Ok((sorted_data, _, usage_data, item_count)) => {
            // Log the total usage for the entire sorting job.
            let label = format!("sort {} items", item_count);
            if let Err(e) = log_usage_turn(job_id, &usage_data, &label, model_name) {
                // We'll just print the error and not fail the whole operation.
                eprintln!("[WARNING] Failed to log sorter usage: {}", e);
            }

            // Convert the BTreeMap<String, Vec<String>> to a Python dictionary.
            Python::with_gil(|py| {
                let dict = PyDict::new_bound(py);
                for (key, value) in sorted_data {
                    dict.set_item(key, value)?;
                }
                Ok(dict.to_object(py))
            })
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            e.to_string(),
        )),
    }
}

#[pyclass(name = "SchemaItems")]
#[derive(Clone)]
pub struct PySchemaItems {
    #[pyo3(get, set)]
    item_type: String,
}

#[pymethods]
impl PySchemaItems {
    #[new]
    fn new(item_type: &str) -> Self {
        PySchemaItems {
            item_type: item_type.to_string(),
        }
    }
}

#[pyclass(name = "SchemaProperty")]
#[derive(Clone)]
pub struct PySchemaProperty {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    property_type: String,
    #[pyo3(get, set)]
    description: String,
    #[pyo3(get, set)]
    items: Option<PySchemaItems>,
}

#[pymethods]
impl PySchemaProperty {
    #[new]
    #[pyo3(signature = (name, property_type, description, items = None))]
    fn new(
        name: &str,
        property_type: &str,
        description: &str,
        items: Option<PySchemaItems>,
    ) -> Self {
        PySchemaProperty {
            name: name.to_string(),
            property_type: property_type.to_string(),
            description: description.to_string(),
            items,
        }
    }
}

/// A simplified, serializable JSON schema for guiding model responses.
#[pyclass(name = "SimpleSchema")]
#[derive(Clone)]
pub struct PySimpleSchema {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    description: String,
    #[pyo3(get, set)]
    properties: Vec<PySchemaProperty>,
}

#[pymethods]
impl PySimpleSchema {
    #[new]
    fn new(name: &str, description: &str, properties: Vec<PySchemaProperty>) -> Self {
        PySimpleSchema {
            name: name.to_string(),
            description: description.to_string(),
            properties,
        }
    }
}

/// A message in a conversation, representing a single turn from a user, assistant, or tool.
/// This is a read-only object exposed to Python.
#[pyclass(name = "Message")]
pub struct PyMessage {
    #[pyo3(get)]
    role: String,
    #[pyo3(get)]
    content: Option<String>,
}

/// A high-level session manager for conducting stateful conversations.
///
/// This is the main entry point for Python developers. It encapsulates the
/// underlying Rust Orchestra and Conversation objects to provide a simple,
/// stateful chat interface.
#[pyclass(name = "Chat", unsendable)]
pub struct PyChat {
    chat: Chat,
    // A small internal Tokio runtime to execute the async `send` method.
    rt: tokio::runtime::Runtime,
}

#[pymethods]
impl PyChat {
    /// Creates a new chat session.
    ///
    /// Args:
    ///     model_name (str): The user-facing name of the model to use (e.g., "GPT 4o MINI").
    ///     system_prompt (Optional[str]): An optional system prompt to guide the model.
    ///     schema (Optional[SimpleSchema]): An optional schema to enforce JSON output.
    ///     native_tools (Optional[bool]): If True, enables the native Rust tool library.
    ///     extra_tools (Optional[list[Tool]]): A list of custom Python tools to add.
    ///     debug_out (Optional[bool]): An optional flag to enable debug output.
    #[new]
    #[pyo3(signature = (model_name, system_prompt = None, schema = None, native_tools = false, extra_tools = None, debug_out = false))]
    fn new(
            model_name: &str,
            system_prompt: Option<String>,
            schema: Option<PySimpleSchema>,
            native_tools: bool,
            extra_tools: Option<Vec<PyTool>>,
            debug_out: bool,
        ) -> PyResult<Self> {
        if schema.is_some() && (native_tools || extra_tools.is_some()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot use a schema and tools (native or extra) at the same time.",
            ));
        }

        let rust_schema = schema.map(|py_schema| SimpleSchema {
            name: py_schema.name,
            description: py_schema.description,
            properties: py_schema
                .properties
                .into_iter()
                .map(|py_prop| SchemaProperty {
                    name: py_prop.name,
                    property_type: py_prop.property_type,
                    description: py_prop.description,
                    items: py_prop.items.map(|py_items| SchemaItems {
                        item_type: py_items.item_type,
                    }),
                })
                .collect(),
        });

        let mut tool_library = ToolLibrary::new();

        if native_tools {
            tool_library.extend(config::get_rust_tool_library());
        }

        if let Some(py_tools) = extra_tools {
            for py_tool in py_tools {
                let params_schema = &py_tool.definition.parameters;
                let parameters_json = json!({
                    "type": "object",
                    "properties": params_schema.properties.iter().map(|p| {
                        (p.name.clone(), json!({
                            "type": p.property_type,
                            "description": p.description,
                        }))
                    }).collect::<serde_json::Map<String, serde_json::Value>>(),
                    "required": params_schema.properties.iter().map(|p| p.name.clone()).collect::<Vec<String>>()
                });

                let definition = ToolDefinition {
                    tool_type: "function".to_string(),
                    function: FunctionDefinition {
                        name: py_tool.definition.name.clone(),
                        description: py_tool.definition.description.clone(),
                        parameters: parameters_json,
                    },
                };
                tool_library.insert(
                    py_tool.definition.name.clone(),
                    Tool::Python {
                        definition,
                        function: py_tool.function,
                    },
                );
            }
        }

        let final_tools = if tool_library.is_empty() {
            None
        } else {
            Some(tool_library)
        };

        let chat =
            Chat::new(model_name, system_prompt, final_tools, rust_schema, Some(debug_out))?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        Ok(PyChat { chat, rt })
    }

    /// Sends a user prompt to the model and updates the conversation state.
    ///
    /// This is the primary method for driving a conversation. It appends the user's
    /// message, calls the underlying `Orchestra`, and then appends the assistant's
    /// response, updating token usage and timestamps.
    ///
    /// Returns:
    ///     Message: The assistant's response message.
    fn send(&mut self, user_prompt: &str) -> PyResult<PyMessage> {
        let assistant_message = self.rt.block_on(self.chat.send(user_prompt))?;

        // We clone the message to pass ownership to the new Python object.
        Ok(PyMessage {
            role: assistant_message.role.clone(),
            content: assistant_message.content.clone(),
        })
    }
}
