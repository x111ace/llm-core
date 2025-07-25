use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};
use pyo3::{pyclass, pymethods, BoundObject, PyErr, PyObject, PyResult, Python};
use serde_json::{Value as JsonValue};
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;
use std::sync::Arc;
use uuid::Uuid;

use crate::config;
use crate::convo::Chat;
use crate::lucky::{SchemaItems, SchemaProperty, SimpleSchema};
use crate::orchestra::Orchestra;
use crate::sorter::{Sorter, SortingInstructions};
use crate::tools::{FunctionDefinition, Tool, ToolDefinition, ToolLibrary};
use crate::usage::log_usage_turn;
use serde_json::json;
use crate::ingest::Ingestor;
use crate::vector::KnowledgeBase;
use crate::error::LLMCoreError;

// --- Python Bindings for Tools ---

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
        Self { name, description, parameters }
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
        Self { definition, function }
    }
}

// --- Python Bindings for Sorter ---

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

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let job_id = Uuid::new_v4();
    
    let result = rt.block_on(async {
        let orchestra = Orchestra::new(model_name, None, None, None, None, Some(debug_out))?;
        let rust_instructions = SortingInstructions {
            data_item_name: instructions.data_item_name,
            data_profile_description: instructions.data_profile_description,
            item_sorting_guidelines: instructions.item_sorting_guidelines,
            provided_categories: instructions.provided_categories,
        };
        Sorter::run_sorting_task(
            Arc::new(orchestra),
            input_path.map(PathBuf::from),
            items_list,
            output_path.map(PathBuf::from),
            rust_instructions,
            swarm_size,
            debug_out,
        ).await
    });

    match result {
        Ok((sorted_data, _, usage_data, item_count)) => {
            let label = format!("sort {} items", item_count);
            if let Err(e) = log_usage_turn(job_id, &usage_data, &label, model_name) {
                eprintln!("[WARNING] Failed to log sorter usage: {}", e);
            }
            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                for (key, value) in sorted_data {
                    dict.set_item(key, value)?;
                }
                Ok(dict.into())
            })
        }
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())),
    }
}

// --- Python Bindings for Chat and Schema ---

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
        PySchemaItems { item_type: item_type.to_string() }
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

#[pyclass(name = "Message")]
pub struct PyMessage {
    #[pyo3(get)]
    role: String,
    #[pyo3(get)]
    content: Option<String>,
    #[pyo3(get)]
    reasoning_content: Option<String>,
}

#[pyclass(name = "Chat", unsendable)]
pub struct PyChat {
    chat: Chat,
    rt: tokio::runtime::Runtime,
}

#[pymethods]
impl PyChat {
    #[new]
    #[pyo3(signature = (model_name, system_prompt = None, schema = None, native_tools = false, extra_tools = None, thinking_mode = None, debug_out = false))]
    fn new(
            model_name: &str,
            system_prompt: Option<String>,
            schema: Option<PySimpleSchema>,
            native_tools: bool,
            extra_tools: Option<Vec<PyTool>>,
            thinking_mode: Option<bool>,
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
            properties: py_schema.properties.into_iter().map(|py_prop| SchemaProperty {
                    name: py_prop.name,
                    property_type: py_prop.property_type,
                    description: py_prop.description,
                    items: py_prop.items.map(|py_items| SchemaItems {
                        item_type: py_items.item_type,
                    }),
            }).collect(),
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
                    Tool::Python { definition, function: py_tool.function },
                );
            }
        }
        let final_tools = if tool_library.is_empty() { None } else { Some(tool_library) };
        let chat = Chat::new(model_name, system_prompt, final_tools, rust_schema, thinking_mode, Some(debug_out))?;
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        Ok(PyChat { chat, rt })
    }

    fn send(&mut self, user_prompt: &str) -> PyResult<PyMessage> {
        let assistant_message = self.rt.block_on(self.chat.send(user_prompt))?;
        Ok(PyMessage {
            role: assistant_message.role.clone(),
            content: assistant_message.content.clone(),
            reasoning_content: assistant_message.reasoning_content.clone(),
        })
    }
}

#[pyclass(name = "KnowledgeBase", unsendable)]
pub struct PyKnowledgeBase {
    db_path: PathBuf,
    index_path: PathBuf,
    embedding_model: String,
    runtime: Runtime,
}

#[pymethods]
impl PyKnowledgeBase {
    #[new]
    fn new(db_path: &str, index_path: &str, embedding_model: &str) -> PyResult<Self> {
        let runtime = Runtime::new().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            db_path: PathBuf::from(db_path),
            index_path: PathBuf::from(index_path),
            embedding_model: embedding_model.to_string(),
            runtime,
        })
    }

    fn search(&mut self, query: &str, limit: usize) -> PyResult<Py<PyAny>> {
        let db_path = self.db_path.clone();
        let index_path = self.index_path.clone();
        let embedding_model = self.embedding_model.clone();

        let search_results = self.runtime.block_on(async move {
            let kb = KnowledgeBase::new(&db_path, &index_path, &embedding_model)?;
            kb.search(query, limit).await
        }).map_err(|e: LLMCoreError| PyValueError::new_err(e.to_string()))?;
        
        Python::with_gil(|py| {
            let json_val = serde_json::to_value(search_results).unwrap();
            json_to_pyobject(py, &json_val)
        })
    }
}

#[pyclass(name = "Ingestor", unsendable)]
pub struct PyIngestor {
    ingestor: Ingestor,
    runtime: Runtime,
}

#[pymethods]
impl PyIngestor {
    #[new]
    fn new(
        db_path: &str,
        index_path: &str,
        embedding_model: &str,
        enrichment_model: &str,
    ) -> PyResult<Self> {
        let runtime =
            Runtime::new().map_err(|e| PyValueError::new_err(format!("Failed to create Tokio runtime: {}", e)))?;
        let ingestor = Ingestor::new(
            Path::new(db_path),
            Path::new(index_path),
            embedding_model,
            enrichment_model,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(Self {
            ingestor,
            runtime,
        })
    }

    fn ingest_from_url(&mut self, url: &str, source_tag: &str) -> PyResult<()> {
        self.runtime
            .block_on(self.ingestor.ingest_from_url(url, source_tag))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn ingest_from_file(&mut self, file_path: &str, source_tag: &str) -> PyResult<()> {
        self.runtime
            .block_on(
                self.ingestor
                    .ingest_from_file(Path::new(file_path), source_tag),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// --- Python <-> Rust Data Conversion Helpers ---

pub fn json_to_pyobject(py: Python, json_val: &JsonValue) -> PyResult<PyObject> {
    match json_val {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(b) => Ok(b.into_pyobject(py)?.into_bound().into()),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_bound().into())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_bound().into())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid number format"))
            }
        }
        JsonValue::String(s) => Ok(s.into_pyobject(py)?.into_bound().into()),
        JsonValue::Array(a) => {
            let elements: Vec<PyObject> =
                a.iter().map(|item| json_to_pyobject(py, item)).collect::<PyResult<_>>()?;
            Ok(PyList::new(py, &elements)?.into())
        }
        JsonValue::Object(o) => {
            let dict = PyDict::new(py);
            for (k, v) in o {
                dict.set_item(k, json_to_pyobject(py, v)?)?;
            }
            Ok(dict.into())
        }
    }
}

pub fn pyobject_to_json(py: Python, obj: &PyObject) -> PyResult<JsonValue> {
    let bound_obj = obj.bind(py);
    if bound_obj.is_none() {
        return Ok(JsonValue::Null);
    }
    if let Ok(b) = bound_obj.extract::<bool>() {
        return Ok(JsonValue::Bool(b));
    }
    if let Ok(i) = bound_obj.extract::<i64>() {
        return Ok(JsonValue::Number(i.into()));
    }
    if let Ok(f) = bound_obj.extract::<f64>() {
        return Ok(JsonValue::Number(serde_json::Number::from_f64(f).unwrap()));
    }
    if let Ok(s) = bound_obj.extract::<String>() {
        return Ok(JsonValue::String(s));
    }

    if let Ok(list) = bound_obj.clone().downcast_into::<PyList>() {
        let mut rust_vec = Vec::new();
        for item in list.iter() {
            rust_vec.push(pyobject_to_json(py, &(item.into_pyobject(py)?.into_bound().into()))?);
        }
        return Ok(JsonValue::Array(rust_vec));
    }

    if let Ok(dict) = bound_obj.clone().downcast_into::<PyDict>() {
        let mut rust_map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key = k.extract::<String>()?;
            let value = pyobject_to_json(py, &(v.into_pyobject(py)?.into_bound().into()))?;
            rust_map.insert(key, value);
        }
        return Ok(JsonValue::Object(rust_map));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Unsupported Python type for JSON conversion",
    ))
} 