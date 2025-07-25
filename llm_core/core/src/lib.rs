use pyo3::prelude::*;

// Declare all library modules to make them accessible.
pub mod bindings;
pub mod client;
pub mod config;
pub mod convo;
pub mod datam;
pub mod embed;
pub mod error;
pub mod ingest;
pub mod lucky;
pub mod modes;
pub mod orchestra;
pub mod providers;
pub mod retrieval;
pub mod sorter;
pub mod tools;
pub mod usage;
pub mod vector;

/// This is the main entry point for the Python module.
/// The `#[pymodule]` macro creates a function that initializes the module.
/// The function name (`_llm_core`) must match the `[lib].name` in `Cargo.toml`.
#[pymodule]
fn _llm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bindings::python_b::PyChat>()?;
    m.add_class::<bindings::python_b::PyMessage>()?;
    m.add_class::<bindings::python_b::PyTool>()?;
    m.add_class::<bindings::python_b::PyToolDefinition>()?;
    m.add_class::<bindings::python_b::PySimpleSchema>()?;
    m.add_class::<bindings::python_b::PySchemaProperty>()?;
    m.add_class::<bindings::python_b::PySchemaItems>()?;
    m.add_class::<bindings::python_b::PySortingInstructions>()?;
    m.add_class::<bindings::python_b::PyKnowledgeBase>()?;
    m.add_class::<bindings::python_b::PyIngestor>()?;
    m.add_function(wrap_pyfunction!(bindings::python_b::run_sorter, m)?)?;
    Ok(())
}
