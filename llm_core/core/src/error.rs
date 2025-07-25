use pyo3::{exceptions::PyValueError, PyErr};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LLMCoreError {
    #[error("API call failed: {0}")]
    ApiError(String),

    #[error("API call failed with status {status}: {body}")]
    ApiErrorDetailed { status: u16, body: String },

    #[error("Failed to parse response from AI: {0}")]
    ResponseParseError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("A chat-related error occurred: {0}")]
    ChatError(String),

    #[error("An error occurred with a tool: {0}")]
    ToolError(String),

    #[error("An I/O error occurred: {0}")]
    IoError(#[from] std::io::Error),

    #[error("A database error occurred: {0}")]
    DatabaseError(String),

    #[error("A vector index error occurred: {0}")]
    VectorIndexError(String),

    #[error("A concurrency-related error occurred: {0}")]
    ConcurrencyError(String),

    #[error("An error occurred in Python interop: {0}")]
    PythonError(String),

    #[error("An error occurred during data retrieval: {0}")]
    RetrievalError(String),

    #[error("An error occurred during a request: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("Vector index error: {0}")]
    HeedError(#[from] heed::Error),

    #[error("Vector search error: {0}")]
    ArroyError(#[from] arroy::Error),

    #[error("SQLite error: {0}")]
    RusqliteError(#[from] rusqlite::Error),

    #[error("Tool execution error: {0}")]
    ToolExecutionError(String),

    #[error("Task join error: {0}")]
    TaskJoinError(#[from] tokio::task::JoinError),

    #[error("Image generation error: {0}")]
    ImageGenerationError(String),
}

impl From<LLMCoreError> for PyErr {
    fn from(err: LLMCoreError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

// --- From Implementations for Ergonomics ---

impl From<serde_json::Error> for LLMCoreError {
    fn from(err: serde_json::Error) -> Self {
        LLMCoreError::ResponseParseError(err.to_string())
    }
}

impl From<String> for LLMCoreError {
    fn from(err: String) -> Self {
        // This is a general fallback, often used for parsing errors from `lucky`.
        LLMCoreError::ResponseParseError(err)
    }
}

impl From<pyo3::PyErr> for LLMCoreError {
    fn from(py_err: pyo3::PyErr) -> Self {
        LLMCoreError::PythonError(py_err.to_string())
    }
}
