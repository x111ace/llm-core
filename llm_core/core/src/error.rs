use pyo3::{exceptions::PyValueError, PyErr};
use tokio::task::JoinError;
use std::fmt;
use rusqlite;

/// The primary error type for the `llm-core` library.
///
/// This enum consolidates all possible failure modes into a single,
/// structured type, allowing library consumers to write robust,
/// programmatic error-handling logic.
#[derive(Debug)]
pub enum LLMCoreError {
    /// An error related to configuration, such as a missing `models.json`
    /// file, a malformed configuration, or a missing environment variable.
    ConfigError(String),

    /// An error specific to the retrieval process (embedding, indexing, or searching).
    RetrievalError(String),

    /// An error that occurs during an API request, typically related to
    /// network issues, timeouts, or DNS problems. Wraps a `reqwest::Error`.
    RequestError(reqwest::Error),

    /// A non-successful response from the provider's API (e.g., 4xx or 5xx).
    /// Includes the HTTP status code and the response body for debugging.
    ApiError { status: u16, body: String },

    /// An error that occurs when parsing the provider's response body,
    /// indicating that the response was not valid JSON or did not match
    /// the expected structure.
    ResponseParseError(String),

    /// An error originating from the Heed key-value store.
    HeedError(heed::Error),

    /// An error from the Arroy vector search library.
    ArroyError(arroy::Error),

    /// An error from the SQLite database.
    RusqliteError(rusqlite::Error),

    /// A general-purpose database error for other database-related issues.
    DatabaseError(String),

    /// An I/O error related to file system operations, such as saving or
    /// loading conversation history.
    IoError(std::io::Error),

    /// An error that occurs during the execution of a provided tool function.
    ToolExecutionError(String),

    /// An error originating from within the stateful `Chat` session manager.
    ChatError(String),
    
    /// An error that occurs when a concurrent task fails to join.
    TaskJoinError(JoinError),

    /// An error that occurs when image generation fails.
    ImageGenerationError(String),
}

impl fmt::Display for LLMCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMCoreError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            LLMCoreError::RetrievalError(msg) => write!(f, "Retrieval error: {}", msg),
            LLMCoreError::RequestError(err) => write!(f, "Request error: {}", err),
            LLMCoreError::ApiError { status, body } => {
                write!(f, "API error (status {}): {}", status, body)
            }
            LLMCoreError::ResponseParseError(msg) => write!(f, "Response parse error: {}", msg),
            LLMCoreError::HeedError(err) => write!(f, "Vector index error: {}", err),
            LLMCoreError::ArroyError(err) => write!(f, "Vector search error: {}", err),
            LLMCoreError::RusqliteError(err) => write!(f, "SQLite error: {}", err),
            LLMCoreError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            LLMCoreError::IoError(err) => write!(f, "I/O error: {}", err),
            LLMCoreError::ToolExecutionError(msg) => write!(f, "Tool execution error: {}", msg),
            LLMCoreError::ChatError(msg) => write!(f, "Chat session error: {}", msg),
            LLMCoreError::TaskJoinError(err) => write!(f, "Task join error: {}", err),
            LLMCoreError::ImageGenerationError(msg) => write!(f, "Image generation error: {}", msg),
        }
    }
}

impl std::error::Error for LLMCoreError {}

impl From<LLMCoreError> for PyErr {
    fn from(err: LLMCoreError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

// --- From Implementations for Ergonomics ---

impl From<heed::Error> for LLMCoreError {
    fn from(err: heed::Error) -> Self {
        LLMCoreError::HeedError(err)
    }
}

impl From<arroy::Error> for LLMCoreError {
    fn from(err: arroy::Error) -> Self {
        LLMCoreError::ArroyError(err)
    }
}

impl From<rusqlite::Error> for LLMCoreError {
    fn from(err: rusqlite::Error) -> Self {
        LLMCoreError::RusqliteError(err)
    }
}

impl From<reqwest::Error> for LLMCoreError {
    fn from(err: reqwest::Error) -> Self {
        LLMCoreError::RequestError(err)
    }
}

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

impl From<std::io::Error> for LLMCoreError {
    fn from(err: std::io::Error) -> Self {
        LLMCoreError::IoError(err)
    }
}

impl From<JoinError> for LLMCoreError {
    fn from(err: JoinError) -> Self {
        LLMCoreError::TaskJoinError(err)
    }
}
