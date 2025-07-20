use crate::lucky::SimpleSchema;
use serde_json::Value as JsonValue;

/// Defines the strategy for obtaining a structured response from an LLM.
#[derive(Debug, Clone)]
pub enum StructuredResponseMode {
    /// Use the provider's native JSON mode or schema enforcement.
    Schema(SimpleSchema),
    /// Use the `lucky.rs` delimiter-based prompt and parsing strategy.
    /// The `JsonValue` represents the desired output format, e.g., `json!({"name": "<type:str>"})`.
    Lucky(JsonValue),
    /// Expect a plain text response with no structure enforcement.
    None,
}

/// Defines the strategy for using tools with an LLM.
#[derive(Debug, Clone)]
pub enum ToolMode {
    /// Use the provider's native `tools` payload for function calling.
    Payload,
    /// Use a ReAct (Reason-Act) prompting loop. (Not yet implemented)
    ReAct,
    /// Do not use tools.
    None,
} 