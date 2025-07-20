use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign}; // Added for Usage aggregation
pub use crate::tools::ToolCall;

/// Represents a single message in a conversation.
/// This struct is compatible with OpenAI's format and serves as our standard.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    // This field is for non-standard output, like reasoning from specific models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// A single choice within the API response.
#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Message,
}

/// Cost details for a specific API call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Cost {
    pub input_price: f32,
    pub output_price: f32,
    pub total: f32,
}

/// Token usage statistics for a request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)] // Added Clone, Default, Serialize
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
}

impl Usage {
    // Calculates the total cost of the API call and sets the `cost` field.
    pub fn calculate_cost(&mut self, input_price: f32, output_price: f32) {
        let input_cost = (self.prompt_tokens as f32 / 1_000_000.0) * input_price;
        let output_cost = (self.completion_tokens as f32 / 1_000_000.0) * output_price;
        self.cost = Some(Cost {
            input_price,
            output_price,
            total: input_cost + output_cost,
        });
    }
}

impl Add for Usage {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let new_cost = match (self.cost, other.cost) {
            (Some(c1), Some(c2)) => Some(Cost {
                input_price: c1.input_price, // Assumes price is constant for the session
                output_price: c1.output_price,
                total: c1.total + c2.total,
            }),
            (Some(c1), None) => Some(c1),
            (None, Some(c2)) => Some(c2),
            (None, None) => None,
        };

        Self {
            prompt_tokens: self.prompt_tokens + other.prompt_tokens,
            completion_tokens: self.completion_tokens + other.completion_tokens,
            total_tokens: self.total_tokens + other.total_tokens,
            cost: new_cost,
        }
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
        if let Some(other_cost) = other.cost {
            if let Some(self_cost) = &mut self.cost {
                self_cost.total += other_cost.total;
            } else {
                self.cost = Some(other_cost);
            }
        }
    }
}

/// Represents the overall structure of a response from a chat completion API.
/// This is based on the standard OpenAI response format.
#[derive(Debug, Deserialize)]
pub struct ResponsePayload {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

// --- Message Formatting Helpers ---

/// Creates a new Message with the "system" role.
pub fn format_system_message(content: String) -> Message {
    Message {
        role: "system".to_string(),
        content: Some(content),
        name: None,
        tool_call_id: None,
        tool_calls: None,
        reasoning_content: None, // Will be None and not serialized
    }
}

/// Creates a new Message with the "user" role.
pub fn format_user_message(content: String) -> Message {
    Message {
        role: "user".to_string(),
        content: Some(content),
        name: None,
        tool_call_id: None,
        tool_calls: None,
        reasoning_content: None, // Will be None and not serialized
    }
}

/// Creates a new Message with the "assistant" role.
pub fn format_assistant_message(content: String) -> Message {
    Message {
        role: "assistant".to_string(),
        content: Some(content),
        name: None,
        tool_call_id: None,
        tool_calls: None,
        reasoning_content: None, // Will be None and not serialized
    }
}

/// A helper function to format a tool response message.
pub fn format_tool_message(content: String, tool_call_id: String, name: String) -> Message {
    Message {
        role: "tool".to_string(),
        name: Some(name),
        content: Some(content),
        tool_call_id: Some(tool_call_id),
        ..Default::default()
    }
}

