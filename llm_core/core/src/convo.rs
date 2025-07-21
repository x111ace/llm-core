use crate::config::{DEFAULT_SORTER_OUTPUT_DIR};
use crate::datam::{Message, Usage};
use crate::usage::log_usage_turn;
use crate::orchestra::Orchestra;
use crate::lucky::SimpleSchema;
use crate::tools::ToolLibrary;

use crate::error::LLMCoreError;

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::path::Path;
use uuid::Uuid;
use std::fs;
use std::io;

/// Represents a single, stateful conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: Uuid,
    /// The user-facing model name used for this conversation.
    pub model_name: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub messages: Vec<Message>,
    pub usage: Usage,
}
impl Conversation {
    /// Creates a new, empty conversation for a specific model.
    pub fn new(model_name: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            model_name,
            title: "Untitled Conversation".to_string(),
            created_at: now,
            updated_at: now,
            messages: Vec::new(),
            usage: Usage::default(),
        }
    }

    /// Loads a conversation from a JSON file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let data = fs::read_to_string(path)?;
        serde_json::from_str(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Saves the current conversation state to a file.
    ///
    /// If the provided path is a directory, a unique filename will be generated.
    /// If the path is a full file path, it will be used directly, creating
    /// parent directories if they don't exist.
    pub fn save(&mut self, path_str: &str) -> io::Result<()> {
        self.updated_at = Utc::now();
        let path = Path::new(path_str);
        let data = serde_json::to_string_pretty(&self)?;

        if path.is_dir() {
            let file_name = format!("convo-{}.json", self.id);
            let file_path = path.join(file_name);
            fs::write(file_path, data)
        } else {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, data)
        }
    }
}
impl Default for Conversation {
    fn default() -> Self {
        Self::new("unknown".to_string())
    }
}

/// A high-level session manager for conducting stateful conversations.
///
/// This struct is the primary entry point for developers building chat applications.
/// It encapsulates an `Orchestra` instance for API calls and a `Conversation`
/// object to manage the state and history.
pub struct Chat {
    pub orchestra: Orchestra,
    pub conversation: Conversation,
    pub thinking_mode: bool,
    // Add fields to track whether tools or schema are being used.
    has_tools: bool,
    has_schema: bool,
}
impl Chat {
    /// Creates a new chat session with a new, empty conversation.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The user-facing name of the model to use (e.g., "GPT 4o MINI").
    /// * `system_prompt` - An optional system prompt to guide the model's behavior.
    /// * `tools` - An optional library of tools for the model to use.
    pub fn new(
            model_name: &str,
            system_prompt: Option<String>,
            tools: Option<ToolLibrary>,
            schema: Option<SimpleSchema>,
            thinking_mode: Option<bool>,
            debug_out: Option<bool>,
        ) -> Result<Self, LLMCoreError> {
        let has_tools = tools.is_some();
        let has_schema = schema.is_some();
        let orchestra = Orchestra::new(model_name, None, tools, schema, thinking_mode, debug_out)?;
        let final_thinking_mode = orchestra.thinking_mode(); // Get the final state from Orchestra
        let mut conversation = Conversation::new(orchestra.user_facing_model_name.clone());

        if let Some(prompt) = system_prompt {
            conversation
                .messages
                .push(crate::datam::format_system_message(prompt));
        }

        Ok(Self {
            orchestra,
            conversation,
            thinking_mode: final_thinking_mode,
            has_tools,
            has_schema,
        })
    }

    /// Resumes a chat session from a previously saved conversation file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the saved conversation JSON file.
    /// * `model_name` - An optional model name. If `None`, the model that was used to
    ///                  create the original session will be used.
    /// * `tools` - An optional library of tools to provide to the session.
    pub fn from_file(
            path: &Path,
            model_name: Option<&str>,
            tools: Option<ToolLibrary>,
            thinking_mode: Option<bool>,
            debug_out: Option<bool>,
        ) -> Result<Self, LLMCoreError> {
        let conversation = Conversation::load(path)?;
        let has_tools = tools.is_some();

        // Use the provided model name, or default to the one stored in the conversation file.
        let final_model_name = model_name.unwrap_or(&conversation.model_name);

        let orchestra = Orchestra::new(final_model_name, None, tools, None, thinking_mode, debug_out)?;
        let final_thinking_mode = orchestra.thinking_mode();

        Ok(Self {
            orchestra,
            conversation,
            thinking_mode: final_thinking_mode,
            has_tools,
            has_schema: false, // Schema cannot be resumed from a file in this implementation.
        })
    }

    /// Saves the current conversation history to a file.
    ///
    /// If `path` is `None`, it saves to a default directory with a unique name.
    /// If `path` is `Some`, it uses the provided path, which can be a directory
    /// or a full file path.
    pub fn save_history(&mut self, path: Option<&str>) -> Result<(), LLMCoreError> {
        let save_path =
            path.map(|p| p.to_string())
                .unwrap_or_else(|| DEFAULT_SORTER_OUTPUT_DIR.to_string_lossy().to_string());
        self.conversation.save(&save_path).map_err(Into::into)
    }

    /// Sends a user prompt to the model and updates the conversation state.
    ///
    /// This is the primary method for driving a conversation. It appends the user's
    /// message, calls the underlying `Orchestra`, and then appends the assistant's
    /// response, updating token usage and timestamps.
    ///
    /// Returns a reference to the assistant's message that was just added to the history.
    pub async fn send(&mut self, user_prompt: &str) -> Result<&Message, LLMCoreError> {
        // 1. Prepare the messages for this specific turn without mutating state yet.
        let user_message = crate::datam::format_user_message(user_prompt.to_string());
        let mut messages_for_call = self.conversation.messages.clone();
        messages_for_call.push(user_message.clone());

        // 2. Call the stateless Orchestra engine.
        let response = self.orchestra.call_ai(messages_for_call).await?;

        // 3. On success, commit the changes to the conversation state.
        let assistant_message = response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or(LLMCoreError::ChatError(
                "API response did not contain any messages.".to_string(),
            ))?;

        self.conversation.messages.push(user_message);
        self.conversation.messages.push(assistant_message);
        self.conversation.updated_at = Utc::now();

        if let Some(usage) = response.usage {
            // Construct the descriptive label for logging.
            let label = match (self.has_tools, self.has_schema) {
                (true, true) => "convo with tools and schema",
                (true, false) => "convo with tools",
                (false, true) => "convo with schema",
                (false, false) => "convo",
            };

            // Log the usage for this specific turn.
            log_usage_turn(self.conversation.id, &usage, label, &self.conversation.model_name)?;
            // Still update the cumulative total for the stateful conversation object.
            self.conversation.usage += usage;
        }

        // 4. Return a reference to the message just added.
        Ok(self
            .conversation
            .messages
            .last()
            .expect("Message was just added, so it must exist."))
    }
}
