use crate::config::USAGE_DATA_DIR;
use crate::error::LLMCoreError;
use crate::datam::Usage;

use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::io::ErrorKind;
use serde_json::json;
use chrono::Utc;
use uuid::Uuid;
use std::fs;

/// Logs token usage for a single API call to a structured JSON file.
///
/// This function maintains a `usage.json` file, creating it if it doesn't exist.
/// It structures the data hierarchically: Day -> Hour -> ID -> {task_label, model_name, events}.
/// This makes it easy to track token consumption over time and by session.
pub fn log_usage_turn(
        id: Uuid,
        turn_usage: &Usage,
        label: &str,
        model_name: &str,
    ) -> Result<(), LLMCoreError> {
    let usage_dir = &*USAGE_DATA_DIR;
    fs::create_dir_all(usage_dir)?;

    let usage_file_path = usage_dir.join("usage.json");

    // Read the existing file or create a new BTreeMap if it doesn't exist.
    // We use BTreeMap to maintain sorted keys (days, hours).
    let mut all_usage: BTreeMap<String, JsonValue> = match fs::read_to_string(&usage_file_path) {
        Ok(contents) => serde_json::from_str(&contents)?,
        Err(e) if e.kind() == ErrorKind::NotFound => BTreeMap::new(),
        Err(e) => return Err(e.into()),
    };

    let now = Utc::now();
    let day_key = now.format("%Y-%m-%d").to_string();
    let hour_key = now.format("%H:00").to_string(); // Format hour with ":00"
    let id_key = id.to_string();

    // Navigate or create the nested structure: Day -> Hour -> Convo
    let day_data = all_usage
        .entry(day_key)
        .or_insert_with(|| JsonValue::Object(Default::default()));

    let hour_data = day_data
        .as_object_mut()
        .unwrap()
        .entry(hour_key)
        .or_insert_with(|| JsonValue::Object(Default::default()));

    // Get or create the entry for this ID (convo or job)
    let usage_entry = hour_data
        .as_object_mut()
        .unwrap()
        .entry(id_key)
        .or_insert_with(|| {
            json!({
                "task_label": label,
                "model_name": model_name,
                "events": []
            })
        });

    // Update the label and model name in case they have changed.
    usage_entry["task_label"] = JsonValue::String(label.to_string());
    usage_entry["model_name"] = JsonValue::String(model_name.to_string());

    // Append the new usage event for this turn to the "events" array.
    usage_entry["events"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::to_value(turn_usage)?);

    // Write the updated data back to the file.
    let updated_json = serde_json::to_string_pretty(&all_usage)?;
    fs::write(&usage_file_path, updated_json)?;

    Ok(())
}
