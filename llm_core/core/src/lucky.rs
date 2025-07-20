use serde_json::{Value as JsonValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use once_cell::sync::Lazy;
use regex::Regex;

/// Represents a simplified, serializable JSON schema for guiding model responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimpleSchema {
    pub name: String,
    pub description: String,
    pub properties: Vec<SchemaProperty>,
}

/// Defines a single property within a SimpleSchema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaProperty {
    pub name: String,
    #[serde(rename = "type")]
    pub property_type: String, // e.g., "string", "number", "array"
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub items: Option<SchemaItems>,
}

/// Defines the type of items within an 'array' property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaItems {
    #[serde(rename = "type")]
    pub item_type: String, // e.g. "string"
}

// --- New Additions for Full Functionality ---

static CODE_LANGUAGE_PATTERNS: Lazy<HashMap<&'static str, Regex>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("python", Regex::new(r"(?i:python|py)").unwrap());
    m.insert("javascript", Regex::new(r"(?i:javascript|js|node)").unwrap());
    m.insert("rust", Regex::new(r"(?i:rust|rs)").unwrap());
    m.insert("json", Regex::new(r"(?i:json)").unwrap());
    // Add other languages as needed
    m
});

/// Cleans code blocks by removing language tags and markdown formatting.
fn clean_code_block(field: &str, language_hint: Option<&str>) -> String {
    let mut cleaned_field = field.to_string();
    let pattern_to_use = language_hint
        .and_then(|lang| CODE_LANGUAGE_PATTERNS.get(lang))
        .map(|re| re.to_string())
        .unwrap_or_else(|| {
            let all_patterns: Vec<&str> =
                CODE_LANGUAGE_PATTERNS.values().map(|re| re.as_str()).collect();
            format!("({})", all_patterns.join("|"))
        });

    let start_pattern = Regex::new(&format!(r"^(\s|`)*{}?\s*", pattern_to_use)).unwrap();
    cleaned_field = start_pattern.replace(&cleaned_field, "").to_string();

    let end_pattern = Regex::new(r"(\s|`)*$").unwrap();
    cleaned_field = end_pattern.replace(&cleaned_field, "").to_string();

    cleaned_field
}

/// Recursively traverses a `JsonValue` and "polishes" it by converting
/// any string that looks like a number or boolean into its proper JSON type.
fn polish_json_values(value: JsonValue) -> JsonValue {
    match value {
        JsonValue::Object(map) => {
            let new_map = map
                .into_iter()
                .map(|(k, v)| (k, polish_json_values(v)))
                .collect();
            JsonValue::Object(new_map)
        }
        JsonValue::Array(arr) => {
            let new_arr = arr.into_iter().map(polish_json_values).collect();
            JsonValue::Array(new_arr)
        }
        JsonValue::String(s) => {
            if let Ok(i) = s.parse::<i64>() {
                JsonValue::Number(i.into())
            } else if let Ok(f) = s.parse::<f64>() {
                serde_json::Number::from_f64(f)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::String(s))
            } else if let Ok(b) = s.parse::<bool>() {
                JsonValue::Bool(b)
            } else {
                JsonValue::String(s)
            }
        }
        _ => value,
    }
}

/// Recursively wraps the keys of a JSON-like structure with delimiters.
/// This prepares the template for the LLM.
fn wrap_with_delimiters_recursive(
        format: &JsonValue,
        delimiter: &str,
        depth: usize,
    ) -> JsonValue {
    let current_delimiter = delimiter.repeat(depth);
    match format {
        JsonValue::Object(map) => {
            let new_map = map
                .into_iter()
                .map(|(k, v)| {
                    let new_key = format!("{}{}{}", current_delimiter, k, current_delimiter);
                    let new_value = wrap_with_delimiters_recursive(v, delimiter, depth + 1);
                    (new_key, new_value)
                })
                .collect();
            JsonValue::Object(new_map)
        }
        JsonValue::Array(arr) => {
            let new_arr = arr
                .iter()
                .map(|v| wrap_with_delimiters_recursive(v, delimiter, depth + 1))
                .collect();
            JsonValue::Array(new_arr)
        }
        JsonValue::String(s) => {
            // Modify the type hint for the prompt if needed, e.g., list -> array
            let modified_type = s.replace("list", "array");
            JsonValue::String(format!("<{}>", modified_type))
        }
        _ => format.clone(),
    }
}

/// Prepares the prompt for the `Lucky` structured response mode.
/// This function now correctly mirrors the logic from `lucky_struct.py`,
/// appending strict JSON instructions to the user's system prompt.
pub fn prepare_lucky_prompt(
        system_prompt: &str,
        user_prompt: &str,
        output_format: &JsonValue,
        delimiter: &str,
        available_tools: Option<&Vec<crate::tools::ToolDefinition>>,
        is_synthesis_turn: bool,
    ) -> (String, String) {
    let new_output_format = wrap_with_delimiters_recursive(output_format, delimiter, 1);
    let wrapped_json_string = serde_json::to_string_pretty(&new_output_format).unwrap();

    let tool_prompt_section = if let Some(tools) = available_tools {
        let tool_list = tools
            .iter()
            .map(|t| format!("- `{}`: {}", t.function.name, t.function.description))
            .collect::<Vec<String>>()
            .join("\n");
        
        // This line is now conditional and uses a clear "Yes".
        let context_line = if is_synthesis_turn {
            "**CURRENT CONTEXT:** The last message is from a tool: **Yes**.\n"
        } else {
            "" // Omit this line if not a synthesis turn.
        };

        format!(
            "#### **DUAL MODE: TOOL CALLING & SYNTHESIS**\n\n\
            You operate in one of two modes:\n\
            1.  **Tool Calling Mode:** If the user's request can be answered by a tool, your ONLY output is a JSON object to call that tool. The schema for this JSON is provided under 'JSON SCHEMA'.\n\
            2.  **Synthesis Mode:** If the last message in the conversation is from a 'tool', your ONLY task is to create a conversational, human-readable answer for the user based on the tool's output. DO NOT output JSON in this mode. DO NOT apologize or refuse to answer.\n\n\
            {}{}\n\
            **Available Tools (for Tool Calling Mode only):**\n\
            {}\n\n",
            context_line,
            if is_synthesis_turn { "SYNTHESIS" } else { "TOOL CALLING" },
            tool_list
        )
    } else {
        "".to_string()
    };

    let json_prompt_section = if !is_synthesis_turn {
        format!(
            "### **JSON MODE** Instructions \n\n\
        You are a machine that outputs a single, valid JSON object. Do not add any text, explanation, or markdown. \n\n\
        DELIMITER = '{del}' (repeat count exactly as in schema).\n\n\
        #### **JSON MODE OUTPUT** Instructions (From the Developer) \n\n\
        You will be provided a schema for the JSON object you must output. You must adhere to the provided schema, exactly as it is detailed. \n\
        REMEMBER: Use the provided JSON SCHEMA KEYS, along with the DELIMITER built out of '#' to wrap the provided keys. \n\
        The delimited keys (e.g., '{del}key{del}') define the exact structure—do NOT change, rename, or invent provided keys. \n\n\
        #### **JSON MODE OUTPUT** Instructions (Formatting Rules (GENERAL)) \n\
        1. Reply begins with '{{' and ends with '}}'. No markdown, no ``` fences.\n\
        2. Use each key **exactly** as shown in the schema, including both delimiter halves (e.g. '{del}key{del}').\n\
        3. The keys in your JSON output MUST be the keys provided in the schema, along with the '{del}' delimiters (e.g., '{del}key{del}'). The key wrapped in a delimiter is a programmatic identifier and must be provided! DO NOT INVENT YOUR OWN KEYS!\n\
        4. Update placeholder values (e.g., '<type:str>') with **PLAIN, UNWRAPPED** information. IMMEDIATELY REMOVE ALL < > CHARACTERS.\n\
        5. Ensure these exact delimited keys are present in your JSON: {key_list}.\n\
        6. DO NOT invent your own keys (like 'answer')or change the structure; USE THE SCHEMA'S KEYS EXACTLY AS THEY ARE PROVIDED.\n\
        7. Make sure to close all arrays and objects properly, and replace ALL placeholders with real values from the input.\n\
        8. For lists (arrays), iterate through the input and create a separate JSON object for each distinct item found.\n\n\
        #### **JSON MODE OUTPUT** Instructions (Formatting Rules (IMPORTANT)) \n\n\
        FOLLOW THIS EXACT OUTPUT PATTERN: {{\"###provided_key###\": \"output_value\"}} \n\
        - The '###provided_key###' is a placeholder for the schema key, and 'output_value' is the value to be output.\n\n\
        REMEMBER: \n\
        - RETURN THE PROVIDED KEYS INSIDE THE DELIMITER; VERBATIM. DO NOT CHANGE, SHORTEN, OR MODIFY THE KEYS OR DELIMITER (e.g., '###provided_key###' is a placeholder for the key that will be shown in the provided schema.).\n\
        - Your entire response must be ONLY the valid JSON object and NOTHING else. Use the provided keys, inside the provided DELIMITER THAT IS BUILT OUT OF MULTIPLE '#' CHARACTERS. \n\
        - All output_values must be simple, unwrapped data matching the type (e.g., a string is just 'output_value', NOT '<type:output_value>' or '<<output_value>>').\n\n\
        - Values are bare and literal—e.g., \"blue\" (string) or 42 (number); no further quotes, tags, or code syntax.\n\
        - Replace placeholders like '<string>' with plain values ONLY—do not add extra '< >', tags, or wrappers.\n\
        #### **JSON MODE OUTPUT** Instructions (Output Rules (VERY IMPORTANT)): \n\
        **OUTPUT EXACTLY:**\n\
        Output a flat JSON object with no extra nesting unless explicitly shown in the schema as follows...\n\\n
        **FOLLOW THIS EXACT OUTPUT PATTERN:** {{\"###provided_key###\": \"output_value\"}} \n\
        - The '###provided_key###' is a placeholder for the schema key, and 'output_value' is the value to be output.\n\n\
        **STRICTLY FORBIDDEN:** FALSELY USING 'answer' AS A JSON KEY. \n\
        **STRICTLY FORBIDDEN:** Using values from the input (like 'red' or 'blue') as keys; keys are structural and come ONLY from the schema below. \n\
        **STRICTLY FORBIDDEN:** Treating the provided schema as executable code—no functions, calls, or syntax in values.\n\
        **STRICTLY FORBIDDEN:** Extra keys, arrays, or nested objects that are not explicitly shown in the schema.\n\
        **STRICTLY FORBIDDEN:** Using colons or syntax in output_values; output_values are simple strings/numbers/booleans.\n\
        **STRICTLY FORBIDDEN:** ABSOLUTELY NO ANGLE BRACKETS (<< >>) IN VALUES. ONLY PLAIN STRINGS.\n\
        #### **JSON SCHEMA:**\n\
        This is the provided schema that details the JSON object format you must output in: \n\
        ```json\n{schema_json}\n```",
        del = delimiter,
        key_list = if let Some(obj) = output_format.as_object() {
            obj.keys()
               .map(|k| format!("'{0}{1}{0}'", delimiter, k))
               .collect::<Vec<String>>()
               .join(", ")
        } else {
            "".to_string()
        },
        schema_json = wrapped_json_string
        )
    } else {
        "".to_string()
    };
    
    let final_system_prompt = format!("{}{}{}", tool_prompt_section, json_prompt_section, system_prompt);

    (final_system_prompt, user_prompt.to_string())
}

/// The main entry point for parsing a response in `Lucky` mode.
///
/// This function takes the raw text from an LLM, isolates the JSON-like part,
/// and then uses a recursive, delimiter-based strategy to parse it into a
/// valid `JsonValue`, performing type validation along the way.
pub fn parse_lucky_response(
        raw_response_text: &str,
        output_format: &JsonValue,
        delimiter: &str,
    ) -> Result<JsonValue, String> {
    // Robustly extract JSON from raw text, handling markdown code blocks.
    let json_str = if let Some(captures) = regex::Regex::new(r"```json\s*(\{[\s\S]*\})\s*```")
        .unwrap()
        .captures(raw_response_text)
    {
        captures.get(1).map_or("", |m| m.as_str())
    } else if let (Some(start), Some(end)) = (
        raw_response_text.find('{'),
        raw_response_text.rfind('}'),
    ) {
        &raw_response_text[start..=end]
    } else {
        return Err(format!(
            "Could not find a valid JSON object in the response: {}",
            raw_response_text
        ));
    };

    // This regex finds any quoted key followed by a colon.
    let key_finder_re = Regex::new(r#""([^"]+)"\s*:"#).unwrap();

    let cleaned_json_text = key_finder_re.replace_all(json_str, |caps: &regex::Captures| {
            // Get the matched key, e.g., "###key##"
            let dirty_key = &caps[1];
            // Get the first character of the delimiter, defaulting to '#'. This is more robust.
            let delimiter_char = delimiter.chars().next().unwrap_or('#');
            // Remove all instances of the delimiter character, not just the exact substring.
            let clean_key = dirty_key.replace(delimiter_char, "");
            // Reconstruct the JSON key with a colon.
            format!(r#""{}":"#, clean_key)
        });

    // Now, parse the cleaned, standard JSON.
    match serde_json::from_str::<JsonValue>(&cleaned_json_text) {
        Ok(standard_json) => {
            // Validate the structure against the original format.
            match validate_structure(&standard_json, output_format) {
                Ok(_) => Ok(polish_json_values(standard_json)),
                Err(validation_err) => Err(format!(
                    "JSON has incorrect structure. Error: {}",
                    validation_err
                )),
            }
        }
        Err(e) => Err(format!(
            "Failed to parse cleaned JSON. Error: {}. Raw Response: {}",
            e, raw_response_text
        )),
    }
}

/// Validates and converts a string to a `JsonValue` based on a type hint.
fn validate_and_convert_type(field_text: &str, format_str: &str) -> Result<JsonValue, String> {
    let clean_text = field_text.trim().trim_matches('"').trim_matches('\'').trim();
    let format_str_clean = format_str.trim().trim_start_matches('<').trim_end_matches('>');

    if let Some(type_info) = format_str_clean.strip_prefix("type:").map(|s| s.trim()) {
        if type_info.starts_with("code") {
            let language_hint = type_info.strip_prefix("code").and_then(|s| s.strip_prefix(':'));
            return Ok(JsonValue::String(clean_code_block(
                clean_text,
                language_hint,
            )));
        }
        if type_info.starts_with("enum") {
            let enum_values_str = type_info
                .strip_prefix("enum")
                .and_then(|s| s.trim().strip_prefix('('))
                .and_then(|s| s.trim().strip_suffix(')'))
                .ok_or_else(|| "Invalid enum format. Expected enum( [...] )".to_string())?;

            let enum_values: Vec<JsonValue> = serde_json::from_str(enum_values_str)
                .map_err(|_| "Enum values are not a valid JSON array.".to_string())?;

            let field_as_value =
                serde_json::from_str(field_text).unwrap_or(JsonValue::String(clean_text.to_string()));

            if enum_values.contains(&field_as_value) {
                return Ok(field_as_value);
            } else {
                return Err(format!(
                    "Field value '{}' is not one of the allowed enum values: {:?}",
                    field_text, enum_values
                ));
            }
        }

        match type_info {
            "str" => Ok(JsonValue::String(clean_text.to_string())),
            "int" => clean_text
                .parse::<i64>()
                .map(JsonValue::from)
                .map_err(|e| e.to_string()),
            "float" => clean_text
                .parse::<f64>()
                .map(JsonValue::from)
                .map_err(|e| e.to_string()),
            "bool" => clean_text
                .parse::<bool>()
                .map(JsonValue::from)
                .map_err(|e| e.to_string()),
            _ => Ok(JsonValue::String(clean_text.to_string())),
        }
    } else {
        Ok(JsonValue::String(clean_text.to_string()))
    }
}

/// Checks if a parsed `JsonValue` conforms to the structure of the `output_format`.
fn validate_structure(value: &JsonValue, format: &JsonValue) -> Result<(), String> {
    match format {
        JsonValue::Object(format_map) => {
            let value_map = value
                .as_object()
                .ok_or_else(|| format!("Expected object, got {:?}", value))?;
            for (key, format_val) in format_map {
                let value_val = value_map
                    .get(key)
                    .ok_or_else(|| format!("Missing key: {}", key))?;
                validate_structure(value_val, format_val)?;
            }
            Ok(())
        }
        JsonValue::Array(format_arr) => {
            let value_arr = value
                .as_array()
                .ok_or_else(|| format!("Expected array, got {:?}", value))?;
            if let Some(format_template) = format_arr.get(0) {
                for item in value_arr {
                    validate_structure(item, format_template)?;
                }
            }
            Ok(())
        }
        JsonValue::String(format_str) => {
            let value_as_string = value.to_string();
            validate_and_convert_type(&value_as_string, format_str).map(|_| ())
        }
        _ => Err("Invalid format shape in the template.".to_string()),
    }
}

#[cfg(test)]
mod tests {
    // Import the function we want to test
    use super::clean_code_block;

    #[test]
    fn test_clean_code_block_removes_tags() {
        let input = "```json\n{\"key\": \"value\"}\n```";
        let expected = "{\"key\": \"value\"}";
        assert_eq!(clean_code_block(input, Some("json")), expected);
    }

    #[test]
    fn test_another_case() {
        // ... more tests ...
    }
}
