// llm-core/src/sorter.rs

use std::collections::{BTreeMap, HashMap, HashSet};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tokio::sync::Semaphore;
use std::path::PathBuf;
use std::sync::Arc;
use std::io::Write; // Import the Write trait
use uuid::Uuid;
use std::fs;

// --- Import from llm-core ---
use crate::config::{DEFAULT_SORTER_INPUT_DIR, DEFAULT_SORTER_OUTPUT_DIR}; // Import default paths from config
use crate::orchestra::Orchestra;
use crate::datam::{
    Message,
    format_system_message, format_user_message, format_assistant_message, Usage,
};
use crate::lucky::{SimpleSchema, SchemaProperty, SchemaItems};
use crate::error::LLMCoreError;

// --- Data Structures ---

#[derive(Deserialize, Debug, Clone, Serialize)]
pub struct SortingInstructions {
    pub data_item_name: String,
    pub data_profile_description: String,
    pub item_sorting_guidelines: Vec<String>,
    #[serde(default)]
    pub provided_categories: Vec<String>,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct SortResponse {
    pub category: String,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct CategoryGenerationResponse {
    pub categories: Vec<String>,
}

// --- Hardcoded Prompts ---

pub const SORTER_SYSTEM_PROMPT: &str = r#"You are a classifier of provided data items.

Your task is to analyze the provided data items (as "Item:") and use the `sorting_response` tool to provide the most appropriate category.
- Call the `sorting_response` tool with the chosen category name.
- Do not include any other text or explanations in your response, only the tool call.
"#;

pub const CATEGORY_GEN_INITIAL_PROMPT: &str = r#"You are an AI assistant tasked with generating a concise list of categories for a given set of data items. The user is having trouble creating categories and needs your help.

### DATA ITEM NAME

{data_item_name}

### DATA ITEMS INTERESTS DESCRIBED

{data_profile_description}

Based on the data profile, analyze the following block of text containing multiple data items, each on a new line. Generate a small list of unique, relevant, generalized, lowercase categories that the items can be sorted into.
Use the `category-generation-schema` tool to return the list of categories.
"#;

pub const CATEGORY_GEN_NEXT_PROMPT: &str = r#"Excellent. You are continuing to help the user generate categories.

You have already processed a previous block of data and generated the following list of categories:

### EXISTING CATEGORIES

---
{existing_categories}
---

Now, analyze the NEW block of text below, containing the next chunk of data items. 

Generate ONLY new, unique, generalized, lowercase categories that are not in the list above.

Return ONLY the new categories. 
"#;

// --- Sorter ---

// This struct encapsulates all logic for sorting data items using an AI model.
pub struct Sorter {
    orchestra: Arc<Orchestra>,
    pub output_path: PathBuf, // Make this field public for direct access
    system_message_template: String,
    sorting_instructions: SortingInstructions,
    category_set: HashSet<String>,
    debug: bool,
    // Removed sorter_schema and category_gen_schema fields
}
impl Sorter {
    pub fn new(
            orchestra: Arc<Orchestra>,
            sorting_instructions: SortingInstructions,
            output_path: Option<PathBuf>,
            debug: bool,
        ) -> Result<Self, LLMCoreError> {
        // Use provided output_path or default from config
        let final_output_path = output_path.unwrap_or_else(|| {
            // Ensure default output directory exists
            if let Err(e) = fs::create_dir_all(&*DEFAULT_SORTER_OUTPUT_DIR) {
                eprintln!("Error creating default sorter output directory: {}", e);
            }
            DEFAULT_SORTER_OUTPUT_DIR.clone()
        });

        // Schemas are now created on-demand in the methods that use them.

        let category_set: HashSet<String> =
            sorting_instructions.provided_categories.iter().cloned().collect();

        Ok(Self {
            orchestra,
            output_path: final_output_path,
            system_message_template: SORTER_SYSTEM_PROMPT.to_string(),
            sorting_instructions,
            category_set,
            debug,
        })
    }

    pub fn output_path(&self) -> &PathBuf {
        &self.output_path
    }

    // --- Input Data Collection (These will be public for library users) ---
    pub async fn collect_items_recursively(path: &PathBuf, items_vec: &mut Vec<String>) -> Result<(), LLMCoreError> {
        let mut entries = tokio::fs::read_dir(path)
            .await
            .map_err(|e| LLMCoreError::IoError(e))?;

        while let Some(res) = entries.next_entry().await.transpose() {
            let entry = res.map_err(|e| LLMCoreError::IoError(e))?;
            let entry_path = entry.path();

            if entry_path.is_file() {
                if entry_path.extension().map_or(false, |ext| ext == "json") {
                    let content = tokio::fs::read_to_string(&entry_path)
                        .await
                        .map_err(|e| LLMCoreError::IoError(e))?;
                    
                    if content.trim().is_empty() {
                        println!("Skipping empty JSON file: '{}'", entry_path.display());
                        continue;
                    }

                    match serde_json::from_str::<Vec<String>>(&content) {
                        Ok(mut parsed_items) => {
                            if parsed_items.is_empty() {
                                println!("Skipping JSON file '{}' as it contains an empty list.", entry_path.display());
                                continue;
                            }
                            println!("✅ Loaded {} items from '{}'", parsed_items.len(), entry_path.display());
                            items_vec.append(&mut parsed_items);
                        },
                        Err(e) => {
                            eprintln!("Error parsing JSON list from file '{}': {}", entry_path.display(), e);
                            eprintln!("File content: {}", content);
                        }
                    }
                } else {
                    println!("Skipping non-JSON file: '{}'", entry_path.display());
                }
            } else if entry_path.is_dir() {
                Box::pin(Self::collect_items_recursively(&entry_path, items_vec)).await?;
            }
        }
        Ok(())
    }

    pub async fn collect_items_from_file(path: &PathBuf, items_vec: &mut Vec<String>) -> Result<(), LLMCoreError> {
        if path.extension().map_or(false, |ext| ext == "json") {
            let content = tokio::fs::read_to_string(path)
                .await
                .map_err(|e| LLMCoreError::IoError(e))?;
            
            if content.trim().is_empty() {
                println!("Skipping empty JSON file: '{}'", path.display());
                return Ok(());
            }

            match serde_json::from_str::<Vec<String>>(&content) {
                Ok(mut parsed_items) => {
                    if parsed_items.is_empty() {
                        println!("Skipping JSON file '{}' as it contains an empty list.", path.display());
                    } else {
                        println!("✅ Loaded {} items from '{}'", parsed_items.len(), path.display());
                        items_vec.append(&mut parsed_items);
                    }
                },
                Err(e) => {
                    return Err(LLMCoreError::ResponseParseError(format!("Error parsing JSON list from file '{}': {}", path.display(), e)));
                }
            }
        } else {
            return Err(LLMCoreError::ConfigError(format!("Skipping non-JSON file: '{}'", path.display())));
        }
        Ok(())
    }

    // --- Category Generation ---
    async fn generate_categories(&mut self, items: &[String], chunk_size: usize) -> Result<(Vec<String>, Usage), LLMCoreError> {
        let category_gen_schema = SimpleSchema {
            name: "category-generation-schema".to_string(),
            description: "Generates a list of categories for data items.".to_string(),
            properties: vec![
                SchemaProperty {
                    name: "categories".to_string(),
                    property_type: "array".to_string(),
                    description: "A list of unique, generalized category names for the provided data items.".to_string(),
                    items: Some(SchemaItems { item_type: "string".to_string() }),
                }
            ]
        };

        // Create a dedicated Orchestra for this task with the correct schema
        let cat_gen_orchestra = Orchestra::new(
            &self.orchestra.user_facing_model_name, // Use the same model name
            Some(0.0), // Use a low temperature for deterministic category generation
            None,
            Some(category_gen_schema),
            None, // thinking_mode
            Some(self.debug),
        )?;
        
        println!("\n--- GENERATING CATEGORIES ---\n");
        let item_chunks: Vec<&[String]> = items.chunks(chunk_size).collect();
        let mut total_usage = Usage::default();

        for (i, chunk) in item_chunks.iter().enumerate() {
            println!("Processing chunk {}/{}...", i + 1, item_chunks.len());
            let mut messages: Vec<Message> = Vec::new();
            let user_content = chunk.join("\n");

            if i == 0 {
                messages.push(format_system_message(CATEGORY_GEN_INITIAL_PROMPT.to_string().replace("{data_item_name}", &self.sorting_instructions.data_item_name).replace("{data_profile_description}", &self.sorting_instructions.data_profile_description)));
                messages.push(format_user_message(user_content));
            } else {
                let existing_categories_str = self.category_set.iter().cloned().collect::<Vec<_>>().join("\n");
                let next_prompt_formatted = CATEGORY_GEN_NEXT_PROMPT.to_string().replace("{existing_categories}", &existing_categories_str);
                
                messages.push(format_system_message(CATEGORY_GEN_INITIAL_PROMPT.to_string().replace("{data_item_name}", &self.sorting_instructions.data_item_name).replace("{data_profile_description}", &self.sorting_instructions.data_profile_description)));
                messages.push(format_user_message("{data_items_chunk_1}".to_string()));
                messages.push(format_assistant_message(existing_categories_str));
                messages.push(format_system_message(next_prompt_formatted));
                messages.push(format_user_message(user_content));
            }

            // Use the dedicated orchestra for the call
            let response = cat_gen_orchestra.call_ai(messages).await?;
            
            if let Some(usage) = response.usage {
                total_usage += usage;
            }

            if let Some(choice) = response.choices.first() {
                if let Some(content) = &choice.message.content {
                    match serde_json::from_str::<CategoryGenerationResponse>(content) {
                    Ok(parsed_response) => {
                        for category in parsed_response.categories {
                            let clean_category = category.trim().to_lowercase().replace(' ', "-");
                            if !clean_category.is_empty() && self.category_set.insert(clean_category.clone()) {
                                println!("  + New category found: {}", clean_category);
                            }
                        }
                    },
                    Err(e) => {
                        eprintln!("JSON Parse Error for category generation: {}", e);
                            eprintln!("Raw Content: {}", content);
                        }
                    }
                }
            }
        }
        self.category_set = self.category_set.iter().cloned().collect();
        Ok((self.category_set.iter().cloned().collect(), total_usage))
    }

    // --- Core Sorting Logic ---
    pub fn build_sorting_instructions_message(&self) -> String {
        let mut final_message = "# SYSTEM MESSAGE\n\n## SORTING INSTRUCTIONS\n\n".to_string() + &self.system_message_template;
        let i_sort = &self.sorting_instructions;

        final_message = final_message.replace("{data_item_name}", &i_sort.data_item_name);

        let guidelines = i_sort
            .item_sorting_guidelines
            .iter()
            .map(|line| format!("- {}", line))
            .collect::<Vec<_>>()
            .join("\n");
        final_message.push_str(&format!("\n### ITEM SORTING GUIDELINES:\n\n{}\n", guidelines));

        if !i_sort.data_profile_description.is_empty() {
            final_message.push_str(&format!(
                "\n### DATA ITEMS INTERESTS DESCRIBED:\n\n{}\n",
                i_sort.data_profile_description
            ));
        }

        if !self.category_set.is_empty() {
            let categories = self
                .category_set
                .iter()
                .map(|cat| format!("- {}", cat))
                .collect::<Vec<_>>()
                .join("\n");
            final_message.push_str(&format!("\n### EXISTING CATEGORIES:\n\n{}\n", categories));
        }

        final_message
    }

    pub async fn sort_items(&mut self, items: &[String], swarm_size: usize) -> Result<(BTreeMap<String, Vec<String>>, Vec<String>, Usage), LLMCoreError> {
        let sorter_schema = SimpleSchema {
            name: "sorting_response".to_string(),
            description: "Sorts a data item into a category based on provided instructions.".to_string(),
            properties: vec![
                SchemaProperty {
                    name: "category".to_string(),
                    property_type: "string".to_string(),
                    description: "The category name for the data item.".to_string(),
                    items: None,
                },
            ],
        };

        // Create a dedicated Orchestra for the sorting task
        let sort_orchestra = Arc::new(Orchestra::new( // Wrap in Arc
            &self.orchestra.user_facing_model_name, // Use the correct user-facing name
            Some(0.0), // Low temperature for sorting
            None,
            Some(sorter_schema),
            None, // thinking_mode
            Some(self.debug),
        )?);

        let system_message_content = self.build_sorting_instructions_message();
        
        let semaphore = Arc::new(Semaphore::new(swarm_size));
        let mut tasks = vec![];
        
        println!("\n--- SORTING ITEMS ---\n");

        // CRITICAL NOTE ON CONCURRENCY:
        // This loop spawns all requests concurrently, controlled by the `swarm_size` semaphore.
        // This is highly efficient for APIs with high rate limits (e.g., paid OpenAI tiers).
        // However, for free or heavily rate-limited APIs (like OpenRouter's free tier),
        // setting a `swarm_size` greater than the API's requests-per-minute limit
        // will likely result in `429 Too Many Requests` errors. For such cases,
        // a `swarm_size` of 1 is recommended to process items sequentially.
        for item in items {
            let orchestra_clone = Arc::clone(&sort_orchestra); // Clone the Arc, not the Orchestra
            let system_message_clone = system_message_content.clone();
            let semaphore_clone = Arc::clone(&semaphore);
            let item_clone = item.clone();
            
            tasks.push(tokio::spawn(async move {
                let _permit = semaphore_clone.acquire().await.unwrap();
                let user_prompt = format!("Item: {}", item_clone);
                let messages = vec![
                    format_system_message(system_message_clone),
                    format_user_message(user_prompt),
                ];

                // The Orchestra instance held by Sorter has the schema already set.
                let result = orchestra_clone
                    .call_ai(messages)
                    .await;
                (item_clone, result)
            }));
        }

        let mut sort_results = HashMap::new();
        let mut total_usage = Usage::default();

        for task in tasks {
            let (item, result) = task.await?;
            match result {
                Ok(response) => {
                    if let Some(usage) = response.usage {
                        total_usage += usage;
                    }

                    if let Some(choice) = response.choices.first() {
                        if let Some(content) = &choice.message.content {

                            // NEW: Optional debug log of full raw content (including thoughts) for analysis.
                            if self.debug {
                                println!("[SORTER DEBUG] Raw content for item '{}': {}\n", item, content);
                            }

                            // Pre-process the content to strip out <think> blocks, which some models add.
                            let content_after_think = if let Some(end_pos) = content.rfind("</think>") {
                                content[end_pos + "</think>".len()..].trim()
                            } else {
                                content.trim()
                            };

                            // First, try to parse as the full SortResponse struct.
                            let sort_response = match serde_json::from_str::<SortResponse>(content_after_think) {
                                Ok(res) => Ok(res),
                                Err(_) => {
                                    // NEW: Handle cases where the model returns a tool call as a string
                                    let new_res = if let Ok(json_val) = serde_json::from_str::<JsonValue>(content_after_think) {
                                        if let Some(args) = json_val.get("arguments") {
                                            serde_json::from_value::<SortResponse>(args.clone()).map_err(|e| e.to_string())
                                        } else {
                                            Err("No arguments field found in JSON".to_string())
                                        }
                                    } else {
                                        Err("Content is not a valid JSON value".to_string())
                                    };

                                    if let Ok(res) = new_res {
                                        Ok(res)
                                    } else {
                                        // Try parsing as a single-element array containing the response
                                        match serde_json::from_str::<Vec<SortResponse>>(content_after_think) {
                                            Ok(mut vec) if !vec.is_empty() => Ok(vec.remove(0)),
                                            _ => {
                                                // Fallback for generic map like {"answer": "..."}
                                                match serde_json::from_str::<HashMap<String, String>>(content_after_think) {
                                                    Ok(map) => {
                                                        if let Some(value) = map.values().next() {
                                                            Ok(SortResponse { category: value.clone() })
                                                        } else {
                                                            Err("JSON object is empty".to_string())
                                                        }
                                                    },
                                                    Err(_) => {
                                                        // Final fallback for raw string "..."
                                                        serde_json::from_str::<String>(content_after_think)
                                                            .map(|s| SortResponse { category: s })
                                                            .map_err(|e| e.to_string())
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            };

                            match sort_response {
                                Ok(res) => {
                                    let category = res.category;
                                    if !self.category_set.contains(&category) {
                                        println!("\n**NEW CATEGORY** -> {}\n", category);
                                        self.category_set.insert(category.clone());
                                    }
                                    println!("ITEM: {} -> SORT: {}", item, category);
                                    sort_results.insert(item, category);
                                }
                                Err(e) => {
                                    eprintln!("JSON Parse Error for item '{}': {}", item, e);
                                    eprintln!("Raw Content: {}", content);
                                }
                            }
                        }
                    }
                }
                Err(e) => eprintln!("API Error for item '{}': {}", item, e),
            }
        }

        let clean_sort_results = self.build_sorting_results(&sort_results, true)?;
        // Return the updated category set
        let updated_categories = self.category_set.iter().cloned().collect();

        Ok((clean_sort_results, updated_categories, total_usage))
    }
    
    fn build_sorting_results(&self, sort_results: &HashMap<String, String>, save: bool) -> Result<BTreeMap<String, Vec<String>>, LLMCoreError> {
        let mut categorized_items: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (item, category) in sort_results {
            categorized_items.entry(category.clone()).or_default().push(item.clone());
        }

        for titles in categorized_items.values_mut() {
            titles.sort();
        }

        if save {
            let final_path: PathBuf;

            // NEW: Check if the provided path is a file or a directory.
            if self.output_path.extension().is_some() && self.output_path.file_name().is_some() {
                // It's a full file path. Use it directly.
                // Ensure its parent directory exists.
                if let Some(parent) = self.output_path.parent() {
                    fs::create_dir_all(parent).map_err(|e| {
                        LLMCoreError::IoError(std::io::Error::new(
                            e.kind(),
                            format!("Failed to create parent directory '{}': {}", parent.display(), e),
                        ))
                    })?;
                }
                final_path = self.output_path.clone();
            } else {
                // It's a directory. Create a unique filename inside it.
                fs::create_dir_all(&self.output_path).map_err(|e| {
                     LLMCoreError::IoError(std::io::Error::new(
                        e.kind(),
                        format!("Failed to create output directory '{}': {}", self.output_path.display(), e),
                    ))
                })?;
                let file_name = format!("sorted-data-{}.json", Uuid::new_v4().to_string().split('-').next().unwrap_or(""));
                final_path = self.output_path.join(file_name);
            }

            let json_str = serde_json::to_string_pretty(&categorized_items)?;
            
            // Use OpenOptions for a more robust file write/overwrite operation.
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true) // This will clear the file if it exists, achieving overwrite.
                .open(&final_path)
                .map_err(|e| {
                    LLMCoreError::IoError(std::io::Error::new(
                        e.kind(),
                        format!("Failed to open or create file '{}': {}", final_path.display(), e),
                    ))
                })?;

            file.write_all(json_str.as_bytes()).map_err(|e| {
                 LLMCoreError::IoError(std::io::Error::new(
                    e.kind(),
                    format!("Failed to write to final path '{}': {}", final_path.display(), e),
                ))
            })?;
            println!("\n✅ Results saved: '{}'", final_path.display());
        }

        Ok(categorized_items)
    }

    // --- Public API for library users ---
    pub async fn run_sorting_task(
            orchestra: Arc<Orchestra>,
            input_path: Option<PathBuf>,
            items_list: Option<Vec<String>>,
            output_path: Option<PathBuf>,
            sorting_instructions: SortingInstructions,
            swarm_size: usize,
            debug: bool,
        ) -> Result<(BTreeMap<String, Vec<String>>, Vec<String>, Usage, usize), LLMCoreError> {
        let mut sorter = Self::new(orchestra, sorting_instructions, output_path, debug)?;
        let items_to_process: Vec<String>;
        let mut total_usage = Usage::default();

        if let Some(items) = items_list {
            println!("📚 Reading {} items provided directly in a list...", items.len());
            items_to_process = items;
        } else {
            let path_to_process = if let Some(path) = input_path {
                path
            } else {
                if let Err(e) = tokio::fs::create_dir_all(&*DEFAULT_SORTER_INPUT_DIR).await {
                    return Err(LLMCoreError::IoError(e));
                }
                DEFAULT_SORTER_INPUT_DIR.clone()
            };
    
            if path_to_process.is_file() {
                println!("📚 Reading items from file: '{}'...", path_to_process.display());
                let mut items_from_file = Vec::new();
                Self::collect_items_from_file(&path_to_process, &mut items_from_file).await?;
                items_to_process = items_from_file;
            } else if path_to_process.is_dir() {
                println!("📚 Reading items from folder: '{}' (including subfolders)...", path_to_process.display());
                let mut items_from_dir = Vec::new();
                Self::collect_items_recursively(&path_to_process, &mut items_from_dir).await?;
                items_to_process = items_from_dir;
            } else {
                return Err(LLMCoreError::ConfigError(format!("Provided path '{}' is neither a file nor a directory.", path_to_process.display())));
            }
        }

        let item_count = items_to_process.len();
        if item_count == 0 {
            return Err(LLMCoreError::ConfigError(format!("No valid items found to process.")));
        }

        // If no categories are provided, generate them before sorting.
        if sorter.sorting_instructions.provided_categories.is_empty() {
            println!("\nNo categories provided. Attempting to generate categories from data items...");
            let (new_categories, cat_gen_usage) = sorter.generate_categories(&items_to_process, 50).await?;
            total_usage += cat_gen_usage;

            if new_categories.is_empty() {
                return Err(LLMCoreError::ChatError(
                    "Category generation resulted in an empty list. Cannot proceed with sorting."
                        .to_string(),
                ));
            }
            println!("\n✅ Generated {} new categories.", new_categories.len());
            // The sorter's internal category_set is already updated by generate_categories
        }

        let (sorted_items, updated_categories, sort_usage) = sorter.sort_items(&items_to_process, swarm_size).await?;
        total_usage += sort_usage;
        Ok((sorted_items, updated_categories, total_usage, item_count))
    }
}

