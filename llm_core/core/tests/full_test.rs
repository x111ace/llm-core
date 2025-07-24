// cargo test -- 
// --ignored
// --nocapture 
// --test-threads=1 

// // // // // // /|\ \\ \\ \\ \\ \\ \\

//||: cargo test

// // // // // // /|\ \\ \\ \\ \\ \\ \\

// cargo test -- --ignored --test-threads=1

use _llm_core::{
    config::get_rust_tool_library,
    config::storage::Storage,
    orchestra::Orchestra,
    convo::Chat,
    embed::Embedder,
    vector::{KnowledgeBase, DocumentSource},
    datam::{format_system_message, format_user_message},
    lucky::{SchemaProperty, SimpleSchema},
    sorter::{Sorter, SortingInstructions},
};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::env;
use std::path::Path;
use std::sync::Arc;
use base64::Engine as _; // Import the Engine trait for base64 decoding
use tempfile::tempdir;

const MODEL_NAME: &str = "GEMINI 2.0 FLASH";

// CONCLUSIONS:
// All models here have been tested. Of them, only deepseek (OpenRouter & Ollama) have failed.
// They are best suited for single turn conversations, with the OpenRouter model being able to handle tools.
// All the other models have passed all tests, and our system is working as expected!

    // Google
        // GEMINI 2.0 FLASH ✓
    // xAI
        // GROK 4 ✓
        // GROK 3 ✓
        // GROK 3 MINI ✓
    // Anthropic
        // CLAUDE OPUS 4 
        // CLAUDE SONNET 4 
        // CLAUDE SONNET 3.7 
        // CLAUDE SONNET 3.5 
        // CLAUDE HAIKU 3.5 
        // CLAUDE HAIKU 3 
    // OpenAI
        // GPT 4o ✓
        // GPT 4o MINI ✓
        // GPT 4.1 ✓
        // GPT 4.1 MINI ✓
        // GPT 4.1 NANO ✓
    // Inception Labs
        // MERCURY CODER ✓
        // MERCURY (X - TOOLS)
    // Ollama
        // QWEN 3:0.6B ✓
        // GRANITE 3.3:2B ✓
        // DEEPSEEK R1-0528:1.5B (X - SCHEMA / TOOLS)
    // OpenRouter
        // DEEPSEEK R1-0528:FREE (X - ASYNC)

// Helper function to create the tool library, so we don't repeat code.
// REMOVED: This function is redundant. We will use the centralized `get_rust_tool_library`.









// --- Test: Normal Chat Mode ---
// Goal: Verify that the Orchestra works for simple, non-structured chat.
#[tokio::test]
#[ignore]
async fn test_normal_mode() {
    println!("\n--- Running Test: Normal Chat Mode ({}) ---\n", MODEL_NAME);

    let orchestra = Orchestra::new(
        MODEL_NAME, 
        Some(0.7), 
        None, 
        None,
        None,
        None
    ).unwrap();
    let messages = vec![format_user_message(
        "Hello! In one short sentence, what is Rust?".to_string(),
    )];
    let response = orchestra.call_ai(messages).await.unwrap();

    let content = response.choices[0].message.content.as_ref().unwrap();
    println!("Standard chat response: {}", content);

    assert!(!content.is_empty(), "Response should not be empty.");

    // Updated assertion: The response should not be a JSON object or array.
    // A simple JSON string (a sentence in quotes) is acceptable.
    match serde_json::from_str::<JsonValue>(content) {
        Ok(json_val) => {
            assert!(
                !json_val.is_object() && !json_val.is_array(),
                "Response should be plain text, not a JSON object or array."
            );
        }
        Err(_) => {
            // If it fails to parse as JSON, it's definitely plain text, which is a pass.
        }
    }
}

// --- Test: Thinking Mode (Prompt-Induced) ---
// Goal: Verify that a standard model can be prompted to produce reasoning content.
#[tokio::test]
#[ignore]
async fn test_thinking_mode() {
    println!("\n--- Running Test: Thinking Mode ({}) ---\n", MODEL_NAME);

    // 1. Create a chat session with thinking_mode explicitly enabled.
    let mut chat = Chat::new(
        MODEL_NAME,
        Some("You are a helpful assistant.".to_string()),
        None,
        None,
        Some(true), // Enable thinking mode
        Some(true), // Enable debug output
    )
    .unwrap();

    // 2. Send a prompt that requires some thought.
    let response_message = chat.send("Yooo dude, whatsup?").await.unwrap();

    // 3. Assert that both reasoning and content were produced.
    let reasoning = response_message.reasoning_content.as_ref();
    let content = response_message.content.as_ref();

    println!("Reasoning: {:?}", reasoning.unwrap_or(&"No reasoning content found.".to_string()));
    println!("Final Response: {}", content.unwrap());

    assert!(reasoning.is_some(), "Should have reasoning content.");
    assert!(!reasoning.unwrap().is_empty(), "Reasoning content should not be empty.");
    
    assert!(content.is_some(), "Should have final response content.");
    assert!(!content.unwrap().is_empty(), "Final response content should not be empty.");

    // 4. Assert that the thinking tags are not present in the final output.
    assert!(!reasoning.unwrap().to_lowercase().contains("<think>"));
    assert!(!content.unwrap().to_lowercase().contains("<think>"));
}

// --- Test: Native Schema Mode ---
// Goal: Verify that the Orchestra can enforce a JSON schema using a model's native ability.
#[tokio::test]
#[ignore]
async fn test_schema_mode() {
    println!("\n--- Running Test: Native Schema Mode ({}) ---\n", MODEL_NAME);

    #[derive(Deserialize, Debug)]
    struct UserDetails {
        name: String,
        age: u8,
    }

    let schema = SimpleSchema {
        name: "extract_user_details".to_string(),
        description: "Extracts the user's name and age from the text.".to_string(),
        properties: vec![
            SchemaProperty {
                name: "name".to_string(),
                property_type: "string".to_string(),
                description: "The name of the user.".to_string(),
                items: None,
            },
            SchemaProperty {
                name: "age".to_string(),
                property_type: "number".to_string(),
                description: "The age of the user.".to_string(),
                items: None,
            },
        ],
    };

    let orchestra = Orchestra::new(
        MODEL_NAME, 
        Some(0.0), 
        None, 
        Some(schema),
        None,
        None
    ).unwrap();
    let messages = vec![format_user_message("My name is Alex and I'm 34 years old.".to_string())];
    let response = orchestra.call_ai(messages).await.unwrap();

    let content = response.choices[0].message.content.as_ref().unwrap();
    println!("Schema-enforced JSON response: {}", content);
    let details: UserDetails = serde_json::from_str(content).unwrap();

    assert_eq!(details.name.to_lowercase(), "alex");
    assert_eq!(details.age, 34);
}

// --- Test: Native Tool Mode ---
// Goal: Verify that the Orchestra can use a model's native tool-calling ability.
#[tokio::test]
#[ignore]
async fn test_tooler_mode() {
    println!("\n--- Running Test: Native Tool Mode ({}) ---\n", MODEL_NAME);
    let tool_library = get_rust_tool_library();

    let orchestra = Orchestra::new(
        MODEL_NAME, 
        Some(0.0), 
        Some(tool_library), 
        None,
        None,
        None
    ).unwrap();
    let messages = vec![format_user_message("What time is it?".to_string())];
    let response = orchestra.call_ai(messages).await.unwrap();

    let content = response.choices[0].message.content.as_ref().unwrap();
    println!("Final synthesized response: {}", content);

    let content_lower = content.to_lowercase();
    assert!(!content_lower.contains("tool_calls"));
    assert!(
        content_lower.contains(":") && (content_lower.contains("am") || content_lower.contains("pm")),
        "Final response should contain the synthesized time from the tool."
    );
}

// --- Test: Automatic Lucky Fallback (Sorter Mode) ---
// Goal: Verify that the Orchestra automatically falls back to Lucky prompting when a model lacks native schema support.
#[tokio::test]
#[ignore]
async fn test_sorter_mode() {
    println!("\n--- Running Test: Sorter Mode (Lucky Fallback) ---\n");

    #[derive(Deserialize, Debug)]
    struct SortResponse {
        category: String,
    }

    let sorter_schema = SimpleSchema {
        name: "sorting_response".to_string(),
        description: "Sorts a data item into a category.".to_string(),
        properties: vec![SchemaProperty {
            name: "category".to_string(),
            property_type: "string".to_string(),
            description: "The category for the data item.".to_string(),
            items: None,
        }],
    };

    // Use a model known not to have native schema support to force the Lucky fallback.
    // The Orchestra should print a warning that it's falling back.
    let orchestra = Orchestra::new(
        MODEL_NAME, 
        Some(0.0), 
        None, 
        Some(sorter_schema),
        None,
        None
    ).unwrap();
    
    // We'll create a dummy Sorter instance just to borrow its instruction-building logic.
    // This ensures our test prompt is consistent with the real Sorter's prompt.
    let sorting_instructions = SortingInstructions {
        data_item_name: "Electronic Device".to_string(),
        data_profile_description: "Consumer electronic devices.".to_string(),
        item_sorting_guidelines: vec!["Sort by primary function.".to_string()],
        provided_categories: vec![
            "fruit".to_string(),
            "technology".to_string(),
            "vehicle".to_string(),
        ],
    };
    
    // Use the actual orchestra instance to build the prompt.
    let sorter_for_prompting = Sorter::new(
        Arc::new(orchestra.clone()), // We clone the orchestra for the prompt builder
        sorting_instructions,
        None,
        false
    ).unwrap();
    
    let system_prompt = sorter_for_prompting.build_sorting_instructions_message();
    let user_prompt = "Please classify the following item: MacBook Pro";

    let messages = vec![
        format_system_message(system_prompt),
        format_user_message(user_prompt.to_string()),
    ];

    let response = orchestra.call_ai(messages).await.unwrap();
    let content = response.choices[0].message.content.as_ref().unwrap();
    println!("Lucky-generated JSON response: {}", content);
    let sort_result: SortResponse = serde_json::from_str(content).unwrap();

    assert_eq!(sort_result.category, "technology");
}









// --- Test: *NEW* Conversation Mode ---
// Goal: Verify that the Chat session manager can maintain context over several turns and save the result.
#[tokio::test]
#[ignore]
async fn test_conversation_mode() {
    println!("\n--- Running Test: Conversation Mode ({}) ---\n", MODEL_NAME);

    // 1. Create a new chat session
    println!("Phase 1: Starting new chat session...");
    
    let mut chat = Chat::new(
        MODEL_NAME,
        Some("You are a helpful assistant who remembers details from the conversation.".to_string()),
        None,
        None,
        None,
        None,
    )
    .unwrap();

    // 2. Send a series of messages to test context retention
    println!("Sending initial message to set context...");
    let response1 = chat.send("Hi! My name is Ace.").await.unwrap();
    println!("Assistant: {}", response1.content.as_ref().unwrap());

    println!("\nSending a distractor message...");
    let response2 = chat.send("What is the capital of France?").await.unwrap();
    println!("Assistant: {}", response2.content.as_ref().unwrap());

    println!("\nAsking a question that requires memory...");
    let response3 = chat.send("What is my name?").await.unwrap();
    println!("Assistant: {}", response3.content.as_ref().unwrap());

    // 3. Assert that the model remembered the context
    let content_lower = response3.content.as_ref().unwrap().to_lowercase();
    assert!(
        content_lower.contains("ace"),
        "Response should contain the name 'Ace'."
    );
    assert_eq!(chat.conversation.messages.len(), 7, "History should have 7 messages (System + 3 pairs of User/Assistant).");

    // 4. Save the conversation to a dedicated output directory within the tests folder.
    // This avoids hardcoding absolute paths and makes the test portable.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let save_dir = Path::new(manifest_dir).join("tests").join("output");
    std::fs::create_dir_all(&save_dir).expect("Failed to create save directory");

    println!("\nPhase 2: Saving conversation history to directory: {}", save_dir.display());
    chat.save_history(Some(save_dir.to_str().unwrap())).unwrap();

    // 5. Verify the file was created and then clean up
    let convo_id = chat.conversation.id;
    let expected_file_path = save_dir.join(format!("convo-{}.json", convo_id));
    println!("Verifying file exists at: {}", expected_file_path.display());

    assert!(
        expected_file_path.exists(),
        "Conversation file should have been created at the specified path."
    );

    // println!("\nPhase 3: Cleaning up test file...");
    // std::fs::remove_file(&expected_file_path).unwrap();
    // assert!(
    //     !expected_file_path.exists(),
    //     "Test file should be deleted."
    // );
    println!("Test complete.");
}

// --- Test: *NEW* Resume Conversation ---
// Goal: Verify that the Chat session manager can resume a conversation from a file and maintain context.
#[tokio::test]
#[ignore]
async fn test_resume_conversation() {
    println!("\n--- Running Test: Resume Conversation ---\n");

    let resume_file_path = "tests/convo-9efb52e6-1f72-49d2-aa55-a91334400596.json";
    let resume_path = Path::new(resume_file_path);

    // 1. Verify the source file exists
    assert!(
        resume_path.exists(),
        "The resume file '{}' must exist for this test.",
        resume_file_path
    );

    // 2. Resume the chat session from the file
    println!("Phase 1: Resuming chat from {}...", resume_file_path);
    let mut chat = Chat::from_file(resume_path, None, None, None, None).unwrap();

    // Verify the initial state
    assert_eq!(chat.conversation.messages.len(), 7, "Should have loaded 7 messages from the file.");
    println!("Chat resumed with model: {}", chat.orchestra.user_facing_model_name);

    // 3. Send distraction messages
    println!("\nSending a distractor message...");
    let response1 = chat.send("What is the square root of 144?").await.unwrap();
    println!("Assistant: {}", response1.content.as_ref().unwrap());

    println!("\nSending another distractor message...");
    let response2 = chat.send("Tell me a one-sentence joke.").await.unwrap();
    println!("Assistant: {}", response2.content.as_ref().unwrap());

    // 4. Ask the context-dependent question
    println!("\nAsking a question that requires memory from the loaded file...");
    let final_response = chat.send("Do you remember my name?").await.unwrap();
    println!("Assistant: {}", final_response.content.as_ref().unwrap());

    // 5. Assert that the model remembered the name from the original session
    let content_lower = final_response.content.as_ref().unwrap().to_lowercase();
    assert!(
        content_lower.contains("ace"),
        "Response should contain the name 'Ace' from the resumed conversation."
    );

    // Final state check
    // 7 messages from original file + (3 turns * 2 messages/turn) = 13 messages total
    assert_eq!(chat.conversation.messages.len(), 13, "History should have 13 messages after resuming and chatting.");
    println!("\nTest complete. Context was successfully maintained.");
}









#[tokio::test]
#[ignore]
async fn test_chat_image_gen() {
    // 1. Set up a chat session with a capable model (GPT 4o Mini) and provide it with our Rust tool library.
    let mut chat = Chat::new(
        MODEL_NAME,
        Some("You are a helpful assistant with access to tools. Your goal is to use the tools to help the user, and then report the results of the tool use.".to_string()),
        Some(get_rust_tool_library()),
        None,
        None, // thinking_mode
        Some(true), // Enable debug output to see the flow
    )
    .expect("Failed to create Chat instance");

    // 2. Simulate a user asking for an image to be generated with a specific path.
    let output_dir = "C:/Users/ac3la/OneDrive/Desktop/CL-CODER/.learn/databases/llm-core/";
    let file_name = "mary_poppins.png";
    let full_path = format!("{}/{}", output_dir, file_name);

    // Clean up any previous run's file to ensure the test is fresh.
    let _ = std::fs::remove_file(&full_path);

    let user_prompt = format!(
        "Please create an image of Mary Poppins and save it to '{}'.",
        full_path
    );
    let result = chat.send(&user_prompt).await;

    // 3. Assert that the multi-step operation (user -> tool -> synthesis) was successful.
    assert!(result.is_ok(), "Chat send failed: {:?}", result.err());

    // 4. Get the final message from the assistant and verify its content.
    let assistant_message = chat.conversation.messages.last().unwrap();
    assert_eq!(assistant_message.role, "assistant");

    let content = assistant_message
        .content
        .as_ref()
        .expect("Assistant message should have content");

    println!("Final assistant response: {}", content);

    // 5. Assert that the assistant's response confirms the image was created and provides the path.
    assert!(content.contains(&file_name));

    // 6. Verify the file exists at the specified path and is not empty.
    let file_metadata = std::fs::metadata(&full_path)
        .unwrap_or_else(|_| panic!("File should exist at specified path: {}", full_path));
    assert!(
        file_metadata.len() > 0,
        "Saved image file is empty"
    );
}

#[tokio::test]
#[ignore]
async fn test_raww_image_gen() {
    let orchestra = Orchestra::new("GEMINI 2.0 FLASH", None, None, None, None, Some(true))
        .expect("Failed to create Orchestra");

    let prompt = "A photorealistic image of a cat wearing a witch hat";
    let result = orchestra
        .generate_image(prompt, "GEMINI 2.0 FLASH IMAGE GEN")
        .await;

    assert!(
        result.is_ok(),
        "Image generation failed: {:?}",
        result.err()
    );
    let image_result = result.unwrap();
    assert!(
        image_result.image_data_b64.is_some(),
        "No image data in result"
    );

    // Decode and save the image to verify it's valid
    let image_data = image_result.image_data_b64.unwrap();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(image_data)
        .expect("Failed to decode base64 image");
    assert!(!bytes.is_empty(), "Decoded image data is empty");

    let output_dir = "tests/output";
    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");
    let file_path = format!("{}/test_image_output.png", output_dir);
    let mut file = std::fs::File::create(&file_path).expect("Failed to create image file");
    std::io::Write::write_all(&mut file, &bytes).expect("Failed to write image to file");

    println!("Image saved to {}", file_path);
    assert!(
        std::fs::metadata(file_path).unwrap().len() > 0,
        "Saved image file is empty"
    );
}

#[tokio::test]
#[ignore]
async fn test_sorter_in_chat() {
    let mut chat = Chat::new(
        MODEL_NAME,
        Some("You are a helpful assistant with access to tools. Your goal is to use the tools to help the user, and then report the results of the tool use.".to_string()),
        Some(get_rust_tool_library()),
        None,
        None, // thinking_mode
        Some(true), // Enable debug output to see the flow
    )
    .expect("Failed to create Chat instance");

    // Define a specific output path for the test.
    let output_path = "tests/output/fruits_and_animals.json";
    // Clean up any file from a previous test run.
    let _ = std::fs::remove_file(output_path);

    // Provide a minimal prompt with only the essential information.
    // The tool should be able to infer the rest.
    let user_prompt = format!(
        "Please sort these items and save them to '{}': [apple, cat, dog, banana]",
        output_path
    );

    let result = chat.send(&user_prompt).await;
    assert!(result.is_ok(), "Chat send failed: {:?}", result.err());

    let assistant_message = chat.conversation.messages.last().unwrap();
    assert_eq!(assistant_message.role, "assistant");

    let content = assistant_message
        .content
        .as_ref()
        .expect("Assistant message should have content");
    println!("Final assistant response: {}", content);

    // Assert that the AI's response confirms the action and references the output.
    assert!(content.to_lowercase().contains("sorted"));
    assert!(content.contains(output_path));

    // Assert that the output file was actually created and contains the correct data.
    assert!(
        Path::new(output_path).exists(),
        "Output file was not created at the specified path."
    );
    let file_content = std::fs::read_to_string(output_path)
        .expect("Failed to read the output file.");
    let sorted_json: JsonValue = serde_json::from_str(&file_content)
        .expect("Failed to parse output file as JSON.");
    
    // Check for the existence of expected categories.
    assert!(sorted_json.get("animal").is_some() || sorted_json.get("animals").is_some(), "Category 'animal(s)' should exist.");
    assert!(sorted_json.get("fruit").is_some() || sorted_json.get("fruits").is_some(), "Category 'fruit(s)' should exist.");
}









#[tokio::test]
// #[ignore]
async fn test_embeddings() {
    let embedder = Embedder::new("TEXT-EMB 3 SMALL", Some(true)).unwrap();
    let embeddings = embedder
        .get_embeddings(vec!["Hello, world!".to_string()])
        .await
        .unwrap();

    println!(
        "Received {} embedding(s), first one with {} dimensions.",
        embeddings.len(),
        embeddings.get(0).map_or(0, |e| e.len())
    );

    // 1. We asked for one embedding, so we should get one back.
    assert_eq!(embeddings.len(), 1);

    // 2. The default dimension for text-embedding-3-small is 1536.
    // Let's assert that the vector has the expected size.
    assert_eq!(embeddings[0].len(), 1536);
}

#[tokio::test]
// #[ignore]
async fn test_create_database() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("rag.db");
    let storage = Storage::new(&db_path);
    assert!(storage.is_ok(), "Storage::new should succeed.");
    assert!(db_path.exists());
}

#[tokio::test]
// #[ignore]
async fn test_database_remove_and_delete() {
    // 1. Setup: Create a temporary database in a known location.
    let temp_dir = std::env::temp_dir();

    // 2. Create a dummy file to simulate a database file.
    let dummy_file_path = temp_dir.join("dummy.db");
    std::fs::File::create(&dummy_file_path).expect("Failed to create dummy file");
    assert!(dummy_file_path.exists());

    // 3. Attempt to remove and delete the file.
    let remove_result = std::fs::remove_file(&dummy_file_path);
    assert!(remove_result.is_ok(), "Failed to remove file: {:?}", remove_result.err());
    assert!(!dummy_file_path.exists(), "File should have been deleted.");

    // 4. Verify that the file is no longer present.
    let verify_result = std::fs::metadata(&dummy_file_path);
    assert!(verify_result.is_err(), "File should not exist after deletion.");
}


#[tokio::test]
// #[ignore]
async fn test_knowledge_base_end_to_end() {
    // 1. Setup temporary paths for the database and vector index.
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_kb.db");
    let index_path = dir.path().join("test_kb_index");

    // Ensure clean state from previous runs
    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_dir_all(&index_path);

    // 2. Initialize the KnowledgeBase.
    let knowledge_base = KnowledgeBase::new(&db_path, &index_path, "TEXT-EMB 3 SMALL")
        .expect("Failed to create KnowledgeBase");

    // 3. Add documents in a batch.
    let metadata = serde_json::json!({});
    let documents = vec![
        DocumentSource {
            url: "doc1".to_string(),
            chunk_number: 1,
            title: "Fruit".to_string(),
            summary: "".to_string(),
            content: "The apple is a sweet, edible fruit produced by an apple tree.".to_string(),
            metadata: metadata.clone(),
        },
        DocumentSource {
            url: "doc2".to_string(),
            chunk_number: 1,
            title: "Tool".to_string(),
            summary: "".to_string(),
            content: "A hammer is a tool, most often a hand tool, consisting of a weighted head fixed to a long handle.".to_string(),
            metadata: metadata.clone(),
        },
        DocumentSource {
            url: "doc3".to_string(),
            chunk_number: 1,
            title: "Animal".to_string(),
            summary: "".to_string(),
            content: "The domestic dog is a domesticated descendant of the wolf.".to_string(),
            metadata: metadata.clone(),
        },
    ];

    knowledge_base
        .add_documents_and_build(documents)
        .await
        .expect("Failed to add documents and build index");

    // 4. Perform a search.
    let search_query = "What is a hammer?";
    let results = knowledge_base
        .search(search_query, 1)
        .await
        .expect("Search failed");

    // 5. Assert the results.
    assert_eq!(results.len(), 1, "Should retrieve one document.");
    let retrieved_doc = &results[0];
    assert_eq!(retrieved_doc.title, "Tool");
    assert!(retrieved_doc.content.contains("hammer"));

    println!(
        "Search for '{}' returned: '{}'",
        search_query, retrieved_doc.content
    );
}
