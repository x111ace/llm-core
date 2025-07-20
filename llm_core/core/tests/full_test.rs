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
    convo::Chat,
    datam::{format_system_message, format_user_message},
    lucky::{SchemaProperty, SimpleSchema},
    orchestra::Orchestra,
    sorter::{Sorter, SortingInstructions},
};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::env;
use std::path::Path;
use std::sync::Arc;

const MODEL_NAME: &str = "GPT 4o MINI";

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


// --- Test: Native Tool Mode ---
// Goal: Verify that the Orchestra can use a model's native tool-calling ability.
#[tokio::test]
// #[ignore]
async fn test_tool_mode() {
    println!("\n--- Running Test: Native Tool Mode ({}) ---\n", MODEL_NAME);
    let tool_library = get_rust_tool_library();

    let orchestra = Orchestra::new(
        MODEL_NAME, 
        Some(0.0), 
        Some(tool_library), 
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
    let mut chat = Chat::from_file(resume_path, None, None, None).unwrap();

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

