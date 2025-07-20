use reqwest::{header, Client, StatusCode};
use serde_json::Value as JsonValue;
use crate::error::LLMCoreError;
use tokio::task::JoinHandle;
use tokio::sync::Semaphore;
use std::time::Duration;
use tokio::time::sleep;
use std::sync::Arc;
use rand::Rng;

/// Defines the retry strategy for API calls.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub jitter: Jitter,
}

/// Defines the type of jitter to apply to retry delays.
#[derive(Debug, Clone, Copy)]
pub enum Jitter {
    Full,
    None,
}

/// Executes a single API call with retry logic.
pub async fn execute_single_call(
        url: String,
        headers: header::HeaderMap,
        body: JsonValue,
        retry_policy: &RetryPolicy,
    ) -> Result<String, LLMCoreError> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30)) // Add a 30-second timeout to prevent stalling
        .build()?;

    for i in 0..retry_policy.max_retries {
        let response_result = client
            .post(&url)
            .headers(headers.clone())
            .json(&body)
            .send()
            .await;

        match response_result {
            Ok(response) => {
                let status = response.status();
                let response_text = response.text().await?;

                if status.is_success() {
                    return Ok(response_text);
                }

                if status == StatusCode::TOO_MANY_REQUESTS {
                    eprintln!(
                        "Rate limit exceeded. Retrying... (Attempt {}/{})",
                        i + 1,
                        retry_policy.max_retries
                    );
                    if i < retry_policy.max_retries - 1 {
                        let mut delay_ms = retry_policy.base_delay_ms * 2_u64.pow(i);
                        if let Jitter::Full = retry_policy.jitter {
                            let jitter_ms = rand::thread_rng().gen_range(0..=delay_ms / 4);
                            delay_ms += jitter_ms;
                        }
                        sleep(Duration::from_millis(delay_ms)).await;
                        continue; // Retry the loop
                    }
                }
                // For other non-success statuses, return our new ApiError.
                return Err(LLMCoreError::ApiError {
                    status: status.as_u16(),
                    body: response_text,
                });
            }
            Err(e) => {
                // Handle network-level errors
                eprintln!(
                    "Network request failed (Attempt {}/{}): {}",
                    i + 1,
                    retry_policy.max_retries,
                    e
                );
                if i >= retry_policy.max_retries - 1 {
                    return Err(e.into());
                }
            }
        }
    }
    // If the loop finishes, it means all retries were exhausted.
    Err(LLMCoreError::ChatError(
        "API call exhausted all retries without success.".to_string(),
    ))
}

/// Executes a swarm of API calls concurrently, with a limit on concurrency.
///
/// This is ideal for batch processing tasks. It spawns, runs, and awaits all
/// tasks, returning a final vector of results.
pub async fn execute_swarm_call(
        url: String,
        headers: header::HeaderMap,
        payloads: Vec<JsonValue>,
        max_concurrent_requests: usize,
        retry_policy: RetryPolicy,
    ) -> Vec<Result<String, LLMCoreError>> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));
    let mut tasks = Vec::new();

    for payload in payloads {
        let semaphore_clone = Arc::clone(&semaphore);
        let url_clone = url.clone();
        let headers_clone = headers.clone();

        let task: JoinHandle<Result<String, LLMCoreError>> = tokio::spawn(async move {
            let _permit = semaphore_clone
                .acquire()
                .await
                .expect("Failed to acquire semaphore permit");
            execute_single_call(url_clone, headers_clone, payload, &retry_policy).await
        });
        tasks.push(task);
    }

    let mut results = Vec::new();
    for task in tasks {
        results.push(task.await.unwrap_or_else(|e| Err(e.into())));
    }
    results
}

pub struct LLMClient {}

impl LLMClient {
    pub fn new() -> Self {
        LLMClient {}
    }
}
