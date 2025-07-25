use std::path::Path;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use tokio::task;
use futures::{stream, StreamExt};
use serde::Deserialize;

use crate::error::LLMCoreError;
use crate::vector::{KnowledgeBase, DocumentSource};
use crate::orchestra::Orchestra;
use crate::datam::{format_user_message};
use crate::lucky::{SimpleSchema, SchemaProperty};

#[derive(Deserialize)]
struct EnrichedContent {
    title: String,
    summary: String,
}

pub struct Ingestor {
    kb: KnowledgeBase,
    orchestra: Arc<Orchestra>,
}

impl Ingestor {
    pub fn new(
            db_path: &Path,
            index_path: &Path,
            embedding_model: &str,
            enrichment_model: &str,
        ) -> Result<Self, LLMCoreError> {
        let kb = KnowledgeBase::new(db_path, index_path, embedding_model)?;
        
        let schema = SimpleSchema {
            name: "enrich_content".to_string(),
            description: "A title and summary for a chunk of text.".to_string(),
            properties: vec![
                SchemaProperty {
                    name: "title".to_string(),
                    property_type: "string".to_string(),
                    description: "A concise title for the text chunk.".to_string(),
                    items: None,
                },
                SchemaProperty {
                    name: "summary".to_string(),
                    property_type: "string".to_string(),
                    description: "A detailed summary of the text chunk.".to_string(),
                    items: None,
                },
            ],
        };

        let orchestra = Arc::new(Orchestra::new(enrichment_model, None, None, Some(schema), None, None)?);
        Ok(Self { kb, orchestra })
    }

    pub async fn ingest_from_url(&self, url: &str, source_tag: &str) -> Result<(), LLMCoreError> {
        let markdown_content = self.extract_content_from_url(url).await?;
        let documents = self.process_markdown(markdown_content, url, source_tag).await?;
        self.kb.add_documents_and_build(documents).await?;
        Ok(())
    }

    pub async fn ingest_from_file(&self, file_path: &Path, source_tag: &str) -> Result<(), LLMCoreError> {
        let markdown_content = self.extract_content_from_file(file_path).await?;
        let documents = self.process_markdown(markdown_content, &file_path.to_string_lossy(), source_tag).await?;
        self.kb.add_documents_and_build(documents).await?;
        Ok(())
    }

    async fn extract_content_from_url(&self, url: &str) -> Result<String, LLMCoreError> {
        let url = url.to_string();
        task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let converter_class = PyModule::import(py, "docling.document_converter")?
                    .getattr("DocumentConverter")?;
                let converter = converter_class.call0()?;
                let result = converter.call_method1("convert", (url,))?;
                let document = result.getattr("document")?;
                let markdown = document.call_method0("export_to_markdown")?;
                markdown.extract()
            })
        })
        .await
        .map_err(|e| LLMCoreError::PythonError(e.to_string()))? // Handles JoinError
        .map_err(|e: PyErr| LLMCoreError::PythonError(e.to_string())) // Handles PyErr
    }

    async fn extract_content_from_file(&self, file_path: &Path) -> Result<String, LLMCoreError> {
        let file_path_str = file_path.to_str().ok_or_else(|| LLMCoreError::PythonError("Invalid file path".to_string()))?.to_string();
        task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let converter_class = PyModule::import(py, "docling.document_converter")?
                    .getattr("DocumentConverter")?;
                let converter = converter_class.call0()?;
                let result = converter.call_method1("convert", (file_path_str,))?;
                let document = result.getattr("document")?;
                let markdown = document.call_method0("export_to_markdown")?;
                markdown.extract()
            })
        })
        .await
        .map_err(|e| LLMCoreError::PythonError(e.to_string()))? // Handles JoinError
        .map_err(|e: PyErr| LLMCoreError::PythonError(e.to_string())) // Handles PyErr
    }

    async fn process_markdown(&self, markdown: String, url: &str, source_tag: &str) -> Result<Vec<DocumentSource>, LLMCoreError> {
        let chunks = chunk_text(&markdown, 4000);
        const CONCURRENT_REQUESTS: usize = 5;
        
        let documents_futures = chunks.into_iter().enumerate().map(|(i, chunk)| {
            let url = url.to_string();
            let source_tag = source_tag.to_string();
            let orchestra = Arc::clone(&self.orchestra);
            
            tokio::spawn(async move {
                let enriched = Ingestor::enrich_chunk(&orchestra, &chunk).await?;
                
                Ok(DocumentSource {
                    url,
                    chunk_number: (i + 1) as i32,
                    title: enriched.title,
                    summary: enriched.summary,
                    content: chunk,
                    metadata: serde_json::json!({ "source": source_tag }),
                })
            })
        });

        let stream = stream::iter(documents_futures);
        let results: Vec<_> = stream.buffer_unordered(CONCURRENT_REQUESTS).collect().await;

        let mut documents = Vec::new();
        for result in results {
            match result {
                Ok(Ok(doc)) => documents.push(doc),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(LLMCoreError::ConcurrencyError(e.to_string())),
            }
        }
        
        Ok(documents)
    }

    async fn enrich_chunk(orchestra: &Arc<Orchestra>, chunk: &str) -> Result<EnrichedContent, LLMCoreError> {
        let prompt = format!(
            r#"Please analyze the following text chunk and provide a concise title and a detailed summary.

            Text to analyze:
            ---
            {}
            ---"#,
            chunk
        );

        let messages = vec![format_user_message(prompt)];
        let response = orchestra.call_ai(messages).await?;

        let content = response.choices.get(0)
            .and_then(|c| c.message.content.as_ref())
            .ok_or_else(|| {
                LLMCoreError::ResponseParseError("No content received from enrichment model".to_string())
            })?;

        let enriched: EnrichedContent = serde_json::from_str(content)
            .map_err(|e| LLMCoreError::ResponseParseError(format!("Failed to parse enrichment JSON: {}", e)))?;
        
        Ok(enriched)
    }
}

fn chunk_text(text: &str, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current_pos = 0;
    let text_len = text.len();

    while current_pos < text_len {
        let end_pos = std::cmp::min(current_pos + chunk_size, text_len);
        let mut chunk_end = end_pos;

        if end_pos < text_len {
            let chunk_slice = &text[current_pos..end_pos];
            if let Some(last_code_block) = chunk_slice.rfind("```") {
                if last_code_block > chunk_size / 2 {
                    chunk_end = current_pos + last_code_block;
                }
            } else if let Some(last_paragraph) = chunk_slice.rfind("\n\n") {
                 if last_paragraph > chunk_size / 2 {
                    chunk_end = current_pos + last_paragraph;
                }
            } else if let Some(last_sentence) = chunk_slice.rfind(". ") {
                if last_sentence > chunk_size / 2 {
                    chunk_end = current_pos + last_sentence + 1;
                }
            }
        }
        
        let final_chunk = text[current_pos..chunk_end].trim().to_string();
        if !final_chunk.is_empty() {
            chunks.push(final_chunk);
        }
        current_pos = chunk_end;
    }

    chunks
} 