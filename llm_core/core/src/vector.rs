use std::fs;
use std::path::Path;

use arroy::distances::DotProduct;
use arroy::{Database, Reader, Writer};
use heed::EnvOpenOptions;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::embed::Embedder;
use crate::error::LLMCoreError;
use crate::config::storage::{Storage, DocumentChunk};

pub struct VectorIndex {
    env: heed::Env,
    db: Database<DotProduct>,
    dimensions: usize,
}

impl VectorIndex {
    pub fn new(path: &Path, dimensions: usize) -> Result<Self, LLMCoreError> {
        fs::create_dir_all(path)?;
        let env = unsafe { EnvOpenOptions::new().open(path)? };
        let mut wtxn = env.write_txn()?;
        let db: Database<DotProduct> = env.create_database(&mut wtxn, None)?;
        wtxn.commit()?;
        Ok(Self { env, db, dimensions })
    }
}

pub struct KnowledgeBase {
    storage: Storage,
    vector_index: VectorIndex,
    embedder: Embedder,
}

#[derive(Clone)]
pub struct DocumentSource {
    pub url: String,
    pub chunk_number: i32,
    pub title: String,
    pub summary: String,
    pub content: String,
    pub metadata: serde_json::Value,
}


impl KnowledgeBase {
    pub fn new(
        db_path: &Path,
        index_path: &Path,
        embedding_model: &str,
    ) -> Result<Self, LLMCoreError> {
        let embedder = Embedder::new(embedding_model, None)?;
        let storage = Storage::new(db_path)?;
        let vector_index = VectorIndex::new(index_path, embedder.dimensions)?;
        Ok(Self { storage, vector_index, embedder })
    }

    pub async fn add_documents_and_build(
        &self,
        documents: Vec<DocumentSource>,
    ) -> Result<(), LLMCoreError> {
        let contents: Vec<String> = documents.iter().map(|d| d.content.clone()).collect();
        let embeddings = self.embedder.get_embeddings(contents).await?;

        let mut wtxn = self.vector_index.env.write_txn()?;
        let writer = Writer::<DotProduct>::new(self.vector_index.db, 0, self.vector_index.dimensions);

        for (i, doc) in documents.iter().enumerate() {
            let vector = embeddings.get(i).ok_or_else(|| {
                LLMCoreError::RetrievalError(format!("Missing embedding for document {}", i))
            })?;
            
            let id = self.storage.insert_chunk(
                &doc.url,
                doc.chunk_number,
                &doc.title,
                &doc.summary,
                &doc.content,
                &doc.metadata,
            )?;
            
            writer.add_item(&mut wtxn, id as u32, vector)?;
        }
        
        let mut rng = StdRng::seed_from_u64(42);
        writer.builder(&mut rng).build(&mut wtxn)?;
        
        wtxn.commit()?;
        Ok(())
    }

    pub async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<DocumentChunk>, LLMCoreError> {
        let embeddings = self.embedder.get_embeddings(vec![query.to_string()]).await?;
        let query_vector = embeddings.get(0).ok_or_else(|| {
            LLMCoreError::RetrievalError("Failed to generate embedding for query".to_string())
        })?;

        let rtxn = self.vector_index.env.read_txn()?;
        let reader = Reader::<DotProduct>::open(&rtxn, 0, self.vector_index.db)?;
        
        let query_builder = reader.nns(limit);
        let result = query_builder.by_vector(&rtxn, query_vector)?;
        let ids: Vec<i64> = result.into_iter().map(|(id, _)| id as i64).collect();
        
        self.storage.get_chunks_by_ids(&ids)
    }
}
