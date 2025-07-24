use crate::error::LLMCoreError;
use serde::{Deserialize, Serialize};
use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};
use std::fs;

/// Represents a single chunk of a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: i64,
    pub url: String,
    pub chunk_number: i32,
    pub title: String,
    pub summary: String,
    pub content: String,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

/// Manages a SQLite database for storing and retrieving document chunks.
pub struct Storage {
    conn: Connection,
    db_path: PathBuf,
}

impl Storage {
    pub fn new(db_path: &Path) -> Result<Self, LLMCoreError> {
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)?;
        let storage = Self { conn, db_path: db_path.to_path_buf() };
        storage.initialize_db()?;
        Ok(storage)
    }
    
    fn initialize_db(&self) -> Result<(), LLMCoreError> {
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY, url TEXT NOT NULL, chunk_number INTEGER NOT NULL,
                title TEXT NOT NULL, summary TEXT NOT NULL, content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                UNIQUE(url, chunk_number)
            )",
            [],
        )?;
        Ok(())
    }

    pub fn insert_chunk(
            &self, url: &str, chunk_number: i32, title: &str, summary: &str,
            content: &str, metadata: &serde_json::Value,
        ) -> Result<i64, LLMCoreError> {
        let metadata_str = serde_json::to_string(metadata)?;

        let mut stmt = self.conn.prepare(
            "INSERT INTO document_chunks (url, chunk_number, title, summary, content, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )?;
        let id = stmt.insert(params![url, chunk_number, title, summary, content, metadata_str])?;
        Ok(id)
    }

    pub fn get_chunk_by_id(&self, id: i64) -> Result<Option<DocumentChunk>, LLMCoreError> {
        let mut stmt = self.conn.prepare("SELECT id, url, chunk_number, title, summary, content, metadata, created_at FROM document_chunks WHERE id = ?1")?;
        let mut chunk_iter = stmt.query_map(params![id], |row| {
            let metadata = serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or(serde_json::Value::Null);
            let created_at = DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?).unwrap().with_timezone(&Utc);
            Ok(DocumentChunk {
                id: row.get(0)?, url: row.get(1)?, chunk_number: row.get(2)?,
                title: row.get(3)?, summary: row.get(4)?, content: row.get(5)?,
                metadata, created_at,
            })
        })?;

        if let Some(result) = chunk_iter.next() {
            Ok(Some(result?))
        } else {
            Ok(None)
        }
    }

    pub fn get_chunks_by_ids(&self, ids: &[i64]) -> Result<Vec<DocumentChunk>, LLMCoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let params_sql = vec!["?"; ids.len()].join(",");
        let sql = format!(
            "SELECT id, url, chunk_number, title, summary, content, metadata, created_at FROM document_chunks WHERE id IN ({})",
            params_sql
        );
        let mut stmt = self.conn.prepare(&sql)?;

        let chunk_iter = stmt.query_map(rusqlite::params_from_iter(ids.iter()), |row| {
            let metadata = serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or(serde_json::Value::Null);
            let created_at = DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?).unwrap().with_timezone(&Utc);

            Ok(DocumentChunk {
                id: row.get(0)?,
                url: row.get(1)?,
                chunk_number: row.get(2)?,
                title: row.get(3)?,
                summary: row.get(4)?,
                content: row.get(5)?,
                metadata,
                created_at,
            })
        })?;

        let mut chunk_map = std::collections::HashMap::new();
        for result in chunk_iter {
            let chunk = result?;
            chunk_map.insert(chunk.id, chunk);
        }

        let ordered_chunks = ids
            .iter()
            .filter_map(|id| chunk_map.remove(id))
            .collect();
        Ok(ordered_chunks)
    }

    /// Removes a document chunk from the database by its unique ID.
    pub fn remove_chunk(&self, id: i64) -> Result<usize, LLMCoreError> {
        let rows_affected = self.conn.execute("DELETE FROM document_chunks WHERE id = ?1", params![id])?;
        Ok(rows_affected)
    }

    /// Deletes the entire SQLite database file from the filesystem.
    /// This method consumes the Storage object, ensuring the file lock is released.
    pub fn delete_database(self) -> Result<(), LLMCoreError> {
        // Deconstruct `self` to take ownership of its fields.
        let Storage { conn, db_path } = self;
        // Explicitly drop the connection. This closes the database and releases the file lock.
        drop(conn);
        // Now that the lock is released, we can safely delete the file.
        fs::remove_file(&db_path)?;
        Ok(())
    }
}
