//! Local SQLite-backed vector store (no server needed).
//!
//! [`LocalChroma`] stores embeddings as f32 BLOBs in SQLite via `rusqlite`.
//! Supports collection management, add/query/update/delete, cosine similarity
//! computed in Rust. In-memory mode available via `:memory:` path.

use anyhow::{Result, anyhow};
use rusqlite::{params, Connection};
use serde_json::Value;
use std::path::Path;
use std::sync::Mutex;
use uuid::Uuid;

use crate::core::embeddings::cosine_similarity;

// ---------------------------------------------------------------------------
// LocalChroma — persistent SQLite-backed vector store
// ---------------------------------------------------------------------------
pub struct LocalChroma {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone)]
pub struct LocalCollection {
    pub id: String,
    pub name: String,
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct LocalEntry {
    pub id: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: Option<Value>,
    pub document: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LocalQueryResult {
    pub ids: Vec<String>,
    pub distances: Vec<f64>,
    pub documents: Vec<Option<String>>,
    pub metadatas: Vec<Option<Value>>,
}

impl LocalChroma {
    pub fn new(path: &str) -> Result<Self> {
        let db_path = Path::new(path);
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)?;
        let slf = Self { conn: Mutex::new(conn) };
        slf.init_tables()?;
        Ok(slf)
    }

    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let slf = Self { conn: Mutex::new(conn) };
        slf.init_tables()?;
        Ok(slf)
    }

    fn init_tables(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS collections (
                id    TEXT PRIMARY KEY,
                name  TEXT UNIQUE NOT NULL,
                metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS entries (
                id              TEXT PRIMARY KEY,
                collection_id   TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                embedding       BLOB,
                metadata        TEXT,
                document        TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_entries_collection
                ON entries(collection_id);",
        )?;
        Ok(())
    }

    fn blob_from_vec(v: &[f32]) -> Vec<u8> {
        let bytes: Vec<u8> = v.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        bytes
    }

    fn vec_from_blob(blob: &[u8]) -> Vec<f32> {
        blob.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // -- collections --------------------------------------------------------
    pub fn list_collections(&self) -> Result<Vec<LocalCollection>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, name, metadata FROM collections ORDER BY id")?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let meta_str: Option<String> = row.get(2)?;
            let metadata = meta_str
                .and_then(|s| serde_json::from_str(&s).ok());
            Ok(LocalCollection { id, name, metadata })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("{e}"))
    }

    pub fn get_collection(&self, name: &str) -> Result<LocalCollection> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, name, metadata FROM collections WHERE name = ?1",
            params![name],
            |row| {
                let id: String = row.get(0)?;
                let name: String = row.get(1)?;
                let meta_str: Option<String> = row.get(2)?;
                let metadata = meta_str.and_then(|s| serde_json::from_str(&s).ok());
        Ok(LocalCollection { id, name: name.to_string(), metadata })
            },
        ).map_err(|e| anyhow!("Collection '{name}' not found: {e}"))
    }

    pub fn create_collection(&self, name: &str, metadata: Option<Value>) -> Result<LocalCollection> {
        let conn = self.conn.lock().unwrap();
        let id = format!("col_{}", Uuid::new_v4().simple());
        let meta_str = metadata.as_ref().map(|m| m.to_string());
        conn.execute(
            "INSERT INTO collections (id, name, metadata) VALUES (?1, ?2, ?3)",
            params![id, name, meta_str],
        )?;
        Ok(LocalCollection { id, name: name.to_string(), metadata })
    }

    pub fn delete_collection(&self, name: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let col = conn.query_row(
            "SELECT id FROM collections WHERE name = ?1",
            params![name],
            |row| row.get::<_, String>(0),
        ).map_err(|e| anyhow!("Collection '{name}' not found: {e}"))?;
        conn.execute("DELETE FROM entries WHERE collection_id = ?1", params![col])?;
        conn.execute("DELETE FROM collections WHERE id = ?1", params![col])?;
        Ok(())
    }

    // -- add ----------------------------------------------------------------
    pub fn add(
        &self,
        collection_id: &str,
        ids: &[&str],
        embeddings: Option<&[Vec<f32>]>,
        metadatas: Option<&[Value]>,
        documents: Option<&[&str]>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        for (i, id) in ids.iter().enumerate() {
            let emb_blob = embeddings
                .and_then(|e| e.get(i))
                .map(|v| Self::blob_from_vec(v));
            let meta_str = metadatas
                .and_then(|m| m.get(i))
                .map(|v| v.to_string());
            let doc = documents.and_then(|d| d.get(i)).map(|s| s.to_string());

            conn.execute(
                "INSERT INTO entries (id, collection_id, embedding, metadata, document)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![id, collection_id, emb_blob, meta_str, doc],
            )?;
        }
        Ok(())
    }

    // -- query --------------------------------------------------------------
    pub fn query(
        &self,
        collection_id: &str,
        query_embeddings: &[Vec<f32>],
        n_results: usize,
        r#where: Option<Value>,
    ) -> Result<LocalQueryResult> {
        let conn = self.conn.lock().unwrap();
        let q_emb = query_embeddings.first()
            .ok_or_else(|| anyhow!("No query embeddings provided"))?;

        let mut sql = String::from(
            "SELECT id, embedding, metadata, document FROM entries WHERE collection_id = ?1"
        );
        let mut where_val: Option<String> = None;
        if let Some(ref w) = r#where {
            if let Some(obj) = w.as_object() {
                for (field, val) in obj.iter().take(1) {
                    if let Some(s) = val.as_str() {
                        sql.push_str(&format!(" AND json_extract(metadata, '$.{field}') = ?2"));
                        where_val = Some(s.to_string());
                    }
                }
            }
        }

        let mut stmt = conn.prepare(&sql)?;
        let rows: Vec<(String, Option<Vec<u8>>, Option<String>, Option<String>)> = {
            let iter: Box<dyn Iterator<Item = _>> = if let Some(ref wv) = where_val {
                Box::new(
                    stmt.query_map(params![collection_id, wv], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, Option<Vec<u8>>>(1)?,
                            row.get::<_, Option<String>>(2)?,
                            row.get::<_, Option<String>>(3)?,
                        ))
                    })?.filter_map(|r| r.ok())
                )
            } else {
                Box::new(
                    stmt.query_map(params![collection_id], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, Option<Vec<u8>>>(1)?,
                            row.get::<_, Option<String>>(2)?,
                            row.get::<_, Option<String>>(3)?,
                        ))
                    })?.filter_map(|r| r.ok())
                )
            };
            iter.collect()
        };

        let mut scored: Vec<(f64, String, Option<String>, Option<Value>)> = rows
            .into_iter()
            .filter_map(|(id, blob, meta_str, doc)| {
                let emb = blob.as_deref().map(Self::vec_from_blob);
                let dist = emb.as_ref()
                    .map(|e| 1.0 - cosine_similarity(q_emb, e) as f64)
                    .unwrap_or(f64::MAX);
                let meta = meta_str.and_then(|s| serde_json::from_str(&s).ok());
                if dist.is_finite() {
                    Some((dist, id, doc, meta))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n_results);

        let ids: Vec<String> = scored.iter().map(|(_, id, _, _)| id.clone()).collect();
        let distances: Vec<f64> = scored.iter().map(|(d, _, _, _)| *d).collect();
        let documents: Vec<Option<String>> = scored.iter().map(|(_, _, doc, _)| doc.clone()).collect();
        let metadatas: Vec<Option<Value>> = scored.into_iter().map(|(_, _, _, meta)| meta).collect();

        Ok(LocalQueryResult { ids, distances, documents, metadatas })
    }

    pub fn count(&self, collection_id: &str) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: usize = conn.query_row(
            "SELECT COUNT(*) FROM entries WHERE collection_id = ?1",
            params![collection_id],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    pub fn delete_where(&self, collection_id: &str, r#where: Value) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        if let Some(field) = r#where.as_object().and_then(|o| o.keys().next()) {
            if let Some(val) = r#where.get(field).and_then(|v| v.as_str()) {
                conn.execute(
                    &format!(
                        "DELETE FROM entries WHERE collection_id = ?1 AND json_extract(metadata, '$.{field}') = ?2"
                    ),
                    params![collection_id, val],
                )?;
            }
        }
        Ok(())
    }

    pub fn get_entries(&self, collection_id: &str) -> Result<Vec<LocalEntry>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, embedding, metadata, document FROM entries WHERE collection_id = ?1"
        )?;
        let rows = stmt.query_map(params![collection_id], |row| {
            let id: String = row.get(0)?;
            let blob: Option<Vec<u8>> = row.get(1)?;
            let meta_str: Option<String> = row.get(2)?;
            let doc: Option<String> = row.get(3)?;
            let embedding = blob.as_deref().map(Self::vec_from_blob);
            let metadata = meta_str.and_then(|s| serde_json::from_str(&s).ok());
            Ok(LocalEntry { id, embedding, metadata, document: doc })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| anyhow!("{e}"))
    }
}
