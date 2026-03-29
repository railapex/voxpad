// SQLite history with FTS5 for transcription search.
// Adapted from LinguaLens D:/dev/lingualens/src-tauri/src/history.rs

use rusqlite::Connection;
use serde::Serialize;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

static DB: OnceLock<Mutex<Connection>> = OnceLock::new();

pub fn init(data_dir: &Path) -> Result<(), String> {
    let _ = std::fs::create_dir_all(data_dir);
    let db_path = data_dir.join("history.db");

    let conn = Connection::open(&db_path)
        .map_err(|e| format!("open db: {e}"))?;

    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            streaming_text TEXT,
            refined_text TEXT,
            final_text TEXT,
            duration_ms INTEGER,
            word_count INTEGER,
            mode TEXT,
            inserted INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_history_ts ON history(timestamp DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(
            streaming_text, refined_text, final_text,
            content='history', content_rowid='id'
        );

        CREATE TRIGGER IF NOT EXISTS history_ai AFTER INSERT ON history BEGIN
            INSERT INTO history_fts(rowid, streaming_text, refined_text, final_text)
            VALUES (new.id, new.streaming_text, new.refined_text, new.final_text);
        END;
        ",
    )
    .map_err(|e| format!("create tables: {e}"))?;

    log::info!("[history] initialized at {}", db_path.display());
    DB.set(Mutex::new(conn)).map_err(|_| "DB already initialized".to_string())
}

/// Insert a transcription record.
pub fn insert(
    streaming_text: Option<&str>,
    refined_text: Option<&str>,
    final_text: Option<&str>,
    duration_ms: u64,
    mode: &str,
    inserted: bool,
) -> Result<i64, String> {
    let db = DB.get().ok_or("DB not initialized")?;
    let conn = db.lock().map_err(|e| format!("lock: {e}"))?;

    let word_count = final_text
        .or(refined_text)
        .or(streaming_text)
        .map(|t| t.split_whitespace().count() as i64)
        .unwrap_or(0);

    conn.execute(
        "INSERT INTO history (streaming_text, refined_text, final_text, duration_ms, word_count, mode, inserted)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params![
            streaming_text,
            refined_text,
            final_text,
            duration_ms as i64,
            word_count,
            mode,
            inserted as i32,
        ],
    )
    .map_err(|e| format!("insert: {e}"))?;

    Ok(conn.last_insert_rowid())
}

/// Query recent history entries.
pub fn query_recent(limit: u32, offset: u32) -> Result<Vec<HistoryEntry>, String> {
    let db = DB.get().ok_or("DB not initialized")?;
    let conn = db.lock().map_err(|e| format!("lock: {e}"))?;

    let mut stmt = conn
        .prepare(
            "SELECT id, timestamp, streaming_text, refined_text, final_text, duration_ms, word_count, mode, inserted
             FROM history ORDER BY timestamp DESC LIMIT ?1 OFFSET ?2",
        )
        .map_err(|e| format!("prepare: {e}"))?;

    let rows = stmt
        .query_map(rusqlite::params![limit, offset], |row| {
            Ok(HistoryEntry {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                streaming_text: row.get(2)?,
                refined_text: row.get(3)?,
                final_text: row.get(4)?,
                duration_ms: row.get(5)?,
                word_count: row.get(6)?,
                mode: row.get(7)?,
                inserted: row.get::<_, i32>(8)? != 0,
            })
        })
        .map_err(|e| format!("query: {e}"))?;

    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("collect: {e}"))
}

/// Search history via FTS5.
pub fn search(query: &str, limit: u32) -> Result<Vec<HistoryEntry>, String> {
    let db = DB.get().ok_or("DB not initialized")?;
    let conn = db.lock().map_err(|e| format!("lock: {e}"))?;

    let mut stmt = conn
        .prepare(
            "SELECT h.id, h.timestamp, h.streaming_text, h.refined_text, h.final_text,
                    h.duration_ms, h.word_count, h.mode, h.inserted
             FROM history h
             WHERE h.id IN (SELECT rowid FROM history_fts WHERE history_fts MATCH ?1)
             ORDER BY h.timestamp DESC
             LIMIT ?2",
        )
        .map_err(|e| format!("prepare search: {e}"))?;

    let rows = stmt
        .query_map(rusqlite::params![query, limit], |row| {
            Ok(HistoryEntry {
                id: row.get(0)?,
                timestamp: row.get(1)?,
                streaming_text: row.get(2)?,
                refined_text: row.get(3)?,
                final_text: row.get(4)?,
                duration_ms: row.get(5)?,
                word_count: row.get(6)?,
                mode: row.get(7)?,
                inserted: row.get::<_, i32>(8)? != 0,
            })
        })
        .map_err(|e| format!("search: {e}"))?;

    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("collect: {e}"))
}

/// Count total history entries.
pub fn count() -> Result<u64, String> {
    let db = DB.get().ok_or("DB not initialized")?;
    let conn = db.lock().map_err(|e| format!("lock: {e}"))?;
    conn.query_row("SELECT COUNT(*) FROM history", [], |row| row.get(0))
        .map_err(|e| format!("count: {e}"))
}

#[derive(Debug, Serialize, Clone)]
pub struct HistoryEntry {
    pub id: i64,
    pub timestamp: String,
    pub streaming_text: Option<String>,
    pub refined_text: Option<String>,
    pub final_text: Option<String>,
    pub duration_ms: Option<i64>,
    pub word_count: Option<i64>,
    pub mode: Option<String>,
    pub inserted: bool,
}
