//! DuckDB writer for benchmark results.
//!
//! This module provides a transactional writer for storing benchmark results
//! in DuckDB, following the schema defined in `docs/DASHBOARD_OVERHAUL_PLAN.md`.
//!
//! # Schema
//!
//! Five tables: `runs`, `models`, `decode`, `prefill_uniform`, `prefill_mixed`.
//! Separate tables per scenario -- no sparse columns.
//!
//! # Usage
//!
//! ```no_run
//! use rwkv_bench::db::{DbWriter, RunRow, DecodeRow, ModelRow};
//!
//! let mut db = DbWriter::open("benchmarks/results.db").unwrap();
//! db.insert_run(&RunRow {
//!     run_id: "20260128T083000Z_a1b2c3".into(),
//!     human_name: "keen-cedar".into(),
//!     profile: "decode-batch-sweep".into(),
//!     gpu_short: "Radeon 8060S".into(),
//!     started_at_utc: "2026-01-28T08:30:00Z".into(),
//!     git_sha: "d7327a6".into(),
//!     git_dirty: false,
//!     crate_version: "0.10.0".into(),
//!     rustc_version: "1.82.0".into(),
//!     host_os: Some("linux".into()),
//!     host_cpu: Some("AMD Ryzen 9".into()),
//!     host_ram_gb: Some(64.0),
//!     gpu_adapter: Some("AMD Radeon".into()),
//!     gpu_backend_api: Some("Vulkan".into()),
//!     gpu_driver_ver: None,
//!     uname: None,
//! }).unwrap();
//! db.upsert_model(&ModelRow {
//!     model_sha: "abc123".into(),
//!     model_name: "rwkv7_g1a_0.1b".into(),
//!     model_size: "0.1b".into(),
//! }).unwrap();
//! // ... insert decode/prefill rows ...
//! db.commit().unwrap();
//! ```

use std::path::Path;

use duckdb::{params, Connection};

/// Storage version string pinned for DuckDB-WASM v1.1.1 compatibility.
pub const STORAGE_VERSION: &str = "v1.0.0";

/// Error type for DuckDB operations.
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    /// DuckDB error
    #[error("duckdb: {0}")]
    DuckDb(#[from] duckdb::Error),
    /// Writer is in an invalid state (e.g., no active transaction)
    #[error("invalid state: {0}")]
    InvalidState(String),
}

/// Result type for DuckDB operations.
pub type DbResult<T> = Result<T, DbError>;

// =============================================================================
// ROW TYPES
// =============================================================================

/// Row data for the `runs` table.
#[derive(Debug, Clone)]
pub struct RunRow {
    pub run_id: String,
    pub human_name: String,
    pub profile: String,
    pub gpu_short: String,
    pub started_at_utc: String,
    pub git_sha: String,
    pub git_dirty: bool,
    pub crate_version: String,
    pub rustc_version: String,
    pub host_os: Option<String>,
    pub host_cpu: Option<String>,
    pub host_ram_gb: Option<f64>,
    pub gpu_adapter: Option<String>,
    pub gpu_backend_api: Option<String>,
    pub gpu_driver_ver: Option<String>,
    pub uname: Option<String>,
}

/// Row data for the `models` table.
#[derive(Debug, Clone)]
pub struct ModelRow {
    pub model_sha: String,
    pub model_name: String,
    pub model_size: String,
}

/// Row data for the `decode` table.
#[derive(Debug, Clone)]
pub struct DecodeRow {
    pub run_id: String,
    pub case_id: String,
    pub repeat_index: i32,
    pub status: String,
    pub model_sha: String,
    pub backend: String,
    pub batch_size: i32,
    pub token_chunk_size: i32,
    pub decode_steps: i32,
    pub decode_total_ms: Option<f64>,
    pub decode_tokens: Option<i32>,
    pub decode_tok_per_s: Option<f64>,
    pub step_ms_p50: Option<f64>,
    pub step_ms_p95: Option<f64>,
    pub error_kind: Option<String>,
    pub error_message: Option<String>,
}

/// Row data for the `prefill_uniform` table.
#[derive(Debug, Clone)]
pub struct PrefillUniformRow {
    pub run_id: String,
    pub case_id: String,
    pub repeat_index: i32,
    pub status: String,
    pub model_sha: String,
    pub backend: String,
    pub batch_size: i32,
    pub token_chunk_size: i32,
    pub seq_len: i32,
    pub prefill_total_ms: Option<f64>,
    pub total_prompt_tokens: Option<i32>,
    pub prefill_tok_per_s: Option<f64>,
    pub num_infer_calls: Option<i32>,
    pub ttft_ms_local: Option<Vec<f64>>,
    pub ttft_min_ms: Option<f64>,
    pub ttft_p50_ms: Option<f64>,
    pub ttft_max_ms: Option<f64>,
    pub error_kind: Option<String>,
    pub error_message: Option<String>,
}

/// Row data for the `prefill_mixed` table.
#[derive(Debug, Clone)]
pub struct PrefillMixedRow {
    pub run_id: String,
    pub case_id: String,
    pub repeat_index: i32,
    pub status: String,
    pub model_sha: String,
    pub backend: String,
    pub batch_size: i32,
    pub token_chunk_size: i32,
    pub mixed_case_id: String,
    pub seq_lens: Vec<i32>,
    pub prefill_total_ms: Option<f64>,
    pub total_prompt_tokens: Option<i32>,
    pub prefill_tok_per_s: Option<f64>,
    pub num_infer_calls: Option<i32>,
    pub ttft_ms_local: Option<Vec<f64>>,
    pub ttft_min_ms: Option<f64>,
    pub ttft_p50_ms: Option<f64>,
    pub ttft_max_ms: Option<f64>,
    pub error_kind: Option<String>,
    pub error_message: Option<String>,
}

/// A case row returned from `query_cases`.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseRow {
    pub scenario: String,
    pub model_name: String,
    pub model_sha: String,
    pub backend: String,
    pub batch_size: i32,
    pub token_chunk_size: i32,
    pub seq_len: Option<i32>,
    pub decode_steps: Option<i32>,
    pub max_batch_size: Option<i32>,
    pub max_token_chunk_size: Option<i32>,
    pub mixed_case_id: Option<String>,
}

/// Expanded case input for `query_cases`.
#[derive(Debug, Clone)]
pub struct ExpandedCaseInput {
    pub scenario: String,
    pub model_name: String,
    pub model_sha: String,
    pub backend: String,
    pub batch_size: i32,
    pub token_chunk_size: i32,
    pub seq_len: Option<i32>,
    pub decode_steps: Option<i32>,
    pub max_batch_size: Option<i32>,
    pub max_token_chunk_size: Option<i32>,
    pub mixed_case_id: Option<String>,
}

/// Skip conditions for SQL-based case filtering.
#[derive(Debug, Clone, Default)]
pub struct SqlSkipConditions {
    /// Apply built-in filter: batch_size <= max_batch_size
    pub skip_batch_exceeds_model_max: bool,
    /// Apply built-in filter: token_chunk_size <= max_token_chunk_size
    pub skip_chunk_exceeds_model_max: bool,
    /// Custom rule SQL fragments (each is a condition to skip, e.g. "batch_size > 16 AND seq_len > 1024")
    pub custom_rules: Vec<String>,
}

// =============================================================================
// HELPERS
// =============================================================================

/// Format an `Option<Vec<f64>>` as a DuckDB list literal string (e.g., `[1.0, 2.0]` or `NULL`).
///
/// The duckdb Rust crate v1.4.4 does not support binding `Value::List` via params,
/// so we embed numeric arrays as SQL literals. Since these are only numbers,
/// there is no SQL injection risk.
fn fmt_f64_list(v: &Option<Vec<f64>>) -> String {
    match v {
        Some(vec) => {
            let inner: Vec<String> = vec.iter().map(|x| format!("{x}")).collect();
            format!("[{}]", inner.join(", "))
        }
        None => "NULL".to_string(),
    }
}

/// Format a `Vec<i32>` as a DuckDB list literal string (e.g., `[1, 2, 3]`).
fn fmt_i32_list(v: &[i32]) -> String {
    let inner: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
    format!("[{}]", inner.join(", "))
}

// =============================================================================
// SQL CONSTANTS
// =============================================================================

const CREATE_RUNS: &str = "
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    human_name      TEXT NOT NULL,
    profile         TEXT NOT NULL,
    gpu_short       TEXT NOT NULL,
    started_at_utc  TIMESTAMP NOT NULL,
    git_sha         TEXT NOT NULL,
    git_dirty       BOOLEAN NOT NULL,
    crate_version   TEXT NOT NULL,
    rustc_version   TEXT NOT NULL,
    host_os         TEXT,
    host_cpu        TEXT,
    host_ram_gb     DOUBLE,
    gpu_adapter     TEXT,
    gpu_backend_api TEXT,
    gpu_driver_ver  TEXT,
    uname           TEXT
)";

const CREATE_MODELS: &str = "
CREATE TABLE IF NOT EXISTS models (
    model_sha       TEXT PRIMARY KEY,
    model_name      TEXT NOT NULL,
    model_size      TEXT NOT NULL
)";

const CREATE_DECODE: &str = "
CREATE TABLE IF NOT EXISTS decode (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    case_id             TEXT NOT NULL,
    repeat_index        INTEGER NOT NULL,
    status              TEXT NOT NULL,
    model_sha           TEXT NOT NULL REFERENCES models(model_sha),
    backend             TEXT NOT NULL,
    batch_size          INTEGER NOT NULL,
    token_chunk_size    INTEGER NOT NULL,
    decode_steps        INTEGER NOT NULL,
    decode_total_ms     DOUBLE,
    decode_tokens       INTEGER,
    decode_tok_per_s    DOUBLE,
    step_ms_p50         DOUBLE,
    step_ms_p95         DOUBLE,
    error_kind          TEXT,
    error_message       TEXT
)";

const CREATE_PREFILL_UNIFORM: &str = "
CREATE TABLE IF NOT EXISTS prefill_uniform (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    case_id             TEXT NOT NULL,
    repeat_index        INTEGER NOT NULL,
    status              TEXT NOT NULL,
    model_sha           TEXT NOT NULL REFERENCES models(model_sha),
    backend             TEXT NOT NULL,
    batch_size          INTEGER NOT NULL,
    token_chunk_size    INTEGER NOT NULL,
    seq_len             INTEGER NOT NULL,
    prefill_total_ms    DOUBLE,
    total_prompt_tokens INTEGER,
    prefill_tok_per_s   DOUBLE,
    num_infer_calls     INTEGER,
    ttft_ms_local       DOUBLE[],
    ttft_min_ms         DOUBLE,
    ttft_p50_ms         DOUBLE,
    ttft_max_ms         DOUBLE,
    error_kind          TEXT,
    error_message       TEXT
)";

const CREATE_PREFILL_MIXED: &str = "
CREATE TABLE IF NOT EXISTS prefill_mixed (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    case_id             TEXT NOT NULL,
    repeat_index        INTEGER NOT NULL,
    status              TEXT NOT NULL,
    model_sha           TEXT NOT NULL REFERENCES models(model_sha),
    backend             TEXT NOT NULL,
    batch_size          INTEGER NOT NULL,
    token_chunk_size    INTEGER NOT NULL,
    mixed_case_id       TEXT NOT NULL,
    seq_lens            INTEGER[] NOT NULL,
    prefill_total_ms    DOUBLE,
    total_prompt_tokens INTEGER,
    prefill_tok_per_s   DOUBLE,
    num_infer_calls     INTEGER,
    ttft_ms_local       DOUBLE[],
    ttft_min_ms         DOUBLE,
    ttft_p50_ms         DOUBLE,
    ttft_max_ms         DOUBLE,
    error_kind          TEXT,
    error_message       TEXT
)";

// =============================================================================
// DB WRITER
// =============================================================================

/// DuckDB writer for benchmark results.
///
/// The entire run is wrapped in a single transaction. Call [`DbWriter::open`]
/// to create/open the database, then insert rows, and finally [`DbWriter::commit`]
/// to commit the transaction atomically.
pub struct DbWriter {
    conn: Connection,
    in_transaction: bool,
}

impl DbWriter {
    /// Open (or create) a DuckDB database at the given path.
    ///
    /// Sets `STORAGE_VERSION` to `v1.0.0` for DuckDB-WASM compatibility,
    /// creates all tables if they do not exist.
    pub fn open<P: AsRef<Path>>(path: P) -> DbResult<Self> {
        let conn = Connection::open(path)?;

        // Pin storage compatibility
        conn.execute_batch(&format!(
            "PRAGMA storage_compatibility_version = '{STORAGE_VERSION}';"
        ))?;

        // Create schema
        conn.execute_batch(CREATE_RUNS)?;
        conn.execute_batch(CREATE_MODELS)?;
        conn.execute_batch(CREATE_DECODE)?;
        conn.execute_batch(CREATE_PREFILL_UNIFORM)?;
        conn.execute_batch(CREATE_PREFILL_MIXED)?;

        Ok(Self {
            conn,
            in_transaction: false,
        })
    }

    /// Begin a transaction (called implicitly by `insert_run`).
    fn ensure_transaction(&mut self) -> DbResult<()> {
        if !self.in_transaction {
            self.conn.execute_batch("BEGIN TRANSACTION")?;
            self.in_transaction = true;
        }
        Ok(())
    }

    /// Insert a run header row. Begins the transaction if not already active.
    pub fn insert_run(&mut self, run: &RunRow) -> DbResult<()> {
        self.ensure_transaction()?;
        self.conn.execute(
            "INSERT INTO runs (
                run_id, human_name, profile, gpu_short, started_at_utc,
                git_sha, git_dirty, crate_version, rustc_version,
                host_os, host_cpu, host_ram_gb, gpu_adapter, gpu_backend_api,
                gpu_driver_ver, uname
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                run.run_id,
                run.human_name,
                run.profile,
                run.gpu_short,
                run.started_at_utc,
                run.git_sha,
                run.git_dirty,
                run.crate_version,
                run.rustc_version,
                run.host_os,
                run.host_cpu,
                run.host_ram_gb,
                run.gpu_adapter,
                run.gpu_backend_api,
                run.gpu_driver_ver,
                run.uname,
            ],
        )?;
        Ok(())
    }

    /// Upsert a model row (INSERT OR IGNORE).
    pub fn upsert_model(&mut self, model: &ModelRow) -> DbResult<()> {
        self.ensure_transaction()?;
        self.conn.execute(
            "INSERT OR IGNORE INTO models (model_sha, model_name, model_size)
             VALUES (?, ?, ?)",
            params![model.model_sha, model.model_name, model.model_size],
        )?;
        Ok(())
    }

    /// Insert a row into the `decode` table.
    pub fn insert_decode(&mut self, row: &DecodeRow) -> DbResult<()> {
        self.ensure_transaction()?;
        self.conn.execute(
            "INSERT INTO decode (
                run_id, case_id, repeat_index, status,
                model_sha, backend, batch_size, token_chunk_size,
                decode_steps, decode_total_ms, decode_tokens, decode_tok_per_s,
                step_ms_p50, step_ms_p95, error_kind, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                row.run_id,
                row.case_id,
                row.repeat_index,
                row.status,
                row.model_sha,
                row.backend,
                row.batch_size,
                row.token_chunk_size,
                row.decode_steps,
                row.decode_total_ms,
                row.decode_tokens,
                row.decode_tok_per_s,
                row.step_ms_p50,
                row.step_ms_p95,
                row.error_kind,
                row.error_message,
            ],
        )?;
        Ok(())
    }

    /// Insert a row into the `prefill_uniform` table.
    pub fn insert_prefill_uniform(&mut self, row: &PrefillUniformRow) -> DbResult<()> {
        self.ensure_transaction()?;
        let ttft_literal = fmt_f64_list(&row.ttft_ms_local);
        let sql = format!(
            "INSERT INTO prefill_uniform (
                run_id, case_id, repeat_index, status,
                model_sha, backend, batch_size, token_chunk_size,
                seq_len, prefill_total_ms, total_prompt_tokens, prefill_tok_per_s,
                num_infer_calls, ttft_ms_local, ttft_min_ms, ttft_p50_ms, ttft_max_ms,
                error_kind, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {ttft_literal}, ?, ?, ?, ?, ?)"
        );
        self.conn.execute(
            &sql,
            params![
                row.run_id,
                row.case_id,
                row.repeat_index,
                row.status,
                row.model_sha,
                row.backend,
                row.batch_size,
                row.token_chunk_size,
                row.seq_len,
                row.prefill_total_ms,
                row.total_prompt_tokens,
                row.prefill_tok_per_s,
                row.num_infer_calls,
                // ttft_ms_local is inlined in the SQL
                row.ttft_min_ms,
                row.ttft_p50_ms,
                row.ttft_max_ms,
                row.error_kind,
                row.error_message,
            ],
        )?;
        Ok(())
    }

    /// Insert a row into the `prefill_mixed` table.
    pub fn insert_prefill_mixed(&mut self, row: &PrefillMixedRow) -> DbResult<()> {
        self.ensure_transaction()?;
        let seq_lens_literal = fmt_i32_list(&row.seq_lens);
        let ttft_literal = fmt_f64_list(&row.ttft_ms_local);
        let sql = format!(
            "INSERT INTO prefill_mixed (
                run_id, case_id, repeat_index, status,
                model_sha, backend, batch_size, token_chunk_size,
                mixed_case_id, seq_lens,
                prefill_total_ms, total_prompt_tokens, prefill_tok_per_s,
                num_infer_calls, ttft_ms_local, ttft_min_ms, ttft_p50_ms, ttft_max_ms,
                error_kind, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {seq_lens_literal}, ?, ?, ?, ?, {ttft_literal}, ?, ?, ?, ?, ?)"
        );
        self.conn.execute(
            &sql,
            params![
                row.run_id,
                row.case_id,
                row.repeat_index,
                row.status,
                row.model_sha,
                row.backend,
                row.batch_size,
                row.token_chunk_size,
                row.mixed_case_id,
                // seq_lens is inlined in the SQL
                row.prefill_total_ms,
                row.total_prompt_tokens,
                row.prefill_tok_per_s,
                row.num_infer_calls,
                // ttft_ms_local is inlined in the SQL
                row.ttft_min_ms,
                row.ttft_p50_ms,
                row.ttft_max_ms,
                row.error_kind,
                row.error_message,
            ],
        )?;
        Ok(())
    }

    /// Insert expanded cases into a temp table, apply built-in and custom skip
    /// conditions via SQL WHERE clauses, and return the surviving cases.
    ///
    /// This replaces the Rust-side expression parser in `skip.rs` with DuckDB SQL.
    pub fn query_cases(
        &self,
        cases: &[ExpandedCaseInput],
        skip: &SqlSkipConditions,
    ) -> DbResult<Vec<CaseRow>> {
        // Create temp table
        self.conn.execute_batch(
            "CREATE TEMPORARY TABLE IF NOT EXISTS _cases (
                scenario            TEXT,
                model_name          TEXT,
                model_sha           TEXT,
                backend             TEXT,
                batch_size          INTEGER,
                token_chunk_size    INTEGER,
                seq_len             INTEGER,
                decode_steps        INTEGER,
                max_batch_size      INTEGER,
                max_token_chunk_size INTEGER,
                mixed_case_id       TEXT
            )",
        )?;
        self.conn.execute_batch("DELETE FROM _cases")?;

        // Insert all expanded cases
        let mut stmt = self.conn.prepare(
            "INSERT INTO _cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )?;
        for c in cases {
            stmt.execute(params![
                c.scenario,
                c.model_name,
                c.model_sha,
                c.backend,
                c.batch_size,
                c.token_chunk_size,
                c.seq_len,
                c.decode_steps,
                c.max_batch_size,
                c.max_token_chunk_size,
                c.mixed_case_id,
            ])?;
        }

        // Build WHERE clause
        let mut conditions: Vec<String> = Vec::new();

        // Built-in filters
        if skip.skip_batch_exceeds_model_max {
            conditions.push(
                "(max_batch_size IS NULL OR batch_size <= max_batch_size)".to_string(),
            );
        }
        if skip.skip_chunk_exceeds_model_max {
            conditions.push(
                "(max_token_chunk_size IS NULL OR token_chunk_size <= max_token_chunk_size)"
                    .to_string(),
            );
        }

        // Custom rules: each is a condition that should cause skipping,
        // so we negate them: NOT(condition)
        for rule in &skip.custom_rules {
            conditions.push(format!("NOT ({})", rule));
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join("\n  AND "))
        };

        let query = format!(
            "SELECT scenario, model_name, model_sha, backend, batch_size,
                    token_chunk_size, seq_len, decode_steps,
                    max_batch_size, max_token_chunk_size, mixed_case_id
             FROM _cases
             {where_clause}
             ORDER BY scenario, model_name, backend, batch_size"
        );

        let mut stmt = self.conn.prepare(&query)?;
        let rows = stmt.query_map([], |row| {
            Ok(CaseRow {
                scenario: row.get(0)?,
                model_name: row.get(1)?,
                model_sha: row.get(2)?,
                backend: row.get(3)?,
                batch_size: row.get(4)?,
                token_chunk_size: row.get(5)?,
                seq_len: row.get(6)?,
                decode_steps: row.get(7)?,
                max_batch_size: row.get(8)?,
                max_token_chunk_size: row.get(9)?,
                mixed_case_id: row.get(10)?,
            })
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }

        // Clean up temp table
        self.conn.execute_batch("DROP TABLE IF EXISTS _cases")?;

        Ok(result)
    }

    /// Commit the whole-run transaction.
    pub fn commit(&mut self) -> DbResult<()> {
        if self.in_transaction {
            self.conn.execute_batch("COMMIT")?;
            self.in_transaction = false;
            Ok(())
        } else {
            Err(DbError::InvalidState(
                "no active transaction to commit".to_string(),
            ))
        }
    }

    /// Get a reference to the underlying connection (for testing/queries).
    pub fn connection(&self) -> &Connection {
        &self.conn
    }
}

impl Drop for DbWriter {
    fn drop(&mut self) {
        // Rollback any uncommitted transaction
        if self.in_transaction {
            let _ = self.conn.execute_batch("ROLLBACK");
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_db_path() -> String {
        let counter = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "test_db_{}_{}.db",
            std::process::id(),
            counter
        ));
        // Clean up any existing file
        let _ = std::fs::remove_file(&path);
        path.to_string_lossy().to_string()
    }

    fn make_run(run_id: &str) -> RunRow {
        RunRow {
            run_id: run_id.to_string(),
            human_name: "keen-cedar".to_string(),
            profile: "decode-batch-sweep".to_string(),
            gpu_short: "Radeon 8060S".to_string(),
            started_at_utc: "2026-01-28 08:30:00".to_string(),
            git_sha: "d7327a6".to_string(),
            git_dirty: false,
            crate_version: "0.10.0".to_string(),
            rustc_version: "1.82.0".to_string(),
            host_os: Some("linux".to_string()),
            host_cpu: Some("AMD Ryzen 9".to_string()),
            host_ram_gb: Some(64.0),
            gpu_adapter: Some("AMD Radeon".to_string()),
            gpu_backend_api: Some("Vulkan".to_string()),
            gpu_driver_ver: None,
            uname: None,
        }
    }

    fn make_model() -> ModelRow {
        ModelRow {
            model_sha: "sha256_abc123".to_string(),
            model_name: "rwkv7_g1a_0.1b".to_string(),
            model_size: "0.1b".to_string(),
        }
    }

    // ---- Schema creation ----

    #[test]
    fn test_open_creates_tables() {
        let path = temp_db_path();
        {
            let _db = DbWriter::open(&path).unwrap();
        }
        // Re-open and verify tables exist by querying them
        let conn = Connection::open(&path).unwrap();
        let count: i64 = conn
            .query_row("SELECT count(*) FROM runs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
        let count: i64 = conn
            .query_row("SELECT count(*) FROM models", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
        let count: i64 = conn
            .query_row("SELECT count(*) FROM decode", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
        let count: i64 = conn
            .query_row("SELECT count(*) FROM prefill_uniform", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
        let count: i64 = conn
            .query_row("SELECT count(*) FROM prefill_mixed", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
        let _ = std::fs::remove_file(&path);
    }

    // ---- Insert + commit ----

    #[test]
    fn test_insert_run_and_commit() {
        let path = temp_db_path();
        {
            let mut db = DbWriter::open(&path).unwrap();
            db.insert_run(&make_run("run_1")).unwrap();
            db.commit().unwrap();
        }
        // Re-open and verify
        let conn = Connection::open(&path).unwrap();
        let name: String = conn
            .query_row(
                "SELECT human_name FROM runs WHERE run_id = 'run_1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(name, "keen-cedar");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_rollback_on_drop() {
        let path = temp_db_path();
        {
            let mut db = DbWriter::open(&path).unwrap();
            db.insert_run(&make_run("run_rollback")).unwrap();
            // Drop without commit
        }
        // Verify the run was NOT persisted
        let conn = Connection::open(&path).unwrap();
        let count: i64 = conn
            .query_row("SELECT count(*) FROM runs WHERE run_id = 'run_rollback'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_upsert_model_idempotent() {
        let path = temp_db_path();
        {
            let mut db = DbWriter::open(&path).unwrap();
            db.insert_run(&make_run("run_m")).unwrap();
            db.upsert_model(&make_model()).unwrap();
            // Insert again -- should not error
            db.upsert_model(&make_model()).unwrap();
            db.commit().unwrap();
        }
        let conn = Connection::open(&path).unwrap();
        let count: i64 = conn
            .query_row("SELECT count(*) FROM models", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_insert_decode() {
        let path = temp_db_path();
        {
            let mut db = DbWriter::open(&path).unwrap();
            db.insert_run(&make_run("run_d")).unwrap();
            db.upsert_model(&make_model()).unwrap();
            db.insert_decode(&DecodeRow {
                run_id: "run_d".to_string(),
                case_id: "decode_only:rwkv7:wgpu/Vulkan:bs4:c128:steps100".to_string(),
                repeat_index: 0,
                status: "ok".to_string(),
                model_sha: "sha256_abc123".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 4,
                token_chunk_size: 128,
                decode_steps: 100,
                decode_total_ms: Some(245.67),
                decode_tokens: Some(400),
                decode_tok_per_s: Some(1628.5),
                step_ms_p50: Some(2.41),
                step_ms_p95: Some(2.89),
                error_kind: None,
                error_message: None,
            })
            .unwrap();
            db.commit().unwrap();
        }
        let conn = Connection::open(&path).unwrap();
        let tok_per_s: f64 = conn
            .query_row(
                "SELECT decode_tok_per_s FROM decode WHERE run_id = 'run_d'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!((tok_per_s - 1628.5).abs() < 0.01);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_insert_prefill_uniform() {
        let path = temp_db_path();
        {
            let mut db = DbWriter::open(&path).unwrap();
            db.insert_run(&make_run("run_pu")).unwrap();
            db.upsert_model(&make_model()).unwrap();
            db.insert_prefill_uniform(&PrefillUniformRow {
                run_id: "run_pu".to_string(),
                case_id: "prefill_uniform:rwkv7:wgpu/Vulkan:bs4:c256:len512".to_string(),
                repeat_index: 0,
                status: "ok".to_string(),
                model_sha: "sha256_abc123".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 4,
                token_chunk_size: 256,
                seq_len: 512,
                prefill_total_ms: Some(89.45),
                total_prompt_tokens: Some(2048),
                prefill_tok_per_s: Some(22897.1),
                num_infer_calls: Some(2),
                ttft_ms_local: Some(vec![89.12, 89.23, 89.34, 89.45]),
                ttft_min_ms: Some(89.12),
                ttft_p50_ms: Some(89.28),
                ttft_max_ms: Some(89.45),
                error_kind: None,
                error_message: None,
            })
            .unwrap();
            db.commit().unwrap();
        }
        let conn = Connection::open(&path).unwrap();
        let seq_len: i32 = conn
            .query_row(
                "SELECT seq_len FROM prefill_uniform WHERE run_id = 'run_pu'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(seq_len, 512);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_insert_prefill_mixed() {
        let path = temp_db_path();
        {
            let mut db = DbWriter::open(&path).unwrap();
            db.insert_run(&make_run("run_pm")).unwrap();
            db.upsert_model(&make_model()).unwrap();
            db.insert_prefill_mixed(&PrefillMixedRow {
                run_id: "run_pm".to_string(),
                case_id: "prefill_mixed:rwkv7:wgpu/Vulkan:bs8:c256:staircase_8".to_string(),
                repeat_index: 0,
                status: "ok".to_string(),
                model_sha: "sha256_abc123".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 8,
                token_chunk_size: 256,
                mixed_case_id: "staircase_8".to_string(),
                seq_lens: vec![16, 32, 64, 128, 192, 256, 512, 1024],
                prefill_total_ms: Some(150.3),
                total_prompt_tokens: Some(2224),
                prefill_tok_per_s: Some(14800.0),
                num_infer_calls: Some(8),
                ttft_ms_local: Some(vec![10.0, 15.0, 25.0, 40.0, 60.0, 80.0, 120.0, 150.3]),
                ttft_min_ms: Some(10.0),
                ttft_p50_ms: Some(50.0),
                ttft_max_ms: Some(150.3),
                error_kind: None,
                error_message: None,
            })
            .unwrap();
            db.commit().unwrap();
        }
        let conn = Connection::open(&path).unwrap();
        let mixed_id: String = conn
            .query_row(
                "SELECT mixed_case_id FROM prefill_mixed WHERE run_id = 'run_pm'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(mixed_id, "staircase_8");
        let _ = std::fs::remove_file(&path);
    }

    // ---- Re-open existing db ----

    #[test]
    fn test_reopen_existing_db() {
        let path = temp_db_path();
        // First: create and insert
        {
            let mut db = DbWriter::open(&path).unwrap();
            db.insert_run(&make_run("run_first")).unwrap();
            db.upsert_model(&make_model()).unwrap();
            db.commit().unwrap();
        }
        // Second: re-open, insert more
        {
            let mut db = DbWriter::open(&path).unwrap();
            db.insert_run(&make_run("run_second")).unwrap();
            db.upsert_model(&make_model()).unwrap();
            db.commit().unwrap();
        }
        // Verify both runs exist
        let conn = Connection::open(&path).unwrap();
        let count: i64 = conn
            .query_row("SELECT count(*) FROM runs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 2);
        let _ = std::fs::remove_file(&path);
    }

    // ---- Case filtering ----

    #[test]
    fn test_query_cases_no_filters() {
        let path = temp_db_path();
        let db = DbWriter::open(&path).unwrap();

        let cases = vec![
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 1,
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: Some(32),
                max_token_chunk_size: Some(512),
                mixed_case_id: None,
            },
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 4,
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: Some(32),
                max_token_chunk_size: Some(512),
                mixed_case_id: None,
            },
        ];

        let skip = SqlSkipConditions::default();
        let result = db.query_cases(&cases, &skip).unwrap();
        assert_eq!(result.len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_query_cases_builtin_batch_filter() {
        let path = temp_db_path();
        let db = DbWriter::open(&path).unwrap();

        let cases = vec![
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 4,
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: Some(32),
                max_token_chunk_size: Some(512),
                mixed_case_id: None,
            },
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 64, // exceeds max_batch_size=32
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: Some(32),
                max_token_chunk_size: Some(512),
                mixed_case_id: None,
            },
        ];

        let skip = SqlSkipConditions {
            skip_batch_exceeds_model_max: true,
            ..Default::default()
        };
        let result = db.query_cases(&cases, &skip).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].batch_size, 4);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_query_cases_builtin_chunk_filter() {
        let path = temp_db_path();
        let db = DbWriter::open(&path).unwrap();

        let cases = vec![
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 4,
                token_chunk_size: 256,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: None,
                max_token_chunk_size: Some(128), // chunk 256 > max 128
                mixed_case_id: None,
            },
        ];

        let skip = SqlSkipConditions {
            skip_chunk_exceeds_model_max: true,
            ..Default::default()
        };
        let result = db.query_cases(&cases, &skip).unwrap();
        assert_eq!(result.len(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_query_cases_custom_rule() {
        let path = temp_db_path();
        let db = DbWriter::open(&path).unwrap();

        let cases = vec![
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 32,
                token_chunk_size: 128,
                seq_len: Some(2048),
                decode_steps: Some(100),
                max_batch_size: None,
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 4,
                token_chunk_size: 128,
                seq_len: Some(512),
                decode_steps: Some(100),
                max_batch_size: None,
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
        ];

        let skip = SqlSkipConditions {
            custom_rules: vec![
                "batch_size > 16 AND seq_len > 1024".to_string(),
            ],
            ..Default::default()
        };
        let result = db.query_cases(&cases, &skip).unwrap();
        // Only the second case should survive (batch_size=4, seq_len=512)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].batch_size, 4);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_query_cases_custom_rule_string_match() {
        let path = temp_db_path();
        let db = DbWriter::open(&path).unwrap();

        let cases = vec![
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv_puzzle15".to_string(),
                model_sha: "sha1".to_string(),
                backend: "hip".to_string(),
                batch_size: 4,
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: None,
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv_puzzle15".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 4,
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: None,
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
        ];

        let skip = SqlSkipConditions {
            custom_rules: vec![
                "model_name = 'rwkv_puzzle15' AND backend = 'hip'".to_string(),
            ],
            ..Default::default()
        };
        let result = db.query_cases(&cases, &skip).unwrap();
        // Only the wgpu/Vulkan case should survive
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].backend, "wgpu/Vulkan");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_query_cases_like_filter() {
        let path = temp_db_path();
        let db = DbWriter::open(&path).unwrap();

        let cases = vec![
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 512,
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: None,
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "hip".to_string(),
                batch_size: 512,
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: None,
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
        ];

        let skip = SqlSkipConditions {
            custom_rules: vec![
                "backend LIKE 'wgpu%' AND batch_size > 256".to_string(),
            ],
            ..Default::default()
        };
        let result = db.query_cases(&cases, &skip).unwrap();
        // Only the hip case should survive
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].backend, "hip");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_query_cases_multiple_rules() {
        let path = temp_db_path();
        let db = DbWriter::open(&path).unwrap();

        let cases = vec![
            // This one gets killed by rule 1: batch_size > 16 AND seq_len > 1024
            ExpandedCaseInput {
                scenario: "prefill_uniform".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 32,
                token_chunk_size: 128,
                seq_len: Some(2048),
                decode_steps: None,
                max_batch_size: None,
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
            // This one gets killed by rule 2: model_name = 'rwkv_puzzle15' AND backend = 'hip'
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv_puzzle15".to_string(),
                model_sha: "sha2".to_string(),
                backend: "hip".to_string(),
                batch_size: 4,
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: None,
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
            // This one survives
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 4,
                token_chunk_size: 128,
                seq_len: Some(512),
                decode_steps: Some(100),
                max_batch_size: None,
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
        ];

        let skip = SqlSkipConditions {
            custom_rules: vec![
                "batch_size > 16 AND seq_len > 1024".to_string(),
                "model_name = 'rwkv_puzzle15' AND backend = 'hip'".to_string(),
            ],
            ..Default::default()
        };
        let result = db.query_cases(&cases, &skip).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].model_name, "rwkv7");
        assert_eq!(result[0].batch_size, 4);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_query_cases_combined_builtin_and_custom() {
        let path = temp_db_path();
        let db = DbWriter::open(&path).unwrap();

        let cases = vec![
            // Killed by builtin: batch_size 64 > max 32
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 64,
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: Some(32),
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
            // Killed by custom rule
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv_puzzle15".to_string(),
                model_sha: "sha2".to_string(),
                backend: "hip".to_string(),
                batch_size: 4,
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: Some(32),
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
            // Survives
            ExpandedCaseInput {
                scenario: "decode_only".to_string(),
                model_name: "rwkv7".to_string(),
                model_sha: "sha1".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 4,
                token_chunk_size: 128,
                seq_len: None,
                decode_steps: Some(100),
                max_batch_size: Some(32),
                max_token_chunk_size: None,
                mixed_case_id: None,
            },
        ];

        let skip = SqlSkipConditions {
            skip_batch_exceeds_model_max: true,
            custom_rules: vec![
                "model_name = 'rwkv_puzzle15' AND backend = 'hip'".to_string(),
            ],
            ..Default::default()
        };
        let result = db.query_cases(&cases, &skip).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].batch_size, 4);
        assert_eq!(result[0].model_name, "rwkv7");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_commit_without_transaction_errors() {
        let path = temp_db_path();
        let mut db = DbWriter::open(&path).unwrap();
        let result = db.commit();
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_insert_error_record() {
        let path = temp_db_path();
        {
            let mut db = DbWriter::open(&path).unwrap();
            db.insert_run(&make_run("run_err")).unwrap();
            db.upsert_model(&make_model()).unwrap();
            db.insert_decode(&DecodeRow {
                run_id: "run_err".to_string(),
                case_id: "decode_only:rwkv7:wgpu/Vulkan:bs64:c128:steps100".to_string(),
                repeat_index: 0,
                status: "error".to_string(),
                model_sha: "sha256_abc123".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 64,
                token_chunk_size: 128,
                decode_steps: 100,
                decode_total_ms: None,
                decode_tokens: None,
                decode_tok_per_s: None,
                step_ms_p50: None,
                step_ms_p95: None,
                error_kind: Some("out_of_memory".to_string()),
                error_message: Some("Allocation failed".to_string()),
            })
            .unwrap();
            db.commit().unwrap();
        }
        let conn = Connection::open(&path).unwrap();
        let status: String = conn
            .query_row(
                "SELECT status FROM decode WHERE run_id = 'run_err'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(status, "error");
        let err_kind: String = conn
            .query_row(
                "SELECT error_kind FROM decode WHERE run_id = 'run_err'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(err_kind, "out_of_memory");
        let _ = std::fs::remove_file(&path);
    }

    /// Write a test DB and return its path, for use in the DuckDB-WASM compatibility check.
    #[test]
    fn test_write_wasm_compat_db() {
        let path = std::env::temp_dir()
            .join("rwkv_bench_wasm_compat_test.db")
            .to_string_lossy()
            .to_string();
        let _ = std::fs::remove_file(&path);
        {
            let mut db = DbWriter::open(&path).unwrap();
            db.insert_run(&make_run("wasm_test_run")).unwrap();
            db.upsert_model(&make_model()).unwrap();
            db.insert_decode(&DecodeRow {
                run_id: "wasm_test_run".to_string(),
                case_id: "decode_only:rwkv7:wgpu/Vulkan:bs4:c128:steps100".to_string(),
                repeat_index: 0,
                status: "ok".to_string(),
                model_sha: "sha256_abc123".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 4,
                token_chunk_size: 128,
                decode_steps: 100,
                decode_total_ms: Some(245.67),
                decode_tokens: Some(400),
                decode_tok_per_s: Some(1628.5),
                step_ms_p50: Some(2.41),
                step_ms_p95: Some(2.89),
                error_kind: None,
                error_message: None,
            })
            .unwrap();
            db.insert_prefill_uniform(&PrefillUniformRow {
                run_id: "wasm_test_run".to_string(),
                case_id: "prefill_uniform:rwkv7:wgpu/Vulkan:bs4:c256:len512".to_string(),
                repeat_index: 0,
                status: "ok".to_string(),
                model_sha: "sha256_abc123".to_string(),
                backend: "wgpu/Vulkan".to_string(),
                batch_size: 4,
                token_chunk_size: 256,
                seq_len: 512,
                prefill_total_ms: Some(89.45),
                total_prompt_tokens: Some(2048),
                prefill_tok_per_s: Some(22897.1),
                num_infer_calls: Some(2),
                ttft_ms_local: Some(vec![89.12, 89.23, 89.34, 89.45]),
                ttft_min_ms: Some(89.12),
                ttft_p50_ms: Some(89.28),
                ttft_max_ms: Some(89.45),
                error_kind: None,
                error_message: None,
            })
            .unwrap();
            db.commit().unwrap();
        }
        // Verify the file was written
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);
        eprintln!("WASM compat test DB written to: {}", path);
    }
}
