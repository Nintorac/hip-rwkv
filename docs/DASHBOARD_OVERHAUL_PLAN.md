# Dashboard Overhaul: Observable Framework + DuckDB

## Context

The current benchmark dashboard is a monolithic static HTML/JS/CSS bundle (`benches/dashboard/`) with a 4300-line `app.js` using raw D3.js. It requires a manually-generated `index.json` manifest, has 5 overly-complex visualization views, and no integration with the docs site. The benchmark runner writes JSONL files with opaque timestamp-based names.

This overhaul replaces it with:
- **DuckDB** as the benchmark results store (Rust writes, Observable reads directly)
- **Observable Framework** for the dashboard site in `docs/`
- **Human-readable run names** generated in Rust (`words · profile · gpu`)
- **3 simplified dashboard pages** (Runs, Detail, Compare)
- **GitHub Pages** deployment via the `docs/` directory

## Architecture

```
Benchmark Run (Rust)                      Dashboard (Observable Framework)
====================                      ================================
web-rwkv-bench crate                      docs/
  |                                         |
  | duckdb crate                            |
  | STORAGE_VERSION 'v1.0.0'                |
  v                                         v
benchmarks/results.db  --promote-->  docs/benchmarks/results.db (git-lfs)
  (dev, gitignored)                    |
                                       v
                                  DuckDBClient.of({base: FileAttachment(...)})
                                       |
                                       v
                                  SQL code blocks + Observable Plot
                                       |
                                       v
                                  Static site --> GitHub Pages
```

### Dev vs Prod databases

- **Dev**: `benchmarks/results.db` — gitignored, written to by default during benchmarks
- **Prod**: `docs/benchmarks/results.db` — git-lfs tracked, serves the published dashboard
- `make bench-promote` copies dev → prod, **requires clean git** (enforced by the Makefile target)
- The benchmark runner also writes JSONL files as secondary output: `{profile}_{gpu}_{name}.jsonl`

## DuckDB Schema

Separate tables per scenario — no sparse columns. `models` table normalized out for a clean model registry. Scenario tables reference `model_sha`.

```sql
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    human_name      TEXT NOT NULL,        -- "keen-cedar"
    profile         TEXT NOT NULL,        -- "decode-batch-sweep"
    gpu_short       TEXT NOT NULL,        -- "Radeon 8060S"
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
);

CREATE TABLE IF NOT EXISTS models (
    model_sha       TEXT PRIMARY KEY,     -- SHA-256 of model weights
    model_name      TEXT NOT NULL,        -- "rwkv7_g1a_0.1b"
    model_size      TEXT NOT NULL         -- "0.1b"
);

CREATE TABLE IF NOT EXISTS decode (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    case_id             TEXT NOT NULL,
    repeat_index        INTEGER NOT NULL,
    status              TEXT NOT NULL,
    -- Case identity
    model_sha           TEXT NOT NULL REFERENCES models(model_sha),
    backend             TEXT NOT NULL,      -- "wgpu/Vulkan", "hip", etc.
    batch_size          INTEGER NOT NULL,
    token_chunk_size    INTEGER NOT NULL,
    -- Decode params
    decode_steps        INTEGER NOT NULL,
    -- Metrics
    decode_total_ms     DOUBLE,
    decode_tokens       INTEGER,
    decode_tok_per_s    DOUBLE,
    step_ms_p50         DOUBLE,
    step_ms_p95         DOUBLE,
    -- Error
    error_kind          TEXT,
    error_message       TEXT
);

CREATE TABLE IF NOT EXISTS prefill_uniform (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    case_id             TEXT NOT NULL,
    repeat_index        INTEGER NOT NULL,
    status              TEXT NOT NULL,
    -- Case identity
    model_sha           TEXT NOT NULL REFERENCES models(model_sha),
    backend             TEXT NOT NULL,      -- "wgpu/Vulkan", "hip", etc.
    batch_size          INTEGER NOT NULL,
    token_chunk_size    INTEGER NOT NULL,
    -- Prefill params
    seq_len             INTEGER NOT NULL,
    -- Metrics
    prefill_total_ms    DOUBLE,
    total_prompt_tokens INTEGER,
    prefill_tok_per_s   DOUBLE,
    num_infer_calls     INTEGER,
    ttft_ms_local       DOUBLE[],
    ttft_min_ms         DOUBLE,
    ttft_p50_ms         DOUBLE,
    ttft_max_ms         DOUBLE,
    -- Error
    error_kind          TEXT,
    error_message       TEXT
);

CREATE TABLE IF NOT EXISTS prefill_mixed (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    case_id             TEXT NOT NULL,
    repeat_index        INTEGER NOT NULL,
    status              TEXT NOT NULL,
    -- Case identity
    model_sha           TEXT NOT NULL REFERENCES models(model_sha),
    backend             TEXT NOT NULL,      -- "wgpu/Vulkan", "hip", etc.
    batch_size          INTEGER NOT NULL,
    token_chunk_size    INTEGER NOT NULL,
    -- Mixed params
    mixed_case_id       TEXT NOT NULL,
    seq_lens            INTEGER[] NOT NULL,
    -- Metrics
    prefill_total_ms    DOUBLE,
    total_prompt_tokens INTEGER,
    prefill_tok_per_s   DOUBLE,
    num_infer_calls     INTEGER,
    ttft_ms_local       DOUBLE[],
    ttft_min_ms         DOUBLE,
    ttft_p50_ms         DOUBLE,
    ttft_max_ms         DOUBLE,
    -- Error
    error_kind          TEXT,
    error_message       TEXT
);
```

## Files to Modify

### Rust: Benchmark runner → DuckDB output

- **Rename crate**: `crates/web-rwkv-bench/` → `crates/rwkv-bench/`
  - Update `Cargo.toml`: `name = "rwkv-bench"`, add `duckdb` dependency
  - Update workspace `Cargo.toml` member path
  - Update `tests/benchmarks.rs` import (`use rwkv_bench::...`)
- **`crates/rwkv-bench/src/jsonl.rs`** — Add:
  - `generate_human_name(run_id: &str) -> String` — deterministic word-pair via SHA-256 (64 adjectives × 64 nouns, const arrays)
  - `shorten_gpu(cpu_str: &str) -> String` — extract GPU name from host CPU string (e.g., "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S" → "Radeon 8060S")
- **New file: `crates/rwkv-bench/src/db.rs`** — DuckDB writer:
  - `DbWriter::open(path)` — opens/creates db with `STORAGE_VERSION 'v1.0.0'`, creates tables if not exist
  - `DbWriter::insert_run(run: &RunRow)` — begins transaction, inserts run header
  - `DbWriter::upsert_model(sha, name, size)` — INSERT OR IGNORE into `models`
  - `DbWriter::insert_decode(m: &DecodeRow)` — inserts into `decode` table
  - `DbWriter::insert_prefill_uniform(m: &PrefillUniformRow)` — inserts into `prefill_uniform` table
  - `DbWriter::insert_prefill_mixed(m: &PrefillMixedRow)` — inserts into `prefill_mixed` table
  - `DbWriter::query_cases(cases, skip_conditions)` — insert expanded cases into temp table, return iterator via SELECT with combined WHERE NOT clauses (see "SQL-Based Case Filtering" section)
  - `DbWriter::commit()` — commits the whole-run transaction
- **`crates/rwkv-bench/src/lib.rs`** — Export `db` module
- **Backend field change**: merge `backend_id` + `wgpu_backend` into single `backend` field:
  - wgpu: `"wgpu/Vulkan"`, `"wgpu/Metal"`, `"wgpu/Dx12"`
  - hip: `"hip"`
  - Affects: `CaseIdentity`, `MeasureRecordSerialized`, `generate_case_id()`, DB schema, config
- **`tests/benchmarks.rs`**:
  - Add DuckDB writer alongside JSONL writer
  - Wrap entire run in a single transaction (`insert_run` ... N × `insert_measurement` ... `commit`)
  - Remove `update_dashboard_index()` function (lines 1141-1173)
  - Store `human_name`, `profile`, `gpu_short` in the runs table
  - Update JSONL filename pattern to `{profile}_{gpu}_{name}.jsonl`
  - **Implement `prefill_mixed` runner** — currently hardcoded to skip (lines 1253-1262 filter to only `decode_only` and `prefill_uniform`). Add execution loop for `prefill_mixed` cases matching the existing `prefill_uniform` loop pattern.
- **`benchmarks/config.yaml`** — update `filename_pattern`, update backend config to use new format

### Makefile (new file)

```makefile
.PHONY: dashboard build bench-promote

# Start Observable Framework dev server (copies prod DB to dev for local preview)
dashboard:
	@mkdir -p benchmarks
	@if [ ! -f benchmarks/results.db ] && [ -f docs/benchmarks/results.db ]; then \
		cp docs/benchmarks/results.db benchmarks/results.db; \
		echo "Copied prod DB → benchmarks/results.db for dev"; \
	fi
	cd docs && npx observable preview

# Build static site for production
build:
	cd docs && npx observable build

# Promote dev benchmark DB to prod (requires clean git)
bench-promote:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: git working tree is not clean"; exit 1; \
	fi
	cp benchmarks/results.db docs/benchmarks/results.db
	@echo "Promoted benchmarks/results.db → docs/benchmarks/results.db"
```

### Observable Framework: Dashboard site

- **`docs/package.json`** — `@observablehq/framework` dependency
- **`docs/observablehq.config.js`** — config:
  - `title: "hip-rwkv"`
  - `theme: "dashboard"`
  - Pages config (excluding `plans/`)
  - `search: true`
  - No `dynamicPaths` — run detail uses client-side routing via query parameter
- **`docs/index.md`** — Landing page with link to benchmarks
- **`docs/style.css`** — Engineering aesthetic (neutral grays, blue accent, monospace numerics)
- **`docs/benchmarks/index.md`** — Runs Index:
  - SQL front matter: `sql: { bench: benchmarks/results.db }`
  - SQL query joins runs with aggregated measurement stats
  - Sortable table: human name (linked), date, git SHA, case count, scenarios, models, best tok/s
- **`docs/benchmarks/run.md`** — Run Detail (client-side routed):
  - Run ID read from URL query parameter: `?run=<run_id>` (e.g., `/benchmarks/run?run=abc123`)
  - Linked from Runs Index table
  - Header card with run metadata from `bench.runs`
  - Bar chart: `tok_per_s` by case config, colored by backend (Observable Plot)
  - Line chart: `tok_per_s` vs `batch_size`, lines per model×backend
  - Measurements table
- **`docs/benchmarks/compare.md`** — Compare Runs:
  - Two `Inputs.select` dropdowns from runs table
  - Match by `case_id`, compute % delta on `tok_per_s`
  - Diverging bar chart (green=improvement, red=regression)
  - Comparison table

### File operations

**Move** (17 planning docs → `docs/plans/`, excluded from sidebar):
- `docs/DASHBOARD_OVERHAUL_PLAN.md`
- `docs/BENCHMARKING_DASHBOARD_PLAN.md`
- `docs/CLEANUP_PLAN.md`
- `docs/FIXTURE_GENERATION_SPEC.md`
- `docs/FLA_CHUNK_SIZE_PROFILING.md`
- `docs/FLA_IMPLEMENTATION_PLAN.md`
- `docs/hip-rwkv-extraction-plan.md`
- `docs/HIP_KERNEL_OPTIMIZATION_RESEARCH.md`
- `docs/HIP_MEMORY_LAYOUT.md`
- `docs/HIP_OPTIMIZATION_FINDINGS.md`
- `docs/HIP_PROBE_SYSTEM_PLAN.md`
- `docs/PACKED_SEQUENCES_PLAN.md`
- `docs/PREFILL_DECODE_SEPARATION_PLAN.md`
- `docs/RWKV7_ARCHITECTURE.md`
- `docs/RWKV7_HIP_BACKEND_PLAN.md`
- `docs/RWKV7_PAPER_METHOD_SECTION.md`
- `docs/THEROCK_INSTALLATION.md`

**Rename** (crate):
- `crates/web-rwkv-bench/` → `crates/rwkv-bench/`
- Update workspace `Cargo.toml` member entry
- Update `tests/benchmarks.rs` imports

**Delete** (old dashboard — entire `benches/` tree, 7 files):
- `benches/dashboard/index.html`
- `benches/dashboard/app.js` (4,292 lines)
- `benches/dashboard/style.css`
- `benches/dashboard/README.md`
- `benches/dashboard/test_parsing.js`
- `benches/dashboard/scripts/gen_manifest.py`
- `benches/dashboard/data/index.json`
- `benches/` directory itself (empty after dashboard removal)

**Delete** (Python benchmark scripts):
- `benchmarks/validate_config.py` (905 lines)
- `benchmarks/test_validate_config.py` (731 lines)

**Delete** (old manifest):
- `benchmarks/results/index.json`

**Keep** (non-benchmark Python scripts, not part of this overhaul):
- `assets/scripts/convert_safetensors.py`
- `assets/scripts/convert_tokenizer.py`
- `scripts/extract_rwkv7_fixtures.py`
- `scripts/rgp2sqlite.py`
- `scripts/rgp_decode_sqtt.py`

**Update** `.gitignore`:
- Add: `docs/.observablehq/`, `docs/node_modules/`, `dist/`, `benchmarks/results.db`

**Add** `.gitattributes`:
- `docs/benchmarks/results.db filter=lfs diff=lfs merge=lfs -text`

### GitHub Pages workflow

**`.github/workflows/docs.yml`**:
- Trigger: push to main, **only when `docs/**` changed** (`paths: ['docs/**']`), plus `workflow_dispatch`
- Steps: checkout (with lfs), setup-node 20, `npm ci` in `docs/`, `make build`, deploy pages

## Implementation Order

### Phase 1: Rust — DuckDB writer + human names + case filtering
1. Rename crate `web-rwkv-bench` → `rwkv-bench`, add `duckdb` dependency
2. Implement `generate_human_name()` with embedded word lists
3. Implement `shorten_gpu()`
4. Create `crates/rwkv-bench/src/db.rs` with `DbWriter` (schema creation, transaction-based insert, STORAGE_VERSION pinning, SQL-based case filtering)
5. Wire into `tests/benchmarks.rs`: open DuckDB writer, use `query_cases()` for filtered iteration, insert measurements, commit at run end
6. Update config.yaml: filename pattern with new placeholders, condition strings to SQL syntax (`=` not `==`, `backend` not `backend_id`)
7. Remove `update_dashboard_index()`
8. Migrate existing 2 JSONL files into `benchmarks/results.db` via duckdb CLI one-liner

### Phase 2: Observable Framework scaffold
1. Move planning docs to `docs/plans/`
2. Create `docs/package.json` + `docs/observablehq.config.js`
3. Create `docs/index.md`
4. Create `docs/style.css`
5. Set up git-lfs for `docs/benchmarks/results.db`
6. Run migration + `make bench-promote` to seed prod db
7. Update `.gitignore`

### Phase 3: Dashboard pages
1. Create `docs/benchmarks/index.md` (Runs Index)
2. Create `docs/benchmarks/run.md` (Run Detail, client-side routed via `?run=` query param)
3. Create `docs/benchmarks/compare.md` (Compare)

### Phase 4: Cleanup + deployment
1. Delete `benches/dashboard/` entirely
2. Delete Python benchmark files
3. Delete `crates/rwkv-bench/src/skip.rs` (925 lines — replaced by SQL-based case filtering in `db.rs`)
4. Remove `pub mod skip` / `pub use skip::{LimitsTracker, SkipReason}` from `lib.rs`, `use crate::skip::{..}` from `sweep.rs`
5. Create Makefile
6. Create `.github/workflows/docs.yml` (path-filtered to `docs/**`)

## SQL-Based Case Filtering (replaces custom expression parser)

The current codebase has a 925-line custom expression parser in `skip.rs` (`should_skip()`, `evaluate_expression()`, `RuleContext`, `SkipReason`, `LimitsTracker`) that evaluates skip condition strings from `config.yaml`. Since DuckDB is now a dependency, we replace this with DuckDB's own SQL engine.

### How it works

1. **Expand** the cartesian product of cases in Rust (profiles × models × backends × params) as before
2. **Insert** all expanded cases into a temp `cases` table in the run's DuckDB connection
3. **Build a single SELECT** with a WHERE clause that negates all skip conditions
4. **Iterate** the query result set as the execution loop — only non-skipped cases come back

```sql
-- Built-in filters become SQL, custom_rules become NOT(...) clauses
SELECT * FROM cases
WHERE batch_size <= max_batch_size
  AND token_chunk_size <= max_token_chunk_size
  AND NOT (batch_size > 16 AND seq_len > 1024)
  AND NOT (model_name = 'rwkv_puzzle15' AND backend = 'hip')
  AND NOT (backend LIKE 'wgpu%' AND batch_size > 256)
ORDER BY scenario, model_name, backend, batch_size
```

### Config changes

The `skip_conditions.custom_rules[].condition` strings are already nearly valid SQL. Two minor adjustments:

- `==` → `=` (SQL equality)
- `backend_id` → `backend` (matches the unified backend field from bd-2x77.2)

Updated config.yaml example:
```yaml
skip_conditions:
  skip_batch_exceeds_model_max: true
  skip_chunk_exceeds_model_max: true
  custom_rules:
    - name: "skip_large_batch_large_seq"
      condition: "batch_size > 16 AND seq_len > 1024"
    - name: "skip_puzzle15_hip"
      condition: "model_name = 'rwkv_puzzle15' AND backend = 'hip'"
    - name: "skip_wgpu_large_batch"
      condition: "backend LIKE 'wgpu%' AND batch_size > 256"
```

### Files deleted

- `crates/rwkv-bench/src/skip.rs` (925 lines) — custom expression parser, `SkipReason`, `LimitsTracker`
- Remove `pub mod skip` and `pub use skip::{LimitsTracker, SkipReason}` from `lib.rs`
- Remove `use crate::skip::{should_skip, LimitsTracker, SkipReason, StopReason}` from `sweep.rs`
- Inline the simple `LimitsTracker` counters (max_cases, max_errors, timeout) into `db.rs` or the runner directly

### Benefits

- Eliminates 925 lines of hand-rolled expression parser
- Condition strings are now real SQL — users get full SQL expression power (`IN`, `LIKE`, arithmetic, `BETWEEN`, etc.)
- Single source of truth: the same DuckDB engine evaluates conditions at run time and queries results in the dashboard
- No impedance mismatch between column names in conditions vs schema

## Key Design Decisions

- **STORAGE_VERSION 'v1.0.0'**: Observable Framework bundles DuckDB-WASM = DuckDB v1.1.1 (storage version 64). Pinning ensures compatibility. Bump when Observable updates.
- **Whole-run transaction**: The entire benchmark run (1 run header + N measurements) is a single DuckDB transaction. Either the whole run lands or none of it does. Simpler and faster than per-row commits.
- **JSONL as secondary output**: JSONL files remain as portable per-run archives with new naming `{profile}_{gpu}_{name}.jsonl`. DuckDB is the primary store.
- **Dev/prod separation**: Dev db in `benchmarks/results.db` (gitignored), prod in `docs/benchmarks/results.db` (git-lfs). Promotion requires clean git.
- **Separate tables per scenario**: `decode`, `prefill_uniform`, `prefill_mixed` — no NULL columns, every row is fully populated. Each table has its own scenario-specific params and metrics.
- **`duckdb` crate only in `rwkv-bench`**: No impact on the main binary.
- **Unified `backend` field**: `"wgpu/Vulkan"`, `"wgpu/Metal"`, `"hip"` — replaces separate `backend_id` + `wgpu_backend` columns.
- **SQL-based skip conditions**: Custom expression parser in `skip.rs` replaced by DuckDB SQL WHERE clauses. Config condition strings become SQL fragments. Eliminates 925 lines.
- **`prefill_mixed` support**: Currently hardcoded to skip in the runner (lines 1253-1262 filter to decode_only + prefill_uniform only). Implementing the missing execution loop.
- **No Python**: Config validation, manifest generation, and data loading scripts all removed.
- **Crate rename**: `web-rwkv-bench` → `rwkv-bench` to match project rename.
- **GitHub Pages path filter**: Workflow only triggers when `docs/**` files change.

## Verification

1. `cargo test --release -p rwkv-bench` — DuckDB writer unit tests pass
2. `duckdb benchmarks/results.db "SELECT * FROM runs"` — shows runs with human names
3. `duckdb benchmarks/results.db "SELECT count(*), avg(decode_tok_per_s) FROM decode"` — per-scenario tables work
4. `make bench-promote` — fails if git dirty, succeeds if clean
5. `make dashboard` — site loads at localhost:3000
6. Runs Index shows all runs with human-readable names
7. Click a run → detail page with charts and table
8. Compare page: select two runs, see % deltas
9. `make build` → `docs/dist/` with all static files
10. Planning docs NOT in sidebar
