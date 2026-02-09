# Dashboard Overhaul: Evidence + DuckDB

## Context

The current benchmark dashboard is a monolithic static HTML/JS/CSS bundle (`benches/dashboard/`) with a 4300-line `app.js` using raw D3.js. It requires a manually-generated `index.json` manifest, has 5 overly-complex visualization views, and no integration with the docs site. The benchmark runner writes JSONL files with opaque timestamp-based names.

This overhaul replaces it with:
- **DuckDB** as the benchmark results store (Rust writes, Evidence reads natively)
- **Evidence** (evidence.dev) for the dashboard site in `docs/` — SQL-first BI framework with built-in interactive components
- **Human-readable run names** generated in Rust (`words · profile · gpu`)
- **3 pages** — one per benchmark type (decode, prefill_uniform, prefill_mixed)
- **GitHub Pages** deployment via static build

## Architecture

```
Benchmark Run (Rust)                      Dashboard (Evidence)
====================                      ====================
rwkv-bench crate                          docs/
  |                                         |
  | duckdb crate                            | @evidence-dev/duckdb
  | STORAGE_VERSION 'v1.0.0'               |
  v                                         v
benchmarks/results/results.db  --copy-->  docs/sources/bench/results.db
  (dev, gitignored)                         |
                                            v
                                      Source queries (.sql files)
                                            |
                                            v
                                      Markdown pages + Evidence components
                                      (BarChart, DataTable, Dropdown, BigValue)
                                            |
                                            v
                                      Static site --> GitHub Pages
```

### Data flow

- Benchmark runner writes to `benchmarks/results/results.db` (dev DB, gitignored)
- `make dashboard` copies DB into `docs/sources/bench/results.db` then starts Evidence dev server
- Evidence source queries (`.sql` files) JOIN scenario tables with `models` and `runs`, cached as Parquet
- Markdown page queries run against the cache using DuckDB dialect
- `evidence build` generates static site with data baked in

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

## Evidence Dashboard Structure

```
docs/
├── package.json                          @evidence-dev/evidence, core-components, duckdb
├── evidence.config.yaml                  DuckDB datasource plugin registration
├── Dockerfile                            node:18-slim, Evidence dev server
├── .dockerignore
├── sources/
│   └── bench/
│       ├── connection.yaml               type: duckdb, filename: results.db
│       ├── results.db                    (copied from benchmarks/results/)
│       ├── decode.sql                    source query: decode JOIN models JOIN runs
│       ├── prefill_uniform.sql           source query: prefill_uniform JOIN models JOIN runs
│       └── prefill_mixed.sql             source query: prefill_mixed JOIN models JOIN runs
└── pages/
    ├── index.md                          landing page with links to 3 pages
    ├── decode.md                         decode benchmark page
    ├── prefill-uniform.md                prefill uniform benchmark page
    └── prefill-mixed.md                  prefill mixed benchmark page
```

### Source queries

Each `.sql` file in `sources/bench/` defines a denormalized view that Evidence caches as Parquet. Pages query against these cached views.

**`decode.sql`**: `SELECT d.*, m.model_name, m.model_size, r.human_name, r.profile, r.gpu_short, r.started_at_utc, r.git_sha FROM decode d JOIN models m ON d.model_sha = m.model_sha JOIN runs r ON d.run_id = r.run_id WHERE d.status = 'ok'`

**`prefill_uniform.sql`**: Same pattern for `prefill_uniform` table.

**`prefill_mixed.sql`**: Same pattern for `prefill_mixed` table.

### Page pattern (same for all 3 pages)

Each page follows the same structure:
1. **Dropdown filters** — run, model, backend (multi-select, all selected by default)
2. **BigValue summary cards** — total cases, median tok/s, best tok/s
3. **BarChart** — primary throughput metric grouped by relevant dimensions
4. **Additional charts** — scenario-specific (latency for decode, TTFT for prefill, etc.)
5. **DataTable** — all filtered rows with formatted columns

Evidence handles interactivity natively: `<Dropdown>` components bind to SQL queries via `${inputs.name.value}`, and all downstream queries/charts re-render automatically.

### Page-specific details

**decode.md**:
- BarChart: avg `decode_tok_per_s` by model/backend/batch_size (horizontal bars)
- BarChart: step latency p50/p95 (if data exists)
- DataTable columns: run, model, backend, batch_size, steps, tok/s, total_ms, p50, p95

**prefill-uniform.md**:
- BarChart: `prefill_tok_per_s` by seq_len, series=backend
- LineChart: TTFT p50 by seq_len
- DataTable columns: run, model, backend, batch_size, seq_len, tok/s, total_ms, ttft_p50

**prefill-mixed.md**:
- BarChart: `prefill_tok_per_s` by mixed_case_id, series=backend
- DataTable columns: run, model, backend, batch_size, mixed_case_id, tok/s, total_ms, ttft_p50

## Files to Modify

### Rust: Benchmark runner → DuckDB output

> **Status**: Tickets .1 (rename), .2 (backend merge), .3 (human names), .4 (DuckDB writer) are DONE. Only .5 (wiring) and .6 (prefill_mixed) remain.

- ~~**Rename crate**: `crates/web-rwkv-bench/` → `crates/rwkv-bench/`~~ ✓ Done (bd-2x77.1)
- ~~**Backend field change**: merge `backend_id` + `wgpu_backend` into single `backend` field~~ ✓ Done (bd-2x77.2)
- ~~**Human names**: `generate_human_name()`, `shorten_gpu()`~~ ✓ Done (bd-2x77.3)
- ~~**DuckDB writer**: `crates/rwkv-bench/src/db.rs` with `DbWriter`~~ ✓ Done (bd-2x77.4)
- **Wire DuckDB writer into `tests/benchmarks.rs`** (bd-2x77.5, in progress):
  - Open `DbWriter` alongside JSONL writer, wrap in `Option` for graceful fallback
  - Insert run header, upsert models, insert decode/prefill rows
  - Commit at run end
  - Remove `update_dashboard_index()`
- **Implement `prefill_mixed` runner** (bd-2x77.6):
  - Currently hardcoded to skip (filter to decode_only + prefill_uniform only)
  - Add execution loop matching prefill_uniform pattern
  - Write to both DuckDB `prefill_mixed` table and JSONL

### Makefile

```makefile
.PHONY: dashboard dashboard-docker

# Copy benchmark DB into Evidence sources and start dev server
dashboard:
	cp benchmarks/results/results.db docs/sources/bench/results.db
	cd docs && npm run sources && npm run dev

# Build and run dashboard in a container (podman-compatible, :z for SELinux)
dashboard-docker:
	cp benchmarks/results/results.db docs/sources/bench/results.db
	podman build -t hip-rwkv-dashboard docs/
	podman run --rm -p 3000:3000 hip-rwkv-dashboard
```

### File operations

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
- Add: `docs/node_modules/`, `docs/.evidence/`, `docs/build/`, `docs/.svelte-kit/`, `docs/sources/bench/results.db`

### GitHub Pages workflow

**`.github/workflows/docs.yml`**:
- Trigger: push to main when `docs/**` changed, plus `workflow_dispatch`
- Steps: checkout, setup-node 18, `npm ci` in `docs/`, copy results.db, `npm run sources`, `npm run build`, deploy pages

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

## Implementation Tickets (children of bd-2x77)

### Already done
- ~~bd-2x77.1: Rename crate~~ ✓
- ~~bd-2x77.2: Merge backend field~~ ✓
- ~~bd-2x77.3: Human names~~ ✓
- ~~bd-2x77.4: DuckDB writer~~ ✓

### Remaining Rust tickets
- **bd-2x77.5**: Wire DuckDB writer into benchmark runner (P1, in progress)
- **bd-2x77.6**: Implement prefill_mixed benchmark execution (P2, depends on .5)
- **bd-2x77.12**: Delete old dashboard and Python benchmark scripts (P2, depends on .5)

### New Evidence tickets
- **T1: Evidence scaffold** (P1) — package.json, evidence.config.yaml, sources/bench/, index.md, .gitignore
- **T2: decode.md** (P1, depends on T1) — dropdowns, BigValues, BarCharts, DataTable
- **T3: prefill-uniform.md** (P1, depends on T1) — same pattern with seq_len + TTFT charts
- **T4: prefill-mixed.md** (P1, depends on T1) — same pattern with mixed_case_id
- **T5: Docker + Makefile** (P2, depends on T1) — Dockerfile, .dockerignore, Makefile targets

### Dependency graph

```
bd-2x77.5 (wire DuckDB)
  └── bd-2x77.6 (prefill_mixed)
  └── bd-2x77.12 (delete old dashboard)

T1 (Evidence scaffold)
  ├── T2 (decode.md)
  ├── T3 (prefill-uniform.md)    ← T2/T3/T4 parallelizable
  ├── T4 (prefill-mixed.md)
  └── T5 (Docker + Makefile)
```

## Key Design Decisions

- **Evidence over Observable**: Observable required JavaScript glue code, had Arrow Table quirks, no loading spinner for SQL blocks, and complex client-side routing. Evidence is SQL-first with built-in components, native DuckDB support, and templated pages.
- **Server-side queries**: Evidence runs SQL at build time (or dev time), not client-side WASM. Data is baked into static HTML. For a benchmark dashboard that updates infrequently, this is ideal.
- **Copy-on-build**: DB is copied from `benchmarks/results/` into Evidence `sources/` before build/dev. No symlinks, no runtime path dependencies.
- **STORAGE_VERSION 'v1.0.0'**: Pinned for DuckDB compatibility across Rust writer and Evidence reader.
- **Whole-run transaction**: The entire benchmark run (1 run header + N measurements) is a single DuckDB transaction. Either the whole run lands or none of it does.
- **JSONL as secondary output**: JSONL files remain as portable per-run archives. DuckDB is the primary store.
- **Separate tables per scenario**: `decode`, `prefill_uniform`, `prefill_mixed` — no NULL columns, every row is fully populated.
- **`duckdb` crate only in `rwkv-bench`**: No impact on the main binary.
- **Unified `backend` field**: `"wgpu/Vulkan"`, `"wgpu/Metal"`, `"hip"` — replaces separate `backend_id` + `wgpu_backend` columns.
- **SQL-based skip conditions**: Custom expression parser in `skip.rs` replaced by DuckDB SQL WHERE clauses. Eliminates 925 lines.
- **No Python**: Config validation, manifest generation, and data loading scripts all removed.

## Risks

- **DuckDB array columns** (`ttft_ms_local` DOUBLE[], `seq_lens` INTEGER[]) may not extract cleanly to Evidence's Parquet cache. Fallback: exclude from source queries or flatten with aggregates.
- **DuckDB version compat** — DB uses storage_version v1.0.0. Evidence's bundled DuckDB must be able to read it. Verify during scaffold setup.
- **Evidence input binding syntax** — multi-select `IN ${inputs.name.value}` needs validation.

## Verification

1. `cargo check --test benchmarks` — DuckDB wiring compiles
2. `cargo test --release --test benchmarks -- --ignored --nocapture` — bench writes to both DuckDB and JSONL
3. `make dashboard` — Evidence dev server starts, all 3 pages load with real data
4. Dropdown filters on each page update charts and tables
5. `make dashboard-docker` — container builds and runs
6. `npm run build` in `docs/` — static site builds
