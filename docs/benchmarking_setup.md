# Benchmarking Setup

## Overview

The benchmark system is config-driven via `benchmarks/config.yaml`. It performs
Cartesian expansion of parameters (models x backends x batch_sizes x ...) into
individual test cases, filters them through DuckDB SQL skip conditions, then
executes each case and records results.

**Key components:**

- **Config file** (`benchmarks/config.yaml`): Defines models, backends, profiles,
  scenarios, skip conditions, and output settings.
- **Test harness** (`tests/benchmarks.rs`): Rust integration test that loads config,
  expands cases, runs benchmarks, and writes results.
- **Benchmark library** (`crates/rwkv-bench/`): Shared types, DuckDB writer,
  JSONL output, and utility functions.
- **Results**: JSONL files + DuckDB database in `benchmarks/results/`.

## Quick Start

```bash
# Run with default config (benchmarks/config.yaml) and smoke profile
cargo test --release --test benchmarks -- --ignored --nocapture

# Run with HIP backend
cargo test --release --features hip --test benchmarks -- --ignored --nocapture

# Select a specific profile
WEB_RWKV_BENCH_PROFILE=dev \
  cargo test --release --features hip --test benchmarks -- --ignored --nocapture

# Run with memory budget (for AIMax 395 or other constrained systems)
WEB_RWKV_MAX_DEVICE_MB=32768 WEB_RWKV_BENCH_PROFILE=aimax395_dashboard \
  cargo test --release --features hip --test benchmarks -- --ignored --nocapture
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WEB_RWKV_BENCH_CONFIG` | `benchmarks/config.yaml` | Path to YAML config file |
| `WEB_RWKV_BENCH_PROFILE` | `smoke` | Profile name to use |
| `WEB_RWKV_FLA_CHUNK_SIZE` | `16` | FLA chunk size for chunked prefill (16/32/64) |
| `WEB_RWKV_MAX_DEVICE_MB` | `65536` (64 GB) | Memory budget safety net in MB. Before allocating GPU scratch buffers, the runner estimates memory and skips cases exceeding this budget. Prevents `hipMalloc` from silently falling through to GTT on APUs. |

## Config Structure (`benchmarks/config.yaml`)

### Top-level sections

```yaml
schema_version: 3      # Bump on breaking changes

profiles:               # Named parameter presets
models:                 # Model definitions with paths and limits
backends:               # Backend definitions (wgpu, hip)
scenarios:              # Scenario type definitions and defaults
mixed_cases:            # Mixed-length patterns for prefill_mixed
output:                 # JSONL output directory and naming
skip_conditions:        # Builtin flags + custom DuckDB SQL rules
limits:                 # Global execution limits
shared_controls:        # Default values for all scenarios
```

### Profiles

Each profile specifies which models, backends, and scenario parameters to use:

```yaml
profiles:
  my_profile:
    description: "What this profile benchmarks"
    models: [rwkv7_g1a_0.1b, rwkv7_g1c_2.9b]
    backends: [hip, wgpu]
    warmup_runs: 1
    repeats: 3
    benchmarks:
      decode_only:
        batch_sizes: [1, 4, 16, 64]
        decode_steps: [32]
      prefill_uniform:
        batch_sizes: [1, 4, 16]
        token_chunk_sizes: [64, 256, 1024]
        seq_lens: [64, 256, 1024]
```

**Available profiles:**

| Profile | Purpose |
|---|---|
| `smoke` | Quick sanity check with tiny model |
| `dev` | Development testing, broader coverage |
| `quick_coverage` | Good dashboard axes, limited runtime |
| `decode_batch_sweep` | Decode-only batch size sweep |
| `prefill_chunk_sweep` | Prefill sweep across chunk sizes |
| `full` | Comprehensive CI/nightly suite |
| `aimax395_discover_decode` | Discover max decode batch_size per model |
| `aimax395_discover_prefill` | Discover max prefill batch*chunk per model |
| `aimax395_discover_wgpu` | Discover max wgpu decode batch_size |
| `aimax395_dashboard` | Full dashboard sweep with warmup and repeats |
| `aimax395_dashboard_validate` | Single-run validation of dashboard profile |

### Models

Each model entry defines:

```yaml
models:
  - model_id: "sha256_hash"
    model_name: "rwkv7_g1a_0.1b"   # Used in profiles and skip conditions
    model_size: "0.1b"
    path: "/workspace/models/model.st"
    max_batch_size: 65536            # Builtin skip if exceeded
    max_token_chunk_size: 4096       # Builtin skip if exceeded
    skip: false
```

### Skip Conditions

**Builtin flags:**
- `skip_batch_exceeds_model_max`: Skip if `batch_size > model.max_batch_size`
- `skip_chunk_exceeds_model_max`: Skip if `token_chunk_size > model.max_token_chunk_size`
- `skip_known_failures`: Skip known bad combinations
- `skip_oom_predicted`: Skip if OOM is predicted

**Custom rules** use DuckDB SQL expressions evaluated against case parameters:

```yaml
skip_conditions:
  custom_rules:
    - name: "my_rule"
      description: "Skip large batches on wgpu"
      condition: "backend LIKE 'wgpu%' AND batch_size > 256"
```

Available columns in conditions: `scenario`, `model_name`, `model_sha`,
`backend`, `batch_size`, `token_chunk_size`, `seq_len`, `decode_steps`,
`max_batch_size`, `max_token_chunk_size`.

Arithmetic expressions are supported (e.g., `batch_size * token_chunk_size > N`).

## Scenario Types

### `decode_only`

Measures decode throughput in steady-state T=1 mode. Each step calls `infer()`
with one token per batch element.

**Parameters:**
- `batch_sizes`: Number of concurrent sequences
- `decode_steps`: Number of decode iterations per measurement

**HIP runtime mode:** `DecodeOnly` (no prefill buffers allocated)

### `prefill_uniform`

Measures prompt processing with uniform prompt lengths across all batch elements.

**Parameters:**
- `batch_sizes`: Number of concurrent sequences
- `token_chunk_sizes`: Max tokens per infer() call (HIP FLA buffer size)
- `seq_lens`: Prompt length for each batch element

**HIP runtime mode:** `PrefillOnly` (no decode buffers allocated)

### `prefill_mixed`

Measures batching behavior with different prompt lengths per batch element.
Uses named mixed-case patterns (staircase, bimodal, etc.) defined in the
`mixed_cases` section.

**HIP runtime mode:** `PrefillOnly`

## HIP Runtime Modes

The benchmark runner uses single-mode runtime allocation to maximize the
batch size that fits in GPU memory:

| Mode | Allocates | Use Case |
|---|---|---|
| `Both` | Prefill + Decode | General inference |
| `DecodeOnly` | Decode only | Saves ~95 MB of prefill scratch |
| `PrefillOnly` | Prefill only | Saves ~12 MB of decode scratch |

## Memory Budget Safety Net

On APUs with unified memory (like the AIMax 395), `hipMalloc` can silently
fall through to GTT (system RAM) instead of failing. Large allocations succeed
but may wedge the system when accessed under memory pressure.

The `WEB_RWKV_MAX_DEVICE_MB` environment variable sets a memory budget. Before
each `create_hip_runtime()` call, the benchmark runner uses static estimation
methods (`DecodeScratch::estimate_memory_bytes()`, `PrefillScratch::estimate_memory_bytes()`)
to predict GPU memory usage. Cases exceeding the budget are skipped with a log message.

**Estimation covers:**
- All scratch buffers (standard, FFN, LoRA, FLA)
- Per-layer persistent state (attention shift, FFN shift, WKV state)
- Pinned host buffers (logits staging, embedding staging)
- Token and sequence length staging buffers

**Note:** The estimate does NOT include model weight memory (loaded via
`hipMallocManaged` into GTT) since weights are shared across runtime
configurations and loaded once.

## AIMax 395 Dashboard

### Hardware

- **GPU**: AMD Radeon 8060S (Strix Halo APU)
- **VRAM**: 1 GB dedicated
- **Unified memory**: 96 GB (GTT/system RAM accessible to GPU)
- Model weights loaded via `hipMallocManaged` into GTT
- Scratch buffers and state via `hipMalloc` (may overflow to GTT)

### Discovery workflow

1. Run discovery profiles with memory budget:
   ```bash
   WEB_RWKV_MAX_DEVICE_MB=32768 WEB_RWKV_BENCH_PROFILE=aimax395_discover_decode \
     cargo test --release --features hip --test benchmarks -- --ignored --nocapture \
     2>&1 | tee /tmp/discover_decode.log
   ```

2. Record the max successful batch_size per model from the logs. Cases that
   fail show `[bench] Failed to create runtime` or `[bench] SKIP: estimated ... > budget`.

3. Update the placeholder values (`999999999`) in the AIMax 395 skip conditions
   in `benchmarks/config.yaml` with the actual discovered limits.

4. Run validation to confirm no OOM:
   ```bash
   WEB_RWKV_MAX_DEVICE_MB=32768 WEB_RWKV_BENCH_PROFILE=aimax395_dashboard_validate \
     cargo test --release --features hip --test benchmarks -- --ignored --nocapture
   ```

5. Run the full dashboard:
   ```bash
   WEB_RWKV_MAX_DEVICE_MB=32768 WEB_RWKV_BENCH_PROFILE=aimax395_dashboard \
     cargo test --release --features hip --test benchmarks -- --ignored --nocapture
   ```

### Discovered limits

| Model | Decode max batch | Prefill max batch*chunk |
|---|---|---|
| rwkv7_g1a_0.1b | TBD | TBD |
| rwkv7_g1c_2.9b | TBD | TBD |
| rwkv7_g1d_7.2b | TBD | TBD |
| rwkv7_g0b_13.3b | TBD | TBD |
| wgpu (0.1b only) | TBD | N/A |

## Output

### JSONL files

Results are written to `benchmarks/results/` as JSONL files. Each file starts
with a run header record, followed by measure records (one per repeat per case).

Filename pattern: `bench_{profile}_{timestamp}.jsonl`

### DuckDB database

Results are also written to `benchmarks/results/results.db` for SQL analysis.
Tables: `runs`, `cases`, `decode`, `prefill_uniform`, `prefill_mixed`, `models`.

## Adding New Profiles

1. Add the profile entry to `profiles:` in `benchmarks/config.yaml`
2. Reference existing model names from the `models:` section
3. Reference existing backend IDs from the `backends:` section
4. Specify per-scenario benchmark parameters in the `benchmarks:` block
5. Optionally add custom skip conditions if needed
6. Test with: `WEB_RWKV_BENCH_PROFILE=my_profile cargo test --release --features hip --test benchmarks -- --ignored --nocapture`
