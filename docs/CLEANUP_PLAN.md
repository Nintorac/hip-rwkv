# Hip-RWKV Codebase Cleanup & Extraction Plan

## Overview

1. Extract hip-rwkv as standalone crate (depend on published web-rwkv)
2. Clean up dead code and naming
3. Fix safety issues and code quality

---

## Phase 0: Extract hip-rwkv as Standalone

### 0.1 Remove web-rwkv source

Delete the local web-rwkv source - we'll use the published crate:

```
DELETE:
├── src/                    # All web-rwkv source
├── examples/               # web-rwkv examples (keep hip-rwkv/examples/)
├── crates/web-rwkv-derive/ # Published separately
```

### 0.2 Update hip-rwkv/Cargo.toml

Change from path dependency to crates.io:

```toml
# Before
web-rwkv = { version = "0.10.19", path = "..", default-features = false }

# After
web-rwkv = { version = "0.10.19", default-features = false }
```

### 0.3 Update root Cargo.toml

Convert to hip-rwkv-only workspace:

```toml
[workspace]
members = [".", "crates/web-rwkv-bench"]

[package]
name = "hip-rwkv"
# ... move hip-rwkv package definition to root
```

Or simpler: make hip-rwkv/ the root (move files up).

### 0.4 Keep local crates

- `crates/web-rwkv-bench/` - Keep (benchmarking harness, 8k lines)
- Profiling code (`profiling.rs`, `prof.rs`) - Not needed for HIP, discard

### 0.5 Update tests & dependencies

**Cargo.toml dependencies:**
```toml
[dependencies]
web-rwkv = { version = "0.10.19", default-features = false }  # Traits only

[dev-dependencies]
web-rwkv = { version = "0.10.19", features = ["native"] }  # Full WebGPU for benchmarks
```

**Tests:**
- `tests/benchmarks.rs` - Keep (WebGPU vs HIP comparison via dev-dep)
- `tests/common/mod.rs` - Keep (fixture loading, shared utilities)
- `tests/functional_metrics.rs` - Review if HIP-specific or remove

**New structure:**
```
hip-rwkv/                    # Repository root
├── Cargo.toml               # Main crate
├── src/hip/                 # HIP backend (current hip-rwkv/src/hip/)
├── build.rs                 # hipcc compilation
├── examples/                # HIP examples
├── tests/                   # HIP tests
├── crates/web-rwkv-bench/   # Benchmarking utilities
├── docs/                    # Documentation
└── .beads/                  # Issue tracker
```

---

## Phase 1: Rename WKV Kernels

| Current | New | Location |
|---------|-----|----------|
| `FlaChunkedWkv` | `ChunkWkv` | `src/hip/model/fla.rs` |
| `FusedT1Wkv` | `RecurrentWkv` | `src/hip/model/prefill.rs` |

**Files to modify:**
- `src/hip/model/fla.rs` - rename struct
- `src/hip/model/prefill.rs` - rename struct
- `src/hip/model/mod.rs` - update re-exports
- `src/hip/model/decode.rs` - update usage
- `src/hip/model/hip_prefill.rs` - update references
- `src/hip/model/dispatch_helpers.rs` - update references
- `src/hip/model/state.rs` - update doc comments

---

## Phase 2: Remove Dead Code

### 2.1 Remove `WaveReduceWkv` (superseded by `ChunkWkv`)

**File:** `src/hip/model/prefill.rs`
- Delete `WaveReduceWkv` struct (~lines 84-118)
- Delete `WkvKernel` impl for `WaveReduceWkv`
- Remove `supports_multi_token()` and `name()` from `WkvKernel` trait

**File:** `src/hip/model/mod.rs`
- Remove `WaveReduceWkv` from exports

**File:** `src/hip/model/fla.rs`
- Remove comment referencing `WaveReduceWkv` (line 33)

### 2.2 Remove unused FFI functions

**File:** `src/hip/ffi.rs`
- `launch_sgemm_ta`
- `launch_sgemv_strided_batched`
- `launch_wkv7_state_update_flat`
- `launch_exp_f16`

**File:** `src/hip/blas.rs`
- `hip_sgemm_ta` function
- Remove import of `launch_sgemm_ta`

**File:** `src/hip/kernels/elementwise.rs`
- `exp_f16` function
- Remove import of `launch_exp_f16`

### 2.3 Remove unused constants

**File:** `src/hip/ffi.rs`
- `HIP_HOST_MALLOC_WRITE_COMBINED`
- `HIP_EVENT_DEFAULT`
- `HIP_STREAM_WAIT_VALUE_EQ`

### 2.4 Remove unused helper

**File:** `src/hip/kernels/rwkv_ops.rs`
- `extract_shift_state_at_lengths` function

### 2.5 Internalize `FLA_CHUNK_THRESHOLD`

**File:** `src/hip/model/fla.rs`
- Make `pub(crate)` or inline into tests

---

## Phase 3: Fix Safety Issues (Clippy Errors)

Wrap raw pointer operations in `unsafe {}` blocks with safety comments:

**File:** `src/hip/blas.rs` (lines 196, 208, 271, 402)
- `rocblas_destroy`, `rocblas_set_stream`, `sgemm_f32`, `hgemm_f16`

**File:** `src/hip/blaslt.rs` (line 276)
- `hipblaslt_destroy`

**File:** `src/hip/kernels/rwkv_ops.rs` (line 762)

---

## Phase 4: Code Quality Fixes

### 4.1 Use stdlib methods

- `x.div_ceil(n)` instead of manual div_ceil
- `x.is_multiple_of(n)` where applicable

### 4.2 Fix unused variables

Add `_` prefix: `sa_len`, `n_embd`, `ffn_hidden`

### 4.3 Fix unused imports

Remove unused imports flagged by compiler

---

## Phase 5: Consolidate Test Infrastructure

### 5.1 Single test common module

After extraction, there will be only one `tests/common/mod.rs`

### 5.2 Remove dead_code allows

Clean up `#[allow(dead_code)]` for actually-dead code

### 5.3 Fix unused imports in tests

- `ground_truth.rs` - remove unused `Tolerances`
- `multi_batch_stress.rs` - remove unused `stable_softmax`
- `hipblaslt_benchmark.rs` - remove unused `Stream`

---

## Phase 6: Create Refactoring Ticket

Create a ticket under the cleanup epic for future code deduplication. The ticket should note that further planning is needed to break into subtasks.

**Ticket content:**

```
Title: Code deduplication and refactoring opportunities

Description:
Analysis identified ~600-900 lines of duplicated code patterns that could be consolidated.
Further exploration and planning needed to create subtasks.

## Identified Opportunities

### 1. F32/F16 Function Pairs (Critical - ~200-300 LOC)
~40 nearly-identical function pairs in kernel wrappers where only the type differs.

Files:
- src/hip/kernels/elementwise.rs (26 pairs)
- src/hip/kernels/norm.rs (6 pairs)
- src/hip/kernels/rwkv_ops.rs (3 pairs)
- src/hip/kernels/wkv.rs (1 pair)

Approach: Create generic trait `KernelType` with f32/f16 impls, or use macros.

### 2. Validation Helpers (Critical - ~140-210 LOC)
70+ repeated validation blocks for contiguity and size checks.

Pattern repeated everywhere:
```rust
if !input.is_contiguous() || !output.is_contiguous() {
    return Err(HipErrorKind { code: -1, message: "..." });
}
if input.len() != output.len() {
    return Err(HipErrorKind { code: -1, message: "..." });
}
```

Approach: Helper functions like validate_contiguous_pair(), validate_same_length()

### 3. Host Convenience Functions (High - ~80-120 LOC)
~20 identical hip_* wrapper functions with same pattern:
- Create stream, shape
- Allocate device tensors from host slice
- Call GPU kernel
- Copy back to host

Approach: Generic helper or macro to eliminate boilerplate.

### 4. rocBLAS Context Duplication (High - ~100-130 LOC)
HipBlasContext (blas.rs) and HipBlasLtContext (blaslt.rs) share:
- Create/destroy patterns
- Stream management
- Drop impl
- Helper functions

Approach: Extract common BlasContextBase trait or generic struct.

### 5. Kernel Launch Patterns (Medium - ~50-100 LOC)
~50 kernel wrapper functions with identical unsafe { check(launch_*(...)) } pattern.

Approach: Generic launcher helper that wraps check() and unsafe block.

### 6. Tensor Dimension Extraction (Medium - ~30-60 LOC)
~30 instances of:
```rust
let c = x.shape()[0];
let t = x.shape()[1];
let b = x.shape()[2];
```

Approach: Helper method on TensorHip or extract_dims_4d() function.

### 7. GEMM Validation (Medium - ~30-50 LOC)
Matrix dimension validation repeated across 5 GEMM functions in blas.rs/blaslt.rs.

Approach: validate_gemm_dims(weight, input) -> Result<(m, k, n)>

## Estimated Impact
Total: 600-920 lines of code could be eliminated
Improved maintainability and reduced bug surface area
```

---

## Verification

After Phase 0 (extraction):
```bash
cargo check -p hip-rwkv --all-targets
cargo test -p hip-rwkv --lib  # Requires HIP hardware
```

After each cleanup phase:
```bash
cargo clippy -p hip-rwkv --all-targets 2>&1 | grep -E "^(warning|error)"
```

Final:
```bash
cargo check --all-targets
cargo clippy --all-targets
```

---

## Files Summary

**Phase 0 - Delete:**
- `src/` (web-rwkv source)
- `examples/` (web-rwkv examples)
- `crates/web-rwkv-derive/`

**Phase 0 - Restructure:**
- Move `hip-rwkv/` contents to root, or update workspace

**Phase 1-5 - Modify:**
- `src/hip/model/fla.rs` - rename `FlaChunkedWkv` → `ChunkWkv`
- `src/hip/model/prefill.rs` - rename `FusedT1Wkv` → `RecurrentWkv`, delete `WaveReduceWkv`
- `src/hip/model/decode.rs` - update references
- `src/hip/model/mod.rs` - update exports
- `src/hip/blas.rs` - remove dead code, fix safety
- `src/hip/blaslt.rs` - fix safety
- `src/hip/ffi.rs` - remove dead FFI declarations
- `src/hip/kernels/elementwise.rs` - remove `exp_f16`
- `src/hip/kernels/rwkv_ops.rs` - remove dead helper, fix safety
