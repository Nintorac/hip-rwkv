# FLA Chunk Size Profiling (HIP, Feb 4 2026)

## Goal
Evaluate whether increasing the FLA inner chunk size (C) above 16 improves prefill throughput on gfx1151 for the 2.9B model.

## Setup
- Model: `rwkv7-g1c-2.9b-20251231-ctx8192.st`
- GPU: gfx1151 (ROCm 7.2.0)
- Workload: batch=256, seq_len=256 (65,536 tokens total), packed prefill path
- Chunk sizes: 16, 32, 64
- Runtime configured with `max_prefill_chunk = total_tokens` to avoid the capacity guard
- Profiling: `rocprofv3` hip+hsa+kernel+memcopy traces (adds overhead but gives per-kernel times)

## Results
| Chunk size | Total prefill ms | Tok/s |
|-----------:|-----------------:|------:|
| 16 | 39,115 | 1,675.5 |
| 32 | 43,823 | 1,495.5 |
| 64 | 63,887 | 1,025.8 |

Decode baseline (batch=256, 32 steps): 5,749 ms, 1,424.9 tok/s.

### FLA kernel time (rocprof totals)
- **C=16:** cumsum_intra 13.7 s; chunk_o 5.6 s; wy_wu 5.9 s; chunk_h 2.6 s.
- **C=32:** cumsum_intra 16.4 s; chunk_o 13.6 s; wy_wu 8.0 s; chunk_h 2.3 s.
- **C=64:** chunk_o 28.8 s; wy_wu 26.4 s; cumsum_intra 22.7 s; chunk_h 2.2 s.

## Interpretation
- Bigger C reduces chunk count, but per-chunk cost grows ~C² in Stages 2/3/5; beyond C=32 the larger matrices and register/LDS footprints dominate.
- Stage 5 (chunk_o) and Stage 3B (wy_wu) become the primary drivers at C=64; cumsum_intra also scales poorly with C because it carries C-wide register arrays and walks C sequentially.
- chunk_h benefits slightly from fewer chunks but is a small slice of total time.
- For this workload, C=16 is fastest; C=32 is ~11% slower; C=64 is ~39% slower.

## Optimization outlook
Likelihood of improvement by tiling high C values is **medium**:
- Tiling C into 16/32-wide blocks in chunk_o and wy_wu would shrink LDS/register footprints, improve occupancy, and cut the dominant C² cost. A 20–30% reduction in these kernels could make C=32 competitive or better for large batches.
- C=64 would need >30% kernel speedup to catch C=16; achievable only if tiling plus better reductions significantly lift occupancy and cut memory traffic.

Tradeoffs and risks:
- More passes/reductions can add overhead; if tiling overhead outweighs footprint savings, gains vanish.
- Increased code complexity and potential numerical drift from additional partial reductions.
- More template variants increase compile time, but manageable (add C>32 tiled path only).

Practical guidance:
- Keep production at C≤32 until tiled kernels are prototyped.
- Prototype by tiling chunk_o and wy_wu for C>32, then re-profile C=32/64 to see if they converge toward C=16 performance.
