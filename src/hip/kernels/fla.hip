// FLA (Flash Linear Attention) kernels for RWKV7 HIP backend.
//
// Stages 1+2A+2B (fused): kernel_fla_cumsum_intra — cumulative decay + gating + attention matrices.
// Stage 3: kernel_fla_wy_inv + kernel_fla_wy_wu — WY representation.
// Stage 4: kernel_fla_chunk_h — inter-chunk state recurrence.
// Stage 5: kernel_fla_chunk_o — output combination.
//
// All per-token tensors use web-rwkv column-major layout [K, H, T, B] with K fastest.
// For packed/varlen sequences, B=1 and T is total packed length.
// Position t in sequence starting at bos, head h, element k:
//   index = k + h * K + (bos + t) * H * K
//
// Attention matrices use layout [C, C, H, total_chunks]:
//   A[i, j, h, chunk] = i + j*C + h*C*C + chunk*H*C*C
//
// Reference: repos/flash-linear-attention/fla/ops/generalized_delta_rule/dplr/chunk_A_fwd.py
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ============================================================================
// Fused Stages 1+2A+2B: kernel_fla_cumsum_intra
//
// Combines cumulative decay scan (Stage 1), decay-scaled gating (Stage 2A),
// and intra-chunk attention matrices (Stage 2B) into a single kernel launch.
//
// This fusion keeps gi/ge intermediates in registers instead of writing them
// to global memory, eliminating 2 kernel launches and 2 x K x T global
// read+write round-trips.
//
// Phase 1 (cumsum + gating): Sequential loop over C=16 positions.
//   Thread kk accumulates gi_vals[C] and ge_vals[C] in registers.
//   Also computes g_last, then writes qg, kg, ag, bg to global memory.
//   gi is ALSO written to global memory since Stage 4 needs it.
//
// Phase 2 (attention matrices): Uses register gi/ge values directly.
//   Same wave-level reduction as the standalone kernel_fla_intra_attn.
//   Computes 4 CxC attention matrices per chunk.
//
// Grid:  dim3(total_chunks, H, 1)
// Block: dim3(K=64) (one thread per head-size element)
// ============================================================================

template <int C_TPL>
__global__ __launch_bounds__(64, 4)
void kernel_fla_cumsum_intra(
    const float*  __restrict__ gk,             // [K, H, T_packed, 1] input log-decay
    const __half* __restrict__ q,              // [K, H, T_packed, 1] f16
    const __half* __restrict__ k,              // [K, H, T_packed, 1] f16
    const __half* __restrict__ a,              // [K, H, T_packed, 1] f16
    const __half* __restrict__ b,              // [K, H, T_packed, 1] f16
    float* __restrict__ gi_out,                // [K, H, T_packed, 1] inclusive cumsum (for Stage 4)
    float* __restrict__ qg_out,                // [K, H, T_packed, 1] decay-scaled query
    float* __restrict__ kg_out,                // [K, H, T_packed, 1] decay-scaled key
    float* __restrict__ ag_out,                // [K, H, T_packed, 1] decay-scaled a
    float* __restrict__ bg_out,                // [K, H, T_packed, 1] decay-scaled b
    float* __restrict__ A_qk,                  // [C, C, H, total_chunks]
    float* __restrict__ A_qb,                  // [C, C, H, total_chunks]
    float* __restrict__ A_ak,                  // [C, C, H, total_chunks]
    float* __restrict__ A_ab,                  // [C, C, H, total_chunks]
    const int* __restrict__ chunk_indices,     // [total_chunks * 2]
    const int* __restrict__ cu_seqlens,        // [N+1]
    const int* __restrict__ batch_offsets,     // [N] data offsets per sequence
    int K_dim,                                 // head_size (64)
    int H                                      // n_heads
) {
    const int C = C_TPL;
    const int chunk_flat = blockIdx.x;
    const int h = blockIdx.y;
    const int kk = threadIdx.x;  // K-dimension element index (0..K-1)

    if (h >= H || kk >= K_dim) return;

    // Resolve chunk -> sequence mapping
    const int seq_id      = chunk_indices[chunk_flat * 2];
    const int local_chunk = chunk_indices[chunk_flat * 2 + 1];

    // Sequence boundaries
    const int bos = batch_offsets[seq_id];
    const int T   = cu_seqlens[seq_id + 1] - cu_seqlens[seq_id];

    // Token stride and base offset
    const int token_stride = H * K_dim;
    const int base_h = h * K_dim + bos * token_stride;

    // Chunk boundaries
    const int chunk_start = local_chunk * C;
    const int valid_len = (chunk_start < T) ? min(C, T - chunk_start) : 0;

    // Output base for attention matrices: [C, C, H, total_chunks]
    const int out_hc = h * C * C + chunk_flat * H * C * C;

    // ================================================================
    // Phase 1: Cumsum + gating (sequential over C positions)
    // ================================================================
    // Register arrays for gi and ge values per position in the chunk.
    // C_TPL floats each in registers.
    float gi_vals[C_TPL];
    float ge_vals[C_TPL];

    // Sequential prefix sum + gating
    float accum = 0.0f;
    float g_last = 0.0f;

    // First pass: compute cumsum and cache gi/ge in registers
    #pragma unroll
    for (int t = 0; t < C; t++) {
        if (t < valid_len) {
            const int t_abs = chunk_start + t;
            const int idx = base_h + kk + t_abs * token_stride;
            const float val = gk[idx];

            // Exclusive: before adding current value
            ge_vals[t] = accum;

            accum += val;

            // Inclusive: after adding current value
            gi_vals[t] = accum;

            // Write gi to global memory (needed by Stage 4)
            gi_out[idx] = accum;
        } else {
            gi_vals[t] = 0.0f;
            ge_vals[t] = 0.0f;
        }
    }

    // g_last = gi at last valid position (for gating)
    if (valid_len > 0) {
        g_last = gi_vals[valid_len - 1];
    }

    // Second pass: compute gated outputs using register values
    if (valid_len > 0) {
        #pragma unroll
        for (int t = 0; t < C; t++) {
            if (t < valid_len) {
                const int t_abs = chunk_start + t;
                const int idx = base_h + kk + t_abs * token_stride;

                // Load f16 inputs and convert to f32
                const float q_val = __half2float(q[idx]);
                const float k_val = __half2float(k[idx]);
                const float a_val = __half2float(a[idx]);
                const float b_val = __half2float(b[idx]);

                // Use register gi/ge values directly (no global memory read!)
                const float gi_val = gi_vals[t];
                const float ge_val = ge_vals[t];

                // Compute decay-scaled outputs
                const float exp_gi     = expf(gi_val);
                const float exp_neg_gi = expf(-gi_val + g_last);
                const float exp_ge     = expf(ge_val);

                qg_out[idx] = q_val * exp_gi;
                kg_out[idx] = k_val * exp_neg_gi;
                ag_out[idx] = a_val * exp_ge;
                bg_out[idx] = b_val * exp_neg_gi;
            }
        }
    }

    // ================================================================
    // Phase 2: Attention matrices using register gi/ge values
    // ================================================================
    // Same wave-level reduction as the standalone kernel_fla_intra_attn,
    // but using register values instead of global memory loads.

    // Shared memory for cross-wavefront reduction (4 floats)
    __shared__ float smem[4];

    const int lane_id = kk & 31;
    const int wave_id = kk >> 5;

    // Early exit: if the entire chunk is beyond the sequence, zero the outputs
    if (chunk_start >= T) {
        if (kk == 0) {
            for (int i = 0; i < C; i++) {
                for (int j = 0; j < C; j++) {
                    const int out_idx = out_hc + i + j * C;
                    A_qk[out_idx] = 0.0f;
                    A_qb[out_idx] = 0.0f;
                    A_ak[out_idx] = 0.0f;
                    A_ab[out_idx] = 0.0f;
                }
            }
        }
        return;
    }

    // Outer loop over rows i
    for (int i = 0; i < C; i++) {
        // If row i is beyond sequence length, zero out the row
        if (i >= valid_len) {
            if (kk == 0) {
                for (int j = 0; j < C; j++) {
                    const int out_idx = out_hc + i + j * C;
                    A_qk[out_idx] = 0.0f;
                    A_qb[out_idx] = 0.0f;
                    A_ak[out_idx] = 0.0f;
                    A_ab[out_idx] = 0.0f;
                }
            }
            continue;
        }

        // Load row i data — coalesced (thread kk reads element kk)
        const int i_abs = chunk_start + i;
        const int row_i_offset = base_h + i_abs * token_stride;
        const float q_i  = __half2float(q[row_i_offset + kk]);
        const float a_i  = __half2float(a[row_i_offset + kk]);

        // Use register gi/ge values instead of global memory loads
        const float gi_i = gi_vals[i];
        const float ge_i = ge_vals[i];

        // Precompute exp(gi_i) and exp(ge_i) — constant across all j
        const float exp_gi_i = expf(gi_i);
        const float exp_ge_i = expf(ge_i);

        // Inner loop over columns j (lower triangular: j <= i)
        for (int j = 0; j < C; j++) {
            const int out_idx = out_hc + i + j * C;

            // Boundary check: j beyond valid positions
            if (j >= valid_len) {
                if (kk == 0) {
                    A_qk[out_idx] = 0.0f;
                    A_qb[out_idx] = 0.0f;
                    A_ak[out_idx] = 0.0f;
                    A_ab[out_idx] = 0.0f;
                }
                continue;
            }

            // Causal masks
            const bool inclusive_mask = (i >= j);
            const bool strict_mask   = (i > j);

            // Upper triangular: all zero
            if (!inclusive_mask) {
                if (kk == 0) {
                    A_qk[out_idx] = 0.0f;
                    A_qb[out_idx] = 0.0f;
                    A_ak[out_idx] = 0.0f;
                    A_ab[out_idx] = 0.0f;
                }
                continue;
            }

            // Load col j data — coalesced
            const int j_abs = chunk_start + j;
            const int row_j_offset = base_h + j_abs * token_stride;
            const float k_j  = __half2float(k[row_j_offset + kk]);
            const float b_j  = __half2float(b[row_j_offset + kk]);

            // Use register gi value for column j
            const float gi_j = gi_vals[j];

            // Compute decay factors
            const float exp_neg_gi_j = expf(-gi_j);
            const float decay_qk = exp_gi_i * exp_neg_gi_j;
            const float decay_ak = exp_ge_i * exp_neg_gi_j;

            // Each thread computes its partial product
            float prod_qk = q_i * k_j * decay_qk;
            float prod_qb = q_i * b_j * decay_qk;
            float prod_ak = a_i * k_j * decay_ak;
            float prod_ab = a_i * b_j * decay_ak;

            // Wave-level reduction via __shfl_xor (5 steps for 32 lanes)
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                prod_qk += __shfl_xor(prod_qk, offset, 32);
                prod_qb += __shfl_xor(prod_qb, offset, 32);
                prod_ak += __shfl_xor(prod_ak, offset, 32);
                prod_ab += __shfl_xor(prod_ab, offset, 32);
            }

            // Cross-wavefront reduction via shared memory
            if (wave_id == 1 && lane_id == 0) {
                smem[0] = prod_qk;
                smem[1] = prod_qb;
                smem[2] = prod_ak;
                smem[3] = prod_ab;
            }
            __syncthreads();

            if (wave_id == 0 && lane_id == 0) {
                const float final_qk = prod_qk + smem[0];
                const float final_qb = prod_qb + smem[1];
                const float final_ak = prod_ak + smem[2];
                const float final_ab = prod_ab + smem[3];

                A_qk[out_idx] = final_qk;
                A_qb[out_idx] = final_qb;

                if (strict_mask) {
                    A_ak[out_idx] = final_ak;
                    A_ab[out_idx] = final_ab;
                } else {
                    A_ak[out_idx] = 0.0f;
                    A_ab[out_idx] = 0.0f;
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Host launch wrappers (extern "C" for FFI from Rust)
// ============================================================================
extern "C" {

// ============================================================================
// Host launch wrapper for fused Stages 1+2A+2B
// ============================================================================

hipError_t launch_fla_cumsum_intra(
    const float*  gk,
    const __half* q,
    const __half* k,
    const __half* a,
    const __half* b,
    float* gi,
    float* qg,
    float* kg,
    float* ag,
    float* bg,
    float* A_qk,
    float* A_qb,
    float* A_ak,
    float* A_ab,
    const int* chunk_indices,
    const int* cu_seqlens,
    const int* batch_offsets,
    int K,
    int H,
    int C,
    int total_chunks,
    hipStream_t stream
) {
    if (K <= 0 || H <= 0 || C <= 0 || total_chunks <= 0) {
        return hipSuccess;
    }

    dim3 grid(total_chunks, H, 1);
    dim3 block(K);

    #define LAUNCH_CUMSUM_INTRA(C_VAL) \
        hipLaunchKernelGGL(kernel_fla_cumsum_intra<C_VAL>, grid, block, 0, stream, \
                           gk, q, k, a, b, \
                           gi, qg, kg, ag, bg, \
                           A_qk, A_qb, A_ak, A_ab, \
                           chunk_indices, cu_seqlens, batch_offsets, \
                           K, H)

    switch (C) {
        case 16: LAUNCH_CUMSUM_INTRA(16); break;
        case 32: LAUNCH_CUMSUM_INTRA(32); break;
        case 64: LAUNCH_CUMSUM_INTRA(64); break;
        default: return hipErrorInvalidValue;
    }

    #undef LAUNCH_CUMSUM_INTRA
    return hipGetLastError();
}

} // extern "C" (pause for template kernels)

// ============================================================================
// Stage 3, Part A: Lower-triangular matrix inversion (kernel_fla_wy_inv)
//
// Computes A_ab_inv = (I - A_ab)^{-1} for each chunk.
//
// A_ab is strict lower triangular (zeros on diagonal and above), as produced
// by Stage 2. The inverse of (I - L) where L is strict lower triangular is
// computed via forward substitution: row by row, accumulate contributions
// from previously-computed rows.
//
// Algorithm (matches prepare_wy_repr_fwd_kernel_chunk32 from FLA reference):
//   1. Load A_ab into shared memory, enforce strict lower triangular
//   2. For i = 1..C-1 (forward substitution):
//      - Extract row i
//      - row[i] += sum(row[i][m] * row[m][j] for m < i)
//      - Write updated row back
//   3. Add identity: A[k][k] += 1.0
//   4. Store to A_ab_inv
//
// Input A_ab:  [C, C, H, total_chunks] column-major (i fastest)
// Output A_ab_inv: [C, C, H, total_chunks] column-major
//
// Grid:  dim3(total_chunks, H, 1)
// Block: dim3(C=16) (one thread per column j of the CxC matrix)
// ============================================================================

template <int C_TPL>
__global__ __launch_bounds__(64, 8)
void kernel_fla_wy_inv(
    const float* __restrict__ A_ab,       // [C, C, H, total_chunks]
    float* __restrict__ A_ab_inv,         // [C, C, H, total_chunks]
    int H                                 // n_heads
) {
    const int C = C_TPL;
    const int chunk_flat = blockIdx.x;    // which chunk
    const int h = blockIdx.y;             // which head
    const int j = threadIdx.x;            // column index (0..C-1)

    if (h >= H || j >= C) return;

    // Base offset into the CxC matrix for this (h, chunk) pair:
    // A[i, j, h, chunk] = i + j*C + h*C*C + chunk*H*C*C
    const int mat_base = h * C * C + chunk_flat * H * C * C;

    // Shared memory for the CxC matrix (dynamic size via template).
    // Layout: smem[row * C + col] — row i, column j.
    __shared__ float smem[C_TPL * C_TPL];

    // Step 1: Load column j of A_ab into shared memory.
    // In column-major A[i, j] = A_ab[mat_base + i + j*C], so we read
    // elements with i = 0..C-1 for our column j.
    // Enforce strict lower triangular: A[i,j] = 0 unless i > j.
    for (int i = 0; i < C; i++) {
        float val = 0.0f;
        if (i > j) {
            val = A_ab[mat_base + i + j * C];
        }
        smem[i * C + j] = val;
    }
    __syncthreads();

    // Step 2: Forward substitution, row by row.
    for (int i = 1; i < C; i++) {
        float b_a_j = smem[i * C + j];  // Current value at (i, j)

        if (j < i) {
            float accum = 0.0f;
            for (int m = 0; m < i; m++) {
                accum += smem[i * C + m] * smem[m * C + j];
            }
            b_a_j += accum;
        }

        __syncthreads();

        // Write the updated row i back to shared memory
        smem[i * C + j] = b_a_j;

        __syncthreads();
    }

    // Step 3: Add identity: A[j][j] += 1.0
    smem[j * C + j] += 1.0f;
    __syncthreads();

    // Step 4: Store the entire matrix column j to A_ab_inv
    for (int i = 0; i < C; i++) {
        A_ab_inv[mat_base + i + j * C] = smem[i * C + j];
    }
}

// ============================================================================
// Stage 3, Part B: Compute w and u (kernel_fla_wy_wu)
//
// Using A_ab_inv (from Part A) and A_ak (from Stage 2), plus ag and v:
//
//   compound = A_ab_inv @ A_ak           [C, C] @ [C, C] = [C, C]
//   w[chunk_pos, k] = A_ab_inv @ ag      [C, C] @ [C, K] = [C, K]
//   u[chunk_pos, k] = compound @ v       [C, C] @ [C, K] = [C, K]
//
// A_ab_inv is lower triangular, A_ak is strict lower triangular.
// compound = A_ab_inv @ A_ak is also lower triangular (strict, actually).
//
// ag is f32 [K, H, T, B] (from Stage 2 gating).
// v is f16 [K, H, T, B] (original value tensor).
// w_wy and u_wy are f32 [K, H, T, B] outputs.
//
// PARALLELISM STRATEGY: Block=K=64, sequential over C=16 rows.
//
// Each thread kk (0..K-1=63) owns one column of the K dimension.
// The two CxC matrices (A_ab_inv and compound) are loaded into LDS once,
// then for each output row i (0..C-1):
//   a. Broadcast inv_coeff[j] and compound_coeff[j] from LDS (same for all threads)
//   b. Each thread kk loads ag[j, kk] and v[j, kk] -- coalesced!
//   c. Accumulate w_val += inv_coeff * ag_val, u_val += compound_coeff * v_val
//   d. Store w_wy[i, kk] and u_wy[i, kk] -- coalesced
//
// Benefits over the old block=C=16 design:
//   - 64 threads = 2 full wave32 wavefronts (was 16 = half a wave, 50% SIMD waste)
//   - All memory accesses coalesced (consecutive threads access consecutive K elements)
//   - No per-thread serial K loop (was K=64 serial FMA per thread)
//
// Grid:  dim3(total_chunks, H, 1)
// Block: dim3(K=64) (one thread per K-dimension element)
// ============================================================================


template <int C>
__global__ __launch_bounds__(64, 8)
void kernel_fla_wy_wu(
    const float*  __restrict__ A_ab_inv,        // [C, C, H, total_chunks]
    const float*  __restrict__ A_ak,            // [C, C, H, total_chunks]
    const float*  __restrict__ ag,              // [K, H, T_packed, 1] f32
    const __half* __restrict__ v,               // [K, H, T_packed, 1] f16
    float* __restrict__ w_wy,                   // [K, H, T_packed, 1] f32 output
    float* __restrict__ u_wy,                   // [K, H, T_packed, 1] f32 output
    const int* __restrict__ chunk_indices,      // [total_chunks * 2]
    const int* __restrict__ cu_seqlens,         // [N+1]
    const int* __restrict__ batch_offsets,      // [N] data offsets per sequence
    int K_dim,                                  // head_size (64)
    int H                                       // n_heads
) {
    const int chunk_flat = blockIdx.x;
    const int i_h = blockIdx.y;            // which head (0..H-1)
    const int kk = threadIdx.x;            // K-dimension element (0..K-1)

    if (i_h >= H || kk >= K_dim) return;

    // Resolve chunk -> sequence mapping
    const int seq_id      = chunk_indices[chunk_flat * 2];
    const int local_chunk = chunk_indices[chunk_flat * 2 + 1];

    // Sequence boundaries
    const int bos = batch_offsets[seq_id];
    const int T   = cu_seqlens[seq_id + 1] - cu_seqlens[seq_id];

    // Token stride in [K, H, T, B] layout
    const int token_stride = H * K_dim;

    // Chunk boundaries
    const int chunk_start = local_chunk * C;
    const int valid_len = min(C, T - chunk_start);

    // Early exit: entire chunk beyond sequence
    if (chunk_start >= T) return;

    // Matrix base: [C, C, H, total_chunks], element (i, j, h, chunk)
    const int mat_base = i_h * C * C + chunk_flat * H * C * C;

    // Shared memory for A_ab_inv and compound matrices (16x16 each)
    // Plus one row of coefficients for broadcast (16 inv + 16 compound = 32 floats)
    // Total LDS: 16*16 + 16*16 + 32 = 544 floats = 2176 bytes
    __shared__ float smem_inv[C][C];
    __shared__ float smem_ak[C][C];
    __shared__ float smem_coeffs[2 * C];  // [0..C-1] = inv_row, [C..2C-1] = compound_row

    // Step 1: Cooperatively load A_ab_inv and A_ak into LDS.
    // 64 threads loading 16x16 = 256 elements each: 4 elements per thread.
    // Layout: column-major A[i, j] = A[mat_base + i + j*C]
    // We flatten to linear index and map back.
    {
        const int total_elems = C * C;
        const int elems_per_thread = (total_elems + 63) / 64;
        for (int e = 0; e < elems_per_thread; e++) {
            const int lin = kk + e * 64;
            if (lin < total_elems) {
                const int row = lin % C;  // i (row index)
                const int col = lin / C;  // j (col index)
                // A_ab_inv: lower triangular (i >= j)
                float inv_val = 0.0f;
                if (row >= col) {
                    inv_val = A_ab_inv[mat_base + row + col * C];
                }
                smem_inv[row][col] = inv_val;
                // A_ak: strict lower triangular (i > j)
                float ak_val = 0.0f;
                if (row > col) {
                    ak_val = A_ak[mat_base + row + col * C];
                }
                smem_ak[row][col] = ak_val;
            }
        }
    }
    __syncthreads();

    // Base offset for head h at sequence start
    const int head_base = i_h * K_dim + bos * token_stride;

    // Step 3: Sequential loop over C output rows
    for (int i = 0; i < valid_len; i++) {
        const int i_abs = chunk_start + i;
        const int tok_offset_i = head_base + i_abs * token_stride;

        // Step 3a: Compute and broadcast row i of A_ab_inv and compound.
        // Threads 0..C-1 each compute one element of the compound row.
        if (kk < C) {
            // inv_coeff = smem_inv[i][kk] (already loaded)
                smem_coeffs[kk] = smem_inv[i][kk];

            // compound[i][kk] = sum_m(smem_inv[i][m] * smem_ak[m][kk])
            float compound_val = 0.0f;
            #pragma unroll
            for (int m = 0; m < C; m++) {
                compound_val += smem_inv[i][m] * smem_ak[m][kk];
            }
            smem_coeffs[C + kk] = compound_val;
        }
        __syncthreads();

        // Step 3b: Each thread kk accumulates w_val and u_val across j
        float w_val = 0.0f;
        float u_val = 0.0f;

        #pragma unroll
        for (int j = 0; j < C; j++) {
            if (j < valid_len) {
                const float inv_coeff = smem_coeffs[j];
                const float compound_coeff = smem_coeffs[C + j];

                // Only do work if at least one coefficient is non-zero
                // (triangular structure means many are zero)
                if (inv_coeff != 0.0f || compound_coeff != 0.0f) {
                    const int j_abs = chunk_start + j;
                    const int tok_idx = kk + head_base + j_abs * token_stride;

                    const float ag_val = ag[tok_idx];
                    const float v_val  = __half2float(v[tok_idx]);

                    w_val += inv_coeff * ag_val;
                    u_val += compound_coeff * v_val;
                }
            }
        }

        // Step 3c: Store w_wy and u_wy -- coalesced (thread kk writes element kk)
        w_wy[kk + tok_offset_i] = w_val;
        u_wy[kk + tok_offset_i] = u_val;

        __syncthreads();
    }
}

// ============================================================================
// Host launch wrapper for Stage 3 (WY representation)
// Launches kernel_fla_wy_inv then kernel_fla_wy_wu sequentially.
// ============================================================================

extern "C" {

hipError_t launch_fla_wy_repr(
    const float* A_ab,
    const float* A_ak,
    float* A_ab_inv,
    const float* ag,
    const __half* v,
    float* w_wy,
    float* u_wy,
    const int* chunk_indices,
    const int* cu_seqlens,
    const int* batch_offsets,
    int K,
    int H,
    int C,
    int total_chunks,
    hipStream_t stream
) {
    if (K <= 0 || H <= 0 || C <= 0 || total_chunks <= 0) {
        return hipSuccess;
    }

    // Part A: Compute A_ab_inv (matrix inversion)
    // Block=C threads (one per column of the CxC matrix)
    {
        dim3 grid(total_chunks, H, 1);
        dim3 block(C);

        #define LAUNCH_WY_INV(C_VAL) \
            hipLaunchKernelGGL(kernel_fla_wy_inv<C_VAL>, grid, block, 0, stream, \
                               A_ab, A_ab_inv, H)

        switch (C) {
            case 16: LAUNCH_WY_INV(16); break;
            case 32: LAUNCH_WY_INV(32); break;
            case 64: LAUNCH_WY_INV(64); break;
            default: return hipErrorInvalidValue;
        }

        #undef LAUNCH_WY_INV

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) return err;
    }

    // Part B: Compute w and u (Block=K=64: one thread per K-dimension element)
    {
        dim3 grid(total_chunks, H, 1);
        dim3 block(K);

        #define LAUNCH_WY_WU(C_VAL) \
            hipLaunchKernelGGL(kernel_fla_wy_wu<C_VAL>, grid, block, 0, stream, \
                               A_ab_inv, A_ak, ag, v, w_wy, u_wy, \
                               chunk_indices, cu_seqlens, batch_offsets, \
                               K, H)

        switch (C) {
            case 16: LAUNCH_WY_WU(16); break;
            case 32: LAUNCH_WY_WU(32); break;
            case 64: LAUNCH_WY_WU(64); break;
            default: return hipErrorInvalidValue;
        }

        #undef LAUNCH_WY_WU

        return hipGetLastError();
    }
}

} // extern "C" (pause for kernel definitions)

// ============================================================================
// Stage 4: Inter-chunk state recurrence (kernel_fla_chunk_h)
//
// The sequential bottleneck: propagates state across chunks within each
// sequence.  One thread-block processes ALL chunks for a single
// (head, sequence) pair.
//
// Grid:  dim3(N_seq * H, 1, 1)   — one block per (sequence, head).
// Block: dim3(K=64)               — one thread per V-column of the state.
//
// Thread v (0..K-1) owns one column of the [K, V]==[K, K] state matrix
// in registers (h_col[K]).  Two shared memory buffers (smem_a, smem_b) are
// used in a double-buffered pattern to overlap loads with computation,
// reducing barrier count from 6 to 4 per position.
//
// Per chunk i_t (sequential over chunks for this sequence):
//   1. Store current state to fla_h:
//      fla_h[k, v, h, chunk_offset + i_t] = h_col[k]
//   2. For each sub-chunk position i (0..C-1) inside the chunk:
//      a. Load w_wy -> smem_a, BARRIER
//      b. Dot product from smem_a; load kg -> smem_b, BARRIER
//      c. Accumulate kg from smem_b; load bg -> smem_a, BARRIER
//      d. Accumulate bg from smem_a (no trailing barrier -- next iter's
//         BARRIER at step (a) ensures all smem_a reads complete first)
//   3. Decay and update:
//      g_last = gi at last valid position in chunk
//      h_col[k] = h_col[k] * exp(g_last[k]) + hc_col[k]
//
// Barrier count: 4 per position x C positions + 2 for g_last = 4*16+2 = 66
// (down from 6*16+2 = 98 with single-buffer approach)
//
// After all chunks: write final state to state_out.
//
// Buffer layouts (column-major, K fastest):
//   state:      [K, K, H, B]            — h_col[k] = state[k + v*K + h*K*K + seq*H*K*K]
//   fla_h:      [K, K, H, total_chunks] — same indexing with chunk_id instead of seq
//   per-token:  [K, H, T, B]            — element(k, h, t) = k + h*K + t*H*K
//
// Reference: repos/flash-linear-attention/fla/ops/generalized_delta_rule/dplr/chunk_h_fwd.py
// ============================================================================

__global__ __launch_bounds__(64, 4)
void kernel_fla_chunk_h(
    const float*  __restrict__ kg,             // [K, H, T, B] decay-scaled key (f32)
    const float*  __restrict__ bg,             // [K, H, T, B] decay-scaled b   (f32)
    const __half* __restrict__ v_in,           // [K, H, T, B] original value   (f16)
    const float*  __restrict__ w_wy,           // [K, H, T, B] WY w output      (f32)
    const float*  __restrict__ u_wy,           // [K, H, T, B] WY u output      (f32)
    const float*  __restrict__ gi,             // [K, H, T, B] inclusive cumsum  (f32)
    float*        __restrict__ state,          // [K, K, H, B] state in/out
    float*        __restrict__ h_out,          // [K, K, H, total_chunks] per-chunk states
    float*        __restrict__ v_new,          // [K, H, T, B] corrected values output
    const int*    __restrict__ chunk_offsets,  // [N+1] cumulative chunk counts
    const int*    __restrict__ cu_seqlens,     // [N+1] sequence boundaries
    const int*    __restrict__ batch_offsets,  // [N] data offsets per sequence
    int K_dim,                                 // head_size (64)
    int H,                                     // n_heads
    int C                                      // chunk_size (16)
) {
    // Decode block index -> (sequence, head)
    const int block_id = blockIdx.x;           // 0 .. N_seq*H - 1
    const int i_h      = block_id % H;         // head
    const int i_n      = block_id / H;         // sequence

    const int v_col    = threadIdx.x;           // column index (0..K-1)

    // Sequence boundaries
    const int bos = batch_offsets[i_n];
    const int T   = cu_seqlens[i_n + 1] - cu_seqlens[i_n];
    if (T <= 0) return;

    // Chunk range for this sequence
    const int boh      = chunk_offsets[i_n];    // first chunk id
    const int n_chunks = chunk_offsets[i_n + 1] - boh;

    // Token stride in [K, H, T, B] layout: advance one token
    const int token_stride = H * K_dim;

    // State base offset: state[k, v, h, seq] = k*K + v + h*K*K + seq*H*K*K
    // V (column) is fastest, matching recurrent WKV and Python fixture layout.
    const int state_base = i_h * K_dim * K_dim + i_n * H * K_dim * K_dim;

    // ---- Load initial state column into registers ----
    float h_col[64];
    #pragma unroll
    for (int k = 0; k < 64; k++) {
        h_col[k] = state[k * K_dim + v_col + state_base];
    }

    // Double-buffered shared memory for overlapping loads with computation.
    // smem_a and smem_b alternate roles: while one is being read for computation,
    // the other can be loaded with the next data.
    // Total: 128 floats = 512 bytes (negligible LDS footprint).
    __shared__ float smem_a[64];
    __shared__ float smem_b[64];

    // ---- Sequential loop over chunks ----
    for (int i_t = 0; i_t < n_chunks; i_t++) {
        const int chunk_id    = boh + i_t;
        const int chunk_start = i_t * C;

        // Step 1: Store current state column to fla_h
        // fla_h layout: [K, K, H, total_chunks], V (column) fastest
        // element (k, v, h, chunk_id) = k*K + v + h*K*K + chunk_id*H*K*K
        const int h_out_base = i_h * K_dim * K_dim + chunk_id * H * K_dim * K_dim;
        #pragma unroll
        for (int k = 0; k < 64; k++) {
            h_out[k * K_dim + v_col + h_out_base] = h_col[k];
        }

        // Zero the chunk contribution accumulator
        float hc_col[64];
        #pragma unroll
        for (int k = 0; k < 64; k++) {
            hc_col[k] = 0.0f;
        }

        // Step 2 & 3: Process each position in the chunk using double-buffered smem.
        //
        // Per position, the pattern is:
        //   BARRIER [A]:  protect smem_a from prior Phase 4 reads
        //   Phase 1: Load w_wy -> smem_a
        //   BARRIER [A2]: smem_a ready with w_wy
        //   Phase 2: Dot from smem_a; load kg -> smem_b
        //   BARRIER [B]:  smem_b ready with kg
        //   Phase 3: Accumulate kg from smem_b; load bg -> smem_a
        //   BARRIER [C]:  smem_a ready with bg
        //   Phase 4: Accumulate bg from smem_a (no trailing barrier)
        //
        // Double-buffering eliminates 2 barriers per position vs single-buffer:
        // kg load into smem_b overlaps with smem_a dot product (saves 1 barrier),
        // bg load into smem_a overlaps with smem_b kg accumulation (saves 1 barrier).
        //
        // 4 barriers per position x C=16 = 64 barriers (down from 96 single-buffer).
        const int valid_len = min(C, T - chunk_start);
        for (int i_pos = 0; i_pos < valid_len; i_pos++) {
            const int t_abs = chunk_start + i_pos;    // position within sequence
            const int tok_offset = (bos + t_abs) * token_stride + i_h * K_dim;

            // ---- Phase 1: Load w_wy -> smem_a ----
            // BARRIER [A] ensures previous iteration's Phase 4 reads of smem_a
            // are complete before we overwrite it. For the first position this
            // is a no-op sync (no prior reads to protect).
            __syncthreads();  // BARRIER [A]: protect smem_a from prior Phase 4
            smem_a[v_col] = w_wy[v_col + tok_offset];
            __syncthreads();  // BARRIER [A2]: smem_a ready with w_wy

            // ---- Phase 2: Dot product from smem_a; load kg -> smem_b ----
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < 64; k++) {
                dot += smem_a[k] * h_col[k];
            }

            float b_v2 = dot + u_wy[v_col + tok_offset];

            // Store v_new (global memory write, no smem conflict)
            v_new[v_col + tok_offset] = b_v2;

            // Overlap: load kg into smem_b while dot product was computed from smem_a.
            // smem_a reads for the dot product are done (all threads read smem_a above).
            // BARRIER [A2] already ensured smem_a was fully written before reads.
            // smem_b was last written in the PREVIOUS iteration's Phase 2 (or never).
            // BARRIER [B] of the previous iteration ensured all smem_b reads in Phase 3
            // are done before we get here. But for the first iteration, smem_b is unused.
            smem_b[v_col] = kg[v_col + tok_offset];
            __syncthreads();  // BARRIER [B]: smem_b ready with kg

            // ---- Phase 3: Accumulate kg from smem_b; load bg -> smem_a ----
            float v_val = __half2float(v_in[v_col + tok_offset]);

            #pragma unroll
            for (int k = 0; k < 64; k++) {
                hc_col[k] += smem_b[k] * v_val;
            }

            // Overlap: load bg into smem_a while kg accumulation used smem_b.
            // smem_a was last read in Phase 2 (dot product). BARRIER [B] above
            // ensures all threads completed Phase 2 before reaching here, so
            // smem_a reads are done and we can safely overwrite.
            smem_a[v_col] = bg[v_col + tok_offset];
            __syncthreads();  // BARRIER [C]: smem_a ready with bg

            // ---- Phase 4: Accumulate bg from smem_a ----
            #pragma unroll
            for (int k = 0; k < 64; k++) {
                hc_col[k] += smem_a[k] * b_v2;
            }
            // No trailing barrier needed here: BARRIER [A] at the start of
            // the next iteration protects these smem_a reads.
        }

        // Step 4: Decay and update
        // g_last = gi at last valid position in chunk
        // Barrier protects Phase 4's smem_a reads before overwriting with g_last.
        __syncthreads();  // ensure Phase 4 reads of smem_a are complete
        const int last_idx = min((i_t + 1) * C, T) - 1;
        const int last_tok_offset = (bos + last_idx) * token_stride + i_h * K_dim;
        smem_a[v_col] = gi[v_col + last_tok_offset];
        __syncthreads();  // smem_a ready with g_last

        #pragma unroll
        for (int k = 0; k < 64; k++) {
            h_col[k] = h_col[k] * expf(smem_a[k]) + hc_col[k];
        }
        // No trailing barrier needed: next chunk's h_out store doesn't use smem,
        // and the position loop starts with BARRIER [A] which syncs all threads.
    }

    // ---- Write final state ----
    // state[k, v, h, seq] = k*K + v + h*K*K + seq*H*K*K  (V fastest)
    #pragma unroll
    for (int k = 0; k < 64; k++) {
        state[k * K_dim + v_col + state_base] = h_col[k];
    }
}

// ============================================================================
// Host launch wrapper for Stage 4 (inter-chunk state recurrence)
// ============================================================================

extern "C" {

hipError_t launch_fla_chunk_h(
    const float*  kg,
    const float*  bg,
    const __half* v,
    const float*  w_wy,
    const float*  u_wy,
    const float*  gi,
    float*        state,
    float*        h_out,
    float*        v_new,
    const int*    chunk_offsets,
    const int*    cu_seqlens,
    const int*    batch_offsets,
    int K,
    int H,
    int C,
    int N_seq,
    hipStream_t   stream
) {
    if (K <= 0 || H <= 0 || C <= 0 || N_seq <= 0) {
        return hipSuccess;
    }

    dim3 grid(N_seq * H, 1, 1);
    dim3 block(K);

    hipLaunchKernelGGL(kernel_fla_chunk_h, grid, block, 0, stream,
                       kg, bg, v, w_wy, u_wy, gi,
                       state, h_out, v_new,
                       chunk_offsets, cu_seqlens, batch_offsets,
                       K, H, C);

    return hipGetLastError();
}

} // extern "C" (pause for template kernel)

// ============================================================================
// Stage 5: Output combination (kernel_fla_chunk_o)
//
// Combines intra-chunk attention with inter-chunk state contributions to
// produce the final WKV7 output.
//
// For each token at position i in chunk:
//   o[i, :] = (qg[i, :] @ h[chunk])          -- state contribution via query
//           + sum_j(A_qk[i,j] * v[j, :])      -- local attention (original values)
//           + sum_j(A_qb[i,j] * v_new[j, :])  -- WY correction (corrected values)
//
// Where:
// - qg is the decay-scaled query from Stage 2 [K, H, T, B] (f32)
// - h is the per-chunk state from Stage 4 [K, K, H, total_chunks] (f32)
// - A_qk is the causal QK attention matrix from Stage 2 [C, C, H, total_chunks]
// - A_qb is the causal QB attention matrix from Stage 2 [C, C, H, total_chunks]
// - v is the original value tensor [K, H, T, B] (f16)
// - v_new is the corrected value tensor from Stage 4 [K, H, T, B] (f32)
// - Output o is [K, H, T, B] (f16) matching existing WKV output layout
//
// Both A_qk and A_qb are applied with causal mask (i >= j) before use.
// Positions beyond sequence length produce zero output.
//
// PARALLELISM STRATEGY: Block=K=64, sequential over C=16 positions.
//
// Each thread kk (0..K-1=63) owns one column of the per-chunk state h:
//   h_col[k] = h[k * K + kk]  for k = 0..63
// This column stays in registers for the entire chunk.
//
// For each position i (0..C-1):
//   a. Load qg[i, :] into smem (64 values, coalesced), barrier
//   b. dot = sum(smem[k] * h_col[k])  -- register-only dot, no cross-thread reduction
//   c. Load A_qk[i, j] and A_qb[i, j] for j = 0..C-1 (broadcast via smem)
//   d. local_attn = sum(a_qk[j] * v[j, kk])   -- coalesced v reads across threads
//   e. wy_corr   = sum(a_qb[j] * v_new[j, kk]) -- coalesced v_new reads
//   f. o[kk + tok_offset_i] = __float2half(dot + local_attn + wy_corr)  -- coalesced write
//
// Benefits over the old block=C=16 design:
//   - 64 threads = 2 full wave32 wavefronts (was 16 = half a wave, 50% SIMD waste)
//   - State dot product is register-local (no O(K^2) serial FMA per thread)
//   - Output writes are coalesced (consecutive threads write consecutive addresses)
//   - Reads of v and v_new are coalesced (thread kk reads element kk)
//
// Grid:  dim3(total_chunks, H, 1)
// Block: dim3(K=64) (one thread per V/K-dimension element)
//
// Reference: repos/flash-linear-attention/fla/ops/generalized_delta_rule/dplr/chunk_o_fwd.py
// ============================================================================

template <int C_TPL>
__global__ __launch_bounds__(64, 8)
void kernel_fla_chunk_o(
    const float*  __restrict__ qg,             // [K, H, T_packed, 1] decay-scaled query (f32)
    const __half* __restrict__ v,              // [K, H, T_packed, 1] original value     (f16)
    const float*  __restrict__ v_new,          // [K, H, T_packed, 1] corrected values   (f32)
    const float*  __restrict__ A_qk,           // [C, C, H, total_chunks] causal attn QK
    const float*  __restrict__ A_qb,           // [C, C, H, total_chunks] causal attn QB
    const float*  __restrict__ h,              // [K, K, H, total_chunks] per-chunk states
    __half*       __restrict__ o,              // [K, H, T_packed, 1] output (f16)
    const int*    __restrict__ chunk_indices,  // [total_chunks * 2]
    const int*    __restrict__ cu_seqlens,     // [N+1]
    const int*    __restrict__ batch_offsets,  // [N] data offsets per sequence
    int K_dim,                                 // head_size (64)
    int H                                      // n_heads
) {
    const int C = C_TPL;
    const int chunk_flat = blockIdx.x;     // which chunk (0..total_chunks-1)
    const int i_h = blockIdx.y;            // which head (0..H-1)
    const int kk = threadIdx.x;            // V/K-dimension element (0..K-1)

    if (i_h >= H || kk >= K_dim) return;

    // Resolve chunk -> sequence mapping
    const int seq_id      = chunk_indices[chunk_flat * 2];
    const int local_chunk = chunk_indices[chunk_flat * 2 + 1];

    // Sequence boundaries
    const int bos = batch_offsets[seq_id];
    const int T   = cu_seqlens[seq_id + 1] - cu_seqlens[seq_id];

    // Token stride in [K, H, T, B] layout
    const int token_stride = H * K_dim;

    // Chunk boundaries
    const int chunk_start = local_chunk * C;
    const int valid_len = min(C, T - chunk_start);

    // Early exit: entire chunk beyond sequence
    if (chunk_start >= T) return;

    // ---- Load state column kk into registers ----
    // h layout: [K, K, H, total_chunks], V (column) fastest
    // h[k_row, k_col, h, chunk] = k_row * K + k_col + h * K * K + chunk * H * K * K
    const int h_base = i_h * K_dim * K_dim + chunk_flat * H * K_dim * K_dim;
    float h_col[64];
    #pragma unroll
    for (int k = 0; k < 64; k++) {
        h_col[k] = h[k * K_dim + kk + h_base];
    }

    // Attention matrix base: [C, C, H, total_chunks]
    // A[row_i, col_j, h, chunk] = row_i + col_j * C + h * C * C + chunk * H * C * C
    const int mat_base = i_h * C * C + chunk_flat * H * C * C;

    // Shared memory: used to broadcast per-position data to all K threads.
    // For C>32 we tile j in chunks of TILE to reduce LDS footprint.
    constexpr int TILE = 16;
    // Layout: smem[0..K-1] for qg row, smem[K..K+TILE-1] for A_qk tile,
    //         smem[K+TILE..K+2*TILE-1] for A_qb tile
    __shared__ float smem[64 + 2 * TILE];

    // Base offset for head h, at sequence start
    const int head_base = i_h * K_dim + bos * token_stride;

    // ---- Sequential loop over C positions in this chunk ----
    for (int i = 0; i < valid_len; i++) {
        const int i_abs = chunk_start + i;
        const int tok_offset_i = head_base + i_abs * token_stride;

        // Step (a): Load qg[i, :] into smem (one element per thread, coalesced)
        if (kk < K_dim) {
            smem[kk] = qg[kk + tok_offset_i];
        }
        __syncthreads();

        // Step (b): dot = sum(qg[i, k] * h_col[k]) for k = 0..63
        // This is entirely register-local: no cross-thread reduction needed.
        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < 64; k++) {
            dot += smem[k] * h_col[k];
        }

        // Step (d,e): tiled accumulation over j to reduce LDS footprint for large C.
        float local_attn = 0.0f;
        float wy_corr = 0.0f;
        for (int j0 = 0; j0 < valid_len; j0 += TILE) {
            const int tile = min(TILE, valid_len - j0);

            // Load tile of A_qk / A_qb into shared memory (threads kk < tile)
            if (kk < tile) {
                const int j_idx = j0 + kk;
                float aqk_val = 0.0f;
                float aqb_val = 0.0f;
                if (j_idx <= i && j_idx < valid_len) {
                    aqk_val = A_qk[mat_base + i + j_idx * C];
                    aqb_val = A_qb[mat_base + i + j_idx * C];
                }
                smem[64 + kk] = aqk_val;
                smem[64 + TILE + kk] = aqb_val;
            }
            __syncthreads();

            for (int tj = 0; tj < tile; tj++) {
                const int j_abs = chunk_start + j0 + tj;
                const float aqk = smem[64 + tj];
                const float aqb = smem[64 + TILE + tj];
                if (aqk != 0.0f) {
                    const float v_val = __half2float(v[kk + head_base + j_abs * token_stride]);
                    local_attn += aqk * v_val;
                }
                if (aqb != 0.0f) {
                    const float vn_val = v_new[kk + head_base + j_abs * token_stride];
                    wy_corr += aqb * vn_val;
                }
            }
            __syncthreads();
        }

        // Step (f): Combine and store as f16 — coalesced write
        const float result = dot + local_attn + wy_corr;
        o[kk + tok_offset_i] = __float2half(result);

        __syncthreads();
    }
}

// ============================================================================
// Host launch wrapper for Stage 5 (output combination)
// ============================================================================

extern "C" {

hipError_t launch_fla_chunk_o(
    const float*  qg,
    const __half* v,
    const float*  v_new,
    const float*  A_qk,
    const float*  A_qb,
    const float*  h,
    __half*       o,
    const int*    chunk_indices,
    const int*    cu_seqlens,
    const int*    batch_offsets,
    int K,
    int H,
    int C,
    int total_chunks,
    hipStream_t   stream
) {
    if (K <= 0 || H <= 0 || C <= 0 || total_chunks <= 0) {
        return hipSuccess;
    }

    dim3 grid(total_chunks, H, 1);
    dim3 block(K);  // Block=K=64: one thread per V/K-dimension element

    #define LAUNCH_CHUNK_O(C_VAL) \
        hipLaunchKernelGGL(kernel_fla_chunk_o<C_VAL>, grid, block, 0, stream, \
                           qg, v, v_new, A_qk, A_qb, h, o, \
                           chunk_indices, cu_seqlens, batch_offsets, \
                           K, H)

    switch (C) {
        case 16: LAUNCH_CHUNK_O(16); break;
        case 32: LAUNCH_CHUNK_O(32); break;
        case 64: LAUNCH_CHUNK_O(64); break;
        default: return hipErrorInvalidValue;
    }

    #undef LAUNCH_CHUNK_O
    return hipGetLastError();
}

// ============================================================================
// Utility: f16 raw log-decay to f32 negative exponential (kernel_fla_neg_exp_f16_to_f32)
//
// Converts att_w (f16, raw log-domain decay = -softplus(...) - 0.5) to
// gk (f32) = -exp(att_w).
//
// This avoids the precision-losing round-trip through f16 exp(-exp(w))
// then log. The raw att_w is passed directly from the decay computation
// in step.rs, before the exp(-exp(w)) conversion.
// ============================================================================

__global__ __launch_bounds__(256, 4)
void kernel_fla_neg_exp_f16_to_f32(
    const __half* __restrict__ att_w,     // [N] f16 raw log-domain decay (w = -softplus(...) - 0.5)
    float* __restrict__ gk,               // [N] f32 output: gk = -exp(w)
    int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        gk[idx] = -expf(__half2float(att_w[idx]));
    }
}

hipError_t launch_fla_neg_exp_f16_to_f32(
    const __half* att_w,
    float* gk,
    int N,
    hipStream_t stream
) {
    if (N <= 0) return hipSuccess;

    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;

    hipLaunchKernelGGL(kernel_fla_neg_exp_f16_to_f32, dim3(grid_size), dim3(block_size), 0, stream,
                       att_w, gk, N);

    return hipGetLastError();
}

// ============================================================================
// State transpose kernel (kernel_state_transpose)
//
// Transposes each 64x64 WKV state block between FLA layout [K_row, V_col]
// and decode layout [V_row, K_col], or vice versa.
//
// Out-of-place: reads from src, writes to dst.
//
// State layout: [K, K, H, B] where each (h, b) pair has one K×K matrix.
// FLA stores state[k * K + v] (K-rows, V-cols).
// Decode reads state[v * K + k] (V-rows, K-cols).
// Transposing converts between these two layouts.
//
// One block handles one K×K matrix (one head of one batch item).
// Block dim: dim3(64, 16) = 1024 threads.
// Thread (tx, ty) handles column tx, rows ty*4 .. ty*4+3 (4 elements each).
//
// Uses shared memory with +1 padding on the inner dimension to avoid bank
// conflicts during the transposed write-back.
//
// Grid:  dim3(num_heads * batch_size, 1, 1)
// Block: dim3(64, 16, 1)
// ============================================================================

__global__ __launch_bounds__(1024, 1)
void kernel_state_transpose(
    const float* __restrict__ src,   // [K, K, H, B] source state
    float* __restrict__ dst,         // [K, K, H, B] destination state (transposed)
    int K,                           // matrix dimension (64)
    int num_heads,                   // H
    int batch_size                   // B
) {
    // Each block handles one K×K matrix for a specific (head, batch) pair.
    const int mat_idx = blockIdx.x;  // 0 .. H*B - 1
    if (mat_idx >= num_heads * batch_size) return;

    const int tx = threadIdx.x;  // column index (0..K-1)
    const int ty = threadIdx.y;  // row group index (0..15)

    // Base offset for this matrix in the flattened [K, K, H, B] array.
    // Each matrix has K*K elements, matrices are packed as (h, b) pairs:
    //   matrix (h, b) starts at offset (h + b * H) * K * K
    const int mat_offset = mat_idx * K * K;

    // Shared memory with +1 padding for bank conflict avoidance.
    // smem[row][col+1_padding] — 64 × 65 floats = 16.25 KB.
    __shared__ float smem[64][65];

    // Step 1: Load src into shared memory (coalesced reads).
    // src[row * K + col] for this matrix → smem[row][col]
    // Each thread handles 4 rows: ty*4 .. ty*4+3
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int row = ty * 4 + r;
        if (row < K && tx < K) {
            smem[row][tx] = src[mat_offset + row * K + tx];
        }
    }

    __syncthreads();

    // Step 2: Write transposed from shared memory to dst (coalesced writes).
    // dst[col * K + row] = smem[row][col]
    // Equivalently: dst[tx * K + row] = smem[row][tx]
    // But for coalesced writes, we want threads in a warp to write consecutive addresses.
    // Thread (tx, ty) writes dst[row * K + tx] = smem[tx][row] (transposed read from smem).
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int row = ty * 4 + r;
        if (row < K && tx < K) {
            dst[mat_offset + row * K + tx] = smem[tx][row];
        }
    }
}

// ============================================================================
// Host launch wrapper for kernel_state_transpose
// ============================================================================

hipError_t launch_state_transpose(
    const float* src,
    float* dst,
    int K,
    int num_heads,
    int batch_size,
    hipStream_t stream
) {
    if (K <= 0 || num_heads <= 0 || batch_size <= 0) {
        return hipSuccess;
    }

    dim3 grid(num_heads * batch_size, 1, 1);
    dim3 block(64, 16, 1);

    hipLaunchKernelGGL(kernel_state_transpose, grid, block, 0, stream,
                       src, dst, K, num_heads, batch_size);

    return hipGetLastError();
}

} // extern "C"
