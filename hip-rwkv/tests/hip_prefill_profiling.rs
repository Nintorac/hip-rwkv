//! HIP prefill profiling (FLA) with selectable chunk sizes.
//! Run with (GPU + model required):
//!   WEB_RWKV_HIP_PROF=1 WEB_RWKV_FLA_CHUNK_SIZE=16 cargo test --release --features hip \
//!       --test hip_prefill_profiling -- --ignored b1_s128_c16
//! You can target a specific parameterization by filtering on the generated
//! test name suffix (e.g., `b4_s256_c32`).

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use test_case::test_case;

use hip_rwkv::hip::{HipRuntime, HipRuntimeConfig, Rwkv7Hip};

const DEFAULT_MODEL_PATH: &str = "/workspace/models/rwkv7-g1a-0.1b-20250728-ctx4096.st";

fn current_model_path() -> &'static str {
    static mut OVERRIDE: Option<&'static str> = None;
    unsafe {
        if let Some(p) = OVERRIDE {
            return p;
        }
        if let Ok(val) = std::env::var("WEB_RWKV_MODEL") {
            OVERRIDE = Some(Box::leak(val.into_boxed_str()));
        } else {
            OVERRIDE = Some(DEFAULT_MODEL_PATH);
        }
        OVERRIDE.unwrap()
    }
}

fn model_exists() -> bool {
    Path::new(current_model_path()).exists()
}

#[test_case(1, 256, 16; "b1_s256_c16")]
#[test_case(1, 256, 32; "b1_s256_c32")]
#[test_case(1, 256, 64; "b1_s256_c64")]
#[test_case(4, 256, 16; "b4_s256_c16")]
#[test_case(256, 256, 16; "b256_s256_c16")]
#[test_case(256, 256, 32; "b256_s256_c32")]
#[test_case(256, 256, 64; "b256_s256_c64")]
// Balanced “8 chunks” sweep: keep ~8 chunks per seq for fair per-chunk comparison.
#[test_case(1, 128, 16; "b1_s128_c16_8chunks")]
#[test_case(1, 256, 32; "b1_s256_c32_8chunks")]
#[test_case(1, 512, 64; "b1_s512_c64_8chunks")]
#[ignore = "requires model file and GPU"]
fn prefill_profile(batch_size: usize, seq_len: usize, chunk_size: usize) -> Result<()> {
    if !model_exists() {
        eprintln!("Skipping: model not found at {}", current_model_path());
        return Ok(());
    }

    // Drive chunk size via env so the runtime picks up the template instantiation.
    std::env::set_var("WEB_RWKV_FLA_CHUNK_SIZE", chunk_size.to_string());

    // Build config sized for total packed tokens (batch * seq_len) to avoid the
    // max_prefill_chunk guard tripping on large batches.
    let t_total = batch_size * seq_len;
    let config = HipRuntimeConfig::new(t_total, batch_size);
    let model = Rwkv7Hip::load(current_model_path())?;
    let runtime = HipRuntime::with_config(model, config)?;

    // Deterministic tokens: 1..=seq_len for each batch slot.
    let batch_tokens: Vec<Vec<u32>> = (0..batch_size)
        .map(|b| (0..seq_len).map(|i| 1 + ((b + i) % 1000) as u32).collect())
        .collect();
    let seq_refs: Vec<&[u32]> = batch_tokens.iter().map(|v| v.as_slice()).collect();

    // Warmup one prefill to stabilize clocks and allocator.
    let _ = runtime.infer(&seq_refs)?;
    runtime.reset_state();

    let start = Instant::now();
    let _ = runtime.infer(&seq_refs)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let total_tokens = batch_size * seq_len;
    let tok_per_s = total_tokens as f64 / (elapsed_ms / 1000.0);

    eprintln!(
        "[prefill] batch={} seq_len={} chunk={} tokens={} time_ms={:.2} tok/s={:.1}",
        batch_size, seq_len, chunk_size, total_tokens, elapsed_ms, tok_per_s
    );

    Ok(())
}
