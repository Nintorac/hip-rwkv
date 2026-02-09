//! Integration test runner harness for RWKV benchmarks.
//!
//! This module provides a config-driven benchmark runner that executes performance
//! sweeps across model, backend, batch size, and sequence length dimensions.
//!
//! # Invocation
//!
//! The benchmark tests are marked `#[ignore]` so they only run on demand:
//!
//! ```bash
//! # Run with default config (benchmarks/config.yaml) and smoke profile
//! cargo test --release --test benchmarks -- --ignored --nocapture
//!
//! # Run with custom config and profile
//! WEB_RWKV_BENCH_CONFIG=benchmarks/config.yaml WEB_RWKV_BENCH_PROFILE=dev \
//!     cargo test --release --test benchmarks -- --ignored --nocapture
//!
//! # Run with HIP backend support
//! WEB_RWKV_BENCH_CONFIG=benchmarks/config.yaml WEB_RWKV_BENCH_PROFILE=full \
//!     cargo test --release --features hip --test benchmarks -- --ignored --nocapture
//! ```
//!
//! # Environment Variables
//!
//! - `WEB_RWKV_BENCH_CONFIG`: Path to YAML config file (default: `benchmarks/config.yaml`)
//! - `WEB_RWKV_BENCH_PROFILE`: Profile name to use (default: `smoke`)
//!
//! # Output
//!
//! Results are written to JSONL files in the directory specified by `output.directory`
//! in the config file. Each run creates a new file with the pattern specified by
//! `output.filename_pattern`.

use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use tokio::fs::File as TokioFile;

use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
        v4, v5, v6, v7, Runtime, TokioRuntime,
    },
};

use rwkv_bench::db::{
    CaseRow, DbWriter, DecodeRow, ExpandedCaseInput, ModelRow, PrefillMixedRow, PrefillUniformRow,
    RunRow, SqlSkipConditions,
};
use rwkv_bench::{
    all_mixed_case_ids, collect_run_metadata, generate_case_id, generate_human_name,
    generate_lengths_from_str, generate_run_id, generate_timestamp_utc, round_chunk_size,
    shorten_gpu, CaseIdParams, CaseIdentity, DecodeConfig, DecodeResults, JsonlGpuInfo,
    JsonlHostInfo as HostInfo, JsonlWriter, MeasureRecord, Metrics, MixedTtftTracker,
    PrefillMetrics, PrefillMixedResult, PrefillResult, PrefillUniformConfig, RunHeader, Scenario,
    ScenarioParams, Status, TokenGenerator,
};

/// Default config file path
const DEFAULT_CONFIG_PATH: &str = "benchmarks/config.yaml";

/// Default profile name
const DEFAULT_PROFILE: &str = "smoke";

// =============================================================================
// CONFIG STRUCTS
// =============================================================================

/// Top-level benchmark configuration
#[derive(Debug, Deserialize)]
pub struct BenchConfig {
    /// Schema version for compatibility checking
    pub schema_version: u32,

    /// Named profiles for different benchmark scopes
    pub profiles: HashMap<String, Profile>,

    /// Model definitions
    pub models: Vec<ModelEntry>,

    /// Backend definitions
    pub backends: Vec<BackendEntry>,

    /// Scenario definitions
    pub scenarios: HashMap<String, ScenarioConfig>,

    /// Mixed case definitions for prefill_mixed scenario
    #[serde(default)]
    pub mixed_cases: HashMap<String, MixedCase>,

    /// Sweep definitions
    pub sweeps: HashMap<String, SweepConfig>,

    /// Output configuration
    pub output: OutputConfig,

    /// Skip conditions
    #[serde(default)]
    pub skip_conditions: SkipConditions,

    /// Global limits
    #[serde(default)]
    pub limits: Limits,

    /// Shared controls (defaults for all scenarios)
    #[serde(default)]
    pub shared_controls: SharedControls,
}

/// A named profile that specifies which models, backends, and scenarios to run
#[derive(Debug, Deserialize)]
pub struct Profile {
    /// Human-readable description
    #[serde(default)]
    pub description: String,

    /// List of model names to include
    pub models: Vec<String>,

    /// List of backend IDs to include
    pub backends: Vec<String>,

    /// List of scenario names to run
    pub scenarios: Vec<String>,

    /// Batch sizes to sweep
    pub batch_sizes: Vec<u32>,

    /// Token chunk sizes to sweep
    pub token_chunk_sizes: Vec<u32>,

    /// Decode steps (for decode_only scenario)
    #[serde(default)]
    pub decode_steps: Vec<u32>,

    /// Sequence lengths (for prefill scenarios)
    #[serde(default)]
    pub seq_lens: Vec<u32>,

    /// Number of warmup runs
    #[serde(default = "default_warmup_runs")]
    pub warmup_runs: u32,

    /// Number of recorded repeats
    #[serde(default = "default_repeats")]
    pub repeats: u32,
}

fn default_warmup_runs() -> u32 {
    1
}

fn default_repeats() -> u32 {
    5
}

/// Construct the unified backend string from a config BackendEntry.
///
/// For wgpu backends: "wgpu/Vulkan", "wgpu/Metal", "wgpu/Dx12"
/// For hip: "hip"
fn unified_backend_string(backend_entry: &BackendEntry) -> String {
    if backend_entry.backend_id == "hip" {
        "hip".to_string()
    } else {
        // Default to first wgpu_backend variant, or "Vulkan" if none specified
        let variant = backend_entry
            .wgpu_backends
            .first()
            .map(|s| s.as_str())
            .unwrap_or("Vulkan");
        format!("wgpu/{}", variant)
    }
}

/// Model entry defining a model to benchmark
#[derive(Debug, Deserialize, Clone)]
pub struct ModelEntry {
    /// SHA256 hash of the model file (stable identity)
    pub model_id: String,

    /// Human-readable model name
    pub model_name: String,

    /// Size label for grouping (e.g., "9m", "0.1b", "2.9b")
    pub model_size: String,

    /// Path to the safetensors file
    pub path: String,

    /// Tags for filtering and metadata
    #[serde(default)]
    pub tags: HashMap<String, serde_yaml::Value>,

    /// Maximum batch size this model supports
    #[serde(default)]
    pub max_batch_size: Option<u32>,

    /// Maximum token chunk size this model supports
    #[serde(default)]
    pub max_token_chunk_size: Option<u32>,

    /// Whether to skip this model
    #[serde(default)]
    pub skip: bool,
}

/// Backend entry defining a backend to benchmark
#[derive(Debug, Deserialize, Clone)]
pub struct BackendEntry {
    /// Backend identifier ("wgpu" or "hip")
    pub backend_id: String,

    /// WGPU backend variants (for wgpu only)
    #[serde(default)]
    pub wgpu_backends: Vec<String>,

    /// Adapter selection strategy
    #[serde(default)]
    pub adapter_selection: Option<String>,

    /// Whether to skip this backend
    #[serde(default)]
    pub skip: bool,
}

/// Scenario configuration
#[derive(Debug, Deserialize)]
pub struct ScenarioConfig {
    /// Human-readable description
    #[serde(default)]
    pub description: String,

    /// Default parameters for this scenario
    #[serde(default)]
    pub defaults: HashMap<String, serde_yaml::Value>,
}

/// Mixed case definition for prefill_mixed scenario
#[derive(Debug, Deserialize)]
pub struct MixedCase {
    /// Human-readable description
    #[serde(default)]
    pub description: String,

    /// Base pattern divisors (relative to C)
    #[serde(default)]
    pub base_pattern_divisors: Vec<f64>,

    /// Short divisor for bimodal/one_long patterns
    #[serde(default)]
    pub short_divisor: Option<f64>,

    /// Long divisor for bimodal/one_long patterns
    #[serde(default)]
    pub long_divisor: Option<f64>,

    /// Distribution type
    #[serde(default)]
    pub distribution: Option<String>,

    /// Number of long prompts
    #[serde(default)]
    pub long_count: Option<u32>,

    /// Adapt rule
    #[serde(default)]
    pub adapt_rule: Option<String>,

    /// Base lengths for realistic patterns
    #[serde(default)]
    pub base_lengths: Vec<u32>,

    /// Base chunk size for scaling
    #[serde(default)]
    pub base_chunk_size: Option<u32>,

    /// Scale rule
    #[serde(default)]
    pub scale_rule: Option<String>,
}

/// Sweep configuration defining the parameter matrix
#[derive(Debug, Deserialize)]
pub struct SweepConfig {
    /// Models to include
    pub models: Vec<String>,

    /// Backends to include
    pub backends: Vec<String>,

    /// Batch sizes to sweep
    pub batch_sizes: Vec<u32>,

    /// Token chunk sizes to sweep
    pub token_chunk_sizes: Vec<u32>,

    /// Sequence lengths to sweep
    #[serde(default)]
    pub seq_lens: Vec<u32>,

    /// Decode steps to sweep
    #[serde(default)]
    pub decode_steps: Vec<u32>,
}

/// Output configuration
#[derive(Debug, Deserialize)]
pub struct OutputConfig {
    /// Directory for benchmark results
    pub directory: String,

    /// File naming pattern
    pub filename_pattern: String,

    /// Output mode ("create_new" or "append")
    #[serde(default = "default_output_mode")]
    pub mode: String,

    /// Whether to compress output
    #[serde(default)]
    pub compress: bool,

    /// Whether to include run header record
    #[serde(default = "default_true")]
    pub include_run_header: bool,

    /// Whether to pretty-print JSON
    #[serde(default)]
    pub pretty_print: bool,
}

fn default_output_mode() -> String {
    "create_new".to_string()
}

fn default_true() -> bool {
    true
}

/// Skip conditions configuration
#[derive(Debug, Deserialize, Default)]
pub struct SkipConditions {
    /// Skip if batch_size exceeds model's max
    #[serde(default = "default_true")]
    pub skip_batch_exceeds_model_max: bool,

    /// Skip if token_chunk_size exceeds model's max
    #[serde(default = "default_true")]
    pub skip_chunk_exceeds_model_max: bool,

    /// Skip known failure combinations
    #[serde(default = "default_true")]
    pub skip_known_failures: bool,

    /// Skip if OOM is predicted
    #[serde(default)]
    pub skip_oom_predicted: bool,

    /// Custom skip rules
    #[serde(default)]
    pub custom_rules: Vec<CustomSkipRule>,
}

/// Custom skip rule
#[derive(Debug, Deserialize)]
pub struct CustomSkipRule {
    /// Rule name
    pub name: String,

    /// Rule description
    #[serde(default)]
    pub description: String,

    /// Condition expression
    pub condition: String,
}

/// Global limits configuration
#[derive(Debug, Deserialize, Default)]
pub struct Limits {
    /// Maximum total cases to run
    #[serde(default = "default_max_cases")]
    pub max_total_cases: u32,

    /// Maximum runtime in seconds (0 = unlimited)
    #[serde(default)]
    pub max_runtime_seconds: u32,

    /// Stop on first error
    #[serde(default)]
    pub fail_fast: bool,

    /// Maximum errors before aborting
    #[serde(default)]
    pub max_errors: u32,

    /// Per-case timeout in seconds
    #[serde(default = "default_case_timeout")]
    pub case_timeout_seconds: u32,

    /// Maximum memory usage percentage
    #[serde(default = "default_max_memory_percent")]
    pub max_memory_percent: u32,
}

fn default_max_cases() -> u32 {
    10000
}

fn default_case_timeout() -> u32 {
    300
}

fn default_max_memory_percent() -> u32 {
    90
}

/// Shared controls (defaults for all scenarios)
#[derive(Debug, Deserialize, Default)]
pub struct SharedControls {
    /// PRNG seed for reproducibility
    #[serde(default = "default_seed")]
    pub seed: u64,

    /// Warmup configuration
    #[serde(default)]
    pub warmup: WarmupConfig,

    /// Number of recorded repeats
    #[serde(default = "default_repeats_u32")]
    pub repeats: u32,

    /// Timing configuration
    #[serde(default)]
    pub timing: TimingConfig,

    /// Error handling policy
    #[serde(default)]
    pub error_policy: ErrorPolicy,
}

fn default_seed() -> u64 {
    42
}

fn default_repeats_u32() -> u32 {
    5
}

/// Warmup configuration
#[derive(Debug, Deserialize, Default)]
pub struct WarmupConfig {
    /// Number of warmup runs (not recorded)
    #[serde(default = "default_warmup_runs")]
    pub warmup_runs: u32,

    /// Warmup steps per run
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: u32,
}

fn default_warmup_steps() -> u32 {
    64
}

/// Timing configuration
#[derive(Debug, Deserialize, Default)]
pub struct TimingConfig {
    /// Exclude model load time from measurements
    #[serde(default = "default_true")]
    pub exclude_model_load: bool,

    /// Settle delay between repeats (milliseconds)
    #[serde(default = "default_settle_delay")]
    pub settle_delay_ms: u32,
}

fn default_settle_delay() -> u32 {
    100
}

/// Error handling policy
#[derive(Debug, Deserialize, Default)]
pub struct ErrorPolicy {
    /// Continue on error (write status=error records)
    #[serde(default = "default_true")]
    pub continue_on_error: bool,

    /// Classify errors in JSONL records
    #[serde(default = "default_true")]
    pub classify_errors: bool,
}

// =============================================================================
// CONFIG LOADING
// =============================================================================

/// Error type for config loading
#[derive(Debug)]
pub enum ConfigError {
    /// File not found
    FileNotFound(String),
    /// Parse error
    ParseError(String),
    /// Profile not found
    ProfileNotFound(String),
    /// Model not found
    ModelNotFound(String),
    /// Backend not found
    BackendNotFound(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::FileNotFound(path) => write!(f, "Config file not found: {}", path),
            ConfigError::ParseError(msg) => write!(f, "Config parse error: {}", msg),
            ConfigError::ProfileNotFound(name) => write!(f, "Profile not found: {}", name),
            ConfigError::ModelNotFound(name) => write!(f, "Model not found: {}", name),
            ConfigError::BackendNotFound(name) => write!(f, "Backend not found: {}", name),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Load benchmark configuration from the specified path
pub fn load_config(path: &str) -> Result<BenchConfig, ConfigError> {
    let path = Path::new(path);
    if !path.exists() {
        return Err(ConfigError::FileNotFound(path.display().to_string()));
    }

    let content = fs::read_to_string(path)
        .map_err(|e| ConfigError::ParseError(format!("Failed to read config: {}", e)))?;

    let config: BenchConfig = serde_yaml::from_str(&content)
        .map_err(|e| ConfigError::ParseError(format!("Failed to parse YAML: {}", e)))?;

    Ok(config)
}

/// Load config from environment variable or default path
pub fn load_config_from_env() -> Result<BenchConfig, ConfigError> {
    let config_path =
        env::var("WEB_RWKV_BENCH_CONFIG").unwrap_or_else(|_| DEFAULT_CONFIG_PATH.to_string());

    println!("[bench] Loading config from: {}", config_path);
    load_config(&config_path)
}

/// Get the selected profile name from environment variable or default
pub fn get_profile_name() -> String {
    env::var("WEB_RWKV_BENCH_PROFILE").unwrap_or_else(|_| DEFAULT_PROFILE.to_string())
}

/// Get a profile from the config by name
pub fn get_profile<'a>(config: &'a BenchConfig, name: &str) -> Result<&'a Profile, ConfigError> {
    config
        .profiles
        .get(name)
        .ok_or_else(|| ConfigError::ProfileNotFound(name.to_string()))
}

/// Resolve model entries from a list of model names
pub fn resolve_models(
    config: &BenchConfig,
    names: &[String],
) -> Result<Vec<ModelEntry>, ConfigError> {
    let mut models = Vec::new();
    for name in names {
        let model = config
            .models
            .iter()
            .find(|m| &m.model_name == name)
            .ok_or_else(|| ConfigError::ModelNotFound(name.clone()))?;
        if !model.skip {
            models.push(model.clone());
        }
    }
    Ok(models)
}

/// Resolve backend entries from a list of backend IDs
pub fn resolve_backends(
    config: &BenchConfig,
    ids: &[String],
) -> Result<Vec<BackendEntry>, ConfigError> {
    let mut backends = Vec::new();
    for id in ids {
        let backend = config
            .backends
            .iter()
            .find(|b| &b.backend_id == id)
            .ok_or_else(|| ConfigError::BackendNotFound(id.clone()))?;
        if !backend.skip {
            backends.push(backend.clone());
        }
    }
    Ok(backends)
}

// =============================================================================
// BENCHMARK CASE
// =============================================================================

/// A single benchmark case to execute
#[derive(Debug, Clone)]
pub struct BenchCase {
    /// Model to benchmark
    pub model: ModelEntry,
    /// Backend to use
    pub backend: BackendEntry,
    /// Scenario name
    pub scenario: String,
    /// Batch size
    pub batch_size: u32,
    /// Token chunk size
    pub token_chunk_size: u32,
    /// Sequence length (for prefill scenarios)
    pub seq_len: Option<u32>,
    /// Decode steps (for decode_only scenario)
    pub decode_steps: Option<u32>,
    /// Number of warmup runs
    pub warmup_runs: u32,
    /// Number of recorded repeats
    pub repeats: u32,
}

impl BenchCase {
    /// Generate a stable case ID for this benchmark case
    pub fn case_id(&self) -> String {
        let mut parts = vec![
            self.model.model_name.clone(),
            unified_backend_string(&self.backend),
            self.scenario.clone(),
            format!("b{}", self.batch_size),
            format!("c{}", self.token_chunk_size),
        ];

        if let Some(seq_len) = self.seq_len {
            parts.push(format!("s{}", seq_len));
        }
        if let Some(decode_steps) = self.decode_steps {
            parts.push(format!("d{}", decode_steps));
        }

        parts.join("_")
    }
}

/// Expand a profile into a list of benchmark cases
pub fn expand_profile(
    config: &BenchConfig,
    profile: &Profile,
) -> Result<Vec<BenchCase>, ConfigError> {
    let models = resolve_models(config, &profile.models)?;
    let backends = resolve_backends(config, &profile.backends)?;

    let mut cases = Vec::new();

    for model in &models {
        for backend in &backends {
            for &batch_size in &profile.batch_sizes {
                for &token_chunk_size in &profile.token_chunk_sizes {
                    for scenario in &profile.scenarios {
                        match scenario.as_str() {
                            "decode_only" => {
                                for &decode_steps in &profile.decode_steps {
                                    cases.push(BenchCase {
                                        model: model.clone(),
                                        backend: backend.clone(),
                                        scenario: scenario.clone(),
                                        batch_size,
                                        token_chunk_size,
                                        seq_len: None,
                                        decode_steps: Some(decode_steps),
                                        warmup_runs: profile.warmup_runs,
                                        repeats: profile.repeats,
                                    });
                                }
                            }
                            "prefill_uniform" | "prefill_mixed" => {
                                for &seq_len in &profile.seq_lens {
                                    // HACK: skip cases where seq_len != chunk_size
                                    if seq_len != token_chunk_size {
                                        continue;
                                    }
                                    cases.push(BenchCase {
                                        model: model.clone(),
                                        backend: backend.clone(),
                                        scenario: scenario.clone(),
                                        batch_size,
                                        token_chunk_size,
                                        seq_len: Some(seq_len),
                                        decode_steps: None,
                                        warmup_runs: profile.warmup_runs,
                                        repeats: profile.repeats,
                                    });
                                }
                            }
                            _ => {
                                println!("[bench] Unknown scenario: {}", scenario);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(cases)
}

// =============================================================================
// BENCHMARK EXECUTION
// =============================================================================

/// Loaded model and runtime ready for inference
struct LoadedModel {
    #[allow(dead_code)]
    context: Option<Context>,
    runtime: Box<dyn Runtime<Rnn>>,
    info: ModelInfo,
    vocab_size: u32,
}

/// Persistent HIP model weights that can be reused across runtime configurations.
struct HipWeights {
    weights: std::sync::Arc<hip_rwkv::hip::Rwkv7Model>,
    info: ModelInfo,
    vocab_size: u32,
}

/// Create a wgpu context for the given model info
async fn create_context(info: &ModelInfo) -> anyhow::Result<Context> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

/// Load HIP model weights from disk (expensive, do once per model).
fn load_hip_weights(model_path: &str) -> anyhow::Result<HipWeights> {
    use hip_rwkv::hip::Rwkv7Hip;

    let model =
        Rwkv7Hip::load(model_path).map_err(|e| anyhow::anyhow!("HIP model load failed: {e:?}"))?;
    let weights = model.model();

    let file = std::fs::File::open(model_path)?;
    let data = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&st)?;
    let vocab_size = info.num_vocab as u32;

    Ok(HipWeights {
        weights,
        info,
        vocab_size,
    })
}

/// Create a HIP runtime from persistent weights (cheap, just allocates scratch buffers).
fn create_hip_runtime(
    hip_weights: &HipWeights,
    batch_size: usize,
    token_chunk_size: usize,
) -> anyhow::Result<LoadedModel> {
    use hip_rwkv::hip::{HipRuntime, HipRuntimeConfig};

    let chunk = if token_chunk_size == 0 {
        1
    } else {
        token_chunk_size
    };
    let config = HipRuntimeConfig::new(chunk, batch_size);
    let runtime = HipRuntime::from_model_arc(hip_weights.weights.clone(), config)
        .map_err(|e| anyhow::anyhow!("HIP runtime init failed: {e:?}"))?;

    Ok(LoadedModel {
        context: None,
        runtime: Box::new(runtime),
        info: hip_weights.info.clone(),
        vocab_size: hip_weights.vocab_size,
    })
}

/// Load a model and create runtime (non-HIP backends, or one-shot HIP load).
async fn load_model(
    model_path: &str,
    batch_size: usize,
    token_chunk_size: usize,
    backend_id: &str,
) -> anyhow::Result<LoadedModel> {
    if backend_id == "hip" {
        let hw = load_hip_weights(model_path)?;
        return create_hip_runtime(&hw, batch_size, token_chunk_size);
    }

    let file = TokioFile::open(model_path).await?;
    let data = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&st)?;
    let vocab_size = info.num_vocab as u32;

    let context = create_context(&info).await?;
    let builder = ModelBuilder::new(&context, st);

    let runtime: Box<dyn Runtime<Rnn>> = match info.version {
        ModelVersion::V4 => {
            let model = builder.build_v4().await?;
            let bundle = v4::Bundle::<f16>::new(model, batch_size);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V5 => {
            let model = builder.build_v5().await?;
            let bundle = v5::Bundle::<f16>::new(model, batch_size);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V6 => {
            let model = builder.build_v6().await?;
            let bundle = v6::Bundle::<f16>::new(model, batch_size);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f16>::new(model, batch_size);
            Box::new(TokioRuntime::new(bundle).await)
        }
    };

    Ok(LoadedModel {
        context: Some(context),
        runtime,
        info,
        vocab_size,
    })
}

/// Run a decode-only benchmark case
///
/// This function manually implements the decode scenario loop because we need to
/// run inference asynchronously. The DecodeScenario::run() API expects a sync closure,
/// but our runtime.infer() is async.
async fn run_decode_benchmark(
    loaded: &LoadedModel,
    batch_size: u32,
    token_chunk_size: usize,
    decode_steps: u32,
    warmup_runs: u32,
    repeats: u32,
) -> anyhow::Result<DecodeResults> {
    use rwkv_bench::{DecodeRepeatResult, TokenRng};
    use std::time::{Duration, Instant};

    let vocab_size = loaded.vocab_size;
    let runtime = &loaded.runtime;
    let warmup_steps = 64.min(decode_steps);
    let settle_ms = 50u64;

    let mut rng = TokenRng::new(42);
    let mut results = Vec::with_capacity(repeats as usize);

    // === Warmup Phase ===
    for _ in 0..warmup_runs {
        for _ in 0..warmup_steps {
            let batches: Vec<RnnInputBatch> = (0..batch_size)
                .map(|_| {
                    let token = rng.next_token(vocab_size) as u32;
                    RnnInputBatch::new(vec![token], RnnOption::Last)
                })
                .collect();
            let input = RnnInput::new(batches, token_chunk_size);
            let (_remaining, _output) = runtime
                .infer(input)
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
        }
    }

    // === Measurement Phase ===
    for repeat_idx in 0..repeats {
        // Optional settle delay
        if repeat_idx > 0 {
            tokio::time::sleep(Duration::from_millis(settle_ms)).await;
        }

        // Reset RNG for reproducibility across repeats
        rng = TokenRng::new(42);

        // Pre-generate all tokens for this repeat
        let all_tokens: Vec<Vec<u32>> = (0..decode_steps)
            .map(|_| {
                (0..batch_size)
                    .map(|_| rng.next_token(vocab_size) as u32)
                    .collect()
            })
            .collect();

        // Timed decode loop
        let start = Instant::now();

        for step_tokens in &all_tokens {
            let batches: Vec<RnnInputBatch> = step_tokens
                .iter()
                .map(|&token| RnnInputBatch::new(vec![token], RnnOption::Last))
                .collect();
            let input = RnnInput::new(batches, token_chunk_size);
            let (_remaining, _output) = runtime
                .infer(input)
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
        }

        let elapsed = start.elapsed();
        let decode_total_ms = elapsed.as_secs_f64() * 1000.0;
        let decode_tokens = batch_size * decode_steps;
        let decode_tok_per_s = decode_tokens as f64 / elapsed.as_secs_f64();

        results.push(DecodeRepeatResult {
            repeat_index: repeat_idx,
            decode_total_ms,
            decode_steps,
            decode_tokens,
            decode_tok_per_s,
            step_latencies_ms: None,
        });
    }

    // Calculate aggregate statistics
    let throughputs: Vec<f64> = results.iter().map(|r| r.decode_tok_per_s).collect();
    let (median, mean, min, max) = calculate_stats(&throughputs);

    let config = DecodeConfig {
        decode_steps,
        batch_size,
        seed: 42,
        warmup_runs,
        warmup_steps,
        repeats,
        prime_prefill_len: None,
        settle_ms: Some(settle_ms),
        collect_per_step_latencies: false,
    };

    Ok(DecodeResults {
        config,
        repeats: results,
        median_tok_per_s: median,
        mean_tok_per_s: mean,
        min_tok_per_s: min,
        max_tok_per_s: max,
    })
}

/// Results from a prefill benchmark run across multiple repeats.
#[derive(Debug)]
struct PrefillBenchResults {
    /// Configuration used for this prefill run
    #[allow(dead_code)] // Stored for debugging/future use
    pub config: PrefillUniformConfig,
    /// Results for each repeat
    pub repeats: Vec<PrefillResult>,
    /// Median throughput across repeats
    pub median_tok_per_s: f64,
    /// Mean throughput across repeats
    pub mean_tok_per_s: f64,
    /// Min throughput across repeats
    #[allow(dead_code)] // Stored for debugging/future use
    pub min_tok_per_s: f64,
    /// Max throughput across repeats
    #[allow(dead_code)] // Stored for debugging/future use
    pub max_tok_per_s: f64,
}

/// Run a prefill-uniform benchmark case
///
/// For prefill, we run ONE inference call with `seq_len` tokens per batch slot.
/// This measures how long it takes to process the full prompt.
async fn run_prefill_benchmark(
    loaded: &LoadedModel,
    batch_size: u32,
    token_chunk_size: usize,
    seq_len: u32,
    warmup_runs: u32,
    repeats: u32,
) -> anyhow::Result<PrefillBenchResults> {
    use std::time::{Duration, Instant};

    let vocab_size = loaded.vocab_size;
    let runtime = &loaded.runtime;
    let settle_ms = 50u64;

    // Create config for deterministic token generation
    let config = PrefillUniformConfig::new(batch_size, seq_len, token_chunk_size as u32)
        .with_seed(42)
        .with_vocab_size(vocab_size);

    let mut results = Vec::with_capacity(repeats as usize);

    // === Warmup Phase ===
    for _ in 0..warmup_runs {
        let mut gen = TokenGenerator::new(42, vocab_size);
        let batch_tokens = gen.generate_batch(batch_size, seq_len);

        let batches: Vec<RnnInputBatch> = batch_tokens
            .iter()
            .map(|tokens| {
                let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                RnnInputBatch::new(tokens_u32, RnnOption::Last)
            })
            .collect();

        let input = RnnInput::new(batches, token_chunk_size);
        // Warmup: process just one chunk to prime GPU caches
        let _ = runtime
            .infer(input)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    // === Measurement Phase ===
    for repeat_idx in 0..repeats {
        // Optional settle delay between repeats
        if repeat_idx > 0 {
            tokio::time::sleep(Duration::from_millis(settle_ms)).await;
        }

        // Generate deterministic tokens (same seed each repeat for reproducibility)
        let mut gen = TokenGenerator::new(42, vocab_size);
        let batch_tokens = gen.generate_batch(batch_size, seq_len);

        let batches: Vec<RnnInputBatch> = batch_tokens
            .iter()
            .map(|tokens| {
                let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                RnnInputBatch::new(tokens_u32, RnnOption::Last)
            })
            .collect();

        let mut input = RnnInput::new(batches, token_chunk_size);

        // Timed prefill: loop over all chunks until input is exhausted
        let mut num_infer_calls = 0u32;
        let start = Instant::now();
        while input.num_token() > 0 {
            let (remaining, _) = runtime
                .infer(input)
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            input = remaining;
            num_infer_calls += 1;
        }
        let elapsed = start.elapsed();

        let prefill_total_ms = elapsed.as_secs_f64() * 1000.0;
        let total_prompt_tokens = batch_size * seq_len;
        let prefill_tok_per_s = total_prompt_tokens as f64 / elapsed.as_secs_f64();

        // For uniform prefill, all batches complete at the same time
        let ttft_ms_local = vec![prefill_total_ms; batch_size as usize];

        let result = PrefillResult {
            prefill_total_ms,
            total_prompt_tokens,
            prefill_tok_per_s,
            num_infer_calls,
            ttft_ms_local,
            ttft_min_ms: prefill_total_ms,
            ttft_p50_ms: prefill_total_ms,
            ttft_max_ms: prefill_total_ms,
        };

        results.push(result);
    }

    // Calculate aggregate statistics
    let throughputs: Vec<f64> = results.iter().map(|r| r.prefill_tok_per_s).collect();
    let (median, mean, min, max) = calculate_stats(&throughputs);

    Ok(PrefillBenchResults {
        config,
        repeats: results,
        median_tok_per_s: median,
        mean_tok_per_s: mean,
        min_tok_per_s: min,
        max_tok_per_s: max,
    })
}

/// Results from a prefill-mixed benchmark run across multiple repeats.
#[derive(Debug)]
struct PrefillMixedBenchResults {
    /// Per-batch sequence lengths used
    #[allow(dead_code)]
    pub seq_lens: Vec<u32>,
    /// Results for each repeat
    pub repeats: Vec<PrefillMixedResult>,
    /// Median throughput across repeats
    pub median_tok_per_s: f64,
    /// Mean throughput across repeats
    pub mean_tok_per_s: f64,
    /// Min throughput across repeats
    #[allow(dead_code)]
    pub min_tok_per_s: f64,
    /// Max throughput across repeats
    #[allow(dead_code)]
    pub max_tok_per_s: f64,
}

/// Run a prefill-mixed benchmark case.
///
/// For mixed prefill, each batch element gets a different sequence length
/// determined by the mixed_case_id pattern. We run inference with variable-length
/// token sequences per batch slot.
async fn run_prefill_mixed_benchmark(
    loaded: &LoadedModel,
    batch_size: u32,
    token_chunk_size: usize,
    seq_lens: &[u32],
    warmup_runs: u32,
    repeats: u32,
) -> anyhow::Result<PrefillMixedBenchResults> {
    use std::time::Duration;

    let vocab_size = loaded.vocab_size;
    let runtime = &loaded.runtime;
    let settle_ms = 50u64;

    if seq_lens.len() != batch_size as usize {
        return Err(anyhow::anyhow!(
            "mixed prefill length count ({}) does not match batch size ({})",
            seq_lens.len(),
            batch_size
        ));
    }

    let mut results = Vec::with_capacity(repeats as usize);

    // === Warmup Phase ===
    for _ in 0..warmup_runs {
        let mut gen = TokenGenerator::new(42, vocab_size);
        // Generate per-batch tokens with variable lengths
        let batches: Vec<RnnInputBatch> = seq_lens
            .iter()
            .map(|&len| {
                let tokens = gen.generate(len as usize);
                let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                RnnInputBatch::new(tokens_u32, RnnOption::Last)
            })
            .collect();

        let input = RnnInput::new(batches, token_chunk_size);
        let _ = runtime
            .infer(input)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    // === Measurement Phase ===
    for repeat_idx in 0..repeats {
        if repeat_idx > 0 {
            tokio::time::sleep(Duration::from_millis(settle_ms)).await;
        }

        // Generate deterministic tokens (same seed each repeat for reproducibility)
        let mut gen = TokenGenerator::new(42, vocab_size);
        let batches: Vec<RnnInputBatch> = seq_lens
            .iter()
            .map(|&len| {
                let tokens = gen.generate(len as usize);
                let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                RnnInputBatch::new(tokens_u32, RnnOption::Last)
            })
            .collect();

        let mut input = RnnInput::new(batches, token_chunk_size);
        let mut ttft_tracker = MixedTtftTracker::new(seq_lens.to_vec());
        ttft_tracker.start();

        // Timed prefill: track per-batch first-output time from each chunk's RnnInfo.
        while input.num_token() > 0 {
            let info = input
                .iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("mixed prefill input unexpectedly exhausted"))?;
            let (remaining, _) = runtime
                .infer(input)
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            ttft_tracker.record_infer(|i| info.get(i).map(|b| b.option.is_some()).unwrap_or(false));
            input = remaining;
        }

        results.push(ttft_tracker.finalize());
    }

    // Calculate aggregate statistics
    let throughputs: Vec<f64> = results.iter().map(|r| r.prefill_tok_per_s).collect();
    let (median, mean, min, max) = calculate_stats(&throughputs);

    Ok(PrefillMixedBenchResults {
        seq_lens: seq_lens.to_vec(),
        repeats: results,
        median_tok_per_s: median,
        mean_tok_per_s: mean,
        min_tok_per_s: min,
        max_tok_per_s: max,
    })
}

/// Calculate statistics from a slice of values.
fn calculate_stats(values: &[f64]) -> (f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let sum: f64 = values.iter().sum();
    let mean = sum / values.len() as f64;

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let min = *sorted.first().unwrap();
    let max = *sorted.last().unwrap();

    (median, mean, min, max)
}

/// Get RWKV version string from ModelVersion
fn rwkv_version_str(version: ModelVersion) -> &'static str {
    match version {
        ModelVersion::V4 => "v4",
        ModelVersion::V5 => "v5",
        ModelVersion::V6 => "v6",
        ModelVersion::V7 => "v7",
    }
}

// =============================================================================
// TESTS
// =============================================================================

/// Smoke test: load config, select profile, expand cases, and run benchmarks
#[test]
#[ignore]
fn bench_smoke() {
    // Build a tokio runtime for async operations
    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    rt.block_on(async {
        bench_smoke_async().await;
    });
}

async fn bench_smoke_async() {
    println!("\n=== web-rwkv Benchmark Runner ===\n");

    // Load configuration
    let config = match load_config_from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[bench] Failed to load config: {}", e);
            eprintln!(
                "[bench] Make sure {} exists or set WEB_RWKV_BENCH_CONFIG",
                DEFAULT_CONFIG_PATH
            );
            return;
        }
    };

    println!("[bench] Config schema version: {}", config.schema_version);
    println!(
        "[bench] Available profiles: {:?}",
        config.profiles.keys().collect::<Vec<_>>()
    );
    println!(
        "[bench] Available models: {:?}",
        config
            .models
            .iter()
            .map(|m| &m.model_name)
            .collect::<Vec<_>>()
    );
    println!(
        "[bench] Available backends: {:?}",
        config
            .backends
            .iter()
            .map(|b| &b.backend_id)
            .collect::<Vec<_>>()
    );

    // Get selected profile
    let profile_name = get_profile_name();
    println!("\n[bench] Selected profile: {}", profile_name);

    let profile = match get_profile(&config, &profile_name) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[bench] {}", e);
            return;
        }
    };

    println!("[bench] Profile description: {}", profile.description);
    println!("[bench] Models: {:?}", profile.models);
    println!("[bench] Backends: {:?}", profile.backends);
    println!("[bench] Scenarios: {:?}", profile.scenarios);
    println!("[bench] Batch sizes: {:?}", profile.batch_sizes);
    println!("[bench] Token chunk sizes: {:?}", profile.token_chunk_sizes);

    // Expand to benchmark cases
    let cases = match expand_profile(&config, profile) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[bench] Failed to expand profile: {}", e);
            return;
        }
    };

    println!(
        "\n[bench] Expanded {} total cases (before filtering)",
        cases.len()
    );

    // Create output directory
    let output_dir = Path::new(&config.output.directory);
    if let Err(e) = fs::create_dir_all(output_dir) {
        eprintln!("[bench] Failed to create output directory: {}", e);
        return;
    }

    // Open DuckDB writer early (needed for query_cases filtering)
    let db_path = Path::new(&config.output.directory).join("results.db");
    let mut db = match DbWriter::open(&db_path) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!(
                "[bench] WARNING: Failed to open DuckDB: {}. Continuing with JSONL only.",
                e
            );
            None
        }
    };

    // Convert expanded cases to DB input format
    let case_inputs: Vec<ExpandedCaseInput> = cases
        .iter()
        .map(|c| {
            let backend_str = unified_backend_string(&c.backend);
            ExpandedCaseInput {
                scenario: c.scenario.clone(),
                model_name: c.model.model_name.clone(),
                model_sha: c.model.model_id.clone(),
                backend: backend_str,
                batch_size: c.batch_size as i32,
                token_chunk_size: c.token_chunk_size as i32,
                seq_len: c.seq_len.map(|v| v as i32),
                decode_steps: c.decode_steps.map(|v| v as i32),
                max_batch_size: c.model.max_batch_size.map(|v| v as i32),
                max_token_chunk_size: c.model.max_token_chunk_size.map(|v| v as i32),
                mixed_case_id: None,
            }
        })
        .collect();

    // Convert skip conditions to SQL format
    let sql_skip = SqlSkipConditions {
        skip_batch_exceeds_model_max: config.skip_conditions.skip_batch_exceeds_model_max,
        skip_chunk_exceeds_model_max: config.skip_conditions.skip_chunk_exceeds_model_max,
        custom_rules: config
            .skip_conditions
            .custom_rules
            .iter()
            .map(|r| r.condition.clone())
            .collect(),
    };

    // Filter cases via DuckDB SQL
    let filtered_cases: Vec<CaseRow> = if let Some(ref db) = db {
        match db.query_cases(&case_inputs, &sql_skip) {
            Ok(rows) => rows,
            Err(e) => {
                eprintln!(
                    "[bench] WARNING: query_cases failed: {}. Using unfiltered cases.",
                    e
                );
                case_inputs
                    .iter()
                    .map(|c| CaseRow {
                        scenario: c.scenario.clone(),
                        model_name: c.model_name.clone(),
                        model_sha: c.model_sha.clone(),
                        backend: c.backend.clone(),
                        batch_size: c.batch_size,
                        token_chunk_size: c.token_chunk_size,
                        seq_len: c.seq_len,
                        decode_steps: c.decode_steps,
                        max_batch_size: c.max_batch_size,
                        max_token_chunk_size: c.max_token_chunk_size,
                        mixed_case_id: c.mixed_case_id.clone(),
                    })
                    .collect()
            }
        }
    } else {
        // No DB -- just convert inputs to CaseRow
        case_inputs
            .iter()
            .map(|c| CaseRow {
                scenario: c.scenario.clone(),
                model_name: c.model_name.clone(),
                model_sha: c.model_sha.clone(),
                backend: c.backend.clone(),
                batch_size: c.batch_size,
                token_chunk_size: c.token_chunk_size,
                seq_len: c.seq_len,
                decode_steps: c.decode_steps,
                max_batch_size: c.max_batch_size,
                max_token_chunk_size: c.max_token_chunk_size,
                mixed_case_id: c.mixed_case_id.clone(),
            })
            .collect()
    };

    let decode_rows: Vec<_> = filtered_cases
        .iter()
        .filter(|c| c.scenario == "decode_only")
        .collect();
    let prefill_rows: Vec<_> = filtered_cases
        .iter()
        .filter(|c| c.scenario == "prefill_uniform")
        .collect();
    let mixed_rows: Vec<_> = filtered_cases
        .iter()
        .filter(|c| c.scenario == "prefill_mixed")
        .collect();

    println!("\n[bench] After filtering: {} cases", filtered_cases.len());
    println!("[bench]   decode_only: {}", decode_rows.len());
    println!("[bench]   prefill_uniform: {}", prefill_rows.len());
    println!("[bench]   prefill_mixed: {}", mixed_rows.len());

    if decode_rows.is_empty() && prefill_rows.is_empty() && mixed_rows.is_empty() {
        println!("[bench] No cases to run");
        return;
    }

    // Generate run ID, human name, and collect metadata
    let run_id = generate_run_id();
    let human_name = generate_human_name(&run_id);
    let timestamp = generate_timestamp_utc();
    let metadata = collect_run_metadata(None);
    let gpu_short = shorten_gpu(metadata.host.cpu.as_deref().unwrap_or("unknown"));

    let output_filename = config
        .output
        .filename_pattern
        .replace("{profile}", &profile_name)
        .replace("{timestamp}", &timestamp.replace(":", "").replace("-", ""))
        .replace("{run_id}", &run_id)
        .replace("{name}", &human_name)
        .replace("{gpu}", &gpu_short.replace(' ', "_"));
    let output_path = output_dir.join(&output_filename);

    println!(
        "[bench] Run: {} / {} / {}",
        human_name, profile_name, gpu_short
    );
    println!("[bench] Output file: {}", output_path.display());

    // Create JSONL writer
    let mut writer = match JsonlWriter::create(&output_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("[bench] Failed to create output file: {}", e);
            return;
        }
    };

    let run_header = RunHeader {
        run_id: run_id.clone(),
        started_at_utc: timestamp.clone(),
        git_sha: metadata
            .git
            .sha
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        git_dirty: metadata.git.dirty.unwrap_or(true),
        crate_version: metadata
            .build
            .crate_version
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        rustc_version: metadata
            .build
            .rustc_version
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        host: HostInfo {
            os: metadata
                .host
                .os
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            cpu: metadata
                .host
                .cpu
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            ram_gb: metadata.host.ram_gb.unwrap_or(0.0),
        },
        gpu: JsonlGpuInfo {
            adapter_name: "pending".to_string(),
            backend_api: if profile.backends.len() == 1 {
                profile.backends[0].clone()
            } else {
                "mixed".to_string()
            },
            driver_version: None,
            driver_info: None,
        },
        uname: metadata.host.uname.clone(),
    };

    if let Err(e) = writer.write_run_header(&run_header) {
        eprintln!("[bench] Failed to write run header: {}", e);
        return;
    }
    println!("[bench] Wrote run header");

    // Insert run header into DuckDB
    if let Some(ref mut db) = db {
        if let Err(e) = db.insert_run(&RunRow {
            run_id: run_id.clone(),
            human_name: human_name.clone(),
            profile: profile_name.clone(),
            gpu_short: gpu_short.clone(),
            started_at_utc: timestamp.clone(),
            git_sha: metadata.git.sha.clone().unwrap_or_else(|| "unknown".into()),
            git_dirty: metadata.git.dirty.unwrap_or(true),
            crate_version: metadata
                .build
                .crate_version
                .clone()
                .unwrap_or_else(|| "unknown".into()),
            rustc_version: metadata
                .build
                .rustc_version
                .clone()
                .unwrap_or_else(|| "unknown".into()),
            host_os: metadata.host.os.clone(),
            host_cpu: metadata.host.cpu.clone(),
            host_ram_gb: metadata.host.ram_gb,
            gpu_adapter: None,
            gpu_backend_api: Some(run_header.gpu.backend_api.clone()),
            gpu_driver_ver: None,
            uname: metadata.host.uname.clone(),
        }) {
            eprintln!("[bench] WARNING: Failed to insert DuckDB run: {}", e);
        }
    }

    // Track current model to avoid redundant reloads.
    // For HIP: weights persist across chunk/batch changes, only runtime is recreated.
    let mut current_model_path: Option<String> = None;
    let mut current_batch_size: Option<u32> = None;
    let mut current_backend_id: Option<String> = None;
    let mut current_token_chunk_size: Option<u32> = None;
    let mut loaded_model: Option<LoadedModel> = None;
    let mut hip_weights: Option<HipWeights> = None;

    let mut total_executed = 0;
    let mut total_errors = 0;

    for row in &decode_rows {
        let decode_steps = match row.decode_steps {
            Some(steps) => steps as u32,
            None => {
                println!("[bench] Skipping case without decode_steps");
                continue;
            }
        };

        // Resolve model entry from config
        let model = match config
            .models
            .iter()
            .find(|m| m.model_name == row.model_name)
        {
            Some(m) => m,
            None => {
                eprintln!("[bench] Model not found in config: {}", row.model_name);
                total_errors += 1;
                continue;
            }
        };

        // Resolve backend entry from config
        let backend = match config
            .backends
            .iter()
            .find(|b| unified_backend_string(b) == row.backend)
        {
            Some(b) => b,
            None => {
                eprintln!("[bench] Backend not found in config: {}", row.backend);
                total_errors += 1;
                continue;
            }
        };

        let batch_size = row.batch_size as u32;
        let token_chunk_size = row.token_chunk_size as u32;

        // Check what changed
        let model_changed = current_model_path.as_ref() != Some(&model.path)
            || current_backend_id.as_ref() != Some(&backend.backend_id);
        let config_changed = current_batch_size != Some(batch_size)
            || current_token_chunk_size != Some(token_chunk_size);

        if model_changed || config_changed {
            if !Path::new(&model.path).exists() {
                eprintln!("[bench] Model file not found: {}", model.path);
                total_errors += 1;
                continue;
            }

            if backend.backend_id == "hip" {
                // Only reload weights from disk when model path changes
                if model_changed {
                    println!(
                        "\n[bench] Loading weights: {} (from disk)",
                        model.model_name
                    );
                    match load_hip_weights(&model.path) {
                        Ok(hw) => {
                            println!("[bench] Weights loaded: {:?}", hw.info.version);
                            hip_weights = Some(hw);
                        }
                        Err(e) => {
                            eprintln!("[bench] Failed to load weights: {}", e);
                            total_errors += 1;
                            continue;
                        }
                    }
                }
                // Recreate runtime (cheap) for new batch/chunk config
                let hw = match &hip_weights {
                    Some(hw) => hw,
                    None => {
                        total_errors += 1;
                        continue;
                    }
                };
                println!(
                    "[bench] Creating runtime: batch={}, chunk={}",
                    batch_size, token_chunk_size
                );
                match create_hip_runtime(hw, batch_size as usize, token_chunk_size as usize) {
                    Ok(m) => {
                        loaded_model = Some(m);
                    }
                    Err(e) => {
                        eprintln!("[bench] Failed to create runtime: {}", e);
                        total_errors += 1;
                        continue;
                    }
                }
            } else {
                println!(
                    "\n[bench] Loading model: {} (batch={})",
                    model.model_name, batch_size
                );
                match load_model(
                    &model.path,
                    batch_size as usize,
                    token_chunk_size as usize,
                    &backend.backend_id,
                )
                .await
                {
                    Ok(m) => {
                        println!("[bench] Model loaded: {:?}", m.info.version);
                        loaded_model = Some(m);
                    }
                    Err(e) => {
                        eprintln!("[bench] Failed to load model: {}", e);
                        total_errors += 1;
                        continue;
                    }
                }
            }
            current_model_path = Some(model.path.clone());
            current_batch_size = Some(batch_size);
            current_backend_id = Some(backend.backend_id.clone());
            current_token_chunk_size = Some(token_chunk_size);

            // Upsert model into DuckDB
            if let Some(ref mut db) = db {
                let _ = db.upsert_model(&ModelRow {
                    model_sha: model.model_id.clone(),
                    model_name: model.model_name.clone(),
                    model_size: model.model_size.clone(),
                });
            }
        }

        let loaded = match &loaded_model {
            Some(m) => m,
            None => {
                eprintln!("[bench] No model loaded");
                continue;
            }
        };

        // Generate case_id
        let effective_chunk_size = round_chunk_size(token_chunk_size);
        let backend_str = unified_backend_string(backend);
        let case_id_params = CaseIdParams {
            scenario: Scenario::DecodeOnly,
            model_id: &model.model_name,
            backend: &backend_str,
            batch_size,
            token_chunk_size_effective: effective_chunk_size,
            decode_steps: Some(decode_steps),
            seq_len: None,
            mixed_case_id: None,
        };
        let case_id = generate_case_id(&case_id_params);

        println!(
            "[bench] Running: {} (steps={}, warmup={}, repeats={})",
            case_id, decode_steps, profile.warmup_runs, profile.repeats
        );

        // Run the benchmark
        match run_decode_benchmark(
            loaded,
            batch_size,
            token_chunk_size as usize,
            decode_steps,
            profile.warmup_runs,
            profile.repeats,
        )
        .await
        {
            Ok(results) => {
                println!(
                    "[bench]   Median: {:.1} tok/s, Mean: {:.1} tok/s",
                    results.median_tok_per_s, results.mean_tok_per_s
                );

                // Write measure records for each repeat
                for (repeat_idx, _repeat) in results.repeats.iter().enumerate() {
                    let metrics = results.to_metrics_for_repeat(repeat_idx);

                    let record = MeasureRecord {
                        run_id: run_id.clone(),
                        case_id: case_id.clone(),
                        repeat_index: repeat_idx as u32,
                        scenario: Scenario::DecodeOnly,
                        status: Status::Ok,
                        error_kind: None,
                        error_message: None,
                        case_identity: CaseIdentity {
                            model_id: model.model_id.clone(),
                            model_name: model.model_name.clone(),
                            model_path: model.path.clone(),
                            model_size: model.model_size.clone(),
                            rwkv_version: rwkv_version_str(loaded.info.version).to_string(),
                            backend: backend_str.clone(),
                            batch_size,
                            token_chunk_size_requested: token_chunk_size,
                            token_chunk_size_effective: effective_chunk_size,
                        },
                        scenario_params: ScenarioParams::Decode { decode_steps },
                        metrics: Some(Metrics::Decode(metrics)),
                    };

                    if let Err(e) = writer.write_measure(&record) {
                        eprintln!("[bench] Failed to write measure record: {}", e);
                        total_errors += 1;
                    }
                }

                // Insert decode rows into DuckDB
                if let Some(ref mut db) = db {
                    for (repeat_idx, _repeat) in results.repeats.iter().enumerate() {
                        let metrics = results.to_metrics_for_repeat(repeat_idx);
                        let _ = db.insert_decode(&DecodeRow {
                            run_id: run_id.clone(),
                            case_id: case_id.clone(),
                            repeat_index: repeat_idx as i32,
                            status: "ok".to_string(),
                            model_sha: model.model_id.clone(),
                            backend: backend_str.clone(),
                            batch_size: batch_size as i32,
                            token_chunk_size: effective_chunk_size as i32,
                            decode_steps: decode_steps as i32,
                            decode_total_ms: Some(metrics.decode_total_ms),
                            decode_tokens: Some(metrics.decode_tokens as i32),
                            decode_tok_per_s: Some(metrics.decode_tok_per_s),
                            step_ms_p50: metrics.decode_step_ms_p50,
                            step_ms_p95: metrics.decode_step_ms_p95,
                            error_kind: None,
                            error_message: None,
                        });
                    }
                }

                total_executed += 1;
            }
            Err(e) => {
                eprintln!("[bench]   Error: {}", e);
                total_errors += 1;
            }
        }
    }

    // =========================================================================
    // PREFILL_UNIFORM CASES
    // =========================================================================
    println!("\n--- Running prefill_uniform cases ---\n");

    // Reset model tracking for prefill cases (keep hip_weights if same model)
    current_model_path = None;
    current_batch_size = None;
    current_backend_id = None;
    current_token_chunk_size = None;
    loaded_model = None;
    hip_weights = None;

    for row in &prefill_rows {
        let seq_len = match row.seq_len {
            Some(len) => len as u32,
            None => {
                println!("[bench] Skipping prefill case without seq_len");
                continue;
            }
        };

        // Resolve model entry from config
        let model = match config
            .models
            .iter()
            .find(|m| m.model_name == row.model_name)
        {
            Some(m) => m,
            None => {
                eprintln!("[bench] Model not found in config: {}", row.model_name);
                total_errors += 1;
                continue;
            }
        };

        // Resolve backend entry from config
        let backend = match config
            .backends
            .iter()
            .find(|b| unified_backend_string(b) == row.backend)
        {
            Some(b) => b,
            None => {
                eprintln!("[bench] Backend not found in config: {}", row.backend);
                total_errors += 1;
                continue;
            }
        };

        let batch_size = row.batch_size as u32;
        let token_chunk_size = row.token_chunk_size as u32;

        // Check what changed
        let model_changed = current_model_path.as_ref() != Some(&model.path)
            || current_backend_id.as_ref() != Some(&backend.backend_id);
        let config_changed = current_batch_size != Some(batch_size)
            || current_token_chunk_size != Some(token_chunk_size);

        if model_changed || config_changed {
            if !Path::new(&model.path).exists() {
                eprintln!("[bench] Model file not found: {}", model.path);
                total_errors += 1;
                continue;
            }

            if backend.backend_id == "hip" {
                // Only reload weights from disk when model path changes
                if model_changed {
                    println!(
                        "\n[bench] Loading weights: {} (from disk)",
                        model.model_name
                    );
                    match load_hip_weights(&model.path) {
                        Ok(hw) => {
                            println!("[bench] Weights loaded: {:?}", hw.info.version);
                            hip_weights = Some(hw);
                        }
                        Err(e) => {
                            eprintln!("[bench] Failed to load weights: {}", e);
                            total_errors += 1;
                            continue;
                        }
                    }
                }
                // Recreate runtime (cheap) for new batch/chunk config
                let hw = match &hip_weights {
                    Some(hw) => hw,
                    None => {
                        total_errors += 1;
                        continue;
                    }
                };
                println!(
                    "[bench] Creating runtime: batch={}, chunk={}",
                    batch_size, token_chunk_size
                );
                match create_hip_runtime(hw, batch_size as usize, token_chunk_size as usize) {
                    Ok(m) => {
                        loaded_model = Some(m);
                    }
                    Err(e) => {
                        eprintln!("[bench] Failed to create runtime: {}", e);
                        total_errors += 1;
                        continue;
                    }
                }
            } else {
                println!(
                    "\n[bench] Loading model: {} (batch={})",
                    model.model_name, batch_size
                );
                match load_model(
                    &model.path,
                    batch_size as usize,
                    token_chunk_size as usize,
                    &backend.backend_id,
                )
                .await
                {
                    Ok(m) => {
                        println!("[bench] Model loaded: {:?}", m.info.version);
                        loaded_model = Some(m);
                    }
                    Err(e) => {
                        eprintln!("[bench] Failed to load model: {}", e);
                        total_errors += 1;
                        continue;
                    }
                }
            }
            current_model_path = Some(model.path.clone());
            current_batch_size = Some(batch_size);
            current_backend_id = Some(backend.backend_id.clone());
            current_token_chunk_size = Some(token_chunk_size);

            // Upsert model into DuckDB
            if let Some(ref mut db) = db {
                let _ = db.upsert_model(&ModelRow {
                    model_sha: model.model_id.clone(),
                    model_name: model.model_name.clone(),
                    model_size: model.model_size.clone(),
                });
            }
        }

        let loaded = match &loaded_model {
            Some(m) => m,
            None => {
                eprintln!("[bench] No model loaded");
                continue;
            }
        };

        // Generate case_id for prefill
        let effective_chunk_size = round_chunk_size(token_chunk_size);
        let backend_str = unified_backend_string(backend);
        let case_id_params = CaseIdParams {
            scenario: Scenario::PrefillUniform,
            model_id: &model.model_name,
            backend: &backend_str,
            batch_size,
            token_chunk_size_effective: effective_chunk_size,
            decode_steps: None,
            seq_len: Some(seq_len),
            mixed_case_id: None,
        };
        let case_id = generate_case_id(&case_id_params);

        println!(
            "[bench] Running: {} (seq_len={}, warmup={}, repeats={})",
            case_id, seq_len, profile.warmup_runs, profile.repeats
        );

        // Run the prefill benchmark
        match run_prefill_benchmark(
            loaded,
            batch_size,
            token_chunk_size as usize,
            seq_len,
            profile.warmup_runs,
            profile.repeats,
        )
        .await
        {
            Ok(results) => {
                println!(
                    "[bench]   Median: {:.1} tok/s, Mean: {:.1} tok/s",
                    results.median_tok_per_s, results.mean_tok_per_s
                );

                // Write measure records for each repeat
                for (repeat_idx, repeat) in results.repeats.iter().enumerate() {
                    let metrics = PrefillMetrics::from(repeat.clone());

                    let record = MeasureRecord {
                        run_id: run_id.clone(),
                        case_id: case_id.clone(),
                        repeat_index: repeat_idx as u32,
                        scenario: Scenario::PrefillUniform,
                        status: Status::Ok,
                        error_kind: None,
                        error_message: None,
                        case_identity: CaseIdentity {
                            model_id: model.model_id.clone(),
                            model_name: model.model_name.clone(),
                            model_path: model.path.clone(),
                            model_size: model.model_size.clone(),
                            rwkv_version: rwkv_version_str(loaded.info.version).to_string(),
                            backend: backend_str.clone(),
                            batch_size,
                            token_chunk_size_requested: token_chunk_size,
                            token_chunk_size_effective: effective_chunk_size,
                        },
                        scenario_params: ScenarioParams::PrefillUniform { seq_len },
                        metrics: Some(Metrics::Prefill(metrics)),
                    };

                    if let Err(e) = writer.write_measure(&record) {
                        eprintln!("[bench] Failed to write measure record: {}", e);
                        total_errors += 1;
                    }
                }

                // Insert prefill_uniform rows into DuckDB
                if let Some(ref mut db) = db {
                    for (repeat_idx, repeat) in results.repeats.iter().enumerate() {
                        let metrics = PrefillMetrics::from(repeat.clone());
                        let _ = db.insert_prefill_uniform(&PrefillUniformRow {
                            run_id: run_id.clone(),
                            case_id: case_id.clone(),
                            repeat_index: repeat_idx as i32,
                            status: "ok".to_string(),
                            model_sha: model.model_id.clone(),
                            backend: backend_str.clone(),
                            batch_size: batch_size as i32,
                            token_chunk_size: effective_chunk_size as i32,
                            seq_len: seq_len as i32,
                            prefill_total_ms: Some(metrics.prefill_total_ms),
                            total_prompt_tokens: Some(metrics.total_prompt_tokens as i32),
                            prefill_tok_per_s: Some(metrics.prefill_tok_per_s),
                            num_infer_calls: Some(metrics.num_infer_calls as i32),
                            ttft_ms_local: Some(metrics.ttft_ms_local.clone()),
                            ttft_min_ms: Some(metrics.ttft_min_ms),
                            ttft_p50_ms: Some(metrics.ttft_p50_ms),
                            ttft_max_ms: Some(metrics.ttft_max_ms),
                            error_kind: None,
                            error_message: None,
                        });
                    }
                }

                total_executed += 1;
            }
            Err(e) => {
                eprintln!("[bench]   Error: {}", e);
                total_errors += 1;
            }
        }
    }

    // =========================================================================
    // PREFILL_MIXED CASES
    // =========================================================================
    println!("\n--- Running prefill_mixed cases ---\n");

    // Collect mixed_case_ids from config (or fall back to all known patterns)
    let mixed_case_ids: Vec<String> = config
        .scenarios
        .get("prefill_mixed")
        .and_then(|sc| sc.defaults.get("mixed_case_ids"))
        .and_then(|v| {
            v.as_sequence().map(|seq| {
                seq.iter()
                    .filter_map(|item| item.as_str().map(|s| s.to_string()))
                    .collect()
            })
        })
        .unwrap_or_else(|| all_mixed_case_ids().iter().map(|s| s.to_string()).collect());

    // Reset model tracking for mixed cases
    current_model_path = None;
    current_batch_size = None;
    current_backend_id = None;
    current_token_chunk_size = None;
    loaded_model = None;
    hip_weights = None;

    for row in &mixed_rows {
        // Resolve model entry from config
        let model = match config
            .models
            .iter()
            .find(|m| m.model_name == row.model_name)
        {
            Some(m) => m,
            None => {
                eprintln!("[bench] Model not found in config: {}", row.model_name);
                total_errors += 1;
                continue;
            }
        };

        // Resolve backend entry from config
        let backend = match config
            .backends
            .iter()
            .find(|b| unified_backend_string(b) == row.backend)
        {
            Some(b) => b,
            None => {
                eprintln!("[bench] Backend not found in config: {}", row.backend);
                total_errors += 1;
                continue;
            }
        };

        let batch_size = row.batch_size as u32;
        let token_chunk_size = row.token_chunk_size as u32;

        // Check what changed
        let model_changed = current_model_path.as_ref() != Some(&model.path)
            || current_backend_id.as_ref() != Some(&backend.backend_id);
        let config_changed = current_batch_size != Some(batch_size)
            || current_token_chunk_size != Some(token_chunk_size);

        if model_changed || config_changed {
            if !Path::new(&model.path).exists() {
                eprintln!("[bench] Model file not found: {}", model.path);
                total_errors += 1;
                continue;
            }

            if backend.backend_id == "hip" {
                if model_changed {
                    println!(
                        "\n[bench] Loading weights: {} (from disk)",
                        model.model_name
                    );
                    match load_hip_weights(&model.path) {
                        Ok(hw) => {
                            println!("[bench] Weights loaded: {:?}", hw.info.version);
                            hip_weights = Some(hw);
                        }
                        Err(e) => {
                            eprintln!("[bench] Failed to load weights: {}", e);
                            total_errors += 1;
                            continue;
                        }
                    }
                }
                let hw = match &hip_weights {
                    Some(hw) => hw,
                    None => {
                        total_errors += 1;
                        continue;
                    }
                };
                println!(
                    "[bench] Creating runtime: batch={}, chunk={}",
                    batch_size, token_chunk_size
                );
                match create_hip_runtime(hw, batch_size as usize, token_chunk_size as usize) {
                    Ok(m) => {
                        loaded_model = Some(m);
                    }
                    Err(e) => {
                        eprintln!("[bench] Failed to create runtime: {}", e);
                        total_errors += 1;
                        continue;
                    }
                }
            } else {
                println!(
                    "\n[bench] Loading model: {} (batch={})",
                    model.model_name, batch_size
                );
                match load_model(
                    &model.path,
                    batch_size as usize,
                    token_chunk_size as usize,
                    &backend.backend_id,
                )
                .await
                {
                    Ok(m) => {
                        println!("[bench] Model loaded: {:?}", m.info.version);
                        loaded_model = Some(m);
                    }
                    Err(e) => {
                        eprintln!("[bench] Failed to load model: {}", e);
                        total_errors += 1;
                        continue;
                    }
                }
            }
            current_model_path = Some(model.path.clone());
            current_batch_size = Some(batch_size);
            current_backend_id = Some(backend.backend_id.clone());
            current_token_chunk_size = Some(token_chunk_size);

            // Upsert model into DuckDB
            if let Some(ref mut db) = db {
                let _ = db.upsert_model(&ModelRow {
                    model_sha: model.model_id.clone(),
                    model_name: model.model_name.clone(),
                    model_size: model.model_size.clone(),
                });
            }
        }

        let loaded = match &loaded_model {
            Some(m) => m,
            None => {
                eprintln!("[bench] No model loaded");
                continue;
            }
        };

        let effective_chunk_size = round_chunk_size(token_chunk_size);
        let backend_str = unified_backend_string(backend);

        // Run each mixed_case_id pattern for this (model, backend, batch_size, chunk_size)
        for mixed_case_id in &mixed_case_ids {
            // Generate per-batch sequence lengths from the mixed pattern
            let seq_lens =
                match generate_lengths_from_str(mixed_case_id, batch_size, token_chunk_size) {
                    Ok(lens) => lens,
                    Err(e) => {
                        eprintln!(
                            "[bench] Failed to generate mixed lengths for {}: {}",
                            mixed_case_id, e
                        );
                        total_errors += 1;
                        continue;
                    }
                };

            // Generate case_id for mixed prefill
            let case_id_params = CaseIdParams {
                scenario: Scenario::PrefillMixed,
                model_id: &model.model_name,
                backend: &backend_str,
                batch_size,
                token_chunk_size_effective: effective_chunk_size,
                decode_steps: None,
                seq_len: None,
                mixed_case_id: Some(mixed_case_id),
            };
            let case_id = generate_case_id(&case_id_params);

            println!(
                "[bench] Running: {} (mixed_case={}, seq_lens={:?}, warmup={}, repeats={})",
                case_id, mixed_case_id, seq_lens, profile.warmup_runs, profile.repeats
            );

            // Run the mixed prefill benchmark
            match run_prefill_mixed_benchmark(
                loaded,
                batch_size,
                token_chunk_size as usize,
                &seq_lens,
                profile.warmup_runs,
                profile.repeats,
            )
            .await
            {
                Ok(results) => {
                    println!(
                        "[bench]   Median: {:.1} tok/s, Mean: {:.1} tok/s",
                        results.median_tok_per_s, results.mean_tok_per_s
                    );

                    // Write JSONL measure records for each repeat
                    for (repeat_idx, repeat) in results.repeats.iter().enumerate() {
                        let metrics = PrefillMetrics::from(repeat.clone());

                        let record = MeasureRecord {
                            run_id: run_id.clone(),
                            case_id: case_id.clone(),
                            repeat_index: repeat_idx as u32,
                            scenario: Scenario::PrefillMixed,
                            status: Status::Ok,
                            error_kind: None,
                            error_message: None,
                            case_identity: CaseIdentity {
                                model_id: model.model_id.clone(),
                                model_name: model.model_name.clone(),
                                model_path: model.path.clone(),
                                model_size: model.model_size.clone(),
                                rwkv_version: rwkv_version_str(loaded.info.version).to_string(),
                                backend: backend_str.clone(),
                                batch_size,
                                token_chunk_size_requested: token_chunk_size,
                                token_chunk_size_effective: effective_chunk_size,
                            },
                            scenario_params: ScenarioParams::PrefillMixed {
                                seq_lens: seq_lens.clone(),
                                mixed_case_id: mixed_case_id.clone(),
                            },
                            metrics: Some(Metrics::Prefill(metrics)),
                        };

                        if let Err(e) = writer.write_measure(&record) {
                            eprintln!("[bench] Failed to write measure record: {}", e);
                            total_errors += 1;
                        }
                    }

                    // Insert prefill_mixed rows into DuckDB
                    if let Some(ref mut db) = db {
                        let seq_lens_i32: Vec<i32> = seq_lens.iter().map(|&v| v as i32).collect();
                        for (repeat_idx, repeat) in results.repeats.iter().enumerate() {
                            let metrics = PrefillMetrics::from(repeat.clone());
                            let _ = db.insert_prefill_mixed(&PrefillMixedRow {
                                run_id: run_id.clone(),
                                case_id: case_id.clone(),
                                repeat_index: repeat_idx as i32,
                                status: "ok".to_string(),
                                model_sha: model.model_id.clone(),
                                backend: backend_str.clone(),
                                batch_size: batch_size as i32,
                                token_chunk_size: effective_chunk_size as i32,
                                mixed_case_id: mixed_case_id.clone(),
                                seq_lens: seq_lens_i32.clone(),
                                prefill_total_ms: Some(metrics.prefill_total_ms),
                                total_prompt_tokens: Some(metrics.total_prompt_tokens as i32),
                                prefill_tok_per_s: Some(metrics.prefill_tok_per_s),
                                num_infer_calls: Some(metrics.num_infer_calls as i32),
                                ttft_ms_local: Some(metrics.ttft_ms_local.clone()),
                                ttft_min_ms: Some(metrics.ttft_min_ms),
                                ttft_p50_ms: Some(metrics.ttft_p50_ms),
                                ttft_max_ms: Some(metrics.ttft_max_ms),
                                error_kind: None,
                                error_message: None,
                            });
                        }
                    }

                    total_executed += 1;
                }
                Err(e) => {
                    eprintln!("[bench]   Error: {}", e);
                    total_errors += 1;
                }
            }
        }
    }

    // Commit DuckDB before flush/summary
    if let Some(ref mut db) = db {
        match db.commit() {
            Ok(()) => println!("[bench] DuckDB committed successfully"),
            Err(e) => eprintln!("[bench] WARNING: DuckDB commit failed: {}", e),
        }
    }

    // Flush and report
    if let Err(e) = writer.flush() {
        eprintln!("[bench] Failed to flush output: {}", e);
    }

    println!("\n=== Benchmark Summary ===");
    println!("[bench] Cases executed: {}", total_executed);
    println!("[bench] Errors: {}", total_errors);
    println!("[bench] Records written: {}", writer.records_written());
    println!("[bench] Output file: {}", output_path.display());
}

/// Test that validates config parsing
#[test]
#[ignore]
fn bench_validate_config() {
    println!("\n=== Config Validation ===\n");

    let config = match load_config_from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[bench] Config validation failed: {}", e);
            panic!("Config validation failed");
        }
    };

    // Validate schema version
    assert!(config.schema_version >= 1, "Schema version must be >= 1");

    // Validate at least one profile exists
    assert!(!config.profiles.is_empty(), "At least one profile required");

    // Validate at least one model exists
    assert!(!config.models.is_empty(), "At least one model required");

    // Validate at least one backend exists
    assert!(!config.backends.is_empty(), "At least one backend required");

    // Validate model paths exist (warning only)
    for model in &config.models {
        let path = Path::new(&model.path);
        if !path.exists() {
            println!(
                "[bench] WARNING: Model file not found: {} ({})",
                model.model_name, model.path
            );
        } else {
            println!("[bench] OK: {} -> {}", model.model_name, model.path);
        }
    }

    // Validate profile references
    for (name, profile) in &config.profiles {
        for model_name in &profile.models {
            let found = config.models.iter().any(|m| &m.model_name == model_name);
            assert!(
                found,
                "Profile '{}' references unknown model: {}",
                name, model_name
            );
        }
        for backend_id in &profile.backends {
            let found = config.backends.iter().any(|b| &b.backend_id == backend_id);
            assert!(
                found,
                "Profile '{}' references unknown backend: {}",
                name, backend_id
            );
        }
    }

    println!("\n[bench] Config validation passed!");
}

/// Test that query_cases() correctly applies skip conditions.
/// Uses an in-memory DuckDB (no file), inserts synthetic cases, and verifies
/// that built-in max checks and custom SQL rules filter correctly.
#[test]
fn test_query_cases_skip_conditions() {
    use rwkv_bench::db::{DbWriter, ExpandedCaseInput, SqlSkipConditions};

    // Open an in-memory DuckDB for testing
    let tmp = std::env::temp_dir().join("test_skip_conditions.db");
    let _ = fs::remove_file(&tmp);
    let db = DbWriter::open(&tmp).expect("Failed to open test DB");

    // Synthetic cases: 2 models × 2 backends × 2 batch sizes = 8 cases
    let cases = vec![
        // model_a, hip, batch=16 — should PASS (under max)
        ExpandedCaseInput {
            scenario: "decode_only".into(),
            model_name: "model_a".into(),
            model_sha: "sha_a".into(),
            backend: "hip".into(),
            batch_size: 16,
            token_chunk_size: 128,
            seq_len: None,
            decode_steps: Some(64),
            max_batch_size: Some(32),
            max_token_chunk_size: Some(256),
            mixed_case_id: None,
        },
        // model_a, hip, batch=64 — should be SKIPPED (batch > max_batch_size=32)
        ExpandedCaseInput {
            scenario: "decode_only".into(),
            model_name: "model_a".into(),
            model_sha: "sha_a".into(),
            backend: "hip".into(),
            batch_size: 64,
            token_chunk_size: 128,
            seq_len: None,
            decode_steps: Some(64),
            max_batch_size: Some(32),
            max_token_chunk_size: Some(256),
            mixed_case_id: None,
        },
        // model_a, wgpu/Vulkan, batch=16 — should PASS
        ExpandedCaseInput {
            scenario: "decode_only".into(),
            model_name: "model_a".into(),
            model_sha: "sha_a".into(),
            backend: "wgpu/Vulkan".into(),
            batch_size: 16,
            token_chunk_size: 128,
            seq_len: None,
            decode_steps: Some(64),
            max_batch_size: Some(32),
            max_token_chunk_size: Some(256),
            mixed_case_id: None,
        },
        // model_a, wgpu/Vulkan, batch=512 — should be SKIPPED by BOTH max AND custom rule
        ExpandedCaseInput {
            scenario: "decode_only".into(),
            model_name: "model_a".into(),
            model_sha: "sha_a".into(),
            backend: "wgpu/Vulkan".into(),
            batch_size: 512,
            token_chunk_size: 128,
            seq_len: None,
            decode_steps: Some(64),
            max_batch_size: Some(32),
            max_token_chunk_size: Some(256),
            mixed_case_id: None,
        },
        // puzzle15, hip, batch=4 — should be SKIPPED by custom rule
        ExpandedCaseInput {
            scenario: "decode_only".into(),
            model_name: "rwkv_puzzle15".into(),
            model_sha: "sha_p".into(),
            backend: "hip".into(),
            batch_size: 4,
            token_chunk_size: 128,
            seq_len: None,
            decode_steps: Some(64),
            max_batch_size: None,
            max_token_chunk_size: None,
            mixed_case_id: None,
        },
        // puzzle15, wgpu/Vulkan, batch=4 — should PASS (custom rule only targets hip)
        ExpandedCaseInput {
            scenario: "decode_only".into(),
            model_name: "rwkv_puzzle15".into(),
            model_sha: "sha_p".into(),
            backend: "wgpu/Vulkan".into(),
            batch_size: 4,
            token_chunk_size: 128,
            seq_len: None,
            decode_steps: Some(64),
            max_batch_size: None,
            max_token_chunk_size: None,
            mixed_case_id: None,
        },
        // model_b, hip, batch=32, seq_len=2048 — should be SKIPPED (batch>16 AND seq_len>1024)
        ExpandedCaseInput {
            scenario: "prefill_uniform".into(),
            model_name: "model_b".into(),
            model_sha: "sha_b".into(),
            backend: "hip".into(),
            batch_size: 32,
            token_chunk_size: 128,
            seq_len: Some(2048),
            decode_steps: None,
            max_batch_size: None,
            max_token_chunk_size: None,
            mixed_case_id: None,
        },
        // model_b, hip, batch=8, seq_len=2048 — should PASS (batch<=16, so rule doesn't apply)
        ExpandedCaseInput {
            scenario: "prefill_uniform".into(),
            model_name: "model_b".into(),
            model_sha: "sha_b".into(),
            backend: "hip".into(),
            batch_size: 8,
            token_chunk_size: 128,
            seq_len: Some(2048),
            decode_steps: None,
            max_batch_size: None,
            max_token_chunk_size: None,
            mixed_case_id: None,
        },
    ];

    let skip = SqlSkipConditions {
        skip_batch_exceeds_model_max: true,
        skip_chunk_exceeds_model_max: true,
        custom_rules: vec![
            "model_name = 'rwkv_puzzle15' AND backend = 'hip'".into(),
            "backend LIKE 'wgpu%' AND batch_size > 256".into(),
            "batch_size > 16 AND seq_len > 1024".into(),
        ],
    };

    let result = db.query_cases(&cases, &skip).expect("query_cases failed");

    // Expected survivors: 4 cases
    // 1. model_a / hip / bs16       (pass)
    // 2. model_a / wgpu / bs16      (pass)
    // 3. puzzle15 / wgpu / bs4      (pass — rule only skips hip)
    // 4. model_b / hip / bs8 / s2048 (pass — batch<=16)
    let names: Vec<String> = result
        .iter()
        .map(|r| format!("{}/{}/bs{}", r.model_name, r.backend, r.batch_size))
        .collect();
    println!("Surviving cases: {:?}", names);

    assert_eq!(
        result.len(),
        4,
        "Expected 4 cases after filtering, got {}: {:?}",
        result.len(),
        names
    );

    // Verify specific cases survived
    assert!(result
        .iter()
        .any(|r| r.model_name == "model_a" && r.backend == "hip" && r.batch_size == 16));
    assert!(result
        .iter()
        .any(|r| r.model_name == "model_a" && r.backend == "wgpu/Vulkan" && r.batch_size == 16));
    assert!(result
        .iter()
        .any(|r| r.model_name == "rwkv_puzzle15" && r.backend == "wgpu/Vulkan"));
    assert!(result
        .iter()
        .any(|r| r.model_name == "model_b" && r.batch_size == 8));

    // Verify specific cases were filtered
    assert!(
        !result
            .iter()
            .any(|r| r.model_name == "model_a" && r.batch_size == 64),
        "batch=64 should be skipped (exceeds max)"
    );
    assert!(
        !result
            .iter()
            .any(|r| r.model_name == "model_a" && r.batch_size == 512),
        "batch=512 should be skipped"
    );
    assert!(
        !result
            .iter()
            .any(|r| r.model_name == "rwkv_puzzle15" && r.backend == "hip"),
        "puzzle15/hip should be skipped"
    );
    assert!(
        !result
            .iter()
            .any(|r| r.model_name == "model_b" && r.batch_size == 32),
        "batch=32 + seq=2048 should be skipped"
    );

    // Cleanup
    let _ = fs::remove_file(&tmp);
    println!("\n[test] All skip conditions validated!");
}
