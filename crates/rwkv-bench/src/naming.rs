//! Human-readable run name generation and GPU name shortening.
//!
//! This module provides:
//! - Deterministic human-readable names from run IDs via SHA-256 hashing
//! - GPU name extraction from host CPU strings
//!
//! # Example
//!
//! ```
//! use rwkv_bench::naming::{generate_human_name, shorten_gpu};
//!
//! let name = generate_human_name("20260208T120000Z_abc123");
//! // Returns something like "keen-cedar" (deterministic for a given run_id)
//!
//! let gpu = shorten_gpu("AMD RYZEN AI MAX+ 395 w/ Radeon 8060S");
//! assert_eq!(gpu, "Radeon 8060S");
//! ```

use sha2::{Digest, Sha256};

/// 64 short, engineering-flavored adjectives.
const ADJECTIVES: [&str; 64] = [
    "able", "bold", "calm", "dark", "eager", "fair", "glad", "hale",
    "iron", "just", "keen", "lean", "mild", "neat", "open", "pale",
    "quick", "rare", "safe", "taut", "used", "vast", "warm", "zero",
    "agile", "brave", "clean", "dense", "exact", "firm", "great", "hard",
    "ideal", "jolly", "known", "level", "major", "noble", "outer", "plain",
    "rapid", "rigid", "sharp", "solid", "tight", "ultra", "vivid", "whole",
    "amber", "blunt", "chief", "dry", "epic", "flat", "grey", "high",
    "inner", "joint", "light", "mint", "new", "odd", "prime", "raw",
];

/// 64 short, engineering-flavored nouns.
const NOUNS: [&str; 64] = [
    "arch", "beam", "bolt", "brace", "cable", "cedar", "chain", "cliff",
    "coil", "crane", "crest", "delta", "drift", "edge", "ember", "flint",
    "forge", "frost", "gate", "grove", "haven", "helix", "hinge", "hull",
    "ingot", "jade", "knot", "lance", "latch", "ledge", "lever", "lunar",
    "marsh", "mesa", "nexus", "notch", "orbit", "oxide", "panel", "peak",
    "pier", "plank", "pulse", "quartz", "ridge", "rivet", "rotor", "scale",
    "shaft", "shelf", "slate", "spark", "spoke", "steel", "stone", "surge",
    "torch", "truss", "valve", "vault", "wedge", "wheel", "yield", "zinc",
];

/// Generate a deterministic human-readable name from a run ID.
///
/// Uses SHA-256 to hash the run ID, then picks one adjective and one noun
/// from 64-element word lists (6 bits each, 12 bits total = 4096 combinations).
///
/// # Example
///
/// ```
/// use rwkv_bench::naming::generate_human_name;
///
/// let name = generate_human_name("20260208T120000Z_abc123");
/// assert!(name.contains('-'));
/// // Same input always produces the same output
/// assert_eq!(name, generate_human_name("20260208T120000Z_abc123"));
/// ```
pub fn generate_human_name(run_id: &str) -> String {
    let hash = Sha256::digest(run_id.as_bytes());

    // Use first two bytes: 6 bits for adjective index, 6 bits for noun index
    let adj_idx = (hash[0] & 0x3F) as usize; // bits 0-5
    let noun_idx = (hash[1] & 0x3F) as usize; // bits 8-13

    format!("{}-{}", ADJECTIVES[adj_idx], NOUNS[noun_idx])
}

/// Extract GPU name from a host CPU string.
///
/// Many APU systems report the CPU as e.g. "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S".
/// This function extracts the part after "w/" or "with" (case-insensitive).
/// If no such pattern is found, the input is returned unchanged (trimmed).
///
/// # Examples
///
/// ```
/// use rwkv_bench::naming::shorten_gpu;
///
/// assert_eq!(
///     shorten_gpu("AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"),
///     "Radeon 8060S"
/// );
/// assert_eq!(
///     shorten_gpu("Intel Core i9 with Intel UHD 770"),
///     "Intel UHD 770"
/// );
/// assert_eq!(
///     shorten_gpu("NVIDIA GeForce RTX 4090"),
///     "NVIDIA GeForce RTX 4090"
/// );
/// ```
pub fn shorten_gpu(cpu_str: &str) -> String {
    // Try "w/" first (more specific)
    if let Some(pos) = cpu_str.find("w/") {
        let after = cpu_str[pos + 2..].trim();
        if !after.is_empty() {
            return after.to_string();
        }
    }

    // Try case-insensitive "with"
    let lower = cpu_str.to_lowercase();
    if let Some(pos) = lower.find(" with ") {
        let after = cpu_str[pos + 6..].trim();
        if !after.is_empty() {
            return after.to_string();
        }
    }

    cpu_str.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_human_name_deterministic() {
        let name1 = generate_human_name("test_run_id_123");
        let name2 = generate_human_name("test_run_id_123");
        assert_eq!(name1, name2, "Same input must produce same output");
    }

    #[test]
    fn test_generate_human_name_format() {
        let name = generate_human_name("some_run_id");
        let parts: Vec<&str> = name.split('-').collect();
        assert_eq!(parts.len(), 2, "Name should be adjective-noun");
        assert!(
            ADJECTIVES.contains(&parts[0]),
            "First word should be an adjective"
        );
        assert!(
            NOUNS.contains(&parts[1]),
            "Second word should be a noun"
        );
    }

    #[test]
    fn test_generate_human_name_different_inputs() {
        let name1 = generate_human_name("run_a");
        let name2 = generate_human_name("run_b");
        // Different inputs should (very likely) produce different names
        // Not guaranteed but extremely likely with SHA-256
        assert_ne!(name1, name2, "Different inputs should produce different names");
    }

    #[test]
    fn test_shorten_gpu_with_w_slash() {
        assert_eq!(
            shorten_gpu("AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"),
            "Radeon 8060S"
        );
    }

    #[test]
    fn test_shorten_gpu_with_with() {
        assert_eq!(
            shorten_gpu("Intel Core i9 with Intel UHD 770"),
            "Intel UHD 770"
        );
    }

    #[test]
    fn test_shorten_gpu_with_uppercase_with() {
        assert_eq!(
            shorten_gpu("AMD CPU WITH Radeon Graphics"),
            "Radeon Graphics"
        );
    }

    #[test]
    fn test_shorten_gpu_no_pattern() {
        assert_eq!(
            shorten_gpu("NVIDIA GeForce RTX 4090"),
            "NVIDIA GeForce RTX 4090"
        );
    }

    #[test]
    fn test_shorten_gpu_empty_after_pattern() {
        // If "w/" is at the end with nothing after, return input unchanged
        assert_eq!(shorten_gpu("Some CPU w/"), "Some CPU w/");
    }

    #[test]
    fn test_shorten_gpu_trims_whitespace() {
        assert_eq!(
            shorten_gpu("  AMD RYZEN AI MAX+ 395 w/  Radeon 8060S  "),
            "Radeon 8060S"
        );
    }

    #[test]
    fn test_word_list_sizes() {
        assert_eq!(ADJECTIVES.len(), 64);
        assert_eq!(NOUNS.len(), 64);
    }

    #[test]
    fn test_word_lists_no_duplicates() {
        let mut adj_set = std::collections::HashSet::new();
        for adj in &ADJECTIVES {
            assert!(adj_set.insert(adj), "Duplicate adjective: {}", adj);
        }
        let mut noun_set = std::collections::HashSet::new();
        for noun in &NOUNS {
            assert!(noun_set.insert(noun), "Duplicate noun: {}", noun);
        }
    }
}
