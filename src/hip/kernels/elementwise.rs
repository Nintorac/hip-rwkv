//! HIP kernel wrappers for elementwise operations.

use half::f16;
use std::ffi::c_int;

use super::validation::{
    validate_binary_op, validate_broadcast_op, validate_ternary_op, validate_unary_op,
};
use crate::hip::buffer::DeviceBuffer;
use crate::hip::device::Stream;
use crate::hip::ffi::{
    check, launch_add_f16, launch_add_f32, launch_broadcast_add_f16, launch_broadcast_add_f32,
    launch_broadcast_mul_f16, launch_broadcast_mul_f32, launch_copy_f16, launch_copy_f32,
    launch_decay_exp_f16, launch_decay_exp_f32, launch_exp_f32, launch_lerp_f16, launch_lerp_f32,
    launch_mul_f16, launch_mul_f32, launch_negate_f16, launch_negate_f32, launch_sigmoid_f16,
    launch_sigmoid_f32, launch_softplus_decay_f16, launch_softplus_decay_f32,
    launch_squared_relu_f16, launch_squared_relu_f32, launch_tanh_f16, launch_tanh_f32,
    HipErrorKind, Result,
};
use crate::hip::tensor::{TensorHip, TensorShape};

// ============================================================================
// Dual-precision macro generators
// ============================================================================

/// Generate f32 and f16 variants of a unary elementwise operation.
macro_rules! dual_precision_unary {
    ($fn_f32:ident, $fn_f16:ident, $launcher_f32:ident, $launcher_f16:ident) => {
        pub fn $fn_f32(
            input: &TensorHip<f32>,
            output: &mut TensorHip<f32>,
            stream: &Stream,
        ) -> Result<()> {
            validate_unary_op(input, output, stringify!($fn_f32))?;
            unsafe {
                check($launcher_f32(
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    input.len() as c_int,
                    stream.handle(),
                ))
            }
        }

        pub fn $fn_f16(
            input: &TensorHip<f16>,
            output: &mut TensorHip<f16>,
            stream: &Stream,
        ) -> Result<()> {
            validate_unary_op(input, output, stringify!($fn_f16))?;
            unsafe {
                check($launcher_f16(
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    input.len() as c_int,
                    stream.handle(),
                ))
            }
        }
    };
}

/// Generate f32 and f16 variants of a binary elementwise operation.
macro_rules! dual_precision_binary {
    ($fn_f32:ident, $fn_f16:ident, $launcher_f32:ident, $launcher_f16:ident) => {
        pub fn $fn_f32(
            a: &TensorHip<f32>,
            b: &TensorHip<f32>,
            output: &mut TensorHip<f32>,
            stream: &Stream,
        ) -> Result<()> {
            validate_binary_op(a, b, output, stringify!($fn_f32))?;
            unsafe {
                check($launcher_f32(
                    a.as_ptr(),
                    b.as_ptr(),
                    output.as_mut_ptr(),
                    a.len() as c_int,
                    stream.handle(),
                ))
            }
        }

        pub fn $fn_f16(
            a: &TensorHip<f16>,
            b: &TensorHip<f16>,
            output: &mut TensorHip<f16>,
            stream: &Stream,
        ) -> Result<()> {
            validate_binary_op(a, b, output, stringify!($fn_f16))?;
            unsafe {
                check($launcher_f16(
                    a.as_ptr(),
                    b.as_ptr(),
                    output.as_mut_ptr(),
                    a.len() as c_int,
                    stream.handle(),
                ))
            }
        }
    };
}

/// Generate f32 and f16 variants of a ternary elementwise operation (lerp).
macro_rules! dual_precision_ternary {
    ($fn_f32:ident, $fn_f16:ident, $launcher_f32:ident, $launcher_f16:ident) => {
        pub fn $fn_f32(
            a: &TensorHip<f32>,
            b: &TensorHip<f32>,
            t: &TensorHip<f32>,
            output: &mut TensorHip<f32>,
            stream: &Stream,
        ) -> Result<()> {
            validate_ternary_op(a, b, t, output, stringify!($fn_f32))?;
            unsafe {
                check($launcher_f32(
                    a.as_ptr(),
                    b.as_ptr(),
                    t.as_ptr(),
                    output.as_mut_ptr(),
                    a.len() as c_int,
                    stream.handle(),
                ))
            }
        }

        pub fn $fn_f16(
            a: &TensorHip<f16>,
            b: &TensorHip<f16>,
            t: &TensorHip<f16>,
            output: &mut TensorHip<f16>,
            stream: &Stream,
        ) -> Result<()> {
            validate_ternary_op(a, b, t, output, stringify!($fn_f16))?;
            unsafe {
                check($launcher_f16(
                    a.as_ptr(),
                    b.as_ptr(),
                    t.as_ptr(),
                    output.as_mut_ptr(),
                    a.len() as c_int,
                    stream.handle(),
                ))
            }
        }
    };
}

/// Generate f32 and f16 variants of a broadcast operation.
macro_rules! dual_precision_broadcast {
    ($fn_f32:ident, $fn_f16:ident, $launcher_f32:ident, $launcher_f16:ident) => {
        pub fn $fn_f32(
            input: &TensorHip<f32>,
            scale: &TensorHip<f32>,
            output: &mut TensorHip<f32>,
            stream: &Stream,
        ) -> Result<()> {
            validate_broadcast_op(input, scale, output, stringify!($fn_f32))?;
            unsafe {
                check($launcher_f32(
                    input.as_ptr(),
                    scale.as_ptr(),
                    output.as_mut_ptr(),
                    input.len() as c_int,
                    scale.len() as c_int,
                    stream.handle(),
                ))
            }
        }

        pub fn $fn_f16(
            input: &TensorHip<f16>,
            scale: &TensorHip<f16>,
            output: &mut TensorHip<f16>,
            stream: &Stream,
        ) -> Result<()> {
            validate_broadcast_op(input, scale, output, stringify!($fn_f16))?;
            unsafe {
                check($launcher_f16(
                    input.as_ptr(),
                    scale.as_ptr(),
                    output.as_mut_ptr(),
                    input.len() as c_int,
                    scale.len() as c_int,
                    stream.handle(),
                ))
            }
        }
    };
}

/// Launch the copy kernel to copy f32 data from input to output
pub fn copy_f32(
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    stream: &Stream,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input {} vs output {}",
                input.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check(launch_copy_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

// GPU-to-GPU copy for TensorHip: output = input
// Both tensors must have the same length and be contiguous.
dual_precision_unary!(copy_tensor_f32, copy_tensor_f16, launch_copy_f32, launch_copy_f16);

/// Copy f32 data from host, through GPU copy kernel, back to host.
/// This is a convenience function for testing.
pub fn hip_copy_kernel(input: &[f32]) -> Result<Vec<f32>> {
    // Use the null stream as a workaround for hipStreamCreate crashes
    let stream = Stream::null();

    let mut d_input = DeviceBuffer::<f32>::new(input.len())?;
    let mut d_output = DeviceBuffer::<f32>::new(input.len())?;

    d_input.copy_from_host(input, &stream)?;
    copy_f32(&d_input, &mut d_output, &stream)?;

    let mut output = vec![0.0f32; input.len()];
    d_output.copy_to_host(&mut output, &stream)?;
    stream.synchronize()?;

    Ok(output)
}

// Decay exponential: out = exp(-exp(x))
// Time decay transformation used in RWKV7. Numerically stable for all finite inputs.
dual_precision_unary!(decay_exp_f32, decay_exp_f16, launch_decay_exp_f32, launch_decay_exp_f16);

/// Compute decay exponential on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_decay_exp(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    decay_exp_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

// Lerp: out = a + t * (b - a) (linear interpolation)
// Used for mixing operations in RWKV7. All tensors must have the same length and be contiguous.
dual_precision_ternary!(lerp_f32, lerp_f16, launch_lerp_f32, launch_lerp_f16);

/// Compute linear interpolation on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_lerp(a: &[f32], b: &[f32], t: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() || a.len() != t.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!("Size mismatch: a={}, b={}, t={}", a.len(), b.len(), t.len()),
        });
    }

    let stream = Stream::null();
    let shape = TensorShape::new(a.len(), 1, 1, 1);

    let d_a = TensorHip::from_slice(a, shape, &stream)?;
    let d_b = TensorHip::from_slice(b, shape, &stream)?;
    let d_t = TensorHip::from_slice(t, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    lerp_f32(&d_a, &d_b, &d_t, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

// Sigmoid: out = 1 / (1 + exp(-x))
// Standard sigmoid activation function.
dual_precision_unary!(sigmoid_f32, sigmoid_f16, launch_sigmoid_f32, launch_sigmoid_f16);

/// Compute sigmoid on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_sigmoid(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    sigmoid_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

// Squared ReLU: out = max(0, x)^2
// Used in RWKV7 channel mixing.
dual_precision_unary!(squared_relu_f32, squared_relu_f16, launch_squared_relu_f32, launch_squared_relu_f16);

/// Compute squared ReLU on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_squared_relu(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    squared_relu_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

// Softplus decay: out = log(sigmoid(x)) - 0.5
// Used for RWKV7 time decay computation. Numerically stable for all finite inputs.
dual_precision_unary!(softplus_decay_f32, softplus_decay_f16, launch_softplus_decay_f32, launch_softplus_decay_f16);

/// Compute softplus decay on host data, returning results.
/// This is a convenience function for testing.
pub fn hip_softplus_decay(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    softplus_decay_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

// Tanh: out = tanh(x)
dual_precision_unary!(tanh_f32, tanh_f16, launch_tanh_f32, launch_tanh_f16);

/// Compute tanh on host data.
pub fn hip_tanh(input: &[f32]) -> Result<Vec<f32>> {
    let stream = Stream::null();
    let shape = TensorShape::new(input.len(), 1, 1, 1);

    let d_input = TensorHip::from_slice(input, shape, &stream)?;
    let mut d_output = TensorHip::<f32>::new(shape)?;

    tanh_f32(&d_input, &mut d_output, &stream)?;

    d_output.to_vec(&stream)
}

// ============================================================================
// Elementwise operations for GPU-native forward pass
// ============================================================================

// Elementwise add: output = a + b
// Both inputs must have the same shape and be contiguous.
dual_precision_binary!(add_f32, add_f16, launch_add_f32, launch_add_f16);

// Elementwise multiply: output = a * b
// Both inputs must have the same shape and be contiguous.
dual_precision_binary!(mul_f32, mul_f16, launch_mul_f32, launch_mul_f16);

// Negate: output = -input
dual_precision_unary!(negate_f32, negate_f16, launch_negate_f32, launch_negate_f16);

/// Exponential: output = exp(input)
/// Note: Only f32 variant exists (no f16 FFI binding).
pub fn exp_f32(input: &TensorHip<f32>, output: &mut TensorHip<f32>, stream: &Stream) -> Result<()> {
    validate_unary_op(input, output, "exp_f32")?;
    unsafe {
        check(launch_exp_f32(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as c_int,
            stream.handle(),
        ))
    }
}

// Broadcast add: output[i] = input[i] + bias[i % bias_len]
// Used for adding per-channel biases to batched data.
dual_precision_broadcast!(broadcast_add_f32, broadcast_add_f16, launch_broadcast_add_f32, launch_broadcast_add_f16);

// Broadcast multiply: output[i] = input[i] * scale[i % scale_len]
// Used for per-channel scaling (e.g., k * k_k in RWKV7).
dual_precision_broadcast!(broadcast_mul_f32, broadcast_mul_f16, launch_broadcast_mul_f32, launch_broadcast_mul_f16);

#[cfg(test)]
mod tests {
    use super::*;

    // === Acceptance Criteria Tests for bd-2sh.4.3 (Decay Exponential Kernel) ===

    #[test]
    fn test_decay_exp() {
        // Test basic decay_exp functionality: out = exp(-exp(x))
        // Also serves as the primary acceptance test when fixtures are loaded

        // Test known values
        let input = vec![
            0.0,  // exp(-exp(0)) = exp(-1) ≈ 0.3679
            -1.0, // exp(-exp(-1)) = exp(-0.3679) ≈ 0.6922
            1.0,  // exp(-exp(1)) = exp(-2.718) ≈ 0.0660
            -5.0, // exp(-exp(-5)) ≈ exp(-0.0067) ≈ 0.9933
            5.0,  // exp(-exp(5)) ≈ exp(-148.4) ≈ 0
        ];

        let output = hip_decay_exp(&input).expect("decay_exp kernel failed");

        // Expected values (computed with Python: np.exp(-np.exp(x)))
        let expected = vec![
            0.36787944, // exp(-1)
            0.69220066, // exp(-exp(-1))
            0.06598804, // exp(-exp(1))
            0.99330715, // exp(-exp(-5))
            0.0,        // exp(-exp(5)) ≈ 0 (underflow)
        ];

        // Check each value with tolerance
        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-3 + 1e-3 * exp.abs(); // rtol=1e-3, atol=1e-3
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!(
            "Basic decay_exp test passed: {} values verified",
            output.len()
        );
    }

    #[test]
    fn test_decay_exp_numerical_stability() {
        // Test edge cases that could cause numerical issues
        let input = vec![
            -50.0, // Very negative: exp(-exp(-50)) ≈ 1
            -10.0, // Negative: exp(-exp(-10)) ≈ 1
            -5.0,  // Moderate negative
            -1.0,  // Small negative
            0.0,   // Zero
            1.0,   // Small positive
            5.0,   // Moderate positive: exp(-exp(5)) ≈ 0
            10.0,  // exp(-exp(10)) ≈ 0 (extreme underflow)
            80.0,  // At clamping boundary
            100.0, // Beyond clamping: should be 0, not NaN/Inf
        ];

        let output = hip_decay_exp(&input).expect("decay_exp stability test failed");

        // Verify no NaN or Inf values
        for (i, &val) in output.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at index {} (input={})", i, input[i]);
            assert!(
                !val.is_infinite(),
                "Inf at index {} (input={})",
                i,
                input[i]
            );
            assert!(
                val >= 0.0 && val <= 1.0,
                "Value out of [0,1] range at index {}: {} (input={})",
                i,
                val,
                input[i]
            );
        }

        // Verify expected behavior at extremes
        assert!(
            output[0] > 0.999,
            "exp(-exp(-50)) should be ≈1, got {}",
            output[0]
        );
        assert!(
            output[1] > 0.999,
            "exp(-exp(-10)) should be ≈1, got {}",
            output[1]
        );
        assert!(
            output[7] < 0.001,
            "exp(-exp(10)) should be ≈0, got {}",
            output[7]
        );
        assert!(
            output[8] < 0.001,
            "exp(-exp(80)) should be ≈0, got {}",
            output[8]
        );
        assert_eq!(output[9], 0.0, "exp(-exp(100)) should be exactly 0");

        println!(
            "Numerical stability test passed: all {} values are finite and in [0,1]",
            output.len()
        );
    }

    // === Acceptance Criteria Tests for bd-2sh.4.4 (Lerp Kernel) ===

    #[test]
    fn test_lerp() {
        // Test basic lerp functionality: out = a + t * (b - a)
        let a = vec![0.0, 1.0, 2.0, 10.0, -5.0];
        let b = vec![10.0, 5.0, 2.0, 0.0, 5.0];
        let t = vec![0.0, 0.5, 1.0, 0.25, 0.5];

        let output = hip_lerp(&a, &b, &t).expect("lerp kernel failed");

        // Expected: lerp(a, b, t) = a + t * (b - a)
        // [0] lerp(0, 10, 0) = 0
        // [1] lerp(1, 5, 0.5) = 1 + 0.5 * 4 = 3
        // [2] lerp(2, 2, 1) = 2
        // [3] lerp(10, 0, 0.25) = 10 + 0.25 * (-10) = 7.5
        // [4] lerp(-5, 5, 0.5) = -5 + 0.5 * 10 = 0
        let expected = vec![0.0, 3.0, 2.0, 7.5, 0.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!("Basic lerp test passed: {} values verified", output.len());
    }

    #[test]
    fn test_lerp_edge_cases() {
        // Test edge cases: t outside [0, 1] (extrapolation)
        let a = vec![0.0, 0.0, 100.0];
        let b = vec![10.0, 10.0, 0.0];
        let t = vec![-0.5, 1.5, 2.0];

        let output = hip_lerp(&a, &b, &t).expect("lerp edge case test failed");

        // Expected with extrapolation:
        // [0] lerp(0, 10, -0.5) = 0 + (-0.5) * 10 = -5
        // [1] lerp(0, 10, 1.5) = 0 + 1.5 * 10 = 15
        // [2] lerp(100, 0, 2.0) = 100 + 2.0 * (-100) = -100
        let expected = vec![-5.0, 15.0, -100.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!("Lerp edge case test passed: extrapolation works correctly");
    }

    // === Acceptance Criteria Tests for bd-2sh.4.1 (Sigmoid Kernel) ===

    #[test]
    fn test_sigmoid() {
        // Test basic sigmoid functionality: out = 1 / (1 + exp(-x))
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];

        let output = hip_sigmoid(&input).expect("sigmoid kernel failed");

        // Expected: sigmoid(x) = 1 / (1 + exp(-x))
        // sigmoid(0) = 0.5
        // sigmoid(1) ≈ 0.7311
        // sigmoid(-1) ≈ 0.2689
        // sigmoid(2) ≈ 0.8808
        // sigmoid(-2) ≈ 0.1192
        let expected = vec![0.5, 0.7310586, 0.26894143, 0.880797, 0.11920292];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!(
            "Basic sigmoid test passed: {} values verified",
            output.len()
        );
    }

    #[test]
    fn test_sigmoid_edge_cases() {
        // Test edge cases that could cause numerical issues
        let input = vec![
            -100.0, // Very negative: sigmoid → 0
            -50.0,  // Large negative
            -10.0,  // Moderate negative
            0.0,    // Zero: sigmoid = 0.5
            10.0,   // Moderate positive
            50.0,   // Large positive
            100.0,  // Very positive: sigmoid → 1
        ];

        let output = hip_sigmoid(&input).expect("sigmoid edge case test failed");

        // Verify no NaN or Inf values
        for (i, &val) in output.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at index {} (input={})", i, input[i]);
            assert!(
                !val.is_infinite(),
                "Inf at index {} (input={})",
                i,
                input[i]
            );
            assert!(
                val >= 0.0 && val <= 1.0,
                "Value out of [0,1] range at index {}: {} (input={})",
                i,
                val,
                input[i]
            );
        }

        // Verify expected behavior at extremes
        assert!(
            output[0] < 1e-10,
            "sigmoid(-100) should be ≈0, got {}",
            output[0]
        );
        assert!(
            output[1] < 1e-10,
            "sigmoid(-50) should be ≈0, got {}",
            output[1]
        );
        assert!(
            (output[3] - 0.5).abs() < 1e-6,
            "sigmoid(0) should be 0.5, got {}",
            output[3]
        );
        assert!(
            output[5] >= 1.0 - 1e-10,
            "sigmoid(50) should be ≈1, got {}",
            output[5]
        );
        assert!(
            output[6] >= 1.0 - 1e-10,
            "sigmoid(100) should be ≈1, got {}",
            output[6]
        );

        println!(
            "Sigmoid edge case test passed: all {} values are finite and in [0,1]",
            output.len()
        );
    }

    // === Acceptance Criteria Tests for bd-2sh.4.2 (Squared ReLU Kernel) ===

    #[test]
    fn test_squared_relu() {
        // Test basic squared ReLU: out = max(0, x)^2
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

        let output = hip_squared_relu(&input).expect("squared_relu kernel failed");

        // Expected: max(0, x)^2
        // [-2] -> 0^2 = 0
        // [-1] -> 0^2 = 0
        // [0]  -> 0^2 = 0
        // [1]  -> 1^2 = 1
        // [2]  -> 2^2 = 4
        // [3]  -> 3^2 = 9
        let expected = vec![0.0, 0.0, 0.0, 1.0, 4.0, 9.0];

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!("Squared ReLU test passed: {} values verified", output.len());
    }

    // === Acceptance Criteria Tests for bd-2sh.4.14 (Softplus Decay Kernel) ===

    #[test]
    fn test_softplus_decay() {
        // Test softplus decay: out = log(sigmoid(x)) - 0.5
        let input = vec![0.0, 1.0, -1.0, 5.0, -5.0];

        let output = hip_softplus_decay(&input).expect("softplus_decay kernel failed");

        // Expected: log(sigmoid(x)) - 0.5
        // log(sigmoid(0)) - 0.5 = log(0.5) - 0.5 ≈ -0.693 - 0.5 = -1.193
        // log(sigmoid(1)) - 0.5 ≈ -0.313 - 0.5 = -0.813
        // log(sigmoid(-1)) - 0.5 ≈ -1.313 - 0.5 = -1.813
        // log(sigmoid(5)) - 0.5 ≈ -0.0067 - 0.5 ≈ -0.507
        // log(sigmoid(-5)) - 0.5 ≈ -5.0067 - 0.5 ≈ -5.507
        let expected: Vec<f32> = input
            .iter()
            .map(|&x| {
                let log_sigmoid = -(1.0f32 + (-x).exp()).ln();
                log_sigmoid - 0.5
            })
            .collect();

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-4;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!(
            "Softplus decay test passed: {} values verified",
            output.len()
        );
    }

    #[test]
    fn test_softplus_decay_numerical_stability() {
        // Test edge cases that could cause numerical issues
        let input = vec![
            -100.0, // Very negative: result ≈ x - 0.5 = -100.5
            -50.0,  // Large negative
            -20.0,  // At clamping boundary
            0.0,    // Zero
            20.0,   // At clamping boundary
            50.0,   // Large positive
            100.0,  // Very positive: result ≈ -0.5
        ];

        let output = hip_softplus_decay(&input).expect("softplus_decay stability test failed");

        // Verify no NaN or Inf values
        for (i, &val) in output.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at index {} (input={})", i, input[i]);
            assert!(
                !val.is_infinite(),
                "Inf at index {} (input={})",
                i,
                input[i]
            );
        }

        // Verify expected behavior at extremes
        // For large negative x: result ≈ x - 0.5
        assert!(
            (output[0] - (-100.5)).abs() < 0.1,
            "softplus_decay(-100) should be ≈-100.5, got {}",
            output[0]
        );
        // For large positive x: result ≈ -0.5
        assert!(
            (output[6] - (-0.5)).abs() < 0.01,
            "softplus_decay(100) should be ≈-0.5, got {}",
            output[6]
        );

        println!(
            "Softplus decay stability test passed: all {} values are finite",
            output.len()
        );
    }

    // === Acceptance Criteria Tests for bd-2sh.4.13 (Tanh Kernel) ===

    #[test]
    fn test_tanh_basic() {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];

        let output = hip_tanh(&input).expect("tanh kernel failed");

        // Expected: tanh(x)
        let expected: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();

        for (i, (actual, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - exp).abs();
            let tol = 1e-5;
            assert!(
                diff <= tol,
                "Mismatch at index {}: actual={:.6}, expected={:.6}, diff={:.6}",
                i,
                actual,
                exp,
                diff
            );
        }
        println!("Basic tanh test passed: {} values verified", output.len());
    }

    #[test]
    fn test_tanh_edge_cases() {
        let input = vec![
            -100.0, // Very negative: tanh → -1
            -10.0, 0.0, // tanh(0) = 0
            10.0, 100.0, // Very positive: tanh → 1
        ];

        let output = hip_tanh(&input).expect("tanh edge cases failed");

        // Verify no NaN or Inf
        for (i, &val) in output.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at index {}", i);
            assert!(!val.is_infinite(), "Inf at index {}", i);
            assert!(
                val >= -1.0 && val <= 1.0,
                "Value out of [-1,1] at index {}: {}",
                i,
                val
            );
        }

        // Check extremes
        assert!(
            (output[0] - (-1.0)).abs() < 1e-6,
            "tanh(-100) should be ≈-1"
        );
        assert!(output[2].abs() < 1e-6, "tanh(0) should be ≈0");
        assert!((output[4] - 1.0).abs() < 1e-6, "tanh(100) should be ≈1");

        println!("Tanh edge cases test passed");
    }
}
