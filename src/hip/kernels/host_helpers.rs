//! Macros for host convenience functions that wrap GPU kernel calls.
//!
//! These macros reduce hip_* functions from ~10-20 LOC to 1-line invocations:
//! - `hip_unary_op!(name, kernel)`: 1 input -> 1 output (flat shape)
//! - `hip_ternary_op!(name, kernel)`: 3 inputs -> 1 output (lerp pattern)

/// Unary op: `fn name(input: &[f32]) -> Result<Vec<f32>>`
macro_rules! hip_unary_op {
    ($fn_name:ident, $kernel:ident) => {
        pub fn $fn_name(input: &[f32]) -> Result<Vec<f32>> {
            let stream = Stream::null();
            let shape = TensorShape::new(input.len(), 1, 1, 1);
            let d_input = TensorHip::from_slice(input, shape, &stream)?;
            let mut d_output = TensorHip::<f32>::new(shape)?;
            $kernel(&d_input, &mut d_output, &stream)?;
            d_output.to_vec(&stream)
        }
    };
}

/// Ternary op (lerp): `fn name(a, b, t: &[f32]) -> Result<Vec<f32>>`
macro_rules! hip_ternary_op {
    ($fn_name:ident, $kernel:ident) => {
        pub fn $fn_name(a: &[f32], b: &[f32], t: &[f32]) -> Result<Vec<f32>> {
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
            $kernel(&d_a, &d_b, &d_t, &mut d_output, &stream)?;
            d_output.to_vec(&stream)
        }
    };
}

pub(crate) use hip_ternary_op;
pub(crate) use hip_unary_op;
