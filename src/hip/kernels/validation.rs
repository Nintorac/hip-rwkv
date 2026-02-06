//! Validation helper functions for kernel wrappers.
//!
//! This module provides reusable validation helpers to reduce code duplication
//! across kernel wrapper functions. All helpers return `Result<()>` and produce
//! consistent, informative error messages.

use crate::hip::ffi::{HipErrorKind, Result};
use crate::hip::tensor::TensorHip;

/// Validate that a single tensor is contiguous.
///
/// Returns an error with the function name in the message if the tensor is not contiguous.
#[inline]
pub fn require_contiguous<T: Copy>(tensor: &TensorHip<T>, fn_name: &str) -> Result<()> {
    if !tensor.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: format!("{} requires contiguous tensors", fn_name),
        });
    }
    Ok(())
}

/// Validate that two tensors are both contiguous.
///
/// This is the most common pattern: checking input and output tensors.
#[inline]
pub fn require_contiguous_pair<T: Copy, U: Copy>(
    a: &TensorHip<T>,
    b: &TensorHip<U>,
    fn_name: &str,
) -> Result<()> {
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: format!("{} requires contiguous tensors", fn_name),
        });
    }
    Ok(())
}

/// Validate that three tensors are all contiguous.
#[inline]
pub fn require_contiguous_3<T: Copy, U: Copy, V: Copy>(
    a: &TensorHip<T>,
    b: &TensorHip<U>,
    c: &TensorHip<V>,
    fn_name: &str,
) -> Result<()> {
    if !a.is_contiguous() || !b.is_contiguous() || !c.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: format!("{} requires contiguous tensors", fn_name),
        });
    }
    Ok(())
}

/// Validate that four tensors are all contiguous.
#[inline]
pub fn require_contiguous_4<T: Copy, U: Copy, V: Copy, W: Copy>(
    a: &TensorHip<T>,
    b: &TensorHip<U>,
    c: &TensorHip<V>,
    d: &TensorHip<W>,
    fn_name: &str,
) -> Result<()> {
    if !a.is_contiguous() || !b.is_contiguous() || !c.is_contiguous() || !d.is_contiguous() {
        return Err(HipErrorKind {
            code: -1,
            message: format!("{} requires contiguous tensors", fn_name),
        });
    }
    Ok(())
}

/// Validate that five tensors are all contiguous.
#[inline]
pub fn require_contiguous_5<A: Copy, B: Copy, C: Copy, D: Copy, E: Copy>(
    a: &TensorHip<A>,
    b: &TensorHip<B>,
    c: &TensorHip<C>,
    d: &TensorHip<D>,
    e: &TensorHip<E>,
    fn_name: &str,
) -> Result<()> {
    if !a.is_contiguous()
        || !b.is_contiguous()
        || !c.is_contiguous()
        || !d.is_contiguous()
        || !e.is_contiguous()
    {
        return Err(HipErrorKind {
            code: -1,
            message: format!("{} requires contiguous tensors", fn_name),
        });
    }
    Ok(())
}

/// Validate that two tensors have the same length.
///
/// Returns an error with "Size mismatch" message if lengths differ.
#[inline]
pub fn require_same_len<T: Copy, U: Copy>(a: &TensorHip<T>, b: &TensorHip<U>) -> Result<()> {
    if a.len() != b.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!("Size mismatch: {} vs {}", a.len(), b.len()),
        });
    }
    Ok(())
}

/// Validate that two tensors have the same length, with custom labels.
#[inline]
pub fn require_same_len_labeled<T: Copy, U: Copy>(
    a: &TensorHip<T>,
    a_name: &str,
    b: &TensorHip<U>,
    b_name: &str,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: {} {} vs {} {}",
                a_name,
                a.len(),
                b_name,
                b.len()
            ),
        });
    }
    Ok(())
}

/// Validate that input and output tensors have the same length.
///
/// This is a convenience wrapper with "input"/"output" labels.
#[inline]
pub fn require_same_len_io<T: Copy, U: Copy>(
    input: &TensorHip<T>,
    output: &TensorHip<U>,
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
    Ok(())
}

/// Validate that three tensors all have the same length.
#[inline]
pub fn require_same_len_3<T: Copy, U: Copy, V: Copy>(
    a: &TensorHip<T>,
    b: &TensorHip<U>,
    c: &TensorHip<V>,
) -> Result<()> {
    let len = a.len();
    if b.len() != len || c.len() != len {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: a={}, b={}, output={}",
                a.len(),
                b.len(),
                c.len()
            ),
        });
    }
    Ok(())
}

/// Validate that four tensors all have the same length.
#[inline]
pub fn require_same_len_4<T: Copy, U: Copy, V: Copy, W: Copy>(
    a: &TensorHip<T>,
    b: &TensorHip<U>,
    c: &TensorHip<V>,
    d: &TensorHip<W>,
) -> Result<()> {
    let len = a.len();
    if b.len() != len || c.len() != len || d.len() != len {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: a={}, b={}, t={}, output={}",
                a.len(),
                b.len(),
                c.len(),
                d.len()
            ),
        });
    }
    Ok(())
}

/// Combined validation: check that input/output have same length AND both are contiguous.
///
/// This is the most common validation pattern for unary operations.
#[inline]
pub fn validate_unary_op<T: Copy, U: Copy>(
    input: &TensorHip<T>,
    output: &TensorHip<U>,
    fn_name: &str,
) -> Result<()> {
    require_same_len_io(input, output)?;
    require_contiguous_pair(input, output, fn_name)
}

/// Combined validation: check that a, b, output have same length AND all are contiguous.
///
/// This is the most common validation pattern for binary operations.
#[inline]
pub fn validate_binary_op<T: Copy, U: Copy, V: Copy>(
    a: &TensorHip<T>,
    b: &TensorHip<U>,
    output: &TensorHip<V>,
    fn_name: &str,
) -> Result<()> {
    require_same_len_3(a, b, output)?;
    require_contiguous_3(a, b, output, fn_name)
}

/// Combined validation: check that a, b, c, output have same length AND all are contiguous.
///
/// Used for ternary operations like lerp(a, b, t).
#[inline]
pub fn validate_ternary_op<T: Copy, U: Copy, V: Copy, W: Copy>(
    a: &TensorHip<T>,
    b: &TensorHip<U>,
    c: &TensorHip<V>,
    output: &TensorHip<W>,
    fn_name: &str,
) -> Result<()> {
    require_same_len_4(a, b, c, output)?;
    require_contiguous_4(a, b, c, output, fn_name)
}

/// Validate that all tensors in a slice are contiguous.
///
/// Use this for functions with many tensor arguments where adding a specific
/// require_contiguous_N variant would be impractical.
#[inline]
pub fn require_all_contiguous(tensors: &[bool], fn_name: &str) -> Result<()> {
    if tensors.iter().any(|&is_cont| !is_cont) {
        return Err(HipErrorKind {
            code: -1,
            message: format!("{} requires contiguous tensors", fn_name),
        });
    }
    Ok(())
}

/// Validate broadcast operation: input divisible by broadcast source, same output length.
///
/// Used for broadcast_add and broadcast_mul where we need input.len() % broadcast.len() == 0.
#[inline]
pub fn validate_broadcast_op<T: Copy, U: Copy, V: Copy>(
    input: &TensorHip<T>,
    broadcast: &TensorHip<U>,
    output: &TensorHip<V>,
    fn_name: &str,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Size mismatch: input={}, output={}",
                input.len(),
                output.len()
            ),
        });
    }
    if !input.len().is_multiple_of(broadcast.len()) {
        return Err(HipErrorKind {
            code: -1,
            message: format!(
                "Broadcast incompatible: input len {} not divisible by broadcast len {}",
                input.len(),
                broadcast.len()
            ),
        });
    }
    require_contiguous_3(input, broadcast, output, fn_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip::tensor::TensorShape;

    #[test]
    fn test_require_contiguous_pair() {
        let shape = TensorShape::new(10, 1, 1, 1);
        let a: TensorHip<f32> = TensorHip::new(shape).expect("alloc");
        let b: TensorHip<f32> = TensorHip::new(shape).expect("alloc");

        // Both contiguous - should pass
        assert!(require_contiguous_pair(&a, &b, "test_fn").is_ok());
    }

    #[test]
    fn test_require_same_len_io() {
        let shape1 = TensorShape::new(10, 1, 1, 1);
        let shape2 = TensorShape::new(20, 1, 1, 1);
        let a: TensorHip<f32> = TensorHip::new(shape1).expect("alloc");
        let b: TensorHip<f32> = TensorHip::new(shape1).expect("alloc");
        let c: TensorHip<f32> = TensorHip::new(shape2).expect("alloc");

        // Same length - should pass
        assert!(require_same_len_io(&a, &b).is_ok());

        // Different length - should fail
        let err = require_same_len_io(&a, &c).unwrap_err();
        assert!(err.message.contains("Size mismatch"));
        assert!(err.message.contains("10"));
        assert!(err.message.contains("20"));
    }

    #[test]
    fn test_validate_unary_op() {
        let shape = TensorShape::new(10, 1, 1, 1);
        let input: TensorHip<f32> = TensorHip::new(shape).expect("alloc");
        let output: TensorHip<f32> = TensorHip::new(shape).expect("alloc");

        // Valid case
        assert!(validate_unary_op(&input, &output, "test_fn").is_ok());
    }
}
