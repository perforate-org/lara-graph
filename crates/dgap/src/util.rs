//! APMA utility helpers ported from DGAP's `util.h`.
//!
//! Only the arithmetic helpers used by the PMA (`compute_capacity`,
//! rebalance windows) are included.

/// Returns the 0-based index of the most significant bit set in `x`.
/// `floor_log2(1) == 0`, `floor_log2(8) == 3`. Panics for `x == 0`
/// (mirrors the C++ `last_bit_set` UB on zero).
#[inline]
pub fn floor_log2(x: u64) -> u32 {
    debug_assert!(x > 0, "floor_log2: x must be nonzero");
    63 - x.leading_zeros()
}

/// Returns `ceil(log2(x))` for `x >= 1`.
/// Mirrors the C++ `ceil_log2`: `ceil_log2(13) == 4`, `ceil_log2(27) == 5`.
#[inline]
pub fn ceil_log2(x: u64) -> u32 {
    debug_assert!(x > 0, "ceil_log2: x must be nonzero");
    if x <= 1 {
        return 0;
    }
    floor_log2(x - 1) + 1
}

/// Returns the largest power of two not greater than `x`
/// (`2^{floor(lg x)}`).
#[inline]
pub fn hyperfloor(x: u64) -> u64 {
    debug_assert!(x > 0, "hyperfloor: x must be nonzero");
    1u64 << floor_log2(x)
}

/// Returns the smallest power of two not less than `x`
/// (`2^{ceil(lg x)}`).
#[inline]
pub fn hyperceil(x: u64) -> u64 {
    1u64 << ceil_log2(x)
}

/// Ceiling division. Matches the C++ `ceil_div` (returns 0 for `a == 0`).
#[inline]
pub fn ceil_div(a: u64, b: u64) -> u64 {
    debug_assert!(b > 0, "ceil_div: division by zero");
    if a == 0 { 0 } else { 1 + ((a - 1) / b) }
}

/// Alias of [`hyperfloor`] kept for readability at call sites that think in
/// terms of "largest power of two ≤ x".
#[inline]
pub fn largest_power_of_two_le(x: u64) -> u64 {
    hyperfloor(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_log2_basics() {
        assert_eq!(floor_log2(1), 0);
        assert_eq!(floor_log2(2), 1);
        assert_eq!(floor_log2(3), 1);
        assert_eq!(floor_log2(8), 3);
        assert_eq!(floor_log2(1 << 40), 40);
    }

    #[test]
    fn ceil_log2_matches_cpp_examples() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(13), 4);
        assert_eq!(ceil_log2(27), 5);
        assert_eq!(ceil_log2(1_048_576), 20);
        assert_eq!(ceil_log2(1_048_577), 21);
    }

    #[test]
    fn hyperfloor_and_hyperceil() {
        assert_eq!(hyperfloor(1), 1);
        assert_eq!(hyperfloor(15), 8);
        assert_eq!(hyperfloor(16), 16);
        assert_eq!(hyperfloor(52_429), 32_768);
        assert_eq!(hyperceil(13), 16);
        assert_eq!(hyperceil(16), 16);
    }

    #[test]
    fn ceil_div_basics() {
        assert_eq!(ceil_div(0, 4), 0);
        assert_eq!(ceil_div(1, 4), 1);
        assert_eq!(ceil_div(4, 4), 1);
        assert_eq!(ceil_div(5, 4), 2);
        assert_eq!(ceil_div(1_048_576, 32_768), 32);
    }
}
