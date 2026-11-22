//! TLSF size classes (Masmano et al. 2004/2008): a first level per power of two and
//! [`SL_COUNT`] linear subclasses inside each, so a fitting free span is found with two bitmap
//! lookups. With 16 subclasses the fit loses at most 1/17 of a span to the class rounding.

use crate::units::SpanLen;

/// log2 of the number of subclasses per power of two.
const SL_LOG2: u32 = 4;
/// Subclasses per first-level class.
pub(super) const SL_COUNT: u32 = 1 << SL_LOG2;
/// First-level classes: 0 holds lengths below [`SL_COUNT`] exactly; 1..=28 hold `2^4..2^32`.
pub(super) const FL_COUNT: u32 = 32 - SL_LOG2 + 1;
/// Number of classes, i.e. free-list heads.
pub(super) const CLASS_COUNT: u32 = FL_COUNT * SL_COUNT;

/// One free list: lengths in `[lower, lower + step)` for its first and second level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Class {
    fl: u32,
    sl: u32,
}

impl Class {
    /// The class a free span of `len` is filed under.
    pub(super) fn containing(len: SpanLen) -> Self {
        Self::of(u64::from(len.get())).unwrap_or(Self::LARGEST)
    }

    /// The smallest class whose every span holds at least `len`; `None` when no class can
    /// guarantee it (only the frontier can serve such a request).
    pub(super) fn guaranteeing(len: SpanLen) -> Option<Self> {
        let len = u64::from(len.get());
        if len < u64::from(SL_COUNT) {
            return Self::of(len);
        }
        let step = 1u64 << (len.ilog2() - SL_LOG2);
        Self::of(len + step - 1)
    }

    /// Position of this class's free-list head, `0..CLASS_COUNT`.
    pub(super) const fn index(self) -> u32 {
        self.fl * SL_COUNT + self.sl
    }

    pub(super) const fn first_level(self) -> u32 {
        self.fl
    }

    pub(super) const fn second_level(self) -> u32 {
        self.sl
    }

    const LARGEST: Self = Self {
        fl: FL_COUNT - 1,
        sl: SL_COUNT - 1,
    };

    fn of(len: u64) -> Option<Self> {
        if len < u64::from(SL_COUNT) {
            return u32::try_from(len).ok().map(|sl| Self { fl: 0, sl });
        }
        let log = len.ilog2();
        let sl = u32::try_from((len >> (log - SL_LOG2)) - u64::from(SL_COUNT)).ok()?;
        let fl = log - SL_LOG2 + 1;
        (fl < FL_COUNT).then_some(Self { fl, sl })
    }
}

/// The lowest non-empty class at or above `from`. `first` has one bit per first level;
/// `second_of(fl)` returns that first level's subclass bits.
pub(super) fn find_at_or_above(
    first: u32,
    from: Class,
    second_of: impl Fn(u32) -> u32,
) -> Option<Class> {
    let here = second_of(from.fl) & (u32::MAX << from.sl);
    if here != 0 {
        return Some(Class {
            fl: from.fl,
            sl: here.trailing_zeros(),
        });
    }
    let above = first & u32::MAX.checked_shl(from.fl + 1).unwrap_or(0);
    if above == 0 {
        return None;
    }
    let fl = above.trailing_zeros();
    Some(Class {
        fl,
        sl: second_of(fl).trailing_zeros(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn len(n: u32) -> SpanLen {
        SpanLen::new(n).unwrap()
    }

    /// The inclusive length range of a class.
    fn bounds(class: Class) -> (u64, u64) {
        if class.fl == 0 {
            return (u64::from(class.sl), u64::from(class.sl));
        }
        let log = class.fl + SL_LOG2 - 1;
        let step = 1u64 << (log - SL_LOG2);
        let lower = (1u64 << log) + u64::from(class.sl) * step;
        (lower, lower + step - 1)
    }

    #[test]
    fn every_length_falls_in_its_class() {
        for n in (1..5_000).chain([u32::MAX / 2, u32::MAX - 1, u32::MAX]) {
            let class = Class::containing(len(n));
            let (lo, hi) = bounds(class);
            assert!(lo <= u64::from(n) && u64::from(n) <= hi, "{n} in {class:?}");
        }
    }

    #[test]
    fn guaranteed_class_holds_the_request_and_is_tight() {
        for n in 1..5_000 {
            let class = Class::guaranteeing(len(n)).unwrap();
            let (lo, _) = bounds(class);
            assert!(lo >= u64::from(n), "{n}: {class:?} starts at {lo}");
            // Rounding loses at most one subclass step: under 1/16 of the request.
            assert!((lo - u64::from(n)) * 16 < u64::from(n).max(16), "{n}: {lo}");
        }
        assert_eq!(Class::guaranteeing(len(u32::MAX)), None);
    }

    #[test]
    fn classes_are_ordered_by_index() {
        let mut previous = 0;
        for n in 1..5_000 {
            let index = Class::containing(len(n)).index();
            assert!(index >= previous && index < CLASS_COUNT);
            previous = index;
        }
    }

    #[test]
    fn search_finds_lowest_non_empty_class_at_or_above() {
        let target = Class::containing(len(100));
        let words = |fl: u32| if fl == target.fl { 1 << target.sl } else { 0 };
        let first = 1 << target.fl;
        let from_below = Class::containing(len(70));
        assert_eq!(find_at_or_above(first, from_below, words), Some(target));
        let from_above = Class::containing(len(120));
        assert_eq!(find_at_or_above(first, from_above, words), None);
    }
}
