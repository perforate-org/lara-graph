//! Internal units. Byte addresses have their own types so arithmetic on them is checked in one
//! place; the slot units below serve the TLSF span allocator kept for measurement.

use core::fmt;

/// Arithmetic left the representable range. Corrupt input or an exhausted address space; never a
/// normal outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Overflow;

impl fmt::Display for Overflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("storage arithmetic overflow")
    }
}

/// A byte address inside one [`crate::Memory`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ByteOffset(u64);

impl ByteOffset {
    pub(crate) const fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }

    /// The offset `len` bytes further.
    pub(crate) fn advance(self, len: u64) -> Result<Self, Overflow> {
        self.0.checked_add(len).map(Self).ok_or(Overflow)
    }
}

#[cfg(feature = "bench-internals")]
pub(crate) use slots::{Slot, SlotRange, SpanLen};

/// Slab slot units of the TLSF span allocator (`bench-internals` only).
#[cfg(feature = "bench-internals")]
mod slots {
    use super::Overflow;

    /// An absolute position in a slab of slots.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(crate) struct Slot(u64);

    /// A physical width in slots.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(crate) struct SlotCount(u32);

    impl Slot {
        pub(crate) const fn new(raw: u64) -> Self {
            Self(raw)
        }

        pub(crate) const fn get(self) -> u64 {
            self.0
        }

        /// The slot `len` positions after `self`.
        pub(crate) fn advance(self, len: SlotCount) -> Result<Self, Overflow> {
            self.0
                .checked_add(u64::from(len.0))
                .map(Self)
                .ok_or(Overflow)
        }
    }

    impl SlotCount {
        pub(crate) const fn get(self) -> u32 {
            self.0
        }

        /// For differences an invariant already keeps non-negative.
        pub(crate) const fn saturating_sub(self, other: Self) -> Self {
            Self(self.0.saturating_sub(other.0))
        }
    }

    /// The length of an allocated or free span. Never zero.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(crate) struct SpanLen(core::num::NonZeroU32);

    impl SpanLen {
        pub(crate) fn new(len: u32) -> Option<Self> {
            core::num::NonZeroU32::new(len).map(Self)
        }

        pub(crate) const fn get(self) -> u32 {
            self.0.get()
        }

        pub(crate) const fn count(self) -> SlotCount {
            SlotCount(self.0.get())
        }

        /// The combined length of two adjacent spans, if it is representable.
        pub(crate) fn join(self, other: Self) -> Option<Self> {
            self.0.checked_add(other.0.get()).map(Self)
        }
    }

    /// A half-open range of slots, `[start, start + len)`. Its end is representable by
    /// construction.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(crate) struct SlotRange {
        start: Slot,
        len: SlotCount,
    }

    impl SlotRange {
        pub(crate) fn new(start: Slot, len: SlotCount) -> Result<Self, Overflow> {
            start.advance(len)?;
            Ok(Self { start, len })
        }

        pub(crate) const fn start(self) -> Slot {
            self.start
        }

        pub(crate) const fn len(self) -> SlotCount {
            self.len
        }

        pub(crate) const fn is_empty(self) -> bool {
            self.len.0 == 0
        }

        /// One past the last slot.
        pub(crate) fn end(self) -> Slot {
            // Checked in `new`; the fields are private, so the sum cannot overflow here.
            Slot(self.start.0.saturating_add(u64::from(self.len.0)))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn range_end_is_checked_at_construction() {
            assert_eq!(
                SlotRange::new(Slot::new(u64::MAX), SpanLen::new(1).unwrap().count()),
                Err(Overflow)
            );
            let range =
                SlotRange::new(Slot::new(u64::MAX - 1), SpanLen::new(1).unwrap().count()).unwrap();
            assert_eq!(range.end(), Slot::new(u64::MAX));
        }
    }
}
