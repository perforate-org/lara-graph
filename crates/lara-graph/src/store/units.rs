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
