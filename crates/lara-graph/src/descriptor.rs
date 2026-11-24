//! A bucket's descriptor (13 bytes): where its edges live. The only place that knows its wire
//! layout.
//!
//! Positions count tombstones: a bucket's `stored` positions are `0..stored`, `live` of them hold
//! an edge. The three modes are exclusive, so they are one enum.
//!
//! # Layouts
//!
//! Each is 13 bytes (104 bits): the largest, a paged bucket, takes them all. The kind is the top
//! two bits of the last byte, where an extent's spare bits fall, and every extent sits in the last
//! five bytes. A slot is a node id's complement (`!index`), 0 for a tombstone; all zeros is an
//! empty tiny bucket.
//!
//! ```text
//! ---------------------------------------- <- Tiny (kind 0)
//! Slot 0, 1, 2                    ↕ 3 × 4 bytes
//! ----------------------------------------
//! Reserved                        ↕ 4 bits
//! Slots in use (0 to 3)           ↕ 2 bits
//! Kind                            ↕ 2 bits
//! ---------------------------------------- <- Run (kind 1): slots at the front of one extent
//! Stored positions                ↕ 11 bits
//! Live positions                  ↕ 11 bits
//! Value width beside the slots    ↕ 13 bits       (0 for none)
//! Reserved                        ↕ 29 bits
//! ----------------------------------------
//! Extent, kind in its spare bits  ↕ 5 bytes
//! ---------------------------------------- <- Paged (kind 2): 1024 slots per data page
//! Stored positions                ↕ 4 bytes
//! ----------------------------------------
//! Live positions                  ↕ 4 bytes
//! ----------------------------------------
//! Page index top extent, kind     ↕ 5 bytes
//! ----------------------------------------
//! ```
//!
//! A list entry's kind 3 refers to a header instead (see `stored`).

use crate::corrupt::corrupt;
use crate::id::{NodeId, Width};
use crate::page_store::{ByteClass, EXTENT_BYTES, Extent, PAGE_BYTES};
use crate::units::ByteOffset;
use core::num::NonZeroU32;

/// Bytes per descriptor.
pub(crate) const DESCRIPTOR_BYTES: usize = 13;
/// Slots stored inline in a tiny descriptor.
pub(crate) const TINY_SLOTS: usize = 3;
/// Slots in a page, and in the largest run.
pub(crate) const PAGE_SLOTS: u32 = 1024;
/// Bytes per slot.
pub(crate) const SLOT_BYTES: u32 = 4;
/// The fewest slots a page-sized run with values beside them must hold: a label whose values are
/// wider keeps them in a value plane instead.
const MIN_BESIDE_CAPACITY: u32 = 32;

/// Whether a small bucket of a label of `width` keeps its values beside its slots.
pub(crate) fn beside(width: Width) -> bool {
    max_beside(width) >= MIN_BESIDE_CAPACITY
}

/// The most slots a run with values of `width` beside them holds: those of a page.
pub(crate) fn max_beside(width: Width) -> u32 {
    PAGE_BYTES / (SLOT_BYTES + u32::from(width.get()))
}

/// A stored slot: an edge, or the tombstone of a removed one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    Live(NodeId),
    Tombstone,
}

/// The stored value of a tombstone: zero, the complement of no node's index (an edge stores its
/// target's complement), so a slot decodes with one comparison and zeroed bytes read as holes.
pub(crate) const TOMBSTONE: u32 = 0;

impl Slot {
    pub(crate) fn encode(self) -> [u8; 4] {
        match self {
            Self::Live(v) => v.complement().get().to_le_bytes(),
            Self::Tombstone => TOMBSTONE.to_le_bytes(),
        }
    }

    pub(crate) fn decode(bytes: [u8; 4]) -> Self {
        match NonZeroU32::new(u32::from_le_bytes(bytes)) {
            Some(complement) => Self::Live(NodeId::from_complement(complement)),
            None => Self::Tombstone,
        }
    }

    /// Decodes consecutive slots.
    pub(crate) fn decode_all(bytes: &[u8]) -> impl Iterator<Item = Self> + '_ {
        bytes.as_chunks::<4>().0.iter().map(|&w| Self::decode(w))
    }
}

/// Where a bucket's edges live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Descriptor {
    /// Up to [`TINY_SLOTS`] slots inside the descriptor.
    Tiny(Tiny),
    /// Slots at the front of one extent of up to one page.
    Run(Run),
    /// More than one page of slots: full pages listed by an index (see `paged`).
    Paged(Paged),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tiny {
    len: u8,
    slots: [Slot; TINY_SLOTS],
}

/// Slots at the front of one extent, and when `values` is set, a value of that width per slot
/// after them: positions `0..capacity` are slots, then `capacity` cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Run {
    pub(crate) extent: Extent,
    pub(crate) stored: u16,
    pub(crate) live: u16,
    pub(crate) values: Option<Width>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Paged {
    /// The top level of the page index.
    pub(crate) index: Extent,
    pub(crate) stored: u32,
    pub(crate) live: u32,
}

impl Run {
    /// Slots the run's extent holds.
    pub(crate) fn capacity(&self) -> u32 {
        self.extent.len() / slot_and_cell(self.values)
    }

    /// Where the cells beside the slots start.
    pub(crate) fn cells_offset(&self) -> ByteOffset {
        ByteOffset::new(self.extent.offset().get() + u64::from(self.capacity() * SLOT_BYTES))
    }

    /// Whether the run can grow into a larger class.
    pub(crate) fn can_grow(&self) -> bool {
        self.extent.class() != ByteClass::PAGE
    }
}

/// Bytes a position takes in a run: its slot, and its cell when values sit beside the slots.
fn slot_and_cell(values: Option<Width>) -> u32 {
    SLOT_BYTES + values.map_or(0, |w| u32::from(w.get()))
}

/// The class of a run of `slots` slots, with values of `values` beside them if set.
pub(crate) fn run_class(slots: u32, values: Option<Width>) -> ByteClass {
    ByteClass::for_bytes(slots * slot_and_cell(values))
        .unwrap_or_else(|| corrupt("run beyond a page"))
}

impl Tiny {
    pub(crate) const EMPTY: Self = Self {
        len: 0,
        slots: [Slot::Tombstone; TINY_SLOTS],
    };

    pub(crate) fn slots(&self) -> &[Slot] {
        self.slots.get(..usize::from(self.len)).unwrap_or(&[])
    }

    pub(crate) fn is_full(&self) -> bool {
        usize::from(self.len) >= TINY_SLOTS
    }

    /// Appends a slot. The caller checks [`Self::is_full`] first.
    pub(crate) fn push(mut self, slot: Slot) -> Self {
        match self.slots.get_mut(usize::from(self.len)) {
            Some(dst) => *dst = slot,
            None => corrupt("tiny bucket overfilled"),
        }
        self.len += 1;
        self
    }

    /// Replaces the stored slot at `at`.
    pub(crate) fn set(mut self, at: usize, slot: Slot) -> Self {
        match self
            .slots
            .get_mut(at)
            .filter(|_| at < usize::from(self.len))
        {
            Some(dst) => *dst = slot,
            None => corrupt("tiny slot out of range"),
        }
        self
    }
}

const KIND_TINY: u8 = 0;
const KIND_RUN: u8 = 1;
const KIND_PAGED: u8 = 2;
/// A list entry's reference to a header (`stored`): never a descriptor's kind.
pub(crate) const KIND_HEADER: u8 = 3;
/// Where a descriptor's kind sits: the top two bits of its last byte.
pub(crate) const KIND_SHIFT: u32 = 6;
const LAST: usize = DESCRIPTOR_BYTES - 1;
/// Bits of a run's stored and live counts (at most a page of slots, 1024) and of its width.
const RUN_COUNT_BITS: u32 = 11;
const RUN_WIDTH_BITS: u32 = 13;
const _: () = assert!(PAGE_SLOTS < 1 << RUN_COUNT_BITS);
const _: () = assert!(4096 < 1 << RUN_WIDTH_BITS);

/// The kind of an encoded descriptor or list entry.
pub(crate) const fn kind_of(bytes: &[u8; DESCRIPTOR_BYTES]) -> u8 {
    bytes[LAST] >> KIND_SHIFT
}

impl Descriptor {
    pub(crate) const EMPTY: Self = Self::Tiny(Tiny::EMPTY);

    /// Positions in use, tombstones included.
    pub(crate) fn stored(&self) -> u32 {
        match self {
            Self::Tiny(tiny) => u32::from(tiny.len),
            Self::Run(run) => u32::from(run.stored),
            Self::Paged(paged) => paged.stored,
        }
    }

    /// Live edges.
    pub(crate) fn live(&self) -> u32 {
        match self {
            Self::Tiny(tiny) => {
                let live = tiny
                    .slots()
                    .iter()
                    .filter(|s| matches!(s, Slot::Live(_)))
                    .count();
                u32::try_from(live).unwrap_or_else(|_| corrupt("tiny live count"))
            }
            Self::Run(run) => u32::from(run.live),
            Self::Paged(paged) => paged.live,
        }
    }

    /// As the module's layout diagrams show.
    pub(crate) fn encode(&self) -> [u8; DESCRIPTOR_BYTES] {
        match self {
            Self::Tiny(tiny) => {
                let [[s0, s1, s2, s3], [s4, s5, s6, s7], [s8, s9, s10, s11]] =
                    tiny.slots.map(Slot::encode);
                let last = (tiny.len << 4) | (KIND_TINY << KIND_SHIFT);
                [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, last]
            }
            Self::Run(run) => {
                let counts = u64::from(run.stored)
                    | (u64::from(run.live) << RUN_COUNT_BITS)
                    | (u64::from(run.values.map_or(0, Width::get)) << (2 * RUN_COUNT_BITS));
                let [c0, c1, c2, c3, c4, ..] = counts.to_le_bytes();
                let [e0, e1, e2, e3, e4] = run.extent.encode();
                [
                    c0,
                    c1,
                    c2,
                    c3,
                    c4,
                    0,
                    0,
                    0,
                    e0,
                    e1,
                    e2,
                    e3,
                    e4 | (KIND_RUN << KIND_SHIFT),
                ]
            }
            Self::Paged(paged) => {
                let [s0, s1, s2, s3] = paged.stored.to_le_bytes();
                let [l0, l1, l2, l3] = paged.live.to_le_bytes();
                let [e0, e1, e2, e3, e4] = paged.index.encode();
                [
                    s0,
                    s1,
                    s2,
                    s3,
                    l0,
                    l1,
                    l2,
                    l3,
                    e0,
                    e1,
                    e2,
                    e3,
                    e4 | (KIND_PAGED << KIND_SHIFT),
                ]
            }
        }
    }

    /// Shifts and masks only, as every bucket lookup goes through here.
    pub(crate) fn decode(bytes: [u8; DESCRIPTOR_BYTES]) -> Self {
        let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12] = bytes;
        match b12 >> KIND_SHIFT {
            KIND_TINY => {
                let words = [[b0, b1, b2, b3], [b4, b5, b6, b7], [b8, b9, b10, b11]];
                Self::Tiny(Tiny {
                    len: (b12 >> 4) & 0b11,
                    slots: words.map(Slot::decode),
                })
            }
            KIND_RUN => {
                let counts = u64::from_le_bytes([b0, b1, b2, b3, b4, 0, 0, 0]);
                let field = |shift: u32, width: u32| (counts >> shift) & ((1 << width) - 1);
                Self::Run(Run {
                    stored: u16::try_from(field(0, RUN_COUNT_BITS)).unwrap_or(0),
                    live: u16::try_from(field(RUN_COUNT_BITS, RUN_COUNT_BITS)).unwrap_or(0),
                    extent: Extent::decode([b8, b9, b10, b11, b12]),
                    values: Width::stored(
                        u16::try_from(field(2 * RUN_COUNT_BITS, RUN_WIDTH_BITS)).unwrap_or(0),
                    ),
                })
            }
            KIND_PAGED => Self::Paged(Paged {
                stored: u32::from_le_bytes([b0, b1, b2, b3]),
                live: u32::from_le_bytes([b4, b5, b6, b7]),
                index: Extent::decode([b8, b9, b10, b11, b12]),
            }),
            _ => corrupt("a header reference where a descriptor belongs"),
        }
    }
}

const _: () = assert!(4 + 4 + EXTENT_BYTES == DESCRIPTOR_BYTES);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_store::{ByteClass, PageId};

    #[test]
    fn descriptors_round_trip() {
        let extent = Extent::new(PageId::new(9), 3, ByteClass::for_bytes(64).unwrap());
        let tiny = Tiny::EMPTY
            .push(Slot::Live(NodeId::new(7)))
            .push(Slot::Tombstone);
        for d in [
            Descriptor::EMPTY,
            Descriptor::Tiny(tiny),
            Descriptor::Run(Run {
                extent,
                stored: 9,
                live: 4,
                values: None,
            }),
            Descriptor::Run(Run {
                extent,
                stored: 3,
                live: 2,
                values: Width::new(8),
            }),
            Descriptor::Paged(Paged {
                index: extent,
                stored: 5000,
                live: 4321,
            }),
        ] {
            assert_eq!(Descriptor::decode(d.encode()), d);
        }
        assert_eq!(Descriptor::Tiny(tiny).live(), 1);
    }

    /// Every field round-trips at its largest value, and the kind bits never disturb the extent.
    #[test]
    fn descriptors_round_trip_at_their_limits() {
        let last = PageId::new(crate::page_store::MAX_PAGES - 1);
        let extent = Extent::new(last, 127, ByteClass::SMALLEST);
        let full = Tiny::EMPTY
            .push(Slot::Live(NodeId::new(u32::MAX - 1)))
            .push(Slot::Tombstone)
            .push(Slot::Live(NodeId::new(0)));
        for d in [
            Descriptor::Tiny(full),
            Descriptor::Run(Run {
                extent,
                stored: 1024,
                live: 1024,
                values: Width::new(Width::MAX),
            }),
            Descriptor::Paged(Paged {
                index: Extent::whole(last),
                stored: u32::MAX,
                live: u32::MAX,
            }),
        ] {
            assert_eq!(Descriptor::decode(d.encode()), d);
        }
    }

    #[test]
    fn an_unwritten_descriptor_is_an_empty_tiny_bucket() {
        assert_eq!(Descriptor::decode([0; DESCRIPTOR_BYTES]), Descriptor::EMPTY);
        assert_eq!(Descriptor::EMPTY.encode(), [0; DESCRIPTOR_BYTES]);
    }
}
