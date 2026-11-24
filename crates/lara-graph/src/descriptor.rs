//! A bucket's descriptor (16 bytes): where its edges live. The only place that knows its wire
//! layout.
//!
//! Positions count tombstones: a bucket's `stored` positions are `0..stored`, `live` of them hold
//! an edge. The three modes are exclusive, so they are one enum.
//!
//! # Layouts
//!
//! Each is 16 bytes. A slot is a node id's complement (`!index`), 0 for a tombstone.
//!
//! ```text
//! ---------------------------------------- <- Tiny
//! Kind (1)                        ↕ 1 byte
//! ----------------------------------------
//! Slots in use                    ↕ 1 byte
//! ----------------------------------------
//! Reserved space                  ↕ 2 bytes
//! ----------------------------------------
//! Slot 0, 1, 2                    ↕ 3 × 4 bytes
//! ---------------------------------------- <- Run: slots at the front of one extent
//! Kind (2)                        ↕ 1 byte
//! ----------------------------------------
//! Stored positions                ↕ 2 bytes
//! ----------------------------------------
//! Live positions                  ↕ 2 bytes
//! ----------------------------------------
//! Extent                          ↕ 7 bytes
//! ----------------------------------------
//! Value width beside the slots    ↕ 2 bytes       (0 for none)
//! ----------------------------------------
//! Reserved space                  ↕ 2 bytes
//! ---------------------------------------- <- Paged: 1024 slots per data page
//! Kind (3)                        ↕ 1 byte
//! ----------------------------------------
//! Stored positions                ↕ 4 bytes
//! ----------------------------------------
//! Live positions                  ↕ 4 bytes
//! ----------------------------------------
//! Page index top extent           ↕ 7 bytes
//! ----------------------------------------
//! ```

use crate::corrupt::corrupt;
use crate::id::{NodeId, Width};
use crate::page_store::{ByteClass, EXTENT_BYTES, Extent, PAGE_BYTES};
use crate::units::ByteOffset;
use core::num::NonZeroU32;

/// Bytes per descriptor.
pub(crate) const DESCRIPTOR_BYTES: usize = 16;
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

const KIND_TINY: u8 = 1;
const KIND_RUN: u8 = 2;
const KIND_PAGED: u8 = 3;

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
                [
                    KIND_TINY, tiny.len, 0, 0, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11,
                ]
            }
            Self::Run(run) => {
                let [s0, s1] = run.stored.to_le_bytes();
                let [l0, l1] = run.live.to_le_bytes();
                let [e0, e1, e2, e3, e4, e5, e6] = run.extent.encode();
                let [w0, w1] = run.values.map_or(0, Width::get).to_le_bytes();
                [
                    KIND_RUN, s0, s1, l0, l1, e0, e1, e2, e3, e4, e5, e6, w0, w1, 0, 0,
                ]
            }
            Self::Paged(paged) => {
                let [s0, s1, s2, s3] = paged.stored.to_le_bytes();
                let [l0, l1, l2, l3] = paged.live.to_le_bytes();
                let [e0, e1, e2, e3, e4, e5, e6] = paged.index.encode();
                [
                    KIND_PAGED, s0, s1, s2, s3, l0, l1, l2, l3, e0, e1, e2, e3, e4, e5, e6,
                ]
            }
        }
    }

    pub(crate) fn decode(bytes: [u8; DESCRIPTOR_BYTES]) -> Self {
        let [
            kind,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
        ] = bytes;
        match kind {
            KIND_TINY => {
                let words = [[b4, b5, b6, b7], [b8, b9, b10, b11], [b12, b13, b14, b15]];
                if usize::from(b1) > TINY_SLOTS {
                    corrupt("tiny length");
                }
                Self::Tiny(Tiny {
                    len: b1,
                    slots: words.map(Slot::decode),
                })
            }
            KIND_RUN => Self::Run(Run {
                stored: u16::from_le_bytes([b1, b2]),
                live: u16::from_le_bytes([b3, b4]),
                extent: Extent::decode([b5, b6, b7, b8, b9, b10, b11]),
                values: Width::stored(u16::from_le_bytes([b12, b13])),
            }),
            KIND_PAGED => Self::Paged(Paged {
                stored: u32::from_le_bytes([b1, b2, b3, b4]),
                live: u32::from_le_bytes([b5, b6, b7, b8]),
                index: Extent::decode([b9, b10, b11, b12, b13, b14, b15]),
            }),
            _ => corrupt("descriptor kind"),
        }
    }
}

const _: () = assert!(1 + 4 + 4 + EXTENT_BYTES == DESCRIPTOR_BYTES);

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

    #[test]
    #[should_panic(expected = "descriptor kind")]
    fn an_unwritten_descriptor_is_corruption() {
        Descriptor::decode([0; DESCRIPTOR_BYTES]);
    }
}
