//! A bucket's descriptor (16 bytes): where its edges live. The only place that knows its wire
//! layout.
//!
//! Positions count tombstones: a bucket's `stored` positions are `0..stored`, `live` of them hold
//! an edge. The three modes are exclusive, so they are one enum.
//!
//! # Layouts
//!
//! Each is 16 bytes. A slot is a node id, `u32::MAX` for a tombstone.
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
//! Reserved space                  ↕ 4 bytes
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
use crate::id::NodeId;
use crate::page_store::{EXTENT_BYTES, Extent};

/// Bytes per descriptor.
pub(crate) const DESCRIPTOR_BYTES: usize = 16;
/// Slots stored inline in a tiny descriptor.
pub(crate) const TINY_SLOTS: usize = 3;
/// Slots in a page, and in the largest run.
pub(crate) const PAGE_SLOTS: u32 = 1024;
/// Bytes per slot.
pub(crate) const SLOT_BYTES: u32 = 4;

/// A stored slot: an edge, or the tombstone of a removed one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    Live(NodeId),
    Tombstone,
}

/// The reserved slot value of a tombstone; never a node id.
pub(crate) const TOMBSTONE: u32 = u32::MAX;

impl Slot {
    pub(crate) fn encode(self) -> [u8; 4] {
        match self {
            Self::Live(v) => v.index().to_le_bytes(),
            Self::Tombstone => TOMBSTONE.to_le_bytes(),
        }
    }

    pub(crate) fn decode(bytes: [u8; 4]) -> Self {
        match u32::from_le_bytes(bytes) {
            TOMBSTONE => Self::Tombstone,
            raw => Self::Live(NodeId::new(raw)),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Run {
    pub(crate) extent: Extent,
    pub(crate) stored: u16,
    pub(crate) live: u16,
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
        self.extent.len() / SLOT_BYTES
    }
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
                [
                    KIND_RUN, s0, s1, l0, l1, e0, e1, e2, e3, e4, e5, e6, 0, 0, 0, 0,
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
