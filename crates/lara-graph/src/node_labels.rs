//! A node's labels: a set of [`Label`]s beside its record, in 5 bytes.
//!
//! Most nodes have one label or two, so the cell holds up to two inline; a larger set goes to an
//! extent in the label bucket list store, sorted, behind its count. The kind sits in the spare
//! bits of an extent's last byte, so all zeros is a node without labels.
//!
//! # Cell
//!
//! ```text
//! ---------------------------------------- <- Bit 0
//! Kind 0: nothing                 ↕ 38 bits
//! Kind 1: a label                 ↕ 16 bits, then zeros
//! Kind 2: two labels, ascending   ↕ 2 × 16 bits, then zeros
//! Kind 3: an extent               ↕ 38 bits
//! ---------------------------------------- <- Bit 38
//! Kind                            ↕ 2 bits
//! ----------------------------------------
//! ```
//!
//! # Spilled set
//!
//! ```text
//! ---------------------------------------- <- Extent start
//! Label count = n                 ↕ 2 bytes
//! ----------------------------------------
//! Labels, ascending               ↕ n × 2 bytes
//! ----------------------------------------
//! ```

use crate::id::Label;
use crate::page_store::{EXTENT_BYTES, Extent, PAGE_BYTES, Pages};

/// Bytes of a node's label cell.
pub(crate) const CELL_BYTES: usize = EXTENT_BYTES;
/// The most labels a node can have: a spilled set fills at most one page.
pub(crate) const MAX_NODE_LABELS: usize = (4096 - 2) / 2;
const _: () = assert!(PAGE_BYTES == 4096);
const KIND_SHIFT: u32 = 6;
const NONE: u8 = 0;
const ONE: u8 = 1;
const TWO: u8 = 2;
const SPILLED: u8 = 3;

/// A node's label cell, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NodeLabels {
    None,
    One(Label),
    Two(Label, Label),
    Spilled(Extent),
}

impl NodeLabels {
    /// As the module's Cell diagram shows.
    pub(crate) fn encode(self) -> [u8; CELL_BYTES] {
        match self {
            Self::None => [0; CELL_BYTES],
            Self::One(a) => {
                let [a0, a1] = a.get().to_le_bytes();
                [a0, a1, 0, 0, ONE << KIND_SHIFT]
            }
            Self::Two(a, b) => {
                let ([a0, a1], [b0, b1]) = (a.get().to_le_bytes(), b.get().to_le_bytes());
                [a0, a1, b0, b1, TWO << KIND_SHIFT]
            }
            Self::Spilled(extent) => {
                let [e0, e1, e2, e3, e4] = extent.encode();
                [e0, e1, e2, e3, e4 | (SPILLED << KIND_SHIFT)]
            }
        }
    }

    /// Shifts and masks only: every label check goes through here.
    pub(crate) fn decode(bytes: [u8; CELL_BYTES]) -> Self {
        let [b0, b1, b2, b3, b4] = bytes;
        match b4 >> KIND_SHIFT {
            NONE => Self::None,
            ONE => Self::One(Label::new(u16::from_le_bytes([b0, b1]))),
            TWO => Self::Two(
                Label::new(u16::from_le_bytes([b0, b1])),
                Label::new(u16::from_le_bytes([b2, b3])),
            ),
            _ => Self::Spilled(Extent::decode(bytes)),
        }
    }

    /// Whether the set holds `label`: a comparison or two inline, a binary search spilled.
    pub(crate) fn contains<M: crate::memory::Memory>(
        self,
        view: &Pages<'_, M>,
        label: Label,
    ) -> bool {
        match self {
            Self::None => false,
            Self::One(a) => a == label,
            Self::Two(a, b) => a == label || b == label,
            Self::Spilled(extent) => {
                let mut found = false;
                spilled(view, extent, |labels| {
                    found = labels
                        .binary_search_by(|&l| u16::from_le_bytes(l).cmp(&label.get()))
                        .is_ok();
                });
                found
            }
        }
    }

    /// Appends the set, ascending, to `out`.
    pub(crate) fn extend_into<M: crate::memory::Memory>(
        self,
        view: &Pages<'_, M>,
        out: &mut alloc::vec::Vec<Label>,
    ) {
        match self {
            Self::None => {}
            Self::One(a) => out.push(a),
            Self::Two(a, b) => out.extend([a, b]),
            Self::Spilled(extent) => {
                spilled(view, extent, |labels| {
                    out.extend(labels.iter().map(|&l| Label::new(u16::from_le_bytes(l))));
                });
            }
        }
    }
}

/// Calls `f` with a spilled set's labels as stored: two bytes each, little-endian, ascending.
fn spilled<M: crate::memory::Memory>(
    view: &Pages<'_, M>,
    extent: Extent,
    f: impl FnOnce(&[[u8; 2]]),
) {
    let count: [u8; 2] = view.read_array(extent.offset());
    let count = usize::from(u16::from_le_bytes(count));
    let at = crate::units::ByteOffset::new(extent.offset().get() + 2);
    let mut buf = crate::read_buf::ReadBuf::new();
    f(buf.fill(view, at, count * 2).as_chunks::<2>().0);
}

/// A spilled set's bytes: its count, then its labels.
pub(crate) fn spill_bytes(labels: &[Label]) -> alloc::vec::Vec<u8> {
    let count = u16::try_from(labels.len()).unwrap_or(u16::MAX);
    count
        .to_le_bytes()
        .into_iter()
        .chain(labels.iter().flat_map(|l| l.get().to_le_bytes()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_store::{ByteClass, PageId};

    #[test]
    fn cells_round_trip_and_zero_is_no_labels() {
        assert_eq!(NodeLabels::decode([0; CELL_BYTES]), NodeLabels::None);
        let extent = Extent::new(PageId::new(77), 5, ByteClass::SMALLEST);
        for cell in [
            NodeLabels::None,
            NodeLabels::One(Label::new(0)),
            NodeLabels::One(Label::new(u16::MAX)),
            NodeLabels::Two(Label::new(3), Label::new(u16::MAX)),
            NodeLabels::Spilled(extent),
        ] {
            assert_eq!(NodeLabels::decode(cell.encode()), cell);
        }
    }
}
