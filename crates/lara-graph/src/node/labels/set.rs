//! A node's labels: a set of [`Label`]s in 5 bytes of the node labels memory, and beyond.
//!
//! Most nodes have one label or two, so the cell holds up to two inline. A larger set goes to the
//! label bucket list store: up to [`MAX_LISTED`] as a sorted list in one extent, more as a bitmap
//! of every label (65,536 bits, two pages) behind a small header. Any set of labels fits, and a
//! replacement copies at most two pages. The kind sits in the spare bits of an extent's last
//! byte, so all zeros is a node without labels.
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
//! # Listed set (an extent of up to a page)
//!
//! ```text
//! ---------------------------------------- <- Extent start
//! Label count = n (3 to 2047)     ↕ 2 bytes
//! ----------------------------------------
//! Labels, ascending               ↕ n × 2 bytes
//! ----------------------------------------
//! ```
//!
//! # Bitmap set (a 32-byte header, two pages)
//!
//! ```text
//! ---------------------------------------- <- Header start
//! Mark 0xFFFF                     ↕ 2 bytes
//! ----------------------------------------
//! Label count                     ↕ 4 bytes
//! ----------------------------------------
//! Page of labels 0 to 32767       ↕ 4 bytes
//! ----------------------------------------
//! Page of labels 32768 to 65535   ↕ 4 bytes
//! ----------------------------------------
//! ```
//!
//! Label `l` is bit `l % 8` of byte `(l % 32768) / 8` of page `l / 32768`.

use crate::id::Label;
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::{
    ByteClass, EXTENT_BYTES, Extent, PAGE_BYTES, PageId, PageStore, Pages,
};
use crate::store::read_buf::ReadBuf;
use crate::store::txn::SpanTxn;
use crate::store::units::ByteOffset;
use alloc::vec::Vec;

/// Bytes of a node's label cell.
pub(crate) const CELL_BYTES: usize = EXTENT_BYTES;
/// The most labels a listed set holds: its count and labels fill one page.
pub(crate) const MAX_LISTED: usize = (4096 - 2) / 2;
const _: () = assert!(PAGE_BYTES == 4096);
const KIND_SHIFT: u32 = 6;
const NONE: u8 = 0;
const ONE: u8 = 1;
const TWO: u8 = 2;
/// The first two bytes of a bitmap set's header: never a listed count.
const BITMAP_MARK: u16 = u16::MAX;
const BITMAP_HEADER_BYTES: usize = 14;
/// Labels per bitmap page: 4096 bytes of 8 bits.
const LABELS_PER_PAGE: u32 = 4096 * 8;
const _: () = assert!(MAX_LISTED < 0xFFFF);

/// A node's label cell, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LabelSet {
    None,
    One(Label),
    Two(Label, Label),
    Spilled(Extent),
}

/// What a spilled set's extent holds.
enum Spill {
    Listed(usize),
    Bitmap([PageId; 2]),
}

impl LabelSet {
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
                [e0, e1, e2, e3, e4 | (0b11 << KIND_SHIFT)]
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

    /// The cell for `set` (sorted, distinct), writing a larger set to `store`. On error nothing
    /// was allocated.
    pub(crate) fn store<M: Memory>(
        store: &mut PageStore<M>,
        set: &[Label],
    ) -> Result<Self, WriteError> {
        Ok(match *set {
            [] => Self::None,
            [a] => Self::One(a),
            [a, b] => Self::Two(a, b),
            _ if set.len() <= MAX_LISTED => Self::Spilled(listed(store, set)?),
            _ => Self::Spilled(bitmap(store, set)?),
        })
    }

    /// Gives back what [`Self::store`] took. Cannot fail.
    pub(crate) fn release<M: Memory>(self, store: &mut PageStore<M>) {
        let Self::Spilled(extent) = self else { return };
        if let Spill::Bitmap(pages) = spill(&store.pages(), extent) {
            for page in pages {
                store.release(Extent::whole(page));
            }
        }
        store.release(extent);
    }

    /// Whether the set holds `label`: a comparison or two inline, a binary search of a listed
    /// set, one bit of a bitmap.
    pub(crate) fn contains<M: Memory>(self, view: &Pages<'_, M>, label: Label) -> bool {
        match self {
            Self::None => false,
            Self::One(a) => a == label,
            Self::Two(a, b) => a == label || b == label,
            Self::Spilled(extent) => match spill(view, extent) {
                Spill::Listed(count) => {
                    let mut buf = ReadBuf::new();
                    let labels = buf.fill_inlined(view, after_count(extent), count * 2);
                    labels
                        .as_chunks::<2>()
                        .0
                        .binary_search_by(|&l| u16::from_le_bytes(l).cmp(&label.get()))
                        .is_ok()
                }
                Spill::Bitmap(pages) => {
                    let (page, byte, bit) = bit_of(label);
                    let page = pages
                        .get(page)
                        .copied()
                        .unwrap_or_else(|| corrupt("bitmap"));
                    let [b] = view.read_array(ByteOffset::new(page.offset().get() + byte));
                    b & (1 << bit) != 0
                }
            },
        }
    }

    /// Appends the set, ascending, to `out`.
    pub(crate) fn extend_into<M: Memory>(self, view: &Pages<'_, M>, out: &mut Vec<Label>) {
        match self {
            Self::None => {}
            Self::One(a) => out.push(a),
            Self::Two(a, b) => out.extend([a, b]),
            Self::Spilled(extent) => match spill(view, extent) {
                Spill::Listed(count) => listed_into(view, extent, count, out),
                Spill::Bitmap(pages) => bitmap_into(view, pages, out),
            },
        }
    }

    /// Every extent the set owns (test invariant checks).
    #[cfg(test)]
    pub(crate) fn extents<M: Memory>(self, view: &Pages<'_, M>) -> Vec<Extent> {
        let Self::Spilled(extent) = self else {
            return Vec::new();
        };
        let mut out = alloc::vec![extent];
        if let Spill::Bitmap(pages) = spill(view, extent) {
            out.extend(pages.map(Extent::whole));
        }
        out
    }
}

/// What a spilled set's extent holds, from its first bytes.
fn spill<M: Memory>(view: &Pages<'_, M>, extent: Extent) -> Spill {
    let [m0, m1, c0, c1, c2, c3, p0, p1, p2, p3, q0, q1, q2, q3]: [u8; BITMAP_HEADER_BYTES] =
        if usize::try_from(extent.len()).is_ok_and(|len| len >= BITMAP_HEADER_BYTES) {
            view.read_array(extent.offset())
        } else {
            corrupt("a spilled label set shorter than a bitmap header")
        };
    // The bitmap's label count is kept for readers that want it without counting bits.
    let _ = [c0, c1, c2, c3];
    match u16::from_le_bytes([m0, m1]) {
        BITMAP_MARK => Spill::Bitmap([
            PageId::new(u32::from_le_bytes([p0, p1, p2, p3])),
            PageId::new(u32::from_le_bytes([q0, q1, q2, q3])),
        ]),
        count => Spill::Listed(usize::from(count)),
    }
}

/// Appends a listed set's `count` labels.
fn listed_into<M: Memory>(view: &Pages<'_, M>, extent: Extent, count: usize, out: &mut Vec<Label>) {
    let bytes = view.read(after_count(extent), count * 2);
    out.extend(
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&l| Label::new(u16::from_le_bytes(l))),
    );
}

/// Appends a bitmap set's labels, ascending: each set bit, lowest first.
fn bitmap_into<M: Memory>(view: &Pages<'_, M>, pages: [PageId; 2], out: &mut Vec<Label>) {
    for (base, page) in (0u32..).step_by(32768).zip(pages) {
        let bytes = view.read(page.offset(), 4096);
        for (at, &byte) in (base..).step_by(8).zip(&bytes) {
            let mut rest = byte;
            while rest != 0 {
                let label = u16::try_from(at + rest.trailing_zeros()).unwrap_or(u16::MAX);
                out.push(Label::new(label));
                rest &= rest - 1;
            }
        }
    }
}

fn after_count(extent: Extent) -> ByteOffset {
    ByteOffset::new(extent.offset().get() + 2)
}

/// Page, byte and bit of `label` in a bitmap set.
fn bit_of(label: Label) -> (usize, u64, u32) {
    let l = u32::from(label.get());
    let page = usize::try_from(l / LABELS_PER_PAGE).unwrap_or(0);
    (page, u64::from(l % LABELS_PER_PAGE / 8), l % 8)
}

/// A listed set's extent: its count, then its labels.
fn listed<M: Memory>(store: &mut PageStore<M>, set: &[Label]) -> Result<Extent, WriteError> {
    let count = u16::try_from(set.len()).unwrap_or(u16::MAX);
    let bytes: Vec<u8> = count
        .to_le_bytes()
        .into_iter()
        .chain(set.iter().flat_map(|l| l.get().to_le_bytes()))
        .collect();
    let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    let extent = store.allocate(ByteClass::for_bytes(len).unwrap_or(ByteClass::PAGE))?;
    store.write(extent.offset(), &bytes);
    Ok(extent)
}

/// A bitmap set: its header, then two pages of bits.
fn bitmap<M: Memory>(store: &mut PageStore<M>, set: &[Label]) -> Result<Extent, WriteError> {
    let mut txn = SpanTxn::begin(store);
    let header = txn.allocate(ByteClass::B32)?;
    let low = txn.allocate(ByteClass::PAGE)?.page();
    let high = txn.allocate(ByteClass::PAGE)?.page();
    txn.commit();
    let mut bits = [[0u8; 4096]; 2];
    for &label in set {
        let (page, byte, bit) = bit_of(label);
        if let Some(b) = bits
            .get_mut(page)
            .and_then(|p| p.get_mut(usize::try_from(byte).unwrap_or(usize::MAX)))
        {
            *b |= 1 << bit;
        }
    }
    store.write(low.offset(), &bits[0]);
    store.write(high.offset(), &bits[1]);
    let [m0, m1] = BITMAP_MARK.to_le_bytes();
    let [c0, c1, c2, c3] = u32::try_from(set.len()).unwrap_or(u32::MAX).to_le_bytes();
    let [p0, p1, p2, p3] = low.get().to_le_bytes();
    let [q0, q1, q2, q3] = high.get().to_le_bytes();
    store.write(
        header.offset(),
        &[m0, m1, c0, c1, c2, c3, p0, p1, p2, p3, q0, q1, q2, q3],
    );
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_round_trip_and_zero_is_no_labels() {
        assert_eq!(LabelSet::decode([0; CELL_BYTES]), LabelSet::None);
        let extent = Extent::new(PageId::new(77), 5, ByteClass::SMALLEST);
        for cell in [
            LabelSet::None,
            LabelSet::One(Label::new(0)),
            LabelSet::One(Label::new(u16::MAX)),
            LabelSet::Two(Label::new(3), Label::new(u16::MAX)),
            LabelSet::Spilled(extent),
        ] {
            assert_eq!(LabelSet::decode(cell.encode()), cell);
        }
    }
}

/// Proofs over every input, checked by Kani (`cargo kani -p lara-graph`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Any five bytes decode to a label cell that encodes back to itself.
    #[kani::proof]
    fn label_cells_round_trip() {
        let cell = LabelSet::decode(kani::any());
        assert_eq!(LabelSet::decode(cell.encode()), cell);
    }
}
