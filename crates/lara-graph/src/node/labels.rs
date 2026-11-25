//! Node labels as one component: each node's set in the node labels memory (see `set`)
//! and each label's nodes (see `postings`). [`LabelStore::set`] is the one writer of both,
//! so the postings are always exactly the inverse of the sets. Larger sets and the postings live
//! in the label bucket list store, which callers lend.
//!
//! # Node labels memory
//!
//! Apart from the node records, so that a batch's line of eight records stays 48 bytes.
//!
//! ```text
//! ---------------------------------------- <- Address 0
//! Magic "LGL"                     ↕ 3 bytes
//! ----------------------------------------
//! Layout version                  ↕ 1 byte
//! ---------------------------------------- <- Address 4 (AT_POSTINGS)
//! Label postings directory root   ↕ 6 bytes       (a value plane root, see `postings`)
//! ----------------------------------------
//! Reserved space                  ↕ 6 bytes
//! ---------------------------------------- <- Address 16 (CELLS_START)
//! Labels of node 0                ↕ 5 bytes       (see `set`)
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Labels of node N - 1            ↕ 5 bytes
//! ----------------------------------------
//! Unallocated space
//! ```

pub(crate) mod postings;
pub(crate) mod set;

use crate::bucket::values::Plane;
use crate::id::{Label, NodeId};
use crate::store::corrupt::reservation_exceeded;
use crate::store::memory::{Memory, PAGE_SIZE, Region, WriteError};
use crate::store::page_store::{PageStore, Pages};
use crate::store::units::ByteOffset;
use alloc::vec::Vec;
use core::ops::ControlFlow;
use postings::PAGES_PER_ADD;
use set::{CELL_BYTES, LabelSet};

/// The node labels memory's magic: "lara graph labels".
pub(crate) const MAGIC: [u8; 3] = *b"LGL";
const AT_POSTINGS: u64 = 4;
const CELLS_START: u64 = 16;

/// The node labels memory, with its header mirrored.
pub(crate) struct LabelStore<M: Memory> {
    memory: M,
    /// Mirrors the label postings directory's root.
    postings: Plane,
    /// Nodes whose cells the memory already covers, so that adding a node grows it only when it
    /// must.
    room: u32,
}

impl<M: Memory> LabelStore<M> {
    /// Formats `memory` as a store with no labels, overwriting whatever it held.
    pub(crate) fn format(memory: M, version: u8) -> Result<Self, WriteError> {
        let [m0, m1, m2] = MAGIC;
        let region = Region::new(&memory);
        region.write(ByteOffset::new(0), &[m0, m1, m2, version])?;
        region.overwrite(ByteOffset::new(AT_POSTINGS), &Plane::Empty.encode());
        Ok(Self::open(memory))
    }

    /// The store `memory` holds; its header was checked.
    pub(crate) fn open(memory: M) -> Self {
        let postings = Plane::decode(Region::new(&memory).read_array(ByteOffset::new(AT_POSTINGS)));
        let room = room(&memory);
        Self {
            memory,
            postings,
            room,
        }
    }

    pub(crate) fn into_memory(self) -> M {
        self.memory
    }

    /// Grows the memory to cover the cells of `nodes` nodes, when it does not yet. A new cell is
    /// zeros: no labels.
    pub(crate) fn reserve(&mut self, nodes: u32) -> Result<(), WriteError> {
        if nodes > self.room {
            Region::new(&self.memory).reserve(cell_at(nodes))?;
            self.room = room(&self.memory);
        }
        Ok(())
    }

    /// The eight cells from node `base` on, when the memory covers them all.
    fn line(&self, base: u32) -> Option<[u8; LINE_BYTES]> {
        (self.room.saturating_sub(base) >= LINE_CELLS)
            .then(|| Region::new(&self.memory).read_array(cell_at(base)))
    }

    /// `v`'s label cell. `v` is covered.
    pub(crate) fn cell(&self, v: NodeId) -> LabelSet {
        // A proxy has no cell: it has no labels.
        if v.is_proxy() {
            return LabelSet::decode([0; CELL_BYTES]);
        }
        LabelSet::decode(Region::new(&self.memory).read_array::<CELL_BYTES>(cell_at(v.index())))
    }

    /// Replaces `v`'s labels with `set` (sorted, distinct) and updates each label's nodes for the
    /// labels gained and lost. On error the labels and postings are unchanged.
    pub(crate) fn set(
        &mut self,
        lists: &mut PageStore<M>,
        v: NodeId,
        set: &[Label],
    ) -> Result<(), WriteError> {
        let old_cell = self.cell(v);
        let mut old = Vec::new();
        old_cell.extend_into(&lists.pages(), &mut old);
        let (gained, lost) = diff(&old, set);
        let next = LabelSet::store(lists, set)?;
        // Gains first, each with its pages reserved, undone on failure; losses never allocate.
        for (done, &label) in gained.iter().enumerate() {
            if let Err(error) = lists.reserve(PAGES_PER_ADD) {
                for &undo in gained.get(..done).unwrap_or(&[]) {
                    postings::remove(lists, self.postings, undo, v);
                }
                next.release(lists);
                self.write_postings_root();
                return Err(error);
            }
            self.postings = postings::add(lists, self.postings, label, v)
                .unwrap_or_else(|_| reservation_exceeded());
        }
        for &label in &lost {
            postings::remove(lists, self.postings, label, v);
        }
        self.write_postings_root();
        self.write_cell(v, next);
        old_cell.release(lists);
        Ok(())
    }

    /// Drops `v`'s labels and postings. Cannot fail: removals never allocate.
    pub(crate) fn clear(&mut self, lists: &mut PageStore<M>, v: NodeId) {
        let old = self.cell(v);
        if old == LabelSet::None {
            return;
        }
        let mut labels = Vec::new();
        old.extend_into(&lists.pages(), &mut labels);
        for label in labels {
            postings::remove(lists, self.postings, label, v);
        }
        self.write_cell(v, LabelSet::None);
        old.release(lists);
    }

    /// Calls `f` with each node below `end` that has `label`, from `from` on, ascending, until it
    /// breaks.
    pub(crate) fn visit<B>(
        &self,
        lists: &Pages<'_, M>,
        (label, from, end): (Label, u32, u32),
        f: &mut impl FnMut(NodeId) -> ControlFlow<B>,
    ) -> Option<B> {
        postings::visit(lists, self.postings, (label, from, end), f)
    }

    /// How many nodes have `label`.
    pub(crate) fn count(&self, lists: &Pages<'_, M>, label: Label) -> u32 {
        postings::count(lists, self.postings, label)
    }

    /// The postings directory's root (test invariant checks).
    #[cfg(test)]
    pub(crate) const fn postings(&self) -> Plane {
        self.postings
    }

    fn write_cell(&self, v: NodeId, cell: LabelSet) {
        Region::new(&self.memory).overwrite(cell_at(v.index()), &cell.encode());
    }

    fn write_postings_root(&self) {
        Region::new(&self.memory).overwrite(ByteOffset::new(AT_POSTINGS), &self.postings.encode());
    }
}

/// How many nodes' cells `memory` covers as it stands.
fn room<M: Memory>(memory: &M) -> u32 {
    let bytes = memory.size().saturating_mul(PAGE_SIZE);
    let cell = u64::try_from(CELL_BYTES).unwrap_or(5);
    u32::try_from(bytes.saturating_sub(CELLS_START) / cell).unwrap_or(u32::MAX)
}

/// Cells a batch reads at once: 40 bytes for eight dense ids.
const LINE_CELLS: u32 = 8;
const LINE_BYTES: usize = CELL_BYTES * 8;

/// A batch read's line of label cells. Local to one call, so no invalidation is needed.
pub(crate) struct CellLines {
    base: u32,
    bytes: [u8; LINE_BYTES],
}

impl Default for CellLines {
    fn default() -> Self {
        Self {
            base: u32::MAX,
            bytes: [0; LINE_BYTES],
        }
    }
}

impl CellLines {
    /// `v`'s cell, reading one 40-byte line per eight dense ids. `v` is covered.
    #[inline]
    pub(crate) fn cell<M: Memory>(&mut self, store: &LabelStore<M>, v: NodeId) -> LabelSet {
        let base = v.index() / LINE_CELLS * LINE_CELLS;
        if base != self.base {
            // A line the memory does not cover whole takes the single-cell path.
            let Some(line) = store.line(base) else {
                return store.cell(v);
            };
            self.bytes = line;
            self.base = base;
        }
        let at = usize::try_from(v.index() - base).unwrap_or(0) * CELL_BYTES;
        let cell = self
            .bytes
            .get(at..at + CELL_BYTES)
            .and_then(|b| b.try_into().ok())
            .unwrap_or_else(|| crate::store::corrupt::corrupt("label cell line"));
        LabelSet::decode(cell)
    }
}

/// Where node `index`'s cell starts, and so where the cells before it end.
fn cell_at(index: u32) -> ByteOffset {
    let cell = u64::try_from(CELL_BYTES).unwrap_or(5);
    ByteOffset::new(CELLS_START + u64::from(index) * cell)
}

/// The labels of `next` not in `old`, and of `old` not in `next`; both are sorted and distinct.
fn diff(old: &[Label], next: &[Label]) -> (Vec<Label>, Vec<Label>) {
    let gained = next
        .iter()
        .filter(|l| old.binary_search(l).is_err())
        .copied()
        .collect();
    let lost = old
        .iter()
        .filter(|l| next.binary_search(l).is_err())
        .copied()
        .collect();
    (gained, lost)
}
