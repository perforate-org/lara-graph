//! Label postings: for each node label, the nodes that have it, in ascending id order.
//!
//! Node ids are cut into chunks of 65,536 (their high 16 bits). Each label keeps a table of its
//! chunks, and each chunk a container of the low 16 bits of its members: a sorted list of up to
//! [`MAX_LISTED`](super::set::MAX_LISTED) in one extent, or a bitmap of the whole chunk in two pages. Both tables are value
//! planes (`values`): a label's chunk table is a plane of chunk cells, and the directory is a plane
//! of label cells whose root the node labels memory's header holds. A missing cell reads as zeros,
//! which is an empty chunk or a label nobody has.
//!
//! An add copies at most two pages; a removal writes in place and never allocates, so a node's
//! deletion can drop its postings without reserving memory. A bitmap stays a bitmap until its
//! chunk empties.
//!
//! # Label cell (directory)
//!
//! ```text
//! ---------------------------------------- <- Offset 0
//! Chunk table root (a plane root) ↕ 6 bytes
//! ----------------------------------------
//! Nodes with the label            ↕ 4 bytes
//! ----------------------------------------
//! ```
//!
//! # Chunk cell
//!
//! ```text
//! ---------------------------------------- <- Offset 0
//! Members - 1                     ↕ 2 bytes
//! ----------------------------------------
//! Kind: 0 empty, 1 list, 2 bitmap ↕ 1 byte
//! ----------------------------------------
//! List extent, or the bitmap's    ↕ 8 bytes       (an extent, or two page ids)
//!   two pages
//! ----------------------------------------
//! ```

use crate::bucket::values::{self, PAGES_PER_WRITE, Plane, ROOT_BYTES};
use crate::id::{NodeId, NodeLabel, Width};
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::{PageStore, Pages};
use core::ops::ControlFlow;

mod chunk;

use chunk::{CHUNK_CELL, Chunk};
#[cfg(test)]
use {crate::store::page_store::Extent, alloc::vec::Vec};

const LABEL_CELL: usize = ROOT_BYTES + 4;
const CHUNK_BITS: u32 = 16;
/// Pages one [`add`] can take: a write to each table, and a container's change.
pub(crate) const PAGES_PER_ADD: u32 = 2 * PAGES_PER_WRITE + 2;

/// Whether a node joins a label's postings or leaves them.
#[derive(Clone, Copy)]
enum Change {
    Joins,
    Leaves,
}

fn width(bytes: usize) -> Width {
    Width::new(u16::try_from(bytes).unwrap_or(0)).unwrap_or_else(|| corrupt("cell width"))
}

/// A label's directory cell.
#[derive(Clone, Copy)]
struct LabelCell {
    chunks: Plane,
    count: u32,
}

impl LabelCell {
    fn read<M: Memory>(view: &Pages<'_, M>, dir: Plane, label: NodeLabel) -> Self {
        let mut bytes = [0u8; LABEL_CELL];
        values::read(
            view,
            dir,
            width(LABEL_CELL),
            u32::from(label.get()),
            &mut bytes,
        );
        let [r0, r1, r2, r3, r4, r5, r6, c0, c1, c2, c3] = bytes;
        Self {
            chunks: Plane::decode([r0, r1, r2, r3, r4, r5, r6]),
            count: u32::from_le_bytes([c0, c1, c2, c3]),
        }
    }

    fn encode(self) -> [u8; LABEL_CELL] {
        let [r0, r1, r2, r3, r4, r5, r6] = self.chunks.encode();
        let [c0, c1, c2, c3] = self.count.to_le_bytes();
        [r0, r1, r2, r3, r4, r5, r6, c0, c1, c2, c3]
    }

    /// Writes the cell back with its chunk `chunk_no` now `chunk` and its count moved by one.
    fn write<M: Memory>(
        self,
        store: &mut PageStore<M>,
        (dir, label): (Plane, NodeLabel),
        (chunk_no, chunk, change): (u32, Chunk, Change),
    ) -> Result<Plane, WriteError> {
        let chunks = values::write(
            store,
            self.chunks,
            width(CHUNK_CELL),
            chunk_no,
            &chunk.encode(),
        )?;
        let count = match change {
            Change::Joins => self.count + 1,
            Change::Leaves => self.count - 1,
        };
        let cell = Self { chunks, count };
        values::write(
            store,
            dir,
            width(LABEL_CELL),
            u32::from(label.get()),
            &cell.encode(),
        )
    }
}

fn split(node: NodeId) -> (u32, u16) {
    let index = node.index();
    (
        index >> CHUNK_BITS,
        u16::try_from(index & 0xFFFF).unwrap_or(0),
    )
}

/// Adds `node` to `label`'s postings, within [`PAGES_PER_ADD`] reserved pages, and returns the
/// directory's new root. `node` must not have `label` yet.
pub(crate) fn add<M: Memory>(
    store: &mut PageStore<M>,
    dir: Plane,
    label: NodeLabel,
    node: NodeId,
) -> Result<Plane, WriteError> {
    let (chunk_no, low) = split(node);
    let cell = LabelCell::read(&store.pages(), dir, label);
    let next = Chunk::read(&store.pages(), cell.chunks, chunk_no).added(store, low)?;
    cell.write(store, (dir, label), (chunk_no, next, Change::Joins))
}

/// Removes `node` from `label`'s postings, in place: the directory's root does not change.
/// `node` must have `label`. Never allocates.
pub(crate) fn remove<M: Memory>(
    store: &mut PageStore<M>,
    dir: Plane,
    label: NodeLabel,
    node: NodeId,
) {
    let (chunk_no, low) = split(node);
    let cell = LabelCell::read(&store.pages(), dir, label);
    let next = Chunk::read(&store.pages(), cell.chunks, chunk_no).removed(store, low);
    // Both cells exist, so neither write allocates.
    if cell
        .write(store, (dir, label), (chunk_no, next, Change::Leaves))
        .ok()
        != Some(dir)
    {
        corrupt("a posting removal that moved a table");
    }
}

/// How many nodes have `label`.
pub(crate) fn count<M: Memory>(view: &Pages<'_, M>, dir: Plane, label: NodeLabel) -> u32 {
    LabelCell::read(view, dir, label).count
}

/// Calls `f` with each node that has `label`, from `from` on, ascending, until it breaks;
/// `nodes` bounds the ids.
pub(crate) fn visit<M: Memory, B>(
    view: &Pages<'_, M>,
    dir: Plane,
    (label, from, nodes): (NodeLabel, u32, u32),
    f: &mut impl FnMut(NodeId) -> ControlFlow<B>,
) -> Option<B> {
    let cell = LabelCell::read(view, dir, label);
    if cell.count == 0 {
        return None;
    }
    let (first, end) = (from >> CHUNK_BITS, nodes.div_ceil(1 << CHUNK_BITS));
    values::visit(
        view,
        cell.chunks,
        width(CHUNK_CELL),
        (first, end, false),
        |at, bytes| {
            (at..)
                .zip(bytes.as_chunks::<CHUNK_CELL>().0)
                .try_for_each(|(chunk_no, raw)| {
                    Chunk::decode(*raw).visit(view, (chunk_no << CHUNK_BITS, from), f)
                })
        },
    )
}

/// Every extent the postings own, over node ids below `nodes` (test invariant checks).
#[cfg(test)]
pub(crate) fn extents<M: Memory>(view: &Pages<'_, M>, dir: Plane, nodes: u32) -> Vec<Extent> {
    let mut out = values::extents(view, dir);
    if dir == Plane::Empty {
        return out;
    }
    for label in 0..=u16::MAX {
        let cell = LabelCell::read(view, dir, NodeLabel::new(label));
        if cell.chunks == Plane::Empty {
            continue;
        }
        out.extend(values::extents(view, cell.chunks));
        for chunk_no in 0..nodes.div_ceil(1 << CHUNK_BITS) {
            out.extend(Chunk::read(view, cell.chunks, chunk_no).extents());
        }
    }
    out
}

/// Every label with postings and how many nodes it counts (test invariant checks).
#[cfg(test)]
pub(crate) fn labels<M: Memory>(view: &Pages<'_, M>, dir: Plane) -> Vec<(NodeLabel, u32)> {
    let mut out = Vec::new();
    values::visit(
        view,
        dir,
        width(LABEL_CELL),
        (0, 1 << 16, false),
        |at, bytes| {
            for (label, raw) in (at..).zip(bytes.as_chunks::<LABEL_CELL>().0) {
                let [.., c0, c1, c2, c3] = *raw;
                let count = u32::from_le_bytes([c0, c1, c2, c3]);
                if count > 0 {
                    out.push((NodeLabel::new(u16::try_from(label).unwrap_or(0)), count));
                }
            }
            ControlFlow::<()>::Continue(())
        },
    );
    out
}
