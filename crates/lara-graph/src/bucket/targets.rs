//! The target index of a paged bucket: where the live entries to each node are.
//!
//! Positions follow insertion order, so finding a bucket's entries to a given node would mean
//! scanning it. A paged bucket (a hub's) keeps a **counted B+ tree of postings** beside its slots:
//! a posting `(target, block, count)` says that `count` live entries to `target` sit among the
//! [`BLOCK`] positions of `block`. It points at blocks, not positions, so a lookup ends with one
//! block scan; parallel edges in a block cost one posting.
//!
//! Leaves are pages of bit-packed postings (`leaf`); internal nodes keep each child's first key
//! and total count, so the rank and selection of a target's entries are logarithmic even for a
//! target with millions of parallel edges. Changes come in batches applied within reserved memory;
//! a counting pass tells how much a large batch needs (`tree`). A removal's single decrement is
//! written in place and needs none.

mod bits;
mod leaf;
mod tree;

use crate::id::{NodeId, Position};

#[cfg(test)]
pub(crate) use tree::extents;
pub(crate) use tree::{
    InPlace, Merged, Nth, Plan, Root, apply_reserved, commit, last_block, lower, merge_step, nth,
    plan, raise, rank, release, visit_blocks,
};
#[cfg(test)]
pub(crate) use tree::{count_before, postings, select, zeros_per_segment};

/// Positions per block.
pub(crate) const BLOCK: u32 = 128;

/// Pages one edge's change to an index can take: a split per level and a new root, or building
/// the index of a bucket that just became paged (1025 postings at most). Tests use small leaves
/// and nodes, so their trees are deeper.
pub(crate) const PAGES_PER_CHANGE: u32 = crate::store::scale::Scale::pick(64, 8);

/// `count` live entries to `target` among the positions of `block`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Posting {
    pub(crate) target: u32,
    pub(crate) block: u32,
    pub(crate) count: u32,
}

impl Posting {
    pub(crate) const fn key(&self) -> (u32, u32) {
        (self.target, self.block)
    }
}

/// A change to the count of the posting at `key`: what the index is told as edges come, go or
/// move. Plain data, like [`Posting`], so that leaves and the tree both speak it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Change {
    pub(crate) key: (u32, u32),
    pub(crate) delta: i64,
}

/// The key of the posting of `target` in `at`'s block.
pub(crate) const fn key_at(target: NodeId, at: Position) -> (u32, u32) {
    (target.index(), at.index() / BLOCK)
}

impl Change {
    /// A change of `delta` to the posting of `target` in `at`'s block.
    pub(crate) const fn at(target: NodeId, at: Position, delta: i64) -> Self {
        Self {
            key: key_at(target, at),
            delta,
        }
    }

    /// The changes of compaction moving an edge to `target` from `from` to `to`. Both are
    /// indexed positions: compaction never merges a bucket's last page, which holds its unindexed
    /// last block, and moves only into earlier pages.
    pub(crate) const fn moved(target: NodeId, (from, to): (Position, Position)) -> [Self; 2] {
        [Self::at(target, from, -1), Self::at(target, to, 1)]
    }
}

#[cfg(test)]
mod tests;
