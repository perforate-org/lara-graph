//! A paged bucket's target index in the graph: keeping it in step with the slots, and the
//! lookups by target it makes logarithmic.
//!
//! The index covers a paged bucket's complete blocks ([`indexed_end`]); its last, partial block
//! is scanned instead. Appends then reach the index once per block, as a batch of the block's
//! live entries, and a lookup scans at most one block beside the index. A detaching vertex's
//! buckets drop their indexes, as all their edges are going: their lookups scan.

use super::{Graph, reserved};
use crate::bucket::{self, Spot};
use crate::corrupt::corrupt;
use crate::descriptor::Descriptor;
use crate::id::{Placement, Position, VertexId};
use crate::memory::Memory;
use crate::stored::{Bucket, Index};
use crate::targets::{self, BLOCK, Change, Nth, Root};
use alloc::vec::Vec;

/// What an insert knows of the edges already between a half's two ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Between {
    /// There are none: the other end's bucket shows it.
    Nothing,
    /// Not known without looking.
    Unknown,
}

/// A block batch planned before the insert that completes the block; it holds when the insert
/// appends at `at`.
pub(super) struct Batch {
    /// Boxed: the insert path passes batches by value and is usually without one.
    plan: alloc::boxed::Box<targets::Plan>,
    at: Position,
}

impl Batch {
    /// The pages committing it takes.
    pub(super) fn pages(&self) -> u32 {
        self.plan.pages()
    }
}

/// The end of the positions a bucket's index covers, its complete blocks; `None` for a bucket
/// too small to have an index.
pub(super) fn indexed_end(slots: &Descriptor) -> Option<u32> {
    match slots {
        Descriptor::Paged(_) => Some(slots.stored() / BLOCK * BLOCK),
        Descriptor::Tiny(_) | Descriptor::Run(_) => None,
    }
}

/// The end of the positions `b`'s index covers; `None` when lookups scan the whole bucket (too
/// small for an index, or its index dropped).
pub(super) fn indexed(b: &Bucket) -> Option<u32> {
    match b.index {
        Index::Dropped => None,
        Index::Empty | Index::Tree(_) => indexed_end(&b.slots),
    }
}

impl<M: Memory> Graph<M> {
    /// The bucket's index after an edge to `target` went in at `at`, the slots having been
    /// `was` before: a bucket that just became paged gets one built from its slots; an append
    /// that completes a block brings the block's live entries in. Within reserved memory.
    pub(super) fn index_inserted(
        &mut self,
        bucket: Bucket,
        was: &Descriptor,
        (target, at): (VertexId, Position),
        batch: Option<Batch>,
    ) -> Bucket {
        let Some(end) = indexed(&bucket) else {
            return bucket;
        };
        let root = bucket.index.tree();
        let index = match indexed_end(was) {
            None => {
                let changes = self.block_changes(&bucket.slots, 0..end);
                self.index_apply(None, &changes)
            }
            Some(before) if before < end => {
                // Only an append completes a block, and its batch was planned before the insert
                // with its pages reserved.
                let b = batch
                    .filter(|b| b.at == at)
                    .unwrap_or_else(|| corrupt("a completed block without its planned batch"));
                Index::of(reserved(targets::commit(&mut self.edges, *b.plan)))
            }
            Some(_) if at.index() < end => self.index_apply(root, &[change(target, at, 1)]),
            Some(_) => bucket.index,
        };
        Bucket { index, ..bucket }
    }

    /// The block batch an insert of an edge to `target` into `bucket` brings into its index when
    /// it appends and so completes a block, planned before the insert: its pages are reserved
    /// with the insert's, and it is committed after.
    pub(super) fn batch_for(&self, bucket: &Bucket, target: VertexId) -> Option<Batch> {
        let end = indexed(bucket)?;
        let next = bucket.slots.stored();
        if !(next + 1).is_multiple_of(BLOCK) {
            return None;
        }
        let mut changes = self.block_changes(&bucket.slots, end..next);
        changes.push(change(target, Position::new(next), 1));
        Some(Batch {
            plan: alloc::boxed::Box::new(targets::plan(
                &self.edges.pages(),
                bucket.index.tree(),
                &changes,
            )),
            at: Position::new(next),
        })
    }

    /// Where an insert of an edge to `t` into `b` goes under `placement`: the first safe hole
    /// when it reuses holes and there is one, else the end. With no edges `between` the two ends,
    /// any hole is safe and the target index is not consulted.
    pub(super) fn spot_for(
        &self,
        b: &Bucket,
        t: VertexId,
        placement: Placement,
        between: Between,
    ) -> Spot {
        let last = || match between {
            Between::Nothing => None,
            Between::Unknown => self.last_to(b, t),
        };
        match placement {
            Placement::Append => Spot::End,
            Placement::ReuseHoles => {
                bucket::hole(&b.slots, t, &self.edges.pages(), last).map_or(Spot::End, Spot::Hole)
            }
        }
    }

    /// The position of the last live edge to `t` in a paged bucket: in the unindexed last block,
    /// else in the last indexed block that holds one.
    fn last_to(&self, b: &Bucket, t: VertexId) -> Option<Position> {
        let view = self.edges.pages();
        let last_in = |span: (u32, u32)| {
            let mut last = None;
            bucket::find_in(&b.slots, &view, span, |e| {
                if e.target() == t {
                    last = Some(e.position());
                }
                false
            });
            last
        };
        let end = indexed(b).unwrap_or(0);
        last_in((end, b.slots.stored())).or_else(|| {
            let block = targets::last_block(&view, b.index.tree()?, t.index())?;
            last_in((block * BLOCK, (block + 1) * BLOCK))
        })
    }

    /// One posting's worth of change for each live entry in `range`.
    fn block_changes(&self, slots: &Descriptor, range: core::ops::Range<u32>) -> Vec<Change> {
        let mut changes = Vec::new();
        bucket::find_in(slots, &self.edges.pages(), (range.start, range.end), |e| {
            changes.push(change(e.target(), e.position(), 1));
            false
        });
        changes
    }

    /// Takes the edge to `target` at `at`, just removed, out of the bucket's index: in place, so
    /// the index stays where it is and no memory is taken.
    pub(super) fn index_removed(&self, bucket: &Bucket, target: VertexId, at: Position) {
        match (indexed(bucket), bucket.index.tree()) {
            (Some(end), Some(root)) if at.index() < end => {
                targets::lower(&self.edges, root, key(target, at));
            }
            (Some(end), None) if at.index() < end => corrupt("an indexed edge without an index"),
            (Some(_) | None, _) => {}
        }
    }

    /// Applies changes within reserved memory.
    pub(super) fn index_apply(&mut self, root: Option<Root>, changes: &[Change]) -> Index {
        Index::of(reserved(targets::apply_reserved(
            &mut self.edges,
            root,
            changes,
        )))
    }

    /// How many live entries to `t` precede position `p`.
    pub(super) fn rank_to(&self, b: &Bucket, t: VertexId, p: Position) -> u32 {
        let view = self.edges.pages();
        let (low, before) = match indexed(b) {
            Some(_) => {
                // The index counts the whole blocks before `p`'s, which all lie below the
                // indexed end (`p` is below the bucket's end, so its block is at most the last);
                // the rest of `p`'s block is scanned.
                let block = p.index() / BLOCK;
                let before = b
                    .index
                    .tree()
                    .map_or(0, |root| targets::rank(&view, root, t.index(), block));
                (block * BLOCK, before)
            }
            None => (0, 0),
        };
        let mut rank = before;
        bucket::find_in(&b.slots, &view, (low, p.index()), |e| {
            rank += u32::from(e.target() == t);
            false
        });
        rank
    }

    /// The position of the `k`-th (0-based) live entry to `t`, if there is one.
    pub(super) fn select_to(&self, b: &Bucket, t: VertexId, k: u32) -> Option<Position> {
        let view = self.edges.pages();
        let stored = b.slots.stored();
        let (span, mut skip) = match (indexed(b), b.index.tree()) {
            (Some(end), Some(root)) => match targets::nth(&view, root, t.index(), k) {
                Nth::At(posting, within) => {
                    let low = posting.block * BLOCK;
                    ((low, low + BLOCK), within)
                }
                // Past the target's indexed entries: the rest are in the last block.
                Nth::Past(indexed) => ((end, stored), k - indexed),
            },
            (Some(end), None) => ((end, stored), k),
            (None, _) => ((0, stored), k),
        };
        bucket::find_in(&b.slots, &view, span, |e| {
            if e.target() != t {
                return false;
            }
            skip.checked_sub(1).map(|left| skip = left).is_none()
        })
        .map(|e| e.position())
    }

    /// Releases a bucket's index, leaving `to` in its place: [`Index::Empty`] when it stops being
    /// paged, [`Index::Dropped`] when its vertex starts removing its edges.
    pub(super) fn drop_index(&mut self, bucket: Bucket, to: Index) -> Bucket {
        if let Some(root) = bucket.index.tree() {
            targets::release(&mut self.edges, root);
        }
        Bucket {
            index: to,
            ..bucket
        }
    }
}

/// The key of the posting of `target` in `at`'s block.
const fn key(target: VertexId, at: Position) -> (u32, u32) {
    (target.index(), at.index() / BLOCK)
}

/// A change of `delta` to the posting of `target` in `at`'s block.
pub(super) const fn change(target: VertexId, at: Position, delta: i64) -> Change {
    Change {
        key: key(target, at),
        delta,
    }
}

/// The index changes of compaction moving an edge to `target` from `from` to `to`. Both are
/// indexed positions: compaction never merges a bucket's last page, which holds its unindexed
/// last block, and moves only into earlier pages.
pub(super) fn moved(target: VertexId, (from, to): (Position, Position)) -> [Change; 2] {
    [change(target, from, -1), change(target, to, 1)]
}

#[cfg(feature = "bench-internals")]
impl<M: Memory> Graph<M> {
    /// Pages of the target index of `v`'s `key` bucket (measurements).
    pub(crate) fn index_pages(&self, v: VertexId, key: crate::labels::BucketKey) -> usize {
        self.find_bucket(v, key).ok().flatten().map_or(0, |found| {
            targets::extents(&self.edges.pages(), found.bucket.index.tree()).len()
        })
    }
}
