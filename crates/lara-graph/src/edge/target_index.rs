//! A paged bucket's target index in the edge store: keeping it in step with the slots, and the
//! lookups by target it makes logarithmic.
//!
//! The index covers a paged bucket's complete blocks (`Descriptor::indexed_end`); its last, partial block
//! is scanned instead. Appends then reach the index once per block, as a batch of the block's
//! live entries, and a lookup scans at most one block beside the index. A deleting node's
//! buckets drop their indexes, as all their edges are going: their lookups scan.

use crate::bucket::descriptor::{Descriptor, Slot};
use crate::bucket::reserved;
use crate::bucket::slots::{self, Spot};
use crate::bucket::stored::Stored;
use crate::bucket::target_index::{self, BLOCK, Change, InPlace, Nth, Root};
use crate::bucket::{Index, LabelBucket};
use crate::edge::{EdgeStore, EdgeView};
use crate::id::{NodeId, Position};
use crate::input::Placement;
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use alloc::vec::Vec;

/// What an insert knows of the edges already between a half's two ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Between {
    /// There are none: the other end's bucket shows it.
    Nothing,
    /// Not known without looking.
    Unknown,
}

impl Between {
    /// What the other end's bucket entry, `None` for no bucket, shows of edges to `target`: a
    /// tiny bucket is read whole, a larger one is not.
    pub(crate) fn of(other: Option<Stored>, target: NodeId) -> Self {
        match other {
            None => Self::Nothing,
            Some(Stored::Plain(Descriptor::Tiny(tiny)))
                if !tiny.slots().contains(&Slot::Live(target)) =>
            {
                Self::Nothing
            }
            Some(_) => Self::Unknown,
        }
    }
}

/// A block batch planned before the insert that completes the block; it holds when the insert
/// appends at `at`.
pub(crate) struct Batch {
    /// Boxed: the insert path passes batches by value and is usually without one.
    plan: alloc::boxed::Box<target_index::Plan>,
    at: Position,
}

impl Batch {
    /// The pages committing it takes.
    pub(crate) fn pages(&self) -> u32 {
        self.plan.pages()
    }
}

impl<M: Memory> EdgeStore<M> {
    /// The bucket's index after an edge to `target` went in at `at`, the slots having been
    /// `was` before: a bucket that just became paged gets one built from its slots; an append
    /// that completes a block brings the block's live entries in. Within reserved memory.
    pub(crate) fn index_inserted(
        &mut self,
        bucket: LabelBucket,
        was: &Descriptor,
        (target, at): (NodeId, Position),
        batch: Option<Batch>,
    ) -> LabelBucket {
        let Some(end) = bucket.indexed() else {
            return bucket;
        };
        let root = bucket.index.tree();
        let index = match was.indexed_end() {
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
                Index::of(reserved(target_index::commit(&mut self.store, *b.plan)))
            }
            // A filled hole: in place when its posting's segment can take it.
            Some(_) if at.index() < end => match root {
                Some(root)
                    if target_index::raise(&self.store, root, target_index::key_at(target, at))
                        == InPlace::Done =>
                {
                    bucket.index
                }
                Some(_) | None => self.index_apply(root, &[Change::at(target, at, 1)]),
            },
            Some(_) => bucket.index,
        };
        LabelBucket { index, ..bucket }
    }

    /// The block batch an insert of an edge to `target` into `bucket` brings into its index when
    /// it appends and so completes a block, planned before the insert: its pages are reserved
    /// with the insert's, and it is committed after.
    pub(crate) fn batch_for(&self, bucket: &LabelBucket, target: NodeId) -> Option<Batch> {
        let end = bucket.indexed()?;
        let next = bucket.slots.stored();
        if !(next + 1).is_multiple_of(BLOCK) {
            return None;
        }
        let mut changes = self.block_changes(&bucket.slots, end..next);
        changes.push(Change::at(target, Position::new(next), 1));
        Some(Batch {
            plan: alloc::boxed::Box::new(target_index::plan(
                &self.store.pages(),
                bucket.index.tree(),
                &changes,
            )),
            at: Position::new(next),
        })
    }

    /// Where an insert of an edge to `t` into `b` goes under `placement`: the first safe hole
    /// when it reuses holes and there is one, else the end. With no edges `between` the two ends,
    /// any hole is safe and the target index is not consulted.
    pub(crate) fn spot_for(
        &self,
        b: &LabelBucket,
        t: NodeId,
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
                slots::hole(&b.slots, t, &self.store.pages(), last).map_or(Spot::End, Spot::Hole)
            }
        }
    }

    /// The position of the last live edge to `t` in a paged bucket: in the unindexed last block,
    /// else in the last indexed block that holds one.
    fn last_to(&self, b: &LabelBucket, t: NodeId) -> Option<Position> {
        let view = self.store.pages();
        let last_in = |span: (u32, u32)| {
            let mut last = None;
            slots::find_in(&b.slots, &view, span, |e| {
                if e.node() == t {
                    last = Some(e.position());
                }
                false
            });
            last
        };
        let end = b.indexed().unwrap_or(0);
        last_in((end, b.slots.stored())).or_else(|| {
            let block = target_index::last_block(&view, b.index.tree()?, t.index())?;
            last_in((block * BLOCK, (block + 1) * BLOCK))
        })
    }

    /// One posting's worth of change for each live entry in `range`.
    fn block_changes(&self, slots: &Descriptor, range: core::ops::Range<u32>) -> Vec<Change> {
        let mut changes = Vec::new();
        slots::find_in(slots, &self.store.pages(), (range.start, range.end), |e| {
            changes.push(Change::at(e.node(), e.position(), 1));
            false
        });
        changes
    }

    /// Takes the edge to `target` at `at`, just removed, out of the bucket's index: in place, so
    /// the index stays where it is and no memory is taken.
    pub(crate) fn index_removed(&self, bucket: &LabelBucket, target: NodeId, at: Position) {
        match (bucket.indexed(), bucket.index.tree()) {
            (Some(end), Some(root)) if at.index() < end => {
                target_index::lower(&self.store, root, target_index::key_at(target, at));
            }
            (Some(end), None) if at.index() < end => corrupt("an indexed edge without an index"),
            (Some(_) | None, _) => {}
        }
    }

    /// Applies changes within reserved memory.
    pub(crate) fn index_apply(&mut self, root: Option<Root>, changes: &[Change]) -> Index {
        Index::of(reserved(target_index::apply_reserved(
            &mut self.store,
            root,
            changes,
        )))
    }

    /// How many live entries to `t` precede position `p`.
    pub(crate) fn rank_to(&self, b: &LabelBucket, t: NodeId, p: Position) -> u32 {
        let view = self.store.pages();
        let (low, before) = match b.indexed() {
            Some(_) => {
                // The index counts the whole blocks before `p`'s, which all lie below the
                // indexed end (`p` is below the bucket's end, so its block is at most the last);
                // the rest of `p`'s block is scanned.
                let block = p.index() / BLOCK;
                let before = b
                    .index
                    .tree()
                    .map_or(0, |root| target_index::rank(&view, root, t.index(), block));
                (block * BLOCK, before)
            }
            None => (0, 0),
        };
        let mut rank = before;
        slots::find_in(&b.slots, &view, (low, p.index()), |e| {
            rank += u32::from(e.node() == t);
            false
        });
        rank
    }

    /// The position of the `k`-th (0-based) live entry to `t`, if there is one.
    pub(crate) fn select_to(&self, b: &LabelBucket, t: NodeId, k: u32) -> Option<Position> {
        let view = self.store.pages();
        let stored = b.slots.stored();
        let (span, mut skip) = match (b.indexed(), b.index.tree()) {
            (Some(end), Some(root)) => match target_index::nth(&view, root, t.index(), k) {
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
        slots::find_in(&b.slots, &view, span, |e| {
            if e.node() != t {
                return false;
            }
            skip.checked_sub(1).map(|left| skip = left).is_none()
        })
        .map(|e| e.position())
    }

    /// Releases a bucket's index, leaving `to` in its place: [`Index::Empty`] when it stops being
    /// paged, [`Index::Dropped`] when its node starts removing its edges.
    pub(crate) fn drop_index(&mut self, bucket: LabelBucket, to: Index) -> LabelBucket {
        if let Some(root) = bucket.index.tree() {
            target_index::release(&mut self.store, root);
        }
        LabelBucket {
            index: to,
            ..bucket
        }
    }
}

impl<M: Memory> EdgeView<'_, M> {
    /// The position ranges of an indexed bucket that can hold entries to `t`, ascending: the
    /// blocks its index lists for `t`, adjacent ones merged so that each range is read in one
    /// pass, and the unindexed last block; `None` for a bucket without an index.
    pub(crate) fn indexed_spans_to(
        &self,
        bucket: &LabelBucket,
        t: NodeId,
    ) -> Option<Vec<(u32, u32)>> {
        let stored = bucket.slots.stored();
        let (Some(end), Some(root)) = (bucket.indexed(), bucket.index.tree()) else {
            return None;
        };
        let mut spans: Vec<(u32, u32)> = Vec::new();
        let mut add = |low: u32, high: u32| match spans.last_mut() {
            Some(last) if last.1 == low => last.1 = high,
            Some(_) | None => spans.push((low, high)),
        };
        target_index::visit_blocks(&self.pages(), root, t.index(), |block| {
            add(block * BLOCK, (block + 1) * BLOCK);
        });
        if end < stored {
            add(end, stored);
        }
        Some(spans)
    }
}
