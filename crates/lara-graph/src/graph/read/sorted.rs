//! One node's targets sorted by node id, and the targets two nodes share: the views that
//! intersection-heavy reads (common neighbours, triangles) want, derived on each read rather than
//! stored, so updates pay nothing for them.
//!
//! Positions follow insertion order, so a sorted view reads the bucket whole and sorts it in the
//! heap: on the IC that costs about 3 to 6 times the plain read (`sorted_*` against `unsorted_*`
//! in canbench). A paged bucket's target index lists its targets in order too, but walking it
//! decodes every posting and cost half again as much as reading and sorting (30.4M against
//! 19.8M instructions for 65536 edges), so it is not used here.
//!
//! Buckets are intersected without sorting them: one bucket's targets make a [`TargetSet`] (a
//! bitmap over a narrow range, else a hash table), and each other bucket keeps those it holds,
//! read once or, for a paged bucket many times larger than the set, through its target index
//! ([`PROBE_RATIO`]). A set is built once and serves any number of buckets, so listing triangles
//! reads each node's bucket once per set rather than once per pair. Only the result is ordered.

mod target_set;

use crate::bucket::slots::{self, Span, Window};
use crate::bucket::{LabelBucket, LabelBucketKey};
use crate::error::Error;
use crate::graph::Graph;
use crate::id::{Direction, Label, NodeId};
use crate::input::Order;
use crate::store::memory::Memory;
use alloc::vec::Vec;
use core::ops::ControlFlow;
pub use target_set::TargetSet;

/// How many times more live edges a paged bucket must hold than the candidates left for an
/// intersection to look each candidate up in its index instead of reading it whole: a lookup cost
/// about 14K instructions, reading an edge and looking it up in the candidates about 110
/// (`common_65536_*` in canbench: 7.4M read against 13.9M looked up for 1024 candidates).
const PROBE_RATIO: u32 = 128;

impl<M: Memory> Graph<M> {
    /// Replaces the contents of `out` with the targets of the live edges of `v`'s
    /// `(label, direction)` bucket in ascending node order, once per edge: parallel edges repeat.
    ///
    /// Nothing is stored for it: the bucket is read and sorted in the heap, at about 3 to 6 times
    /// the instructions of reading it in position order.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn sorted_targets(
        &self,
        v: NodeId,
        (label, direction): (Label, Direction),
        out: &mut Vec<NodeId>,
    ) -> Result<(), Error> {
        out.clear();
        if let Some(bucket) = self.read_label_bucket(v, LabelBucketKey { label, direction })? {
            self.sorted_into(&bucket, out);
        }
        Ok(())
    }

    /// Replaces the contents of `out` with the nodes that both `u`'s and `w`'s
    /// `(label, direction)` buckets hold a live edge to, ascending, each once:
    /// [`Self::common_targets_of`] for two nodes.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for `u` or `w`.
    pub fn common_targets(
        &self,
        u: NodeId,
        w: NodeId,
        (label, direction): (Label, Direction),
        out: &mut Vec<NodeId>,
    ) -> Result<(), Error> {
        self.common_targets_of(&[u, w], (label, direction), out)
    }

    /// Replaces the contents of `out` with the nodes that every one of `nodes`' `(label,
    /// direction)` buckets holds a live edge to, ascending, each once; empty for no nodes.
    ///
    /// The smallest bucket's targets make a [`TargetSet`]; each larger bucket, smallest first,
    /// keeps in it those it holds (see [`Self::common_with`]), the set narrowed in place.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any of `nodes`.
    pub fn common_targets_of(
        &self,
        nodes: &[NodeId],
        (label, direction): (Label, Direction),
        out: &mut Vec<NodeId>,
    ) -> Result<(), Error> {
        out.clear();
        let Some(mut buckets) = self.buckets_of(nodes, LabelBucketKey { label, direction })? else {
            return Ok(());
        };
        buckets.sort_by_key(|b| b.slots.live());
        let mut buckets = buckets.into_iter();
        let Some(smallest) = buckets.next() else {
            return Ok(());
        };
        let mut set = self.set_of(&smallest);
        let mut rest = buckets.peekable();
        let mut found = Vec::new();
        if rest.peek().is_none() {
            found = set.keys();
            sort_indexes(&mut found);
        }
        // Each round keeps in the set what the bucket holds; the last one lists it.
        while let Some(bucket) = rest.next() {
            self.match_bucket(&mut set, &bucket, 0);
            if rest.peek().is_none() {
                set.finish(&mut found);
                break;
            }
            set.narrow();
            if set.is_empty() {
                break;
            }
        }
        out.extend(found.into_iter().map(NodeId::new));
        Ok(())
    }

    /// The targets of `v`'s `(label, direction)` bucket, each once, as a set to intersect with
    /// other buckets through [`Self::common_with`]: built once, it serves any number of them, as
    /// listing triangles or common neighbours wants. Empty when `v` has no such bucket.
    ///
    /// A set whose targets lie within a range of 256 node ids per target is a bitmap, looked up
    /// with a shift and a mask; a sparser one is a hash table.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn target_set(
        &self,
        v: NodeId,
        (label, direction): (Label, Direction),
    ) -> Result<TargetSet, Error> {
        Ok(
            match self.read_label_bucket(v, LabelBucketKey { label, direction })? {
                Some(bucket) => self.set_of(&bucket),
                None => TargetSet::of(&[]),
            },
        )
    }

    /// Replaces the contents of `out` with the nodes both `set` and `w`'s `(label, direction)`
    /// bucket hold, ascending, each once, and only those past `above` when it is given: listing
    /// each triangle once, as `u < w < x`, wants those past `w`. `set` is left as it was, ready
    /// for the next bucket; [`ReadBatch::common_with_each`](crate::ReadBatch::common_with_each)
    /// does this for many nodes at a lower cost each.
    ///
    /// `w`'s bucket is read once and each target past `above` looked up in `set`; a paged bucket
    /// over a hundred times larger than `set` instead has each of `set`'s targets looked up in
    /// its target index.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for `w`.
    pub fn common_with(
        &self,
        set: &mut TargetSet,
        (w, above): (NodeId, Option<NodeId>),
        (label, direction): (Label, Direction),
        out: &mut Vec<NodeId>,
    ) -> Result<(), Error> {
        out.clear();
        let bucket = self.read_label_bucket(w, LabelBucketKey { label, direction })?;
        let Some(bucket) = bucket.filter(|_| !set.is_empty()) else {
            return Ok(());
        };
        let mut found = Vec::new();
        self.intersect_from(set, &bucket, first_past(above), &mut found);
        out.extend(found.into_iter().map(NodeId::new));
        Ok(())
    }

    /// The `key` buckets of every one of `nodes`; `None` when any lacks one, or for no nodes.
    fn buckets_of(
        &self,
        nodes: &[NodeId],
        key: LabelBucketKey,
    ) -> Result<Option<Vec<LabelBucket>>, Error> {
        let mut buckets = Vec::with_capacity(nodes.len());
        let mut missing = nodes.is_empty();
        for &v in nodes {
            match self.read_label_bucket(v, key)? {
                Some(bucket) => buckets.push(bucket),
                None => missing = true,
            }
        }
        Ok((!missing).then_some(buckets))
    }

    /// `bucket`'s targets as a set.
    fn set_of(&self, bucket: &LabelBucket) -> TargetSet {
        let mut targets = Vec::with_capacity(usize::try_from(bucket.slots.live()).unwrap_or(0));
        self.visit_targets(bucket, |t| targets.push(t));
        TargetSet::of(&targets)
    }

    /// Appends the targets both `set` and `bucket` hold, ascending, to `out`.
    /// Appends the targets both `set` and `bucket` hold from index `from` on, ascending, to
    /// `out`.
    pub(in crate::graph) fn intersect_from(
        &self,
        set: &mut TargetSet,
        bucket: &LabelBucket,
        from: u32,
        out: &mut Vec<u32>,
    ) {
        self.match_bucket(set, bucket, from);
        set.finish_from(from, out);
    }

    /// A query of `set` matching the targets `bucket` holds from index `from` on: `bucket` read
    /// whole and looked up in `set`, or for a paged bucket [`PROBE_RATIO`] times larger, each of
    /// `set`'s targets looked up in its index.
    fn match_bucket(&self, set: &mut TargetSet, bucket: &LabelBucket, from: u32) {
        set.begin();
        let left = u32::try_from(set.len()).unwrap_or(u32::MAX);
        let probe =
            bucket.index.tree().is_some() && left.saturating_mul(PROBE_RATIO) < bucket.slots.live();
        if probe {
            for t in set.keys() {
                if t >= from && self.holds(bucket, NodeId::new(t)) {
                    set.hit(t);
                }
            }
        } else if from == 0 {
            // Unbounded, the loop keeps no comparison: with it, intersections cost 2-11 % more.
            self.visit_targets(bucket, |t| set.hit(t));
        } else {
            self.visit_targets(bucket, |t| {
                if t >= from {
                    set.hit(t);
                }
            });
        }
    }

    /// Calls `f` with the index of each live target of `bucket`, in position order.
    fn visit_targets(&self, bucket: &LabelBucket, mut f: impl FnMut(u32)) {
        slots::visit(
            &bucket.slots,
            &self.edge_view().pages(),
            Span::all(&bucket.slots, Order::Ascending),
            Window::Page,
            |e| {
                f(e.target().index());
                ControlFlow::<()>::Continue(())
            },
        );
    }

    /// `bucket`'s live targets, ascending, once per edge, appended to `out`.
    fn sorted_into(&self, bucket: &LabelBucket, out: &mut Vec<NodeId>) {
        let mut keys: Vec<u32> =
            Vec::with_capacity(usize::try_from(bucket.slots.live()).unwrap_or(0));
        slots::visit(
            &bucket.slots,
            &self.edge_view().pages(),
            Span::all(&bucket.slots, Order::Ascending),
            Window::Page,
            |e| {
                keys.push(e.target().index());
                ControlFlow::<()>::Continue(())
            },
        );
        sort_indexes(&mut keys);
        out.extend(keys.into_iter().map(NodeId::new));
    }

    /// Whether the paged `bucket` holds a live edge to `t`: the blocks its index lists for `t` and
    /// its unindexed last block, scanned for it.
    fn holds(&self, bucket: &LabelBucket, t: NodeId) -> bool {
        let view = self.edge_view().pages();
        let spans = self
            .edge_view()
            .indexed_spans_to(bucket, t)
            .unwrap_or_else(|| alloc::vec![(0, bucket.slots.stored())]);
        spans.iter().any(|&(low, high)| {
            slots::visit(
                &bucket.slots,
                &view,
                Span::new(low, high, Order::Ascending),
                Window::Page,
                |e| {
                    if e.target() == t {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                },
            )
            .is_some()
        })
    }
}

/// The first node index past `above`: all of them for none.
pub(in crate::graph) fn first_past(above: Option<NodeId>) -> u32 {
    above.map_or(0, |a| a.index().saturating_add(1))
}

/// From this many keys a radix sort beats a comparison sort on the IC: at 1024 keys 166K
/// instructions against 270K, at 256 even, at 64 twice the cost.
const RADIX_FROM: usize = 512;

/// Sorts node indexes ascending: a few by a comparison sort, many by an LSD radix sort on bytes
/// that counts all four bytes in one pass and skips a byte every key shares (most graphs leave the
/// top one zero).
fn sort_indexes(keys: &mut Vec<u32>) {
    if keys.len() < RADIX_FROM {
        keys.sort_unstable();
        return;
    }
    let mut counts = [[0u32; 256]; 4];
    for &k in keys.iter() {
        for (shift, c) in (0u32..).step_by(8).zip(counts.iter_mut()) {
            if let Some(count) = c.get_mut(digit(k, shift)) {
                *count += 1;
            }
        }
    }
    let mut spare = alloc::vec![0u32; keys.len()];
    for (shift, c) in (0u32..).step_by(8).zip(counts.iter_mut()) {
        if c.iter()
            .any(|&x| usize::try_from(x).is_ok_and(|x| x == keys.len()))
        {
            continue;
        }
        scatter(keys, &mut spare, c, shift);
        core::mem::swap(keys, &mut spare);
    }
}

/// One radix pass: `keys` into `into` by their byte at `shift`, whose counts are `counts`.
#[expect(
    clippy::indexing_slicing,
    reason = "digits are masked to the 256 counters, and their prefix sums stay below the key count"
)]
fn scatter(keys: &[u32], into: &mut [u32], counts: &mut [u32; 256], shift: u32) {
    let mut next = 0u32;
    for x in counts.iter_mut() {
        let here = *x;
        *x = next;
        next += here;
    }
    for &k in keys {
        let slot = &mut counts[digit(k, shift)];
        into[usize::try_from(*slot).unwrap_or(usize::MAX)] = k;
        *slot += 1;
    }
}

/// The byte of `key` at `shift`, as a counter index.
fn digit(key: u32, shift: u32) -> usize {
    usize::try_from((key >> shift) & 0xFF).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both sorts agree with the standard one on keys that use every byte, at sizes around the
    /// switch.
    #[test]
    fn indexes_sort_by_either_sort() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        for n in [0, 1, 63, 511, 512, 513, 2000] {
            let mut keys: Vec<u32> = (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    u32::try_from(x >> 32).unwrap()
                })
                .collect();
            let mut want = keys.clone();
            want.sort_unstable();
            sort_indexes(&mut keys);
            assert_eq!(keys, want, "{n} keys");
        }
    }
}
