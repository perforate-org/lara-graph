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
//! Buckets are intersected without sorting them: one bucket's targets make a [`NeighborSet`] (a
//! bitmap over a narrow range, else a hash table), and each other bucket keeps those it holds,
//! read once or, for a paged bucket many times larger than the set, through its target index
//! ([`PROBE_RATIO`]). A set is built once and serves any number of buckets, so listing triangles
//! reads each node's bucket once per set rather than once per pair. Only the result is ordered.

mod target_set;

use crate::bucket::slots::{self, Span, Window};
use crate::bucket::{LabelBucket, LabelBucketKey};
use crate::edge::EdgeView;
use crate::error::Error;
use crate::graph::Graph;
use crate::graph::GraphMemories;
use crate::id::{Adjacency, NodeId};
use crate::input::Order;
use crate::store::memory::Memory;
use alloc::vec::Vec;
use core::ops::ControlFlow;
pub use target_set::NeighborSet;
use target_set::Targets;

/// How many times more live edges a paged bucket must hold than the candidates left for an
/// intersection to look each candidate up in its index instead of reading it whole: a lookup cost
/// about 14K instructions, reading an edge and looking it up in the candidates about 110
/// (`common_65536_*` in canbench: 7.4M read against 13.9M looked up for 1024 candidates).
const PROBE_RATIO: u32 = 128;

impl<S: GraphMemories> Graph<S> {
    /// Replaces the contents of `out` with the targets of the live edges of `v`'s
    /// `adjacency` bucket in ascending node order, once per edge: parallel edges repeat.
    ///
    /// Nothing is stored for it: the bucket is read and sorted in the heap, at about 3 to 6 times
    /// the instructions of reading it in position order.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn sorted_neighbors(
        &self,
        v: NodeId,
        adjacency: Adjacency,
        out: &mut Vec<NodeId>,
    ) -> Result<(), Error> {
        let (label, direction) = (adjacency.label(), adjacency.direction());
        out.clear();
        if let Some(bucket) = self.read_label_bucket(v, LabelBucketKey { label, direction })? {
            self.sorted_into(&bucket, out);
        }
        Ok(())
    }

    /// Replaces the contents of `out` with the nodes that every one of `nodes`' `(label,
    /// direction)` buckets holds a live edge to, ascending, each once; empty for no nodes.
    ///
    /// The smallest bucket's targets make a [`NeighborSet`]; each larger bucket, smallest first,
    /// keeps in it those it holds (see [`Self::common_with`]), the set narrowed in place.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any of `nodes`.
    pub fn common_neighbors(
        &self,
        nodes: &[NodeId],
        adjacency: Adjacency,
        out: &mut Vec<NodeId>,
    ) -> Result<(), Error> {
        let (label, direction) = (adjacency.label(), adjacency.direction());
        out.clear();
        let Some(mut buckets) = self.buckets_of(nodes, LabelBucketKey { label, direction })? else {
            return Ok(());
        };
        buckets.sort_by_key(|b| b.slots.live());
        let mut buckets = buckets.iter();
        let Some(smallest) = buckets.next() else {
            return Ok(());
        };
        let mut set = NeighborSet::new();
        set.refill(&self.targets_of(smallest));
        let mut found = Vec::new();
        set.narrow_to_common(buckets.map(|b| self.targets_of(b)), &mut found);
        out.extend(found.into_iter().map(NodeId::new));
        Ok(())
    }

    /// The targets of `v`'s `adjacency` bucket, each once, as a set to intersect with
    /// other buckets through [`Self::common_with`]: built once, it serves any number of them, as
    /// listing triangles or common neighbours wants. Empty when `v` has no such bucket.
    ///
    /// A set whose targets lie within a range of 256 node ids per target is a bitmap, looked up
    /// with a shift and a mask; a sparser one is a hash table.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn neighbor_set(&self, v: NodeId, adjacency: Adjacency) -> Result<NeighborSet, Error> {
        let mut set = NeighborSet::new();
        self.neighbor_set_into(&mut set, v, adjacency)?;
        Ok(set)
    }

    /// [`Self::neighbor_set`] into `set`, replacing what it held and reusing the memory it took:
    /// building a set for each of many nodes then allocates only as the largest grows.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`]; `set` is then left as it was.
    pub fn neighbor_set_into(
        &self,
        set: &mut NeighborSet,
        v: NodeId,
        adjacency: Adjacency,
    ) -> Result<(), Error> {
        let (label, direction) = (adjacency.label(), adjacency.direction());
        match self.read_label_bucket(v, LabelBucketKey { label, direction })? {
            Some(bucket) => set.refill(&self.targets_of(&bucket)),
            None => set.refill::<[u32]>(&[]),
        }
        Ok(())
    }

    /// Replaces the contents of `out` with the nodes both `set` and `w`'s `adjacency`
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
        set: &mut NeighborSet,
        w: NodeId,
        above: Option<NodeId>,
        adjacency: Adjacency,
        out: &mut Vec<NodeId>,
    ) -> Result<(), Error> {
        let (label, direction) = (adjacency.label(), adjacency.direction());
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

    /// Appends the targets both `set` and `bucket` hold from index `from` on, ascending, to
    /// `out`.
    pub(in crate::graph) fn intersect_from(
        &self,
        set: &mut NeighborSet,
        bucket: &LabelBucket,
        from: u32,
        out: &mut Vec<u32>,
    ) {
        set.common_from(&self.targets_of(bucket), from, out);
    }

    /// `bucket`'s targets, to fill or intersect a [`NeighborSet`] with.
    fn targets_of<'a>(
        &'a self,
        bucket: &'a LabelBucket,
    ) -> BucketTargets<'a, S::EdgePages, S::EdgeDirectory> {
        BucketTargets {
            view: self.edge_view(),
            bucket,
        }
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
                keys.push(e.node().index());
                ControlFlow::<()>::Continue(())
            },
        );
        sort_indexes(&mut keys);
        out.extend(keys.into_iter().map(NodeId::new));
    }
}

/// The live targets of a bucket: read whole, or, for a paged bucket [`PROBE_RATIO`] times larger
/// than the candidates, each candidate looked up in its target index.
struct BucketTargets<'a, M: Memory, D: Memory> {
    view: EdgeView<'a, M, D>,
    bucket: &'a LabelBucket,
}

impl<M: Memory, D: Memory> Targets for BucketTargets<'_, M, D> {
    fn count(&self) -> usize {
        usize::try_from(self.bucket.slots.live()).unwrap_or(0)
    }

    fn probes(&self, candidates: usize) -> bool {
        let candidates = u32::try_from(candidates).unwrap_or(u32::MAX);
        self.bucket.index.tree().is_some()
            && candidates.saturating_mul(PROBE_RATIO) < self.bucket.slots.live()
    }

    /// In position order.
    fn visit(&self, mut f: impl FnMut(u32)) {
        slots::visit(
            &self.bucket.slots,
            &self.view.pages(),
            Span::all(&self.bucket.slots, Order::Ascending),
            Window::Page,
            |e| {
                f(e.node().index());
                ControlFlow::<()>::Continue(())
            },
        );
    }

    /// For a paged bucket: the blocks its index lists for `t` and its unindexed last block,
    /// scanned for it.
    fn holds(&self, t: u32) -> bool {
        let t = NodeId::new(t);
        let bucket = self.bucket;
        let pages = self.view.pages();
        let spans = self
            .view
            .indexed_spans_to(bucket, t)
            .unwrap_or_else(|| alloc::vec![(0, bucket.slots.stored())]);
        spans.iter().any(|&(low, high)| {
            slots::visit(
                &bucket.slots,
                &pages,
                Span::new(low, high, Order::Ascending),
                Window::Page,
                |e| {
                    if e.node() == t {
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

/// Below this many keys a comparison sort is used whatever their range.
const RADIX_FROM: usize = 64;

/// The most counters a radix sort may take per key: past them, clearing and summing counters
/// costs more than a comparison sort saves. With node ids spread over a million nodes (two
/// passes of 1024 counters) it starts at 512 keys; with dense ones far lower: 64 keys within 128
/// ids sorted 6 % cheaper, 256 within 512 26 %, 1024 within 2048 19 % (one pass).
const COUNTERS_PER_KEY: usize = 4;

/// The widest digit a radix pass sorts by: 2048 counters.
const MAX_DIGIT_BITS: u32 = 11;

/// Sorts node indexes ascending: by an LSD radix sort on their offsets from the smallest, in as
/// few passes of up to [`MAX_DIGIT_BITS`] as their range needs (one for keys within 2048 of each
/// other, two within four million), counted in one sweep; or, for few keys or few for their
/// range, by a comparison sort.
fn sort_indexes(keys: &mut Vec<u32>) {
    if keys.len() < RADIX_FROM {
        keys.sort_unstable();
        return;
    }
    let (mut min, mut max) = (u32::MAX, 0);
    for &k in keys.iter() {
        min = min.min(k);
        max = max.max(k);
    }
    let bits = u32::BITS - (max - min).leading_zeros();
    if bits == 0 {
        return;
    }
    let passes = bits.div_ceil(MAX_DIGIT_BITS);
    let width = bits.div_ceil(passes);
    let radix = 1usize << width;
    let counters = radix * usize::try_from(passes).unwrap_or(1);
    if counters > keys.len().saturating_mul(COUNTERS_PER_KEY) {
        keys.sort_unstable();
        return;
    }
    let mask = (1u32 << width) - 1;
    let mut counts = alloc::vec![0u32; counters];
    count_digits(keys, &mut counts, (min, width, mask, passes));
    let mut spare = alloc::vec![0u32; keys.len()];
    let first = keys.first().copied().unwrap_or(min);
    for (pass, c) in (0..passes).zip(counts.chunks_exact_mut(radix)) {
        // A pass every key shares a digit of moves nothing: if one does, the first key's does.
        let shared = c
            .get(digit((first - min) >> (pass * width), mask))
            .is_some_and(|&x| usize::try_from(x).is_ok_and(|x| x == keys.len()));
        if shared {
            continue;
        }
        scatter(keys, &mut spare, c, (min, pass * width, mask));
        core::mem::swap(keys, &mut spare);
    }
}

/// Counts every pass's digits of `keys` in one sweep: pass `p`'s counters are the `p`-th run of
/// `mask + 1` in `counts`.
#[expect(
    clippy::indexing_slicing,
    reason = "digits are masked below the radix, and there are that many counters per pass"
)]
fn count_digits(
    keys: &[u32],
    counts: &mut [u32],
    (min, width, mask, passes): (u32, u32, u32, u32),
) {
    let radix = usize::try_from(mask).unwrap_or(0) + 1;
    match passes {
        1 => {
            for &k in keys {
                counts[digit(k - min, mask)] += 1;
            }
        }
        2 => {
            for &k in keys {
                let off = k - min;
                counts[digit(off, mask)] += 1;
                counts[radix + digit(off >> width, mask)] += 1;
            }
        }
        _ => {
            for &k in keys {
                let off = k - min;
                counts[digit(off, mask)] += 1;
                counts[radix + digit(off >> width, mask)] += 1;
                counts[2 * radix + digit(off >> (2 * width), mask)] += 1;
            }
        }
    }
}

/// One radix pass: `keys` into `into` by the digit of their offset from `min` at `shift`, whose
/// counts are `counts`.
#[expect(
    clippy::indexing_slicing,
    reason = "digits are masked to the counters, and their prefix sums stay below the key count"
)]
fn scatter(
    keys: &[u32],
    into: &mut [u32],
    counts: &mut [u32],
    (min, shift, mask): (u32, u32, u32),
) {
    let mut next = 0u32;
    for x in counts.iter_mut() {
        let here = *x;
        *x = next;
        next += here;
    }
    for &k in keys {
        let slot = &mut counts[digit((k - min) >> shift, mask)];
        into[usize::try_from(*slot).unwrap_or(usize::MAX)] = k;
        *slot += 1;
    }
}

/// The digit `mask` keeps of `shifted`, as a counter index.
fn digit(shifted: u32, mask: u32) -> usize {
    usize::try_from(shifted & mask).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both sorts agree with the standard one on keys that use every byte, at sizes around the
    /// switch.
    #[test]
    fn indexes_sort_by_either_sort() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        for n in [0, 1, 63, 64, 300, 512, 2000, 5000, 9000] {
            let keys: Vec<u32> = (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    u32::try_from(x >> 32).unwrap()
                })
                .collect();
            // Full width (three passes), within four million (two), within 2048 (one), and all
            // one key (none), each well above zero.
            for (span, from) in [(u32::MAX, 0), (0x3F_FFFF, 7), (0x7FF, 1 << 31), (0, 99)] {
                let mut keys: Vec<u32> = keys.iter().map(|&k| from + (k & span)).collect();
                let mut want = keys.clone();
                want.sort_unstable();
                sort_indexes(&mut keys);
                assert_eq!(keys, want, "{n} keys within {span:#x} of {from}");
            }
        }
    }
}
