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
//! Two buckets are intersected by merging their sorted views, unless one is a paged bucket many
//! times larger than the other: then each of the small one's targets is looked up in the large
//! one's index ([`PROBE_RATIO`]).

use crate::bucket::slots::{self, Span, Window};
use crate::bucket::{LabelBucket, LabelBucketKey};
use crate::error::Error;
use crate::graph::Graph;
use crate::id::{Direction, Label, NodeId};
use crate::input::Order;
use crate::store::memory::Memory;
use alloc::vec::Vec;
use core::ops::ControlFlow;

/// How many times more live edges a paged bucket must hold than the other bucket for an
/// intersection to look the other's targets up in its index instead of reading and sorting it:
/// a lookup cost about 14K instructions, reading and sorting about 300 an edge.
const PROBE_RATIO: u32 = 48;

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
    /// `(label, direction)` buckets hold a live edge to, ascending, each once.
    ///
    /// Merges the two buckets' sorted views ([`Self::sorted_targets`]); when one is a paged bucket
    /// many times larger than the other, looks each of the smaller one's targets up in its target
    /// index instead, so the cost follows the smaller end.
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
        out.clear();
        let key = LabelBucketKey { label, direction };
        let first = self.read_label_bucket(u, key)?;
        let second = self.read_label_bucket(w, key)?;
        let (Some(first), Some(second)) = (first, second) else {
            return Ok(());
        };
        let (small, large) = if first.slots.live() <= second.slots.live() {
            (first, second)
        } else {
            (second, first)
        };
        let mut mine = Vec::new();
        self.sorted_into(&small, &mut mine);
        mine.dedup();
        let probe = large.index.tree().is_some()
            && small.slots.live().saturating_mul(PROBE_RATIO) < large.slots.live();
        if probe {
            out.extend(mine.into_iter().filter(|&t| self.holds(&large, t)));
            return Ok(());
        }
        let mut theirs = Vec::new();
        self.sorted_into(&large, &mut theirs);
        intersect(&mine, &theirs, out);
        Ok(())
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

/// The nodes in both ascending lists, each once, appended to `out`; `mine` holds no repeats.
fn intersect(mine: &[NodeId], theirs: &[NodeId], out: &mut Vec<NodeId>) {
    let (mut i, mut j) = (0, 0);
    while let (Some(a), Some(b)) = (mine.get(i), theirs.get(j)) {
        match a.index().cmp(&b.index()) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => {
                out.push(*a);
                i += 1;
                j += 1;
                // Skip the rest of a target's parallel edges.
                while theirs.get(j).is_some_and(|c| c == a) {
                    j += 1;
                }
            }
        }
    }
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
