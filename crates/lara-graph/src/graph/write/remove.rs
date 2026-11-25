//! Removing edges: both halves of each, with the value they carried.

use crate::bucket::label_bucket;
use crate::graph::bucket::{Found, LastBucket, Pair, key_of, opposite, order};
use crate::graph::{BatchError, Error, Graph, Removal};
use crate::id::{EdgeRef, NodeId, RemovedEdge};
use crate::node::label_bucket_list::LabelBucketKey;
use crate::store::memory::Memory;
use alloc::vec::Vec;

impl<M: Memory> Graph<M> {
    /// Removes the edge `half` belongs to: both its halves. Other edges keep their positions.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`]. A removal takes no memory.
    pub fn remove_edge(&mut self, half: EdgeRef) -> Result<Removal, Error> {
        let Some(pair) = self.pair(half)? else {
            return Ok(Removal::Absent);
        };
        let (removal, own_after, theirs_after) = self.take_removal(half, pair);
        self.publish(own_after.entry, own_after.bucket);
        if let Some(theirs) = theirs_after {
            self.publish(theirs.entry, theirs.bucket);
        }
        Ok(removal)
    }

    /// Removes every edge identified by `halves`, in order. All halves are checked before the
    /// first removal; if a check fails, the graph is unchanged, so the batch is all-or-none.
    /// Once the checks pass, no removal can fail. A half whose edge is already gone yields
    /// [`Removal::Absent`].
    ///
    /// Consecutive halves in the same bucket reuse its `Found`, and parallel edges reuse the
    /// opposite bucket.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any half at or past [`Graph::node_count`].
    pub fn remove_edges(&mut self, halves: &[EdgeRef]) -> Result<Vec<Removal>, BatchError> {
        for (index, half) in halves.iter().enumerate() {
            self.check(half.node())
                .map_err(|error| BatchError::unchanged(index, error))?;
        }
        let mut out = Vec::with_capacity(halves.len());
        let mut pending = Pending::default();
        for (index, &half) in halves.iter().enumerate() {
            let removal = self
                .remove_pending(half, &mut pending)
                .map_err(|error| BatchError::after_prefix(index, error))?;
            out.push(removal);
        }
        self.flush_pending(&mut pending.own);
        self.flush_pending(&mut pending.theirs);
        Ok(out)
    }

    /// Removes the edge `half` belongs to, keeping both changed buckets in `pending` and
    /// publishing a pending bucket only when the batch moves on from it.
    fn remove_pending(&mut self, half: EdgeRef, pending: &mut Pending) -> Result<Removal, Error> {
        let node = half.node();
        let key = key_of(half);
        self.flush_unless(&mut pending.own, node, key);
        let own = match pending.own.get(node, key) {
            Some(found) => Some(found),
            None => self.find_label_bucket(node, key)?,
        };
        let Some(own) = own else {
            pending.own.clear();
            return Ok(Removal::Absent);
        };
        let other = {
            let view = self.edges.pages();
            label_bucket::get(&own.bucket.slots, half.position(), &view)
        };
        let Some(other) = other else {
            pending.own.set(node, key, own);
            return Ok(Removal::Absent);
        };
        let their_key = LabelBucketKey {
            label: half.label(),
            direction: opposite(half.direction()),
        };
        self.flush_unless(&mut pending.theirs, other, their_key);
        let theirs = pending
            .theirs
            .get_or_else(other, their_key, || self.present(other, their_key));
        let pair = self.pair_with_theirs(half, own, other, their_key, theirs);
        let (removal, own_after, theirs_after) = self.take_removal(half, pair);
        pending.own.set(node, key, own_after);
        pending.theirs = LastBucket::holding(theirs_after.map(|found| (other, their_key, found)));
        Ok(removal)
    }

    /// Publishes the bucket `cache` holds unless it is `(node, key)`'s.
    fn flush_unless(&mut self, cache: &mut LastBucket<Found>, node: NodeId, key: LabelBucketKey) {
        if cache.holds_other(node, key) {
            self.flush_pending(cache);
        }
    }

    fn flush_pending(&mut self, cache: &mut LastBucket<Found>) {
        if let Some(found) = cache.take() {
            self.publish(found.entry, found.bucket);
        }
    }

    /// Applies one found pair and returns the removal plus the updated own and opposite
    /// buckets, so a batch can carry them to the next edge.
    fn take_removal(
        &mut self,
        half: EdgeRef,
        Pair { own, other, theirs }: Pair,
    ) -> (Removal, Found, Option<Found>) {
        let halves = order(half, other);
        // Only a label with a width has a value to report.
        let value = if self.widths.get(half.label()).is_some() {
            let canonical = if halves.canonical() == half {
                own
            } else {
                theirs
            };
            self.read_value(&canonical, half.label(), halves.canonical().position())
        } else {
            Vec::new()
        };
        let (own_bucket, _) = self.remove_half(own, half.position());
        let own_after = Found {
            entry: own.entry,
            bucket: own_bucket,
        };
        let theirs_after = if other == half {
            None
        } else {
            let (bucket, _) = self.remove_half(theirs, other.position());
            Some(Found {
                entry: theirs.entry,
                bucket,
            })
        };
        (
            Removal::Removed(RemovedEdge::new(halves, value)),
            own_after,
            theirs_after,
        )
    }
}

impl<M: Memory> Graph<M> {}

/// A removal batch's changed buckets, published once the batch moves on from them.
#[derive(Default)]
struct Pending {
    own: LastBucket<Found>,
    theirs: LastBucket<Found>,
}
