//! Removing edges: both halves of each, with the value they carried.

use crate::bucket::LabelBucketKey;
use crate::bucket::slots;
use crate::error::{BatchError, Error};
use crate::graph::Graph;
use crate::graph::locate::Found;
use crate::graph::locate::LastBucket;
use crate::graph::locate::Pair;
use crate::id::{EdgeRef, NodeId};
use crate::outcome::Removal;
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
        let key = LabelBucketKey::of(half);
        self.flush_unless(&mut pending.own, node, key);
        let own = match pending.own.get(node, key) {
            Some(found) => Some(found),
            None => self.find_label_bucket(node, key)?,
        };
        let Some(own) = own else {
            pending.own = LastBucket::default();
            return Ok(Removal::Absent);
        };
        let other = {
            let view = self.edge_view().pages();
            slots::get(&own.bucket.slots, half.position(), &view)
        };
        let Some(other) = other else {
            pending.own.set(node, key, own);
            return Ok(Removal::Absent);
        };
        let their_key = LabelBucketKey::of(half).opposite();
        self.flush_unless(&mut pending.theirs, other, their_key);
        let theirs = pending
            .theirs
            .get_or_else(other, their_key, || self.present(other, their_key));
        let pair = self.pair_with_theirs(half, own, other, their_key, theirs);
        let (removal, own_after, theirs_after) = self.take_removal(half, pair);
        pending.own.set(node, key, own_after);
        pending.theirs = LastBucket::default();
        if let Some(found) = theirs_after {
            pending.theirs.set(other, their_key, found);
        }
        Ok(removal)
    }

    /// Publishes the bucket `cache` holds unless it is `(node, key)`'s.
    fn flush_unless(&mut self, cache: &mut LastBucket<Found>, node: NodeId, key: LabelBucketKey) {
        if let Some(found) = cache.take_other(node, key) {
            self.publish(found.entry, found.bucket);
        }
    }

    fn flush_pending(&mut self, cache: &mut LastBucket<Found>) {
        if let Some(found) = cache.take() {
            self.publish(found.entry, found.bucket);
        }
    }

    fn take_removal(
        &mut self,
        half: EdgeRef,
        Pair {
            mut own,
            other,
            mut theirs,
        }: Pair,
    ) -> (Removal, Found, Option<Found>) {
        let schema = self.widths.schema(half.label());
        let edge =
            self.edges
                .remove_pair((&mut own.bucket, half), (&mut theirs.bucket, other), schema);
        (
            Removal::Removed(edge),
            own,
            (other != half).then_some(theirs),
        )
    }
}

/// A removal batch's changed buckets, published once the batch moves on from them.
#[derive(Default)]
struct Pending {
    own: LastBucket<Found>,
    theirs: LastBucket<Found>,
}
