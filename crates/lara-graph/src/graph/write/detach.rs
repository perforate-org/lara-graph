//! Deleting a node: removing its edges a bounded step at a time.

mod deleting_set;

pub use deleting_set::DeletingSet;

use crate::bucket::LabelBucketKey;
use crate::bucket::slots::{self, Span, Window};
use crate::bucket::{Index, LabelBucket};
use crate::error::{BatchError, Error};
use crate::graph::Graph;
use crate::graph::GraphMemories;
use crate::graph::locate::Found;
use crate::id::{Adjacency, EdgeRef, NodeId, NodeState, Position};
use crate::input::Order;
use crate::node::label_bucket_list;
use crate::node::record::Record;
use crate::outcome::{Deletion, Step, Sweep};
use crate::outcome::{Progress, RemovedEdge};
use crate::store::corrupt::corrupt;
use alloc::vec::Vec;
use core::num::NonZeroU32;
use core::ops::ControlFlow;

impl<S: GraphMemories> Graph<S> {
    /// Starts deleting `v` (DETACH DELETE): from now on it takes no new edges, and
    /// [`Self::delete_step`] removes its edges, then marks it deleted. Constant time; calling it
    /// again while it is deleting changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], or [`Error::Deleted`] if it is already deleted.
    pub fn begin_delete(&mut self, v: NodeId) -> Result<(), Error> {
        let record = self.record(v)?;
        if record.state == NodeState::Deleted {
            return Err(Error::Deleted(v));
        }
        self.write_record(
            v,
            Record {
                state: NodeState::Deleting,
                ..record
            },
        );
        Ok(())
    }

    /// Starts deleting every node in `nodes` (DETACH DELETE). All nodes are checked before the
    /// first state change, so an unknown or already deleted node leaves the graph unchanged.
    /// A node already deleting is left as it is.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] or [`Error::Deleted`] at the first failing position, as
    /// [`BatchError::Unchanged`].
    pub fn begin_delete_many(&mut self, nodes: &[NodeId]) -> Result<(), BatchError> {
        for (index, &v) in nodes.iter().enumerate() {
            let record = self
                .record(v)
                .map_err(|error| BatchError::unchanged(index, error))?;
            if record.state == NodeState::Deleted {
                return Err(BatchError::unchanged(index, Error::Deleted(v)));
            }
        }
        for &v in nodes {
            let record = self.record(v).unwrap_or_else(|_| corrupt("a checked node"));
            if record.state == NodeState::Live {
                self.write_record(
                    v,
                    Record {
                        state: NodeState::Deleting,
                        ..record
                    },
                );
            }
        }
        Ok(())
    }

    /// Removes up to `budget` of a deleting node's edges, both halves of each, and reports
    /// them. Between steps the graph is whole: every edge is either present at both ends or gone.
    /// When `v` has no edges left it is [`NodeState::Deleted`] and the step is
    /// [`Progress::Done`].
    ///
    /// Each edge costs a read of `v`'s next live entry and a scan of the neighbour's bucket up to
    /// the other half.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], or [`Error::NotDeleting`] unless [`Self::begin_delete`] was called.
    pub fn delete_step(&mut self, v: NodeId, budget: NonZeroU32) -> Result<Deletion, Error> {
        if self.record(v)?.state != NodeState::Deleting {
            return Err(Error::NotDeleting(v));
        }
        let mut removed = Vec::new();
        // In `v`'s bucket being drained (always its first, until it goes): positions before
        // hold no live entry.
        let mut cursor: Option<(LabelBucketKey, Position)> = None;
        // In the neighbour bucket of the last edge: positions before hold no live entry to `v`.
        // Parallel edges to one neighbour follow each other there; any other bucket is scanned
        // from its start, or reached through its target index.
        let mut hint: Option<((NodeId, LabelBucketKey), Position)> = None;
        // The opposite bucket of the last edge, while parallel edges stay on one neighbour.
        let mut theirs_cache: Option<((NodeId, LabelBucketKey), Found)> = None;
        // `v`'s bucket as the last removal published it: it stays `v`'s first, at the same list
        // entry, until compaction drops it, and the other half is never in it. Reading it again
        // for each edge cost about 5 % of deleting a hub.
        let mut drained: Option<(Found, LabelBucketKey)> = None;
        for _ in 0..budget.get() {
            let (found, key) = if let Some(carried) = drained.take() {
                carried
            } else {
                let list = self.label_bucket_list(v)?;
                let Some((entry, key, stored)) = list
                    .map(|list| label_bucket_list::first(&self.label_bucket_lists.pages(), list))
                else {
                    break;
                };
                let found = self.draining(Found {
                    entry,
                    bucket: LabelBucket::resolve(&self.edge_view().pages(), stored),
                });
                (found, key)
            };
            let from = start_in(cursor, &key);
            let first = slots::find(&found.bucket.slots, &self.edge_view().pages(), from, |_| {
                true
            });
            let Some(edge) = first else {
                // All holes: publish the pending bucket, then compaction drops it.
                self.publish(found.entry, found.bucket);
                self.compact(v, Adjacency::new(key.label, key.direction))?;
                (cursor, hint) = (None, None);
                continue;
            };
            cursor = Some((key, edge.position()));
            let half = EdgeRef::of(v, key.label, key.direction, edge.position());
            let theirs_key = (edge.node(), LabelBucketKey::of(half).opposite());
            let from = start_in(hint, &theirs_key);
            let (theirs, other) = theirs_cache
                .filter(|(cached, _)| *cached == theirs_key)
                .map_or_else(
                    || self.first_half_from(v, half, (edge.node(), from)),
                    |(_, theirs)| self.first_half_in(v, half, (edge.node(), from), theirs),
                );
            // Now a hole, so the next scan there starts at it.
            hint = Some((theirs_key, other.position()));
            let (edge, found, next_theirs) = self.take_pair((found, half), (theirs, other));
            removed.push(edge);
            drained = Some((found, key));
            theirs_cache = next_theirs.map(|theirs| (theirs_key, theirs));
        }
        if let Some((found, _)) = drained {
            self.publish(found.entry, found.bucket);
        }
        let progress = if self.label_bucket_list(v)?.is_none() {
            self.clear_node_labels(v);
            self.write_record(
                v,
                Record {
                    list: None,
                    state: NodeState::Deleted,
                },
            );
            Progress::Done
        } else {
            Progress::Remaining
        };
        Ok(Step::new(removed, progress))
    }

    /// The set of `nodes` for [`Self::sweep`], each checked once to be deleting or deleted. A
    /// deleted one is fine: it has no edges left.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], or [`Error::NotDeleting`] for a node that is live.
    pub fn deleting_set(
        &self,
        nodes: impl IntoIterator<Item = NodeId>,
    ) -> Result<DeletingSet, Error> {
        let set = DeletingSet::new(nodes);
        for d in set.iter() {
            if self.record(d)?.state == NodeState::Live {
                return Err(Error::NotDeleting(d));
            }
        }
        Ok(set)
    }

    /// Removes the edges of `v`'s `adjacency` bucket whose other ends are in `deleting`,
    /// both halves of each, looking at up to `budget` positions from `from`. One pass over a
    /// bucket that many deleting nodes share (a hub's, say) removes their edges to it for less
    /// than [`Self::delete_step`] does one node at a time, which pairs each edge through the
    /// shared bucket's target index. Their own buckets are left with holes, which their
    /// [`Self::delete_step`]s then pass over.
    ///
    /// Call again from [`Sweep::next`] until it is `None`. A self-loop at `v` is left to `v`'s own
    /// deletion.
    ///
    /// The cost of a call follows `budget`, not the size of `deleting`: its members were checked
    /// when [`Self::deleting_set`] built it.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for `v`.
    pub fn sweep(
        &mut self,
        v: NodeId,
        adjacency: Adjacency,
        from: Position,
        budget: NonZeroU32,
        deleting: &DeletingSet,
    ) -> Result<Sweep, Error> {
        let (label, direction) = (adjacency.label(), adjacency.direction());
        let key = LabelBucketKey { label, direction };
        let Some(found) = self.find_label_bucket(v, key)? else {
            return Ok(Sweep::new(Vec::new(), None));
        };
        let stored = found.bucket.slots.stored();
        let high = from.index().saturating_add(budget.get()).min(stored);
        // One pass finds the edges; positions stay put as they go, so they are removed after.
        let mut doomed = Vec::new();
        let span = Span::new(from.index(), high, Order::Ascending);
        slots::visit(
            &found.bucket.slots,
            &self.edge_view().pages(),
            span,
            Window::Page,
            |e| {
                if e.node() != v && deleting.contains(e.node()) {
                    doomed.push(e);
                }
                ControlFlow::<()>::Continue(())
            },
        );
        let mut removed = Vec::with_capacity(doomed.len());
        // Each removal publishes the shared bucket; carry it to the next edge instead of
        // looking the node and its list up again.
        let mut found = found;
        for edge in doomed {
            let half = EdgeRef::of(v, label, direction, edge.position());
            let (theirs, other) = self.first_half_from(v, half, (edge.node(), Position::new(0)));
            let (pair, next, _) = self.take_pair((found, half), (theirs, other));
            removed.push(pair);
            found = next;
        }
        if !removed.is_empty() {
            self.publish(found.entry, found.bucket);
        }
        let next = (high < stored).then(|| Position::new(high));
        Ok(Sweep::new(removed, next))
    }

    /// Removes both halves of an edge, `half` in `found` and `other` in `theirs`, and reports it
    /// with its value, and with `half`'s bucket as published.
    fn take_pair(
        &mut self,
        (mut own, half): (Found, EdgeRef),
        (mut theirs, other): (Found, EdgeRef),
    ) -> (RemovedEdge, Found, Option<Found>) {
        let schema = self.widths.schema(half.label());
        let edge =
            self.edges
                .remove_pair((&mut own.bucket, half), (&mut theirs.bucket, other), schema);
        let theirs = (other != half).then(|| Found {
            bucket: self.publish(theirs.entry, theirs.bucket),
            ..theirs
        });
        (edge, own, theirs)
    }

    /// A deleting node's bucket without its target index: all its edges are going, so
    /// keeping the index in step would only cost. Lookups into it scan instead. Within reserved
    /// memory (releasing allocates nothing).
    fn draining(&mut self, found: Found) -> Found {
        let Index::Tree(_) = found.bucket.index else {
            return found;
        };
        let bucket = self.edges.drop_index(found.bucket, Index::Dropped);
        let bucket = self.publish(found.entry, bucket);
        Found { bucket, ..found }
    }

    /// The other half of `half`, `v`'s first live edge in its bucket: the first live entry to
    /// `v` in `other`'s opposite bucket, since `v`'s earlier entries to `other` are gone. For an
    /// undirected self-loop that is `half` itself. A bucket without an index is scanned from
    /// `from`: positions before it hold no live entry to `v`.
    fn first_half_from(
        &self,
        v: NodeId,
        half: EdgeRef,
        (other, from): (NodeId, Position),
    ) -> (Found, EdgeRef) {
        let key = LabelBucketKey::of(half).opposite();
        self.first_half_in(v, half, (other, from), self.present(other, key))
    }

    /// [`Self::first_half_from`] with the opposite bucket already found.
    fn first_half_in(
        &self,
        v: NodeId,
        half: EdgeRef,
        other: (NodeId, Position),
        theirs: Found,
    ) -> (Found, EdgeRef) {
        let at = self
            .edge_view()
            .first_half_to(v, half, other, &theirs.bucket);
        (theirs, at)
    }
}

/// Where a scan of the bucket `key` starts: at `mark` when it is that bucket's, else at the start.
fn start_in<K: PartialEq>(mark: Option<(K, Position)>, key: &K) -> Position {
    mark.filter(|(k, _)| k == key)
        .map_or(Position::new(0), |(_, at)| at)
}
