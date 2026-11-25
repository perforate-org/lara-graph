//! Detaching a node: removing its edges a bounded step at a time.

use crate::bucket::label_bucket::{self, Span, Window};
use crate::bucket::stored::{Index, LabelBucket};
use crate::graph::bucket::{Found, indexed, opposite, order};
use crate::graph::{BatchError, Detachment, Error, Graph, Step, Sweep};
use crate::id::{
    Direction, EdgeRef, Label, NodeId, NodeState, Order, Position, Progress, RemovedEdge,
};
use crate::node::detaching_set::DetachingSet;
use crate::node::label_bucket_list::{self, LabelBucketKey};
use crate::node::record::Record;
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use alloc::vec::Vec;
use core::num::NonZeroU32;
use core::ops::ControlFlow;

impl<M: Memory> Graph<M> {
    /// Starts deleting `v` (DETACH DELETE): from now on it takes no new edges, and
    /// [`Self::detach_step`] removes its edges, then marks it deleted. Constant time; calling it
    /// again while it is detaching changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], or [`Error::Deleted`] if it is already deleted.
    pub fn detach(&mut self, v: NodeId) -> Result<(), Error> {
        let record = self.record(v)?;
        if record.state == NodeState::Deleted {
            return Err(Error::Deleted(v));
        }
        self.write_record(
            v,
            Record {
                state: NodeState::Detaching,
                ..record
            },
        );
        Ok(())
    }

    /// Starts deleting every node in `nodes` (DETACH DELETE). All nodes are checked before the
    /// first state change, so an unknown or already deleted node leaves the graph unchanged.
    /// A node already detaching is left as it is.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] or [`Error::Deleted`] at the first failing position, as
    /// [`BatchError::Unchanged`].
    pub fn detach_many(&mut self, nodes: &[NodeId]) -> Result<(), BatchError> {
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
                        state: NodeState::Detaching,
                        ..record
                    },
                );
            }
        }
        Ok(())
    }

    /// Removes up to `budget` of a detaching node's edges, both halves of each, and reports
    /// them. Between steps the graph is whole: every edge is either present at both ends or gone.
    /// When `v` has no edges left it is [`NodeState::Deleted`] and the step is
    /// [`Progress::Done`].
    ///
    /// Each edge costs a read of `v`'s next live entry and a scan of the neighbour's bucket up to
    /// the other half.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], or [`Error::NotDetaching`] unless [`Self::detach`] was called.
    pub fn detach_step(&mut self, v: NodeId, budget: NonZeroU32) -> Result<Detachment, Error> {
        if self.record(v)?.state != NodeState::Detaching {
            return Err(Error::NotDetaching(v));
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
        // for each edge cost about 5 % of detaching a hub.
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
                    bucket: LabelBucket::resolve(&self.edges.pages(), stored),
                });
                (found, key)
            };
            let from = start_in(cursor, &key);
            let first =
                label_bucket::find(&found.bucket.slots, &self.edges.pages(), from, |_| true);
            let Some(edge) = first else {
                // All holes: publish the pending bucket, then compaction drops it.
                self.publish(found.entry, found.bucket);
                self.compact(v, key.label, key.direction)?;
                (cursor, hint) = (None, None);
                continue;
            };
            cursor = Some((key, edge.position()));
            let half = EdgeRef::new(v, key.label, key.direction, edge.position());
            let theirs_key = (edge.target(), opposite_key(half));
            let from = start_in(hint, &theirs_key);
            let (theirs, other) = theirs_cache
                .filter(|(cached, _)| *cached == theirs_key)
                .map_or_else(
                    || self.first_half_from(v, half, (edge.target(), from)),
                    |(_, theirs)| self.first_half_in(v, half, (edge.target(), from), theirs),
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

    /// The set of `nodes` for [`Self::sweep`], each checked once to be detaching or deleted. A
    /// deleted one is fine: it has no edges left.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], or [`Error::NotDetaching`] for a node that is live.
    pub fn detaching_set(
        &self,
        nodes: impl IntoIterator<Item = NodeId>,
    ) -> Result<DetachingSet, Error> {
        let set = DetachingSet::new(nodes);
        for d in set.iter() {
            if self.record(d)?.state == NodeState::Live {
                return Err(Error::NotDetaching(d));
            }
        }
        Ok(set)
    }

    /// Removes the edges of `v`'s `(label, direction)` bucket whose other ends are in `detaching`,
    /// both halves of each, looking at up to `budget` positions from `from`. One pass over a
    /// bucket that many detaching nodes share (a hub's, say) removes their edges to it for less
    /// than [`Self::detach_step`] does one node at a time, which pairs each edge through the
    /// shared bucket's target index. Their own buckets are left with holes, which their
    /// [`Self::detach_step`]s then pass over.
    ///
    /// Call again from [`Sweep::next`] until it is `None`. A self-loop at `v` is left to `v`'s own
    /// detach.
    ///
    /// The cost of a call follows `budget`, not the size of `detaching`: its members were checked
    /// when [`Self::detaching_set`] built it.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for `v`.
    pub fn sweep(
        &mut self,
        v: NodeId,
        (label, direction): (Label, Direction),
        (from, budget): (Position, NonZeroU32),
        detaching: &DetachingSet,
    ) -> Result<Sweep, Error> {
        let key = LabelBucketKey { label, direction };
        let Some(found) = self.find_label_bucket(v, key)? else {
            return Ok(Sweep::new(Vec::new(), None));
        };
        let stored = found.bucket.slots.stored();
        let high = from.index().saturating_add(budget.get()).min(stored);
        // One pass finds the edges; positions stay put as they go, so they are removed after.
        let mut doomed = Vec::new();
        let span = Span::new(from.index(), high, Order::Ascending);
        label_bucket::visit(
            &found.bucket.slots,
            &self.edges.pages(),
            span,
            Window::Page,
            |e| {
                if e.target() != v && detaching.contains(e.target()) {
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
            let half = EdgeRef::new(v, label, direction, edge.position());
            let (theirs, other) = self.first_half_from(v, half, (edge.target(), Position::new(0)));
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
        (found, half): (Found, EdgeRef),
        (theirs, other): (Found, EdgeRef),
    ) -> (RemovedEdge, Found, Option<Found>) {
        let halves = order(half, other);
        let canonical = if halves.canonical() == half {
            found
        } else {
            theirs
        };
        let value = self.read_value(&canonical, half.label(), halves.canonical().position());
        let (bucket, _) = self.remove_half(found, half.position());
        let next_theirs = if other == half {
            None
        } else {
            let bucket = self.tombstone(theirs, other.position());
            Some(Found { bucket, ..theirs })
        };
        (
            RemovedEdge::new(halves, value),
            Found { bucket, ..found },
            next_theirs,
        )
    }

    /// A detaching node's bucket without its target index: all its edges are going, so
    /// keeping the index in step would only cost. Lookups into it scan instead. Within reserved
    /// memory (releasing allocates nothing).
    fn draining(&mut self, found: Found) -> Found {
        let Index::Tree(_) = found.bucket.index else {
            return found;
        };
        let bucket = self.drop_index(found.bucket, Index::Dropped);
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
        let key = opposite_key(half);
        self.first_half_in(v, half, (other, from), self.present(other, key))
    }

    /// [`first_half_from`] with the opposite bucket already resolved.
    fn first_half_in(
        &self,
        v: NodeId,
        half: EdgeRef,
        (other, from): (NodeId, Position),
        theirs: Found,
    ) -> (Found, EdgeRef) {
        let key = opposite_key(half);
        // A paged bucket finds it through its target index; a smaller one is scanned.
        let at = if indexed(&theirs.bucket).is_some() {
            self.select_to(&theirs.bucket, v, 0)
        } else {
            label_bucket::find(&theirs.bucket.slots, &self.edges.pages(), from, |e| {
                e.target() == v
            })
            .map(|e| e.position())
        }
        .unwrap_or_else(|| corrupt("an edge half without its counterpart"));
        (theirs, EdgeRef::new(other, key.label, key.direction, at))
    }
}

/// The key of the bucket that holds the other half of `half`.
const fn opposite_key(half: EdgeRef) -> LabelBucketKey {
    LabelBucketKey {
        label: half.label(),
        direction: opposite(half.direction()),
    }
}

/// Where a scan of the bucket `key` starts: at `mark` when it is that bucket's, else at the start.
fn start_in<K: PartialEq>(mark: Option<(K, Position)>, key: &K) -> Position {
    mark.filter(|(k, _)| k == key)
        .map_or(Position::new(0), |(_, at)| at)
}
