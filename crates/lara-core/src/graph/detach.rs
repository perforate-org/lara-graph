//! Detaching a vertex: removing its edges a bounded step at a time.

use super::index::indexed;
use super::pair::{Found, opposite, order};
use super::{Detachment, Error, Graph, Step, Sweep};
use crate::bucket::{self, Span, Window};
use crate::corrupt::corrupt;
use crate::id::{
    EdgeRef, Label, Order, Position, Progress, RemovedEdge, Side, VertexId, VertexState,
};
use crate::labels::{self, BucketKey};
use crate::memory::Memory;
use crate::stored::{Bucket, Index};
use crate::vertex::Record;
use crate::vertex_set::VertexSet;
use alloc::collections::BTreeMap;
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
    /// [`Error::UnknownVertex`], or [`Error::Deleted`] if it is already deleted.
    pub fn detach(&mut self, v: VertexId) -> Result<(), Error> {
        let record = self.record(v)?;
        if record.state == VertexState::Deleted {
            return Err(Error::Deleted(v));
        }
        self.write_record(
            v,
            Record {
                state: VertexState::Detaching,
                ..record
            },
        );
        Ok(())
    }

    /// Removes up to `budget` of a detaching vertex's edges, both halves of each, and reports
    /// them. Between steps the graph is whole: every edge is either present at both ends or gone.
    /// When `v` has no edges left it is [`VertexState::Deleted`] and the step is
    /// [`Progress::Done`].
    ///
    /// Each edge costs a read of `v`'s next live entry and a scan of the neighbour's bucket up to
    /// the other half.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`], or [`Error::NotDetaching`] unless [`Self::detach`] was called.
    pub fn detach_step(&mut self, v: VertexId, budget: NonZeroU32) -> Result<Detachment, Error> {
        if self.record(v)?.state != VertexState::Detaching {
            return Err(Error::NotDetaching(v));
        }
        let mut removed = Vec::new();
        // Per bucket of `v`: positions before hold no live entry.
        let mut cursors: BTreeMap<BucketKey, Position> = BTreeMap::new();
        // Per neighbour bucket: positions before hold no live entry to `v`.
        let mut hints = BTreeMap::new();
        for _ in 0..budget.get() {
            let list = self.list(v)?;
            let Some((entry, key, stored)) =
                list.map(|list| labels::first(&self.lists.pages(), list))
            else {
                break;
            };
            let found = self.draining(Found {
                entry,
                bucket: Bucket::resolve(&self.edges.pages(), stored),
            });
            let from = cursors.get(&key).copied().unwrap_or(Position::new(0));
            let first = bucket::find(&found.bucket.slots, &self.edges.pages(), from, |_| true);
            let Some(edge) = first else {
                // All holes: compaction drops the bucket without moving anything.
                self.compact(v, key.label, key.side)?;
                cursors.clear();
                hints.clear();
                continue;
            };
            cursors.insert(key, edge.position());
            let half = EdgeRef::new(v, key.label, key.side, edge.position());
            let key = (edge.target(), opposite_key(half));
            let from = hints.get(&key).copied().unwrap_or(Position::new(0));
            let (theirs, other) = self.first_half_from(v, half, (edge.target(), from));
            // Now a hole, so the next scan starts there.
            hints.insert(key, other.position());
            removed.push(self.take_pair((found, half), (theirs, other)));
        }
        let progress = if self.list(v)?.is_none() {
            self.write_record(
                v,
                Record {
                    list: None,
                    state: VertexState::Deleted,
                },
            );
            Progress::Done
        } else {
            Progress::Remaining
        };
        Ok(Step::new(removed, progress))
    }

    /// Removes the edges of `v`'s `(label, side)` bucket whose other ends are in `detaching`,
    /// both halves of each, looking at up to `budget` positions from `from`. One pass over a
    /// bucket that many detaching vertices share (a hub's, say) removes their edges to it for less
    /// than [`Self::detach_step`] does one vertex at a time, which pairs each edge through the
    /// shared bucket's target index. Their own buckets are left with holes, which their
    /// [`Self::detach_step`]s then pass over.
    ///
    /// Call again from [`Sweep::next`] until it is `None`. A self-loop at `v` is left to `v`'s own
    /// detach.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`], or [`Error::NotDetaching`] for a vertex in `detaching` that is
    /// live; then nothing changed. A deleted one is fine: it has no edges left.
    pub fn sweep(
        &mut self,
        v: VertexId,
        (label, side): (Label, Side),
        (from, budget): (Position, NonZeroU32),
        detaching: &VertexSet,
    ) -> Result<Sweep, Error> {
        for d in detaching.iter() {
            if self.record(d)?.state == VertexState::Live {
                return Err(Error::NotDetaching(d));
            }
        }
        let key = BucketKey { label, side };
        let Some(found) = self.find_bucket(v, key)? else {
            return Ok(Sweep::new(Vec::new(), None));
        };
        let stored = found.bucket.slots.stored();
        let high = from.index().saturating_add(budget.get()).min(stored);
        // One pass finds the edges; positions stay put as they go, so they are removed after.
        let mut doomed = Vec::new();
        let span = Span::new(from.index(), high, Order::Ascending);
        bucket::visit(
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
        for edge in doomed {
            let found = self.present(v, key);
            let half = EdgeRef::new(v, label, side, edge.position());
            let (theirs, other) = self.first_half_from(v, half, (edge.target(), Position::new(0)));
            removed.push(self.take_pair((found, half), (theirs, other)));
        }
        let next = (high < stored).then(|| Position::new(high));
        Ok(Sweep::new(removed, next))
    }

    /// Removes both halves of an edge, `half` in `found` and `other` in `theirs`, and reports it
    /// with its value.
    fn take_pair(
        &mut self,
        (found, half): (Found, EdgeRef),
        (theirs, other): (Found, EdgeRef),
    ) -> RemovedEdge {
        let halves = order(half, other);
        let canonical = if halves.canonical() == half {
            found
        } else {
            theirs
        };
        let value = self.read_value(&canonical, half.label(), halves.canonical().position());
        self.tombstone(found, half.position());
        if other != half {
            self.tombstone(theirs, other.position());
        }
        RemovedEdge::new(halves, value)
    }

    /// A detaching vertex's bucket without its target index: all its edges are going, so
    /// keeping the index in step would only cost. Lookups into it scan instead. Within reserved
    /// memory (releasing allocates nothing).
    fn draining(&mut self, found: Found) -> Found {
        let Index::Tree(_) = found.bucket.index else {
            return found;
        };
        let bucket = self.drop_index(found.bucket, Index::Dropped);
        self.publish(found.entry, bucket);
        Found { bucket, ..found }
    }

    /// The other half of `half`, `v`'s first live edge in its bucket: the first live entry to
    /// `v` in `other`'s opposite bucket, since `v`'s earlier entries to `other` are gone. For an
    /// undirected self-loop that is `half` itself. A bucket without an index is scanned from
    /// `from`: positions before it hold no live entry to `v`.
    fn first_half_from(
        &self,
        v: VertexId,
        half: EdgeRef,
        (other, from): (VertexId, Position),
    ) -> (Found, EdgeRef) {
        let key = opposite_key(half);
        let theirs = self.present(other, key);
        // A paged bucket finds it through its target index; a smaller one is scanned.
        let at = if indexed(&theirs.bucket).is_some() {
            self.select_to(&theirs.bucket, v, 0)
        } else {
            bucket::find(&theirs.bucket.slots, &self.edges.pages(), from, |e| {
                e.target() == v
            })
            .map(|e| e.position())
        }
        .unwrap_or_else(|| corrupt("an edge half without its counterpart"));
        (theirs, EdgeRef::new(other, key.label, key.side, at))
    }
}

/// The key of the bucket that holds the other half of `half`.
const fn opposite_key(half: EdgeRef) -> BucketKey {
    BucketKey {
        label: half.label(),
        side: opposite(half.side()),
    }
}
