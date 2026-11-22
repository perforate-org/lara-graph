//! Detaching a vertex: removing its edges a bounded step at a time.

use super::pair::{Found, opposite, order};
use super::{Detachment, Error, Graph, Step};
use crate::bucket;
use crate::corrupt::corrupt;
use crate::id::{EdgeRef, Position, Progress, RemovedEdge, VertexId, VertexState};
use crate::labels::{self, BucketKey};
use crate::memory::Memory;
use crate::stored::Bucket;
use crate::vertex::Record;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::num::NonZeroU32;

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
            let found = Found {
                entry,
                bucket: Bucket::resolve(&self.edges.pages(), stored),
            };
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
            let (theirs, other) = self.first_half_to(v, half, edge.target(), &mut hints);
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
            removed.push(RemovedEdge::new(halves, value));
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

    /// The other half of `half`, `v`'s first live edge in its bucket: the first live entry to
    /// `v` in `other`'s opposite bucket, since `v`'s earlier entries to `other` are gone. For an
    /// undirected self-loop that is `half` itself. `hints` remembers where each scan stopped
    /// within one step; positions before it hold no live entry to `v`.
    fn first_half_to(
        &self,
        v: VertexId,
        half: EdgeRef,
        other: VertexId,
        hints: &mut BTreeMap<(VertexId, BucketKey), Position>,
    ) -> (Found, EdgeRef) {
        let key = BucketKey {
            label: half.label(),
            side: opposite(half.side()),
        };
        let from = hints
            .get(&(other, key))
            .copied()
            .unwrap_or(Position::new(0));
        let theirs = self.present(other, key);
        let found = bucket::find(&theirs.bucket.slots, &self.edges.pages(), from, |e| {
            e.target() == v
        })
        .unwrap_or_else(|| corrupt("an edge half without its counterpart"));
        // Now a hole, so the next scan starts there.
        hints.insert((other, key), found.position());
        (
            theirs,
            EdgeRef::new(other, key.label, key.side, found.position()),
        )
    }
}
