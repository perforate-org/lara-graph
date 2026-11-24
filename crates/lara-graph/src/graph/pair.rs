//! Finding and removing both halves of an edge.

use super::index::indexed;
use super::{BatchError, Error, Graph, Removal, reserved};
use crate::corrupt::corrupt;
use crate::descriptor::Descriptor;
use crate::id::{Direction, EdgeRef, Halves, NodeId, Position, RemovedEdge};
use crate::label_bucket;
use crate::label_bucket_list::{self, EntryAt, LabelBucketKey, Lookup};
use crate::memory::Memory;
use crate::page_store::Pages;
use crate::stored::{HEADER_CLASS, LabelBucket};
use alloc::vec::Vec;

impl<M: Memory> Graph<M> {
    /// The other half of the edge `half` belongs to, or `None` when `half` is not live.
    ///
    /// When the other bucket is small and holds one entry back, that entry is the answer;
    /// otherwise the halves are matched by rank. A paged bucket answers rank and selection from
    /// its target index (logarithmic, then one block scan); a smaller one is scanned.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn counterpart(&self, half: EdgeRef) -> Result<Option<EdgeRef>, Error> {
        Ok(self.pair(half)?.map(|pair| pair.other))
    }

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
        pending.theirs = LastBucket(theirs_after.map(|found| (other, their_key, found)));
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

    /// Both halves of the edge `half` belongs to, with their buckets; `None` when `half` is not
    /// live.
    pub(super) fn pair(&self, half: EdgeRef) -> Result<Option<Pair>, Error> {
        let Some(own) = self.find_label_bucket(half.node(), key_of(half))? else {
            return Ok(None);
        };
        Ok(self.pair_from(half, own))
    }

    /// [`pair`] with `half`'s own bucket already found.
    pub(super) fn pair_from(&self, half: EdgeRef, own: Found) -> Option<Pair> {
        let view = self.edges.pages();
        let other = label_bucket::get(&own.bucket.slots, half.position(), &view)?;
        let key = LabelBucketKey {
            label: half.label(),
            direction: opposite(half.direction()),
        };
        let theirs = self.present(other, key);
        Some(self.pair_with_theirs(half, own, other, key, theirs))
    }

    /// [`pair_from`] with the target and its bucket already found.
    pub(super) fn pair_with_theirs(
        &self,
        half: EdgeRef,
        own: Found,
        other: NodeId,
        key: LabelBucketKey,
        theirs: Found,
    ) -> Pair {
        let view = self.edges.pages();
        let owner = half.node();
        // A small other bucket with one entry back answers alone; else match by rank.
        let only = (indexed(&theirs.bucket).is_none()
            && theirs.bucket.slots.stored() <= own.bucket.slots.stored())
        .then(|| only_entry_to(&theirs.bucket.slots, &view, owner))
        .flatten();
        let at = only.unwrap_or_else(|| {
            let rank = self.rank_to(&own.bucket, other, half.position());
            self.select_to(&theirs.bucket, owner, rank)
                .unwrap_or_else(|| corrupt("an edge half without its counterpart"))
        });
        Pair {
            own,
            other: EdgeRef::new(other, key.label, key.direction, at),
            theirs,
        }
    }

    /// Tombstones the live half at `at` in `found` and takes it out of the target index. Writes
    /// no value and takes no memory.
    pub(super) fn tombstone(&mut self, found: Found, at: Position) -> LabelBucket {
        let (bucket, _) = self.remove_half(found, at);
        self.publish(found.entry, bucket)
    }

    /// Applies one tombstone to a bucket without publishing it yet.
    pub(super) fn remove_half(&mut self, found: Found, at: Position) -> (LabelBucket, NodeId) {
        let (next, target) = label_bucket::remove(found.bucket.slots, at, &self.edges)
            .unwrap_or_else(|| corrupt("live half not in its bucket"));
        let bucket = LabelBucket {
            slots: next,
            ..found.bucket
        };
        self.index_removed(&bucket, target, at);
        (bucket, target)
    }

    /// Publishes a bucket's new state: its header, and its list entry when it is plain or just
    /// changed form. Returns the bucket as published. Within reserved memory.
    pub(super) fn publish(&mut self, entry: EntryAt, bucket: LabelBucket) -> LabelBucket {
        let had_header = bucket.header.is_some();
        let bucket = self.settle(bucket);
        bucket.write_header(&self.edges);
        if !had_header || bucket.header.is_none() {
            label_bucket_list::set(&self.label_bucket_lists, entry, &bucket.stored());
        }
        bucket
    }

    /// Gives a bucket with values or an index a header, and releases a header nothing needs any
    /// more. Within reserved memory.
    pub(super) fn settle(&mut self, bucket: LabelBucket) -> LabelBucket {
        match (bucket.extended(), bucket.header) {
            (true, None) => LabelBucket {
                header: Some(reserved(self.edges.allocate(HEADER_CLASS))),
                ..bucket
            },
            (false, Some(header)) => {
                self.edges.release(header);
                LabelBucket {
                    header: None,
                    ..bucket
                }
            }
            (true, Some(_)) | (false, None) => bucket,
        }
    }

    /// The bucket a live half proves exists.
    pub(super) fn present(&self, v: NodeId, key: LabelBucketKey) -> Found {
        self.find_label_bucket(v, key)
            .ok()
            .flatten()
            .unwrap_or_else(|| corrupt("an edge half without its bucket"))
    }

    pub(super) fn find_label_bucket(
        &self,
        v: NodeId,
        key: LabelBucketKey,
    ) -> Result<Option<Found>, Error> {
        let list = self.label_bucket_list(v)?;
        Ok(
            match label_bucket_list::lookup(&self.label_bucket_lists.pages(), list, key) {
                Lookup::Present { entry, stored } => Some(Found {
                    entry,
                    bucket: LabelBucket::resolve(&self.edges.pages(), stored),
                }),
                Lookup::Absent { .. } => None,
            },
        )
    }
}

impl<M: Memory> Graph<M> {
    /// `v`'s bucket `key` for a read: without the list entry's position, which only an update
    /// needs.
    pub(super) fn read_label_bucket(
        &self,
        v: NodeId,
        key: LabelBucketKey,
    ) -> Result<Option<LabelBucket>, Error> {
        let list = self.label_bucket_list(v)?;
        Ok(
            label_bucket_list::lookup_stored(&self.label_bucket_lists.pages(), list, key)
                .map(|stored| LabelBucket::resolve(&self.edges.pages(), stored)),
        )
    }
}

/// A label bucket found in its node's label bucket list.
#[derive(Clone, Copy)]
pub(super) struct Found {
    pub(super) entry: EntryAt,
    pub(super) bucket: LabelBucket,
}

/// The bucket last looked up for one `(node, key)`, so consecutive halves in that bucket reuse
/// the lookup.
#[derive(Clone, Copy)]
pub(super) struct LastBucket<T>(Option<(NodeId, LabelBucketKey, T)>);

impl<T> Default for LastBucket<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T: Copy> LastBucket<T> {
    /// The bucket held for `(node, key)`, if that is the one held.
    #[inline]
    pub(super) fn get(&self, node: NodeId, key: LabelBucketKey) -> Option<T> {
        match self.0 {
            Some((last_node, last_key, found)) if last_node == node && last_key == key => {
                Some(found)
            }
            _ => None,
        }
    }

    /// The bucket held for `(node, key)`, else `find`'s, which is then held.
    #[inline]
    pub(super) fn get_or_try<E>(
        &mut self,
        node: NodeId,
        key: LabelBucketKey,
        find: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        if let Some(found) = self.get(node, key) {
            return Ok(found);
        }
        let found = find()?;
        self.set(node, key, found);
        Ok(found)
    }

    /// The bucket held for `(node, key)`, else `find`'s, which is then held.
    #[inline]
    pub(super) fn get_or_else(
        &mut self,
        node: NodeId,
        key: LabelBucketKey,
        find: impl FnOnce() -> T,
    ) -> T {
        if let Some(found) = self.get(node, key) {
            return found;
        }
        let found = find();
        self.set(node, key, found);
        found
    }

    #[inline]
    pub(super) fn set(&mut self, node: NodeId, key: LabelBucketKey, found: T) {
        self.0 = Some((node, key, found));
    }

    fn clear(&mut self) {
        self.0 = None;
    }

    fn take(&mut self) -> Option<T> {
        self.0.take().map(|(_, _, found)| found)
    }

    /// Whether a bucket other than `(node, key)`'s is held.
    fn holds_other(&self, node: NodeId, key: LabelBucketKey) -> bool {
        self.0
            .is_some_and(|(last_node, last_key, _)| last_node != node || last_key != key)
    }
}

/// A removal batch's changed buckets, published once the batch moves on from them.
#[derive(Default)]
struct Pending {
    own: LastBucket<Found>,
    theirs: LastBucket<Found>,
}

/// Both halves of one edge: the given half's bucket, the other half, and its bucket.
pub(super) struct Pair {
    pub(super) own: Found,
    pub(super) other: EdgeRef,
    pub(super) theirs: Found,
}

/// The position of the one live entry to `owner` in `d`, or `None` when there are none or
/// several. Stops at the second.
fn only_entry_to<M: Memory>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    owner: NodeId,
) -> Option<Position> {
    let mut first = None;
    let mut seen = 0u32;
    label_bucket::find(d, view, Position::new(0), |e| {
        if e.target() != owner {
            return false;
        }
        seen += 1;
        first.get_or_insert(e.position());
        seen > 1
    });
    first.filter(|_| seen == 1)
}

pub(super) const fn key_of(half: EdgeRef) -> LabelBucketKey {
    LabelBucketKey {
        label: half.label(),
        direction: half.direction(),
    }
}

/// The side the other half of an edge is on.
pub(super) const fn opposite(direction: Direction) -> Direction {
    match direction {
        Direction::Outgoing => Direction::Incoming,
        Direction::Incoming => Direction::Outgoing,
        Direction::Undirected => Direction::Undirected,
    }
}

/// An edge's halves from any half and its counterpart.
pub(super) fn order(half: EdgeRef, other: EdgeRef) -> Halves {
    let half_first = match half.direction() {
        Direction::Outgoing => true,
        Direction::Incoming => false,
        Direction::Undirected => half.node() >= other.node(),
    };
    if half_first {
        Halves::new(half, other)
    } else {
        Halves::new(other, half)
    }
}
