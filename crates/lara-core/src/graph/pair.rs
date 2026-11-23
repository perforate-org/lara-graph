//! Finding and removing both halves of an edge.

use super::index::indexed;
use super::{Error, Graph, Removal, reserved};
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
        let Some(Pair { own, other, theirs }) = self.pair(half)? else {
            return Ok(Removal::Absent);
        };
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
        self.tombstone(own, half.position());
        if other != half {
            self.tombstone(theirs, other.position());
        }
        Ok(Removal::Removed(RemovedEdge::new(halves, value)))
    }

    /// Both halves of the edge `half` belongs to, with their buckets; `None` when `half` is not
    /// live.
    pub(super) fn pair(&self, half: EdgeRef) -> Result<Option<Pair>, Error> {
        let Some(own) = self.find_label_bucket(half.node(), key_of(half))? else {
            return Ok(None);
        };
        let view = self.edges.pages();
        let Some(other) = label_bucket::get(&own.bucket.slots, half.position(), &view) else {
            return Ok(None);
        };
        let key = LabelBucketKey {
            label: half.label(),
            direction: opposite(half.direction()),
        };
        let theirs = self.present(other, key);
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
        Ok(Some(Pair {
            own,
            other: EdgeRef::new(other, key.label, key.direction, at),
            theirs,
        }))
    }

    /// Tombstones the live half at `at` in `found` and takes it out of the target index. Writes
    /// no value and takes no memory.
    pub(super) fn tombstone(&mut self, found: Found, at: Position) {
        let (next, target) = label_bucket::remove(found.bucket.slots, at, &self.edges)
            .unwrap_or_else(|| corrupt("live half not in its bucket"));
        let bucket = LabelBucket {
            slots: next,
            ..found.bucket
        };
        self.index_removed(&bucket, target, at);
        self.publish(found.entry, bucket);
    }

    /// Publishes a bucket's new state: its header, and its list entry when it is plain or just
    /// changed form. Within reserved memory.
    pub(super) fn publish(&mut self, entry: EntryAt, bucket: LabelBucket) {
        let had_header = bucket.header.is_some();
        let bucket = self.settle(bucket);
        bucket.write_header(&self.edges);
        if !had_header || bucket.header.is_none() {
            label_bucket_list::set(&self.label_bucket_lists, entry, &bucket.stored());
        }
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

/// A label bucket found in its node's label bucket list.
#[derive(Clone, Copy)]
pub(super) struct Found {
    pub(super) entry: EntryAt,
    pub(super) bucket: LabelBucket,
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
