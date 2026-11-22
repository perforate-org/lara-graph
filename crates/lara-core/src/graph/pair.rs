//! Finding and removing both halves of an edge.

use super::{Error, Graph, Removal};
use crate::bucket;
use crate::corrupt::corrupt;
use crate::descriptor::Descriptor;
use crate::id::{EdgeRef, Halves, Position, Side, VertexId};
use crate::labels::{self, BucketKey, BucketList, Lookup};
use crate::memory::Memory;
use crate::page_store::Pages;

impl<M: Memory> Graph<M> {
    /// The other half of the edge `half` belongs to, or `None` when `half` is not live.
    ///
    /// When the other bucket is no larger and holds one entry back, that entry is the answer;
    /// otherwise the halves are matched by rank, scanning both buckets up to the edge.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn counterpart(&self, half: EdgeRef) -> Result<Option<EdgeRef>, Error> {
        Ok(self.pair(half)?.map(|pair| pair.other))
    }

    /// Removes the edge `half` belongs to: both its halves. Other edges keep their positions.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn remove_edge(&mut self, half: EdgeRef) -> Result<Removal, Error> {
        let Some(Pair { own, other, theirs }) = self.pair(half)? else {
            return Ok(Removal::Absent);
        };
        self.tombstone(own, half.position());
        if other != half {
            self.tombstone(theirs, other.position());
        }
        Ok(Removal::Removed(order(half, other)))
    }

    /// Both halves of the edge `half` belongs to, with their buckets; `None` when `half` is not
    /// live.
    fn pair(&self, half: EdgeRef) -> Result<Option<Pair>, Error> {
        let Some(own) = self.find_bucket(half.vertex(), key_of(half))? else {
            return Ok(None);
        };
        let view = self.edges.pages();
        let Some(other) = bucket::get(&own.descriptor, half.position(), &view) else {
            return Ok(None);
        };
        let key = BucketKey {
            label: half.label(),
            side: opposite(half.side()),
        };
        let theirs = self.present(other, key);
        let owner = half.vertex();
        let only = (theirs.descriptor.stored() <= own.descriptor.stored())
            .then(|| only_entry_to(&theirs.descriptor, &view, owner))
            .flatten();
        let at = only.unwrap_or_else(|| {
            partner_by_rank(half, other, (&own.descriptor, &theirs.descriptor), &view)
        });
        Ok(Some(Pair {
            own,
            other: EdgeRef::new(other, key.label, key.side, at),
            theirs,
        }))
    }

    /// Tombstones the live half at `at` in `found`. Cannot fail.
    pub(super) fn tombstone(&self, found: Found, at: Position) {
        let (next, _) = bucket::remove(found.descriptor, at, &self.edges)
            .unwrap_or_else(|| corrupt("live half not in its bucket"));
        labels::set(&self.lists, found.list, found.index, &next);
    }

    /// The bucket a live half proves exists.
    pub(super) fn present(&self, v: VertexId, key: BucketKey) -> Found {
        self.find_bucket(v, key)
            .ok()
            .flatten()
            .unwrap_or_else(|| corrupt("an edge half without its bucket"))
    }

    pub(super) fn find_bucket(&self, v: VertexId, key: BucketKey) -> Result<Option<Found>, Error> {
        let list = self.list(v)?;
        Ok(match labels::lookup(&self.lists.pages(), list, key) {
            Lookup::Present { index, descriptor } => Some(Found {
                list: list.unwrap_or_else(|| corrupt("present bucket without a list")),
                index,
                descriptor,
            }),
            Lookup::Absent { .. } => None,
        })
    }
}

/// A bucket found in its vertex's list.
#[derive(Clone, Copy)]
pub(super) struct Found {
    pub(super) list: BucketList,
    pub(super) index: usize,
    pub(super) descriptor: Descriptor,
}

/// Both halves of one edge: the given half's bucket, the other half, and its bucket.
struct Pair {
    own: Found,
    other: EdgeRef,
    theirs: Found,
}

/// The position of the one live entry to `owner` in `d`, or `None` when there are none or
/// several. Stops at the second.
fn only_entry_to<M: Memory>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    owner: VertexId,
) -> Option<Position> {
    let mut first = None;
    let mut seen = 0u32;
    bucket::find(d, view, Position::new(0), |e| {
        if e.target() != owner {
            return false;
        }
        seen += 1;
        first.get_or_insert(e.position());
        seen > 1
    });
    first.filter(|_| seen == 1)
}

/// The other half's position by rank: as many live entries to `owner` precede it in the other
/// bucket as entries to `other` precede `half` in its own.
fn partner_by_rank<M: Memory>(
    half: EdgeRef,
    other: VertexId,
    (own, theirs): (&Descriptor, &Descriptor),
    view: &Pages<'_, M>,
) -> Position {
    let mut rank = 0usize;
    bucket::find(own, view, Position::new(0), |e| {
        let reached = e.position() >= half.position();
        if !reached && e.target() == other {
            rank += 1;
        }
        reached
    });
    let owner = half.vertex();
    let mut seen = 0usize;
    bucket::find(theirs, view, Position::new(0), |e| {
        if e.target() != owner {
            return false;
        }
        seen += 1;
        seen > rank
    })
    .map_or_else(
        || corrupt("an edge half without its counterpart"),
        |e| e.position(),
    )
}

pub(super) const fn key_of(half: EdgeRef) -> BucketKey {
    BucketKey {
        label: half.label(),
        side: half.side(),
    }
}

/// The side the other half of an edge is on.
pub(super) const fn opposite(side: Side) -> Side {
    match side {
        Side::Out => Side::In,
        Side::In => Side::Out,
        Side::Undirected => Side::Undirected,
    }
}

/// An edge's halves from any half and its counterpart.
pub(super) fn order(half: EdgeRef, other: EdgeRef) -> Halves {
    let half_first = match half.side() {
        Side::Out => true,
        Side::In => false,
        Side::Undirected => half.vertex() <= other.vertex(),
    };
    if half_first {
        Halves::new(half, other)
    } else {
        Halves::new(other, half)
    }
}
