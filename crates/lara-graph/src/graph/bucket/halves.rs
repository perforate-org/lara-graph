//! The two halves of an edge: pairing one with the other by rank, and taking a half out of its
//! bucket.

use crate::bucket::descriptor::Descriptor;
use crate::bucket::label_bucket;
use crate::bucket::stored::LabelBucket;
use crate::graph::bucket::Found;
use crate::graph::{Error, Graph};
use crate::id::{EdgeRef, NodeId, Position};
use crate::node::label_bucket_list::LabelBucketKey;
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use crate::store::page_store::Pages;

impl<M: Memory> Graph<M> {
    /// Both halves of the edge `half` belongs to, with their buckets; `None` when `half` is not
    /// live.
    pub(in crate::graph) fn pair(&self, half: EdgeRef) -> Result<Option<Pair>, Error> {
        let Some(own) = self.find_label_bucket(half.node(), LabelBucketKey::of(half))? else {
            return Ok(None);
        };
        Ok(self.pair_from(half, own))
    }

    /// [`pair`] with `half`'s own bucket already found.
    pub(in crate::graph) fn pair_from(&self, half: EdgeRef, own: Found) -> Option<Pair> {
        let view = self.edges.pages();
        let other = label_bucket::get(&own.bucket.slots, half.position(), &view)?;
        let key = LabelBucketKey::of(half).opposite();
        let theirs = self.present(other, key);
        Some(self.pair_with_theirs(half, own, other, key, theirs))
    }

    /// [`pair_from`] with the target and its bucket already found.
    pub(in crate::graph) fn pair_with_theirs(
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
        let only = (theirs.bucket.indexed().is_none()
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
    pub(in crate::graph) fn tombstone(&mut self, found: Found, at: Position) -> LabelBucket {
        let (bucket, _) = self.remove_half(found, at);
        self.publish(found.entry, bucket)
    }

    /// Applies one tombstone to a bucket without publishing it yet.
    pub(in crate::graph) fn remove_half(
        &mut self,
        found: Found,
        at: Position,
    ) -> (LabelBucket, NodeId) {
        let (next, target) = label_bucket::remove(found.bucket.slots, at, &self.edges)
            .unwrap_or_else(|| corrupt("live half not in its bucket"));
        let bucket = LabelBucket {
            slots: next,
            ..found.bucket
        };
        self.index_removed(&bucket, target, at);
        (bucket, target)
    }
}

/// Both halves of one edge: the given half's bucket, the other half, and its bucket.
pub(in crate::graph) struct Pair {
    pub(in crate::graph) own: Found,
    pub(in crate::graph) other: EdgeRef,
    pub(in crate::graph) theirs: Found,
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
