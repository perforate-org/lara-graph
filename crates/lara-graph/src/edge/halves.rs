//! The two halves of an edge: pairing one with the other by rank, and taking a half out of its
//! bucket.

use super::values::value_at;
use crate::bucket::LabelBucket;
use crate::bucket::LabelBucketKey;
use crate::bucket::Schema;
use crate::bucket::descriptor::Descriptor;
use crate::bucket::slots;
use crate::edge::{EdgeStore, EdgeView};
use crate::id::{EdgeRef, Halves, NodeId, Position, ValueVersion};
use crate::outcome::{BucketFill, RemovedEdge};
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use crate::store::page_store::Pages;

impl<M: Memory> EdgeStore<M> {
    /// Takes both halves of an edge out of their buckets, `half` from `own` and `other` from
    /// `theirs`, and reports the edge with its value as the canonical half held it (empty for a
    /// label without a width). The buckets are left as they are after, not yet published; an
    /// undirected self-loop's one half is in `own`, and `theirs` stays as it was. Takes no memory.
    pub(crate) fn remove_pair(
        &mut self,
        (own, half): (&mut LabelBucket, EdgeRef),
        (theirs, other): (&mut LabelBucket, EdgeRef),
        schema: Option<Schema>,
    ) -> RemovedEdge {
        let halves = Halves::of(half, other);
        let canonical = if halves.canonical() == half {
            &*own
        } else {
            &*theirs
        };
        let value = value_at(
            &self.store.pages(),
            schema.map(|s| s.width),
            canonical,
            halves.canonical().position(),
        );
        let version = schema.map_or(ValueVersion::FIRST, |s| {
            canonical
                .cells()
                .map_or(s.version, |c| c.version_given(s.version))
        });
        self.remove_half(own, half.position());
        if other != half {
            self.remove_half(theirs, other.position());
        }
        let (own, theirs) = (
            fill(own),
            if other == half {
                fill(own)
            } else {
                fill(theirs)
            },
        );
        let fills = if halves.canonical() == half {
            [own, theirs]
        } else {
            [theirs, own]
        };
        RemovedEdge::new(halves, (value, version), fills)
    }

    /// Applies one tombstone to a bucket without publishing it yet.
    fn remove_half(&mut self, bucket: &mut LabelBucket, at: Position) {
        let (next, target) = slots::remove(bucket.slots, at, &self.store)
            .unwrap_or_else(|| corrupt("live half not in its bucket"));
        bucket.slots = next;
        self.index_removed(bucket, target, at);
    }
}

/// How full `bucket` is.
fn fill(bucket: &LabelBucket) -> BucketFill {
    let live = bucket.slots.live();
    BucketFill::new(live, bucket.slots.stored() - live)
}

impl<M: Memory> EdgeView<'_, M> {
    /// The other half of `half`, an edge to `other`, whose own bucket is `own`; `theirs` is
    /// `other`'s bucket `key`, which holds it.
    pub(crate) fn other_half(
        &self,
        half: EdgeRef,
        own: &LabelBucket,
        (other, key): (NodeId, LabelBucketKey),
        theirs: &LabelBucket,
    ) -> EdgeRef {
        let view = self.pages();
        let owner = half.node();
        // A small other bucket with one entry back answers alone; else match by rank.
        let only = (theirs.indexed().is_none() && theirs.slots.stored() <= own.slots.stored())
            .then(|| only_entry_to(&theirs.slots, &view, owner))
            .flatten();
        let at = only.unwrap_or_else(|| {
            let rank = self.edges.rank_to(own, other, half.position());
            self.edges
                .select_to(theirs, owner, rank)
                .unwrap_or_else(|| corrupt("an edge half without its counterpart"))
        });
        EdgeRef::new(other, key.label, key.direction, at)
    }
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
    slots::find(d, view, Position::new(0), |e| {
        if e.target() != owner {
            return false;
        }
        seen += 1;
        first.get_or_insert(e.position());
        seen > 1
    });
    first.filter(|_| seen == 1)
}

impl<M: Memory> EdgeView<'_, M> {
    /// The other half of `half`, `v`'s first live edge in its bucket, in `theirs`, `other`'s
    /// opposite bucket: its first live entry to `v`, scanned from `from` when the bucket has no
    /// index.
    pub(crate) fn first_half_to(
        &self,
        v: NodeId,
        half: EdgeRef,
        (other, from): (NodeId, Position),
        theirs: &LabelBucket,
    ) -> EdgeRef {
        let key = LabelBucketKey::of(half).opposite();
        // A paged bucket finds it through its target index; a smaller one is scanned.
        let at = if theirs.indexed().is_some() {
            self.edges.select_to(theirs, v, 0)
        } else {
            slots::find(&theirs.slots, &self.pages(), from, |e| e.target() == v)
                .map(|e| e.position())
        }
        .unwrap_or_else(|| corrupt("an edge half without its counterpart"));
        EdgeRef::new(other, key.label, key.direction, at)
    }
}
