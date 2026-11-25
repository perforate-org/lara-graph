//! Applying an admitted insert: plan each half's spot, reserve once, then write every half.

use super::cache::Cached;
use super::{Half, Place, Target, pages_per_half};
use crate::bucket::descriptor::{Descriptor, Slot};
use crate::bucket::label_bucket::{self, Spot};
use crate::bucket::stored::{LabelBucket, Stored, Values};
use crate::bucket::values;
use crate::graph::bucket::{Batch, Between};
use crate::graph::{Error, Graph, reserved};
use crate::id::{EdgeRef, NodeId, Placement, Position, Width};
use crate::node::label_bucket_list::{self, EntryAt, LabelBucketKey, LabelBucketList};
use crate::node::record::Record;
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;

/// A batch-local cursor into a paged bucket's holes. It is only used for fresh targets, so the
/// holes before `next` have been filled by this run.
pub(super) struct HoleCursor {
    key: Option<(NodeId, LabelBucketKey)>,
    next: Position,
}

impl Default for HoleCursor {
    fn default() -> Self {
        Self {
            key: None,
            next: Position::new(0),
        }
    }
}

/// Each half's spot, and the block batch of an append that completes a block of a hub's bucket.
type Spots<const N: usize> = [(Spot, Option<Batch>); N];

impl<M: Memory> Graph<M> {
    /// The one insert path: a cache supplies any half already admitted, then the batch plans
    /// spots, reserves once and applies each half.
    pub(super) fn insert_halves_with<const N: usize, C: Cached<N>>(
        &mut self,
        halves: [Half; N],
        value: &[u8],
        cached: &C,
        mut cursor: Option<(usize, &mut HoleCursor)>,
    ) -> Result<([EdgeRef; N], [Option<Place>; N]), Error> {
        let label = halves
            .first()
            .map_or_else(|| corrupt("an edge without halves"), |h| h.key.label);
        let width = self.value_width(label, value)?;
        let places = self.admit_with(&halves, cached)?;
        let n = u32::try_from(N).map_err(|_| Error::AddressSpaceExhausted)?;
        // Each half's spot, and the block batch of an append that completes a block of a hub's
        // bucket: planned now, so that its pages are reserved with the rest.
        let spots: Spots<N> =
            core::array::from_fn(|i| self.plan_spot(&halves, &places, i, &mut cursor));
        self.reserve_halves(width, n, &spots)?;
        let mut places = places.into_iter();
        let mut spots = spots.into_iter();
        let mut touched: Option<NodeId> = None;
        let mut updated: [Option<Place>; N] = [None; N];
        let out = core::array::from_fn(|i| {
            let half = halves
                .get(i)
                .copied()
                .unwrap_or_else(|| corrupt("an admitted half"));
            let planned = places.next().unwrap_or_else(|| corrupt("an admitted half"));
            // An earlier half at the same node may have changed its list: look again.
            let place = if touched == Some(half.node) {
                self.place(half)
                    .unwrap_or_else(|_| corrupt("an admitted node"))
            } else {
                planned
            };
            touched = Some(half.node);
            let (spot, batch) = spots.next().unwrap_or_else(|| corrupt("an admitted half"));
            let (edge, place) =
                self.insert_half_place(half, place, (spot, batch), width.map(|w| (w, value)));
            if let Some(slot) = updated.get_mut(i) {
                *slot = place;
            }
            edge
        });
        Ok((out, updated))
    }

    /// Half `i`'s spot and the block batch its append completes, if any.
    #[inline]
    fn plan_spot<const N: usize>(
        &self,
        halves: &[Half; N],
        places: &[Place; N],
        i: usize,
        cursor: &mut Option<(usize, &mut HoleCursor)>,
    ) -> (Spot, Option<Batch>) {
        let (Some(half), Some(place)) = (halves.get(i), places.get(i)) else {
            corrupt("an admitted half");
        };
        let Target::Present(_, stored) = place.target else {
            return (Spot::End, None);
        };
        let bucket = LabelBucket::resolve(&self.edges.pages(), stored);
        // The other half's bucket, as it stands, tells whether the two ends have
        // edges between them already.
        let partner = (N == 2)
            .then(|| halves.get(1 - i).zip(places.get(1 - i)))
            .flatten();
        let between = partner.map_or(Between::Unknown, |(h, p)| between(h, p));
        let spot = match cursor.as_mut() {
            Some((at, cursor)) if *at == i => self.spot_for_cursor(&bucket, *half, between, cursor),
            _ => self.spot_for(&bucket, half.target, half.placement, between),
        };
        let batch = match spot {
            Spot::End => self.batch_for(&bucket, half.target),
            Spot::Hole(_) => None,
        };
        (spot, batch)
    }

    /// Reserves the pages every half and its planned block batches can take.
    #[inline]
    fn reserve_halves<const N: usize>(
        &mut self,
        width: Option<Width>,
        n: u32,
        spots: &Spots<N>,
    ) -> Result<(), Error> {
        let batch_pages: u32 = spots
            .iter()
            .filter_map(|(_, b)| b.as_ref())
            .map(Batch::pages)
            .sum();
        self.edges
            .reserve(pages_per_half(width) * n + batch_pages)?;
        self.label_bucket_lists
            .reserve(label_bucket_list::PAGES_PER_ADD * n)?;
        Ok(())
    }

    fn spot_for_cursor(
        &self,
        bucket: &LabelBucket,
        half: Half,
        between: Between,
        cursor: &mut HoleCursor,
    ) -> Spot {
        if half.placement != Placement::ReuseHoles
            || between != Between::Nothing
            || !matches!(bucket.slots, Descriptor::Paged(_))
        {
            return self.spot_for(bucket, half.target, half.placement, between);
        }
        let key = (half.node, half.key);
        if cursor.key != Some(key) {
            cursor.key = Some(key);
            cursor.next = Position::new(0);
        }
        let Some(position) =
            label_bucket::hole_from(&bucket.slots, &self.edges.pages(), cursor.next)
        else {
            cursor.next = Position::new(bucket.slots.stored());
            return Spot::End;
        };
        cursor.next = Position::new(position.index().saturating_add(1));
        Spot::Hole(position)
    }

    /// Applies one admitted half within reserved memory. Cannot fail.
    /// A half of a label without values. A plain bucket that stays below a page takes the slot
    /// alone; anything else goes through [`Self::insert_extended`].
    /// A half into a bucket with a header (values, an index or both), or one that needs one.
    /// Publishes a half's bucket: in place when it existed, else as a new list entry.
    fn insert_half_place(
        &mut self,
        half: Half,
        place: Place,
        spot: (Spot, Option<Batch>),
        value: Option<(Width, &[u8])>,
    ) -> (EdgeRef, Option<Place>) {
        let (at, updated) = match value {
            None => self.insert_plain_place(half, place, spot),
            Some(value) => self.insert_extended_place(half, place, spot, Some(value)),
        };
        (
            EdgeRef::new(half.node, half.key.label, half.key.direction, at),
            updated,
        )
    }

    /// A half of a label without values. A plain bucket that stays below a page takes the slot
    /// alone; anything else goes through [`Self::insert_extended`].
    fn insert_plain_place(
        &mut self,
        half: Half,
        place: Place,
        (destination, batch): (Spot, Option<Batch>),
    ) -> (Position, Option<Place>) {
        let (slot, slots) = match place.target {
            // A run with values beside its slots is of a label with a width, never here.
            Target::Present(entry, Stored::Plain(slots)) => (Ok(entry), slots),
            Target::Absent(gap) => (Err(gap), Descriptor::EMPTY),
            Target::Present(_, Stored::Extended(_)) => {
                return self.insert_extended_place(half, place, (destination, batch), None);
            }
        };
        let (next, at) = reserved(label_bucket::insert(
            slots,
            half.target,
            destination,
            &mut self.edges,
        ));
        // A run that just became paged gets its target index.
        let bucket =
            self.index_inserted(LabelBucket::plain(next), &slots, (half.target, at), batch);
        let updated = self.finish_place(half, place, slot, bucket);
        (at, updated)
    }

    /// A half into a bucket with a header (values, an index or both), or one that needs one:
    /// the slot, its value cell, and its index posting. A bucket of a label with values but no
    /// value plane yet (new, or made before the width was declared) gets one.
    fn insert_extended_place(
        &mut self,
        half: Half,
        place: Place,
        (destination, batch): (Spot, Option<Batch>),
        value: Option<(Width, &[u8])>,
    ) -> (Position, Option<Place>) {
        let (slot, bucket) = match place.target {
            Target::Present(entry, stored) => {
                (Ok(entry), LabelBucket::resolve(&self.edges.pages(), stored))
            }
            Target::Absent(gap) => (Err(gap), LabelBucket::plain(Descriptor::EMPTY)),
        };
        let mut bucket = match value {
            Some((width, _)) if bucket.cells().is_none() => self.with_values(bucket, width),
            Some(_) if destination == Spot::End => self.room_to_append(bucket),
            Some(_) | None => bucket,
        };
        let (slots, at) = reserved(label_bucket::insert(
            bucket.slots,
            half.target,
            destination,
            &mut self.edges,
        ));
        let was = core::mem::replace(&mut bucket.slots, slots);
        if let Some((width, bytes)) = value
            && let Some(cells) = bucket.cells()
        {
            let written = reserved(values::write(
                &mut self.edges,
                cells.plane,
                width,
                at.index(),
                bytes,
            ));
            bucket.set_plane(written);
        }
        let bucket = self.index_inserted(bucket, &was, (half.target, at), batch);
        let updated = self.finish_place(half, place, slot, bucket);
        (at, updated)
    }

    /// `bucket` with room to append: a full page-sized run with values beside its slots moves them
    /// into a value plane, so that it can grow on into pages. Within reserved memory.
    fn room_to_append(&mut self, bucket: LabelBucket) -> LabelBucket {
        let Descriptor::Run(run) = bucket.slots else {
            return bucket;
        };
        if !label_bucket::full_beside(&bucket.slots) {
            return bucket;
        }
        let width = run
            .values
            .unwrap_or_else(|| corrupt("a full run without values beside its slots"));
        let (slots, plane) = reserved(label_bucket::values_out(run, &mut self.edges));
        LabelBucket {
            slots,
            values: Some(Values { plane, width }),
            ..bucket
        }
    }

    /// Publishes a half's bucket: in place when it existed, else as a new list entry.
    fn finish_place(
        &mut self,
        half: Half,
        place: Place,
        slot: Result<EntryAt, EntryAt>,
        bucket: LabelBucket,
    ) -> Option<Place> {
        match slot {
            Ok(entry) => {
                let bucket = self.publish(entry, bucket);
                Some(Place {
                    record: place.record,
                    target: Target::Present(entry, bucket.stored()),
                })
            }
            Err(gap) => {
                let bucket = self.settle(bucket);
                bucket.write_header(&self.edges);
                let list = reserved(label_bucket_list::add(
                    &mut self.label_bucket_lists,
                    place.record.list,
                    gap,
                    half.key,
                    &bucket.stored(),
                ));
                self.set_record_label_bucket_list(half.node, place.record, list);
                None
            }
        }
    }

    /// Publishes `node`'s new label bucket list, keeping the rest of the record it was admitted
    /// with.
    fn set_record_label_bucket_list(&self, node: NodeId, record: Record, list: LabelBucketList) {
        self.write_record(
            node,
            Record {
                list: Some(list),
                ..record
            },
        );
    }
}

/// Whether `half`'s bucket, at `place`, shows the two ends without edges between them: absent, or
/// tiny without an entry to `half`'s target. Larger buckets are not read for this.
fn between(half: &Half, place: &Place) -> Between {
    match place.target {
        Target::Absent(_) => Between::Nothing,
        Target::Present(_, Stored::Plain(Descriptor::Tiny(tiny)))
            if !tiny.slots().contains(&Slot::Live(half.target)) =>
        {
            Between::Nothing
        }
        Target::Present(..) => Between::Unknown,
    }
}
