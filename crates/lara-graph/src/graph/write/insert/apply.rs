//! Applying an admitted insert: plan each half's spot, reserve once, then write every half.

use super::{Half, Place, Target};
use crate::bucket::LabelBucket;
use crate::bucket::Schema;
use crate::bucket::reserved;
use crate::bucket::slots::Spot;
use crate::edge::EdgeView;
use crate::edge::Writes;
use crate::edge::{Batch, Between, HoleCursor};
use crate::error::Error;
use crate::graph::Graph;
use crate::graph::GraphMemories;
use crate::id::{EdgeRef, NodeId, Position, ValueVersion, Width};
use crate::node::label_bucket_list::{self, EntryAt};
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;

/// Each half's spot, and the block batch of an append that completes a block of a hub's bucket.
type Spots<const N: usize> = [(Spot, Option<Batch>); N];

impl<S: GraphMemories> Graph<S> {
    /// The one insert path: a cache supplies any half already admitted, then the batch plans
    /// spots, reserves once and applies each half.
    pub(super) fn insert_halves_with<const N: usize>(
        &mut self,
        halves: [Half; N],
        value: &[u8],
        cached: &[Option<Place>; N],
        mut cursor: Option<(usize, &mut HoleCursor)>,
    ) -> Result<([EdgeRef; N], [Option<Place>; N]), Error> {
        let label = halves
            .first()
            .map_or_else(|| corrupt("an edge without halves"), |h| h.key.label);
        let schema = self.value_schema(label, value)?;
        let width = schema.map(|s| s.width);
        let places = self.admit_with(&halves, cached)?;
        if let Some(schema) = schema
            && schema.version != ValueVersion::FIRST
        {
            self.ensure_places_current(&halves, &places, schema)?;
        }
        let n = u32::try_from(N).map_err(|_| Error::AddressSpaceExhausted)?;
        // Each half's spot, and the block batch of an append that completes a block of a hub's
        // bucket: planned now, so that its pages are reserved with the rest.
        let planned: [Result<(Spot, Option<Batch>), Error>; N] =
            core::array::from_fn(|i| plan(&self.edge_view(), &halves, &places, i, &mut cursor));
        if let Some(Err(error)) = planned.iter().find(|p| p.is_err()) {
            return Err(*error);
        }
        let spots: Spots<N> = planned.map(|p| p.unwrap_or_else(|_| corrupt("a planned half")));
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
                self.insert_half_place(half, place, (spot, batch), schema.map(|s| (s, value)));
            if let Some(slot) = updated.get_mut(i) {
                *slot = place;
            }
            edge
        });
        Ok((out, updated))
    }

    /// Fails with [`Error::StaleValues`] when a half's bucket holds values in an older version
    /// than `schema`'s: a label never changed has none, so only changed labels pay the header read.
    /// Kept out of line, so that the insert path it sits on compiles as before.
    #[cold]
    #[inline(never)]
    fn ensure_places_current<const N: usize>(
        &self,
        halves: &[Half; N],
        places: &[Place; N],
        schema: Schema,
    ) -> Result<(), Error> {
        let view = self.edge_view().pages();
        for (half, place) in halves.iter().zip(places) {
            if let Target::Present(_, stored) = place.target {
                let bucket = LabelBucket::resolve(&view, stored);
                let at = EdgeRef::of(
                    half.node,
                    half.key.label,
                    half.key.direction,
                    Position::new(0),
                );
                Self::ensure_current(at, &bucket, schema)?;
            }
        }
        Ok(())
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
        self.edges.reserve(Writes::Halves {
            width,
            count: n,
            batches: batch_pages,
        })?;
        self.label_bucket_lists
            .reserve(label_bucket_list::PAGES_PER_ADD * n)?;
        Ok(())
    }

    /// Applies one admitted half within reserved memory. Cannot fail.
    fn insert_half_place(
        &mut self,
        half: Half,
        place: Place,
        spot: (Spot, Option<Batch>),
        value: Option<(Schema, &[u8])>,
    ) -> (EdgeRef, Option<Place>) {
        let (entry, stored) = match place.target {
            Target::Present(entry, stored) => (Ok(entry), Some(stored)),
            Target::Absent(gap) => (Err(gap), None),
        };
        let (at, bucket) = self.edges.insert(stored, half.target, spot, value);
        let updated = self.finish_place(half, place, entry, bucket);
        (
            EdgeRef::of(half.node, half.key.label, half.key.direction, at),
            updated,
        )
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
                let bucket = self.edges.seal(bucket);
                let list = reserved(label_bucket_list::add(
                    &mut self.label_bucket_lists,
                    place.record.list,
                    gap,
                    half.key,
                    &bucket.stored(),
                ));
                self.set_label_bucket_list(half.node, place.record, Some(list));
                None
            }
        }
    }
}

/// Half `i`'s spot and the block batch its append completes, if any; the cursor, when it is
/// half `i`'s, finds holes for a run of fresh targets.
#[inline]
fn plan<M: Memory, D: Memory, const N: usize>(
    edges: &EdgeView<'_, M, D>,
    halves: &[Half; N],
    places: &[Place; N],
    i: usize,
    cursor: &mut Option<(usize, &mut HoleCursor)>,
) -> Result<(Spot, Option<Batch>), Error> {
    let (Some(half), Some(place)) = (halves.get(i), places.get(i)) else {
        corrupt("an admitted half");
    };
    let Target::Present(_, stored) = place.target else {
        return Ok((Spot::End, None));
    };
    // The other half's bucket, as it stands, tells whether the two ends have edges between them
    // already.
    let partner = (N == 2)
        .then(|| halves.get(1 - i).zip(places.get(1 - i)))
        .flatten();
    let between = partner.map_or(Between::Unknown, |(h, p)| {
        Between::of(
            match p.target {
                Target::Present(_, stored) => Some(stored),
                Target::Absent(_) => None,
            },
            h.target,
        )
    });
    let cursor = match cursor.as_mut() {
        Some((at, cursor)) if *at == i => Some(&mut **cursor),
        _ => None,
    };
    edges.plan(
        stored,
        (half.node, half.key),
        (half.target, half.placement),
        between,
        cursor,
    )
}
