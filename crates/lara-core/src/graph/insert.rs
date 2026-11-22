//! Inserting edges: every half or none, each with its value when the label has a width.

use super::index::Batch;
use super::{Error, Graph, reserved};
use crate::bucket;
use crate::corrupt::corrupt;
use crate::descriptor::Descriptor;
use crate::id::{EdgeRef, Halves, Label, Placement, Position, Side, VertexId, VertexState, Width};
use crate::labels::{self, BucketKey, BucketList, EntryAt, Lookup};
use crate::memory::Memory;
use crate::page_store::Pages;
use crate::stored::{Bucket, Stored, Values};
use crate::targets;
use crate::values::{self, Plane};
use crate::vertex::Record;

impl<M: Memory> Graph<M> {
    /// Adds the edge `from -[label]-> to`, placed by `placement` in both buckets, with `value`
    /// in both halves. Parallel edges keep their relative order either way.
    ///
    /// `value` is exactly the label's width ([`Self::declare_width`]) long, or empty for a label
    /// without one.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`] for either endpoint, [`Error::Detaching`] or [`Error::Deleted`]
    /// for an endpoint being or already deleted, [`Error::WrongValueLength`],
    /// [`Error::TooManyBuckets`], [`Error::OutOfMemory`] or [`Error::AddressSpaceExhausted`].
    /// On error the graph is unchanged.
    pub fn insert_directed(
        &mut self,
        from: VertexId,
        label: Label,
        to: VertexId,
        placement: Placement,
        value: &[u8],
    ) -> Result<Halves, Error> {
        let out = Half::new(from, (label, Side::Out), to, placement);
        let into = Half::new(to, (label, Side::In), from, placement);
        let [canonical, other] = self.insert_halves([out, into], value)?;
        Ok(Halves::new(canonical, other))
    }

    /// Adds the undirected edge `a -[label]- b`, placed by `placement` in both buckets, with
    /// `value` at each end. Parallel edges keep their relative order either way.
    ///
    /// # Errors
    ///
    /// As [`Self::insert_directed`]. On error the graph is unchanged.
    pub fn insert_undirected(
        &mut self,
        a: VertexId,
        label: Label,
        b: VertexId,
        placement: Placement,
        value: &[u8],
    ) -> Result<Halves, Error> {
        let (low, high) = (a.min(b), a.max(b));
        let at_high = Half::new(high, (label, Side::Undirected), low, placement);
        if low == high {
            let [half] = self.insert_halves([at_high], value)?;
            return Ok(Halves::new(half, half));
        }
        let at_low = Half::new(low, (label, Side::Undirected), high, placement);
        let [canonical, other] = self.insert_halves([at_high, at_low], value)?;
        Ok(Halves::new(canonical, other))
    }

    /// Inserts every half or none: checks and reserves memory first, then applies them.
    fn insert_halves<const N: usize>(
        &mut self,
        halves: [Half; N],
        value: &[u8],
    ) -> Result<[EdgeRef; N], Error> {
        let label = halves
            .first()
            .map_or_else(|| corrupt("an edge without halves"), |h| h.key.label);
        let width = self.value_width(label, value)?;
        let places = self.admit(&halves)?;
        let n = u32::try_from(N).map_err(|_| Error::AddressSpaceExhausted)?;
        // An append that completes a block of a hub's bucket indexes the whole block: planned
        // now, so that its pages are reserved with the rest.
        let batches: [Option<Batch>; N] = core::array::from_fn(|i| {
            let (half, place) = (halves.get(i)?, places.get(i)?);
            match place.target {
                Target::Present(_, stored) => {
                    let bucket = Bucket::resolve(&self.edges.pages(), stored);
                    self.batch_for(&bucket, half.target)
                }
                Target::Absent(_) => None,
            }
        });
        let batch_pages: u32 = batches.iter().flatten().map(Batch::pages).sum();
        self.edges
            .reserve(pages_per_half(width) * n + batch_pages)?;
        self.lists.reserve(labels::PAGES_PER_ADD * n)?;
        let mut places = places.into_iter();
        let mut batches = batches.into_iter();
        let mut touched: Option<VertexId> = None;
        Ok(halves.map(|half| {
            let planned = places.next().unwrap_or_else(|| corrupt("an admitted half"));
            // An earlier half at the same vertex may have changed its list: look again.
            let place = if touched == Some(half.vertex) {
                self.place(half)
                    .unwrap_or_else(|_| corrupt("an admitted vertex"))
            } else {
                planned
            };
            touched = Some(half.vertex);
            let batch = batches.next().flatten();
            self.insert_half(half, place, width.map(|w| (w, value)), batch)
        }))
    }

    /// The label's width, checking that `value` matches it.
    pub(super) fn value_width(&self, label: Label, value: &[u8]) -> Result<Option<Width>, Error> {
        let width = self.widths.get(label);
        let expected = width.map_or(0, Width::get);
        if usize::from(expected) == value.len() {
            Ok(width)
        } else {
            Err(Error::WrongValueLength {
                label,
                expected,
                actual: value.len(),
            })
        }
    }

    /// Fails as an insert of `halves` would, before anything changes; otherwise returns where
    /// each half goes. Reads each vertex record once, and allocates nothing.
    fn admit<const N: usize>(&self, halves: &[Half; N]) -> Result<[Place; N], Error> {
        let mut records = Records::default();
        let mut places: [Option<Place>; N] = [None; N];
        let mut fresh: [Option<(VertexId, Option<BucketList>)>; N] = [None; N];
        for ((&half, place), new) in halves.iter().zip(&mut places).zip(&mut fresh) {
            let own = records.get(self, half.vertex)?;
            records.get(self, half.target)?;
            let planned = self.place_in(own, half);
            match planned.target {
                Target::Present(_, stored) if self.stored_positions(stored) == u32::MAX => {
                    return Err(Error::AddressSpaceExhausted);
                }
                Target::Present(..) => {}
                Target::Absent(_) => *new = Some((half.vertex, own.list)),
            }
            *place = Some(planned);
        }
        check_room(&self.lists.pages(), &fresh)?;
        Ok(places.map(|p| p.unwrap_or_else(|| corrupt("an admitted half"))))
    }

    fn place(&self, half: Half) -> Result<Place, Error> {
        Ok(self.place_in(self.record(half.vertex)?, half))
    }

    fn place_in(&self, record: Record, half: Half) -> Place {
        let target = match labels::lookup(&self.lists.pages(), record.list, half.key) {
            Lookup::Present { entry, stored } => Target::Present(entry, stored),
            Lookup::Absent { gap } => Target::Absent(gap),
        };
        Place { record, target }
    }

    /// Applies one admitted half within reserved memory. Cannot fail.
    fn insert_half(
        &mut self,
        half: Half,
        place: Place,
        value: Option<(Width, &[u8])>,
        batch: Option<Batch>,
    ) -> EdgeRef {
        let at = match value {
            None => self.insert_plain(half, place, batch),
            Some(value) => self.insert_extended(half, place, Some(value), batch),
        };
        EdgeRef::new(half.vertex, half.key.label, half.key.side, at)
    }

    /// A half of a label without values. A plain bucket that stays below a page takes the slot
    /// alone; anything else goes through [`Self::insert_extended`].
    fn insert_plain(&mut self, half: Half, place: Place, batch: Option<Batch>) -> Position {
        let (slot, slots) = match place.target {
            Target::Present(entry, Stored::Plain(slots)) => (Ok(entry), slots),
            Target::Absent(gap) => (Err(gap), Descriptor::EMPTY),
            Target::Present(_, Stored::Extended(_)) => {
                return self.insert_extended(half, place, None, batch);
            }
        };
        let (next, at) = reserved(bucket::insert(
            slots,
            half.target,
            half.placement,
            &mut self.edges,
        ));
        // A run that just became paged gets its target index.
        let bucket = self.index_inserted(Bucket::plain(next), &slots, (half.target, at), batch);
        self.finish(half, place, slot, bucket);
        at
    }

    /// A half into a bucket with a header (values, an index or both), or one that needs one:
    /// the slot, its value cell, and its index posting. A bucket of a label with values but no
    /// value plane yet (new, or made before the width was declared) gets one.
    fn insert_extended(
        &mut self,
        half: Half,
        place: Place,
        value: Option<(Width, &[u8])>,
        batch: Option<Batch>,
    ) -> Position {
        let (slot, bucket) = match place.target {
            Target::Present(entry, stored) => {
                (Ok(entry), Bucket::resolve(&self.edges.pages(), stored))
            }
            Target::Absent(gap) => (Err(gap), Bucket::plain(Descriptor::EMPTY)),
        };
        let mut bucket = match value {
            Some((width, _)) if bucket.values.is_none() => self.with_values(bucket, width),
            Some(_) | None => bucket,
        };
        let (slots, at) = {
            reserved(bucket::insert(
                bucket.slots,
                half.target,
                half.placement,
                &mut self.edges,
            ))
        };
        let was = core::mem::replace(&mut bucket.slots, slots);
        if let (Some(cells), Some((width, bytes))) = (bucket.values.as_mut(), value) {
            cells.plane = reserved(values::write(
                &mut self.edges,
                cells.plane,
                width,
                at.index(),
                bytes,
            ));
        }
        let bucket = self.index_inserted(bucket, &was, (half.target, at), batch);
        self.finish(half, place, slot, bucket);
        at
    }

    /// Publishes a half's bucket: in place when it existed, else as a new list entry.
    fn finish(&mut self, half: Half, place: Place, slot: Result<EntryAt, EntryAt>, bucket: Bucket) {
        match slot {
            Ok(entry) => self.publish(entry, bucket),
            Err(gap) => {
                let bucket = self.settle(bucket);
                bucket.write_header(&self.edges);
                let list = reserved(labels::add(
                    &mut self.lists,
                    place.record.list,
                    gap,
                    half.key,
                    &bucket.stored(),
                ));
                self.set_record_list(half.vertex, place.record, list);
            }
        }
    }

    /// How many positions a bucket stores, from its list entry.
    fn stored_positions(&self, stored: Stored) -> u32 {
        match stored {
            Stored::Plain(d) => d.stored(),
            Stored::Extended(_) => Bucket::resolve(&self.edges.pages(), stored).slots.stored(),
        }
    }

    /// Publishes `vertex`'s new list, keeping the rest of the record it was admitted with.
    fn set_record_list(&self, vertex: VertexId, record: Record, list: BucketList) {
        self.write_record(
            vertex,
            Record {
                list: Some(list),
                ..record
            },
        );
    }

    /// `bucket` with an empty value plane, never tiny; [`Self::publish`] gives it a header.
    /// Within reserved memory.
    pub(super) fn with_values(&mut self, bucket: Bucket, width: Width) -> Bucket {
        let slots = reserved(bucket::without_tiny(bucket.slots, &mut self.edges));
        Bucket {
            slots,
            values: Some(Values {
                plane: Plane::Empty,
                width,
            }),
            ..bucket
        }
    }
}

/// Pages one half can take from the edge store: its slots, a header, the target index's change,
/// and for a label with a width, a run replacing a tiny bucket and the value's write.
pub(super) fn pages_per_half(width: Option<Width>) -> u32 {
    bucket::PAGES_PER_INSERT
        + 1
        + targets::PAGES_PER_CHANGE
        + width.map_or(0, |_| 1 + values::PAGES_PER_WRITE)
}

/// Where a half goes: its vertex's record and its bucket, or where the bucket belongs.
#[derive(Clone, Copy)]
struct Place {
    /// The vertex's record when the half was admitted.
    record: Record,
    target: Target,
}

#[derive(Clone, Copy)]
enum Target {
    /// The bucket's list entry, resolved only where its header matters.
    Present(EntryAt, Stored),
    Absent(EntryAt),
}

/// One half to insert: an entry to `target` in `vertex`'s bucket `key`.
#[derive(Clone, Copy)]
struct Half {
    vertex: VertexId,
    key: BucketKey,
    target: VertexId,
    placement: Placement,
}

impl Half {
    const fn new(
        vertex: VertexId,
        (label, side): (Label, Side),
        target: VertexId,
        placement: Placement,
    ) -> Self {
        Self {
            vertex,
            key: BucketKey { label, side },
            target,
            placement,
        }
    }
}

/// The records of an insert's vertices (an edge touches at most two), each read once; a
/// detaching or deleted vertex fails the insert.
#[derive(Default)]
struct Records([Option<(VertexId, Record)>; 2]);

impl Records {
    fn get<M: Memory>(&mut self, g: &Graph<M>, v: VertexId) -> Result<Record, Error> {
        if let Some((_, record)) = self.0.iter().flatten().find(|(w, _)| *w == v) {
            return Ok(*record);
        }
        let record = g.record(v)?;
        match record.state {
            VertexState::Live => {}
            VertexState::Detaching => return Err(Error::Detaching(v)),
            VertexState::Deleted => return Err(Error::Deleted(v)),
        }
        let free = self
            .0
            .iter_mut()
            .find(|slot| slot.is_none())
            .unwrap_or_else(|| corrupt("an edge with more than two vertices"));
        *free = Some((v, record));
        Ok(record)
    }
}

/// Fails when new buckets would not fit a vertex's list. `fresh` lists each new bucket's vertex
/// with its current list.
fn check_room<M: Memory>(
    pages: &Pages<'_, M>,
    fresh: &[Option<(VertexId, Option<BucketList>)>],
) -> Result<(), Error> {
    for &(v, list) in fresh.iter().flatten() {
        let adding = fresh.iter().flatten().filter(|(w, _)| *w == v).count();
        if !labels::has_room(pages, list, adding) {
            return Err(Error::TooManyBuckets(v));
        }
    }
    Ok(())
}
