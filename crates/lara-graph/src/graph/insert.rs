//! Inserting edges: every half or none, each with its value when the label has a width.

use super::index::{Batch, Between};
use super::{Error, Graph, reserved};
use crate::corrupt::corrupt;
use crate::descriptor::{Descriptor, Slot};
use crate::id::{Direction, EdgeRef, Halves, Label, NodeId, NodeState, Placement, Position, Width};
use crate::label_bucket::{self, Spot};
use crate::label_bucket_list::{self, EntryAt, LabelBucketKey, LabelBucketList, Lookup};
use crate::memory::Memory;
use crate::node::Record;
use crate::page_store::Pages;
use crate::stored::{LabelBucket, Stored, Values};
use crate::targets;
use crate::values::{self, Plane};

impl<M: Memory> Graph<M> {
    /// Adds the edge `from -[label]-> to`, placed by `placement` in both buckets, with `value`
    /// in both halves. Parallel edges keep their relative order either way.
    ///
    /// `value` is exactly the label's width ([`Self::declare_width`]) long, or empty for a label
    /// without one.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for either endpoint, [`Error::Detaching`] or [`Error::Deleted`]
    /// for an endpoint being or already deleted, [`Error::WrongValueLength`],
    /// [`Error::TooManyLabelBuckets`], [`Error::OutOfMemory`] or [`Error::AddressSpaceExhausted`].
    /// On error the graph is unchanged.
    pub fn add_edge(
        &mut self,
        from: NodeId,
        label: Label,
        to: NodeId,
        placement: Placement,
        value: &[u8],
    ) -> Result<Halves, Error> {
        let out = Half::new(from, (label, Direction::Outgoing), to, placement);
        let into = Half::new(to, (label, Direction::Incoming), from, placement);
        let [canonical, other] = self.insert_halves([out, into], value)?;
        Ok(Halves::new(canonical, other))
    }

    /// Adds the undirected edge `a -[label]- b`, placed by `placement` in both buckets, with
    /// `value` at each end. Parallel edges keep their relative order either way.
    ///
    /// # Errors
    ///
    /// As [`Self::add_edge`]. On error the graph is unchanged.
    pub fn add_undirected_edge(
        &mut self,
        a: NodeId,
        label: Label,
        b: NodeId,
        placement: Placement,
        value: &[u8],
    ) -> Result<Halves, Error> {
        let (low, high) = (a.min(b), a.max(b));
        let at_high = Half::new(high, (label, Direction::Undirected), low, placement);
        if low == high {
            let [half] = self.insert_halves([at_high], value)?;
            return Ok(Halves::new(half, half));
        }
        let at_low = Half::new(low, (label, Direction::Undirected), high, placement);
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
        // Each half's spot, and the block batch of an append that completes a block of a hub's
        // bucket: planned now, so that its pages are reserved with the rest.
        let spots: [(Spot, Option<Batch>); N] = core::array::from_fn(|i| {
            let (Some(half), Some(place)) = (halves.get(i), places.get(i)) else {
                corrupt("an admitted half");
            };
            match place.target {
                Target::Present(_, stored) => {
                    let bucket = LabelBucket::resolve(&self.edges.pages(), stored);
                    // The other half's bucket, as it stands, tells whether the two ends have
                    // edges between them already.
                    let partner = (N == 2)
                        .then(|| halves.get(1 - i).zip(places.get(1 - i)))
                        .flatten();
                    let between = partner.map_or(Between::Unknown, |(h, p)| between(h, p));
                    let spot = self.spot_for(&bucket, half.target, half.placement, between);
                    let batch = match spot {
                        Spot::End => self.batch_for(&bucket, half.target),
                        Spot::Hole(_) => None,
                    };
                    (spot, batch)
                }
                Target::Absent(_) => (Spot::End, None),
            }
        });
        let batch_pages: u32 = spots
            .iter()
            .filter_map(|(_, b)| b.as_ref())
            .map(Batch::pages)
            .sum();
        self.edges
            .reserve(pages_per_half(width) * n + batch_pages)?;
        self.label_bucket_lists
            .reserve(label_bucket_list::PAGES_PER_ADD * n)?;
        let mut places = places.into_iter();
        let mut spots = spots.into_iter();
        let mut touched: Option<NodeId> = None;
        Ok(halves.map(|half| {
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
            self.insert_half(half, place, (spot, batch), width.map(|w| (w, value)))
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
    /// each half goes. Reads each node record once, and allocates nothing.
    fn admit<const N: usize>(&self, halves: &[Half; N]) -> Result<[Place; N], Error> {
        let mut records = Records::default();
        let mut places: [Option<Place>; N] = [None; N];
        let mut fresh: [Option<(NodeId, Option<LabelBucketList>)>; N] = [None; N];
        for ((&half, place), new) in halves.iter().zip(&mut places).zip(&mut fresh) {
            let own = records.get(self, half.node)?;
            records.get(self, half.target)?;
            let planned = self.place_in(own, half);
            match planned.target {
                Target::Present(_, stored) if self.stored_positions(stored) == u32::MAX => {
                    return Err(Error::AddressSpaceExhausted);
                }
                Target::Present(..) => {}
                Target::Absent(_) => *new = Some((half.node, own.list)),
            }
            *place = Some(planned);
        }
        check_room(&self.label_bucket_lists.pages(), &fresh)?;
        Ok(places.map(|p| p.unwrap_or_else(|| corrupt("an admitted half"))))
    }

    fn place(&self, half: Half) -> Result<Place, Error> {
        Ok(self.place_in(self.record(half.node)?, half))
    }

    fn place_in(&self, record: Record, half: Half) -> Place {
        let target = match label_bucket_list::lookup(
            &self.label_bucket_lists.pages(),
            record.list,
            half.key,
        ) {
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
        spot: (Spot, Option<Batch>),
        value: Option<(Width, &[u8])>,
    ) -> EdgeRef {
        let at = match value {
            None => self.insert_plain(half, place, spot),
            Some(value) => self.insert_extended(half, place, spot, Some(value)),
        };
        EdgeRef::new(half.node, half.key.label, half.key.direction, at)
    }

    /// A half of a label without values. A plain bucket that stays below a page takes the slot
    /// alone; anything else goes through [`Self::insert_extended`].
    fn insert_plain(
        &mut self,
        half: Half,
        place: Place,
        (destination, batch): (Spot, Option<Batch>),
    ) -> Position {
        let (slot, slots) = match place.target {
            Target::Present(entry, Stored::Plain(slots)) => (Ok(entry), slots),
            Target::Absent(gap) => (Err(gap), Descriptor::EMPTY),
            Target::Present(_, Stored::Extended(_)) => {
                return self.insert_extended(half, place, (destination, batch), None);
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
        (destination, batch): (Spot, Option<Batch>),
        value: Option<(Width, &[u8])>,
    ) -> Position {
        let (slot, bucket) = match place.target {
            Target::Present(entry, stored) => {
                (Ok(entry), LabelBucket::resolve(&self.edges.pages(), stored))
            }
            Target::Absent(gap) => (Err(gap), LabelBucket::plain(Descriptor::EMPTY)),
        };
        let mut bucket = match value {
            Some((width, _)) if bucket.values.is_none() => self.with_values(bucket, width),
            Some(_) | None => bucket,
        };
        let (slots, at) = reserved(label_bucket::insert(
            bucket.slots,
            half.target,
            destination,
            &mut self.edges,
        ));
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
    fn finish(
        &mut self,
        half: Half,
        place: Place,
        slot: Result<EntryAt, EntryAt>,
        bucket: LabelBucket,
    ) {
        match slot {
            Ok(entry) => self.publish(entry, bucket),
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
            }
        }
    }

    /// How many positions a bucket stores, from its list entry.
    fn stored_positions(&self, stored: Stored) -> u32 {
        match stored {
            Stored::Plain(d) => d.stored(),
            Stored::Extended(_) => LabelBucket::resolve(&self.edges.pages(), stored)
                .slots
                .stored(),
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

    /// `bucket` with an empty value plane, never tiny; [`Self::publish`] gives it a header.
    /// Within reserved memory.
    pub(super) fn with_values(&mut self, bucket: LabelBucket, width: Width) -> LabelBucket {
        let slots = reserved(label_bucket::without_tiny(bucket.slots, &mut self.edges));
        LabelBucket {
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
    label_bucket::PAGES_PER_INSERT
        + 1
        + targets::PAGES_PER_CHANGE
        + width.map_or(0, |_| 1 + values::PAGES_PER_WRITE)
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

/// Where a half goes: its node's record and its bucket, or where the bucket belongs.
#[derive(Clone, Copy)]
struct Place {
    /// The node's record when the half was admitted.
    record: Record,
    target: Target,
}

#[derive(Clone, Copy)]
enum Target {
    /// The bucket's list entry, resolved only where its header matters.
    Present(EntryAt, Stored),
    Absent(EntryAt),
}

/// One half to insert: an entry to `target` in `node`'s bucket `key`.
#[derive(Clone, Copy)]
struct Half {
    node: NodeId,
    key: LabelBucketKey,
    target: NodeId,
    placement: Placement,
}

impl Half {
    const fn new(
        node: NodeId,
        (label, direction): (Label, Direction),
        target: NodeId,
        placement: Placement,
    ) -> Self {
        Self {
            node,
            key: LabelBucketKey { label, direction },
            target,
            placement,
        }
    }
}

/// The records of an insert's nodes (an edge touches at most two), each read once; a
/// detaching or deleted node fails the insert.
#[derive(Default)]
struct Records([Option<(NodeId, Record)>; 2]);

impl Records {
    fn get<M: Memory>(&mut self, g: &Graph<M>, v: NodeId) -> Result<Record, Error> {
        if let Some((_, record)) = self.0.iter().flatten().find(|(w, _)| *w == v) {
            return Ok(*record);
        }
        let record = g.record(v)?;
        match record.state {
            NodeState::Live => {}
            NodeState::Detaching => return Err(Error::Detaching(v)),
            NodeState::Deleted => return Err(Error::Deleted(v)),
        }
        let free = self
            .0
            .iter_mut()
            .find(|slot| slot.is_none())
            .unwrap_or_else(|| corrupt("an edge with more than two nodes"));
        *free = Some((v, record));
        Ok(record)
    }
}

/// Fails when new buckets would not fit a node's label bucket list. `fresh` lists each new bucket's
/// node with its current list.
fn check_room<M: Memory>(
    pages: &Pages<'_, M>,
    fresh: &[Option<(NodeId, Option<LabelBucketList>)>],
) -> Result<(), Error> {
    for &(v, list) in fresh.iter().flatten() {
        let adding = fresh.iter().flatten().filter(|(w, _)| *w == v).count();
        if !label_bucket_list::has_room(pages, list, adding) {
            return Err(Error::TooManyLabelBuckets(v));
        }
    }
    Ok(())
}
