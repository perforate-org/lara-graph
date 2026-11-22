//! Inserting edges: every half or none.

use super::{Error, Graph};
use crate::bucket;
use crate::corrupt::{corrupt, reservation_exceeded};
use crate::descriptor::Descriptor;
use crate::id::{EdgeRef, Halves, Label, Side, VertexId};
use crate::labels::{self, BucketKey, BucketList, Lookup};
use crate::memory::{Memory, WriteError};
use crate::vertex::{Record, State};
use alloc::vec::Vec;

impl<M: Memory> Graph<M> {
    /// Adds the edge `from -[label]-> to`. Parallel edges are kept in insertion order.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`] for either endpoint, [`Error::TooManyBuckets`],
    /// [`Error::OutOfMemory`] or [`Error::AddressSpaceExhausted`]. On error the graph is
    /// unchanged.
    pub fn insert_directed(
        &mut self,
        from: VertexId,
        label: Label,
        to: VertexId,
    ) -> Result<Halves, Error> {
        let out = Half::new(from, label, Side::Out, to);
        let into = Half::new(to, label, Side::In, from);
        let [source, target] = self.insert_halves([out, into])?;
        Ok(Halves::new(source, target))
    }

    /// Adds the undirected edge `a -[label]- b`. Parallel edges are kept in insertion order.
    ///
    /// # Errors
    ///
    /// As [`Self::insert_directed`]. On error the graph is unchanged.
    pub fn insert_undirected(
        &mut self,
        a: VertexId,
        label: Label,
        b: VertexId,
    ) -> Result<Halves, Error> {
        let (low, high) = (a.min(b), a.max(b));
        let at_low = Half::new(low, label, Side::Undirected, high);
        if low == high {
            let [half] = self.insert_halves([at_low])?;
            return Ok(Halves::new(half, half));
        }
        let at_high = Half::new(high, label, Side::Undirected, low);
        let [source, target] = self.insert_halves([at_low, at_high])?;
        Ok(Halves::new(source, target))
    }

    /// Inserts every half or none: checks and reserves memory first, then applies them.
    fn insert_halves<const N: usize>(&mut self, halves: [Half; N]) -> Result<[EdgeRef; N], Error> {
        let places = self.admit(&halves)?;
        let n = u32::try_from(N).map_err(|_| Error::AddressSpaceExhausted)?;
        self.edges.reserve(bucket::PAGES_PER_INSERT * n)?;
        self.lists.reserve(labels::PAGES_PER_ADD * n)?;
        let mut places = places.into_iter();
        let mut touched: Vec<VertexId> = Vec::new();
        Ok(halves.map(|half| {
            let planned = places.next().unwrap_or_else(|| corrupt("an admitted half"));
            // An earlier half at the same vertex may have moved its list: look again.
            let place = if touched.contains(&half.vertex) {
                self.place(half)
                    .unwrap_or_else(|_| corrupt("an admitted vertex"))
            } else {
                planned
            };
            touched.push(half.vertex);
            self.insert_half(half, place)
        }))
    }

    /// Fails as an insert of `halves` would, before anything changes; otherwise returns where
    /// each half goes. Reads each vertex record once.
    fn admit(&self, halves: &[Half]) -> Result<Vec<Place>, Error> {
        let mut records = Records::default();
        let mut places = Vec::with_capacity(halves.len());
        let mut fresh: Vec<(VertexId, Option<BucketList>)> = Vec::new();
        for &half in halves {
            let own = records.get(self, half.vertex)?;
            records.get(self, half.target)?;
            let lookup = labels::lookup(&self.lists.pages(), own.list, half.key);
            match lookup {
                Lookup::Present { descriptor, .. } if descriptor.stored() == u32::MAX => {
                    return Err(Error::AddressSpaceExhausted);
                }
                Lookup::Present { .. } => {}
                Lookup::Absent { .. } => fresh.push((half.vertex, own.list)),
            }
            places.push(Place {
                record: own,
                lookup,
            });
        }
        check_room(&fresh)?;
        Ok(places)
    }

    fn place(&self, half: Half) -> Result<Place, Error> {
        let record = self.record(half.vertex)?;
        Ok(Place {
            record,
            lookup: labels::lookup(&self.lists.pages(), record.list, half.key),
        })
    }

    /// Applies one admitted half within reserved memory. Cannot fail.
    fn insert_half(&mut self, half: Half, place: Place) -> EdgeRef {
        let Half {
            vertex,
            key,
            target,
        } = half;
        let Place { record, lookup } = place;
        let list = record.list;
        let at = match lookup {
            Lookup::Present { index, descriptor } => {
                let (next, at) = reserved(bucket::insert(descriptor, target, &mut self.edges));
                let list = list.unwrap_or_else(|| corrupt("present bucket without a list"));
                labels::set(&self.lists, list, index, &next);
                at
            }
            Lookup::Absent { index } => {
                let (next, at) =
                    reserved(bucket::insert(Descriptor::EMPTY, target, &mut self.edges));
                let list = reserved(labels::add(&mut self.lists, list, index, key, &next));
                let list = Some(list);
                self.write_record(vertex, Record { list, ..record });
                at
            }
        };
        EdgeRef::new(vertex, key.label, key.side, at)
    }
}

/// Where a half goes: its vertex's list and the lookup of its bucket there.
#[derive(Clone, Copy)]
struct Place {
    /// The vertex's record when the half was admitted.
    record: Record,
    lookup: Lookup,
}

/// One half to insert: an entry to `target` in `vertex`'s bucket `key`.
#[derive(Clone, Copy)]
struct Half {
    vertex: VertexId,
    key: BucketKey,
    target: VertexId,
}

impl Half {
    const fn new(vertex: VertexId, label: Label, side: Side, target: VertexId) -> Self {
        Self {
            vertex,
            key: BucketKey { label, side },
            target,
        }
    }
}

/// The result of an update that runs within reserved memory.
/// The records of an insert's vertices, each read once; a detaching vertex fails the insert.
#[derive(Default)]
struct Records(Vec<(VertexId, Record)>);

impl Records {
    fn get<M: Memory>(&mut self, g: &Graph<M>, v: VertexId) -> Result<Record, Error> {
        if let Some(&(_, record)) = self.0.iter().find(|(w, _)| *w == v) {
            return Ok(record);
        }
        let record = g.record(v)?;
        if record.state == State::Detaching {
            return Err(Error::Detaching(v));
        }
        self.0.push((v, record));
        Ok(record)
    }
}

/// Fails when new buckets would take a vertex past [`labels::MAX_BUCKETS`]. `fresh` lists each
/// new bucket's vertex with its current list.
fn check_room(fresh: &[(VertexId, Option<BucketList>)]) -> Result<(), Error> {
    for &(v, list) in fresh {
        let adding = fresh.iter().filter(|(w, _)| *w == v).count();
        if labels::len(list) + adding > labels::MAX_BUCKETS {
            return Err(Error::TooManyBuckets(v));
        }
    }
    Ok(())
}

fn reserved<T>(result: Result<T, WriteError>) -> T {
    result.unwrap_or_else(|_| reservation_exceeded())
}
