//! Admitting an insert: every check that can fail, before anything changes.

use super::cache::Cached;
use super::{Half, Place, Target};
use crate::bucket::stored::{LabelBucket, Stored};
use crate::graph::{Error, Graph};
use crate::id::{NodeId, NodeState};
use crate::node::label_bucket_list::{self, LabelBucketList, Lookup};
use crate::node::record::Record;
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use crate::store::page_store::Pages;

impl<M: Memory> Graph<M> {
    /// Fails as an insert of `halves` would, before anything changes; otherwise returns where
    /// each half goes. A cached half skips its node and list read.
    pub(super) fn admit_with<const N: usize, C: Cached<N>>(
        &self,
        halves: &[Half; N],
        cached: &C,
    ) -> Result<[Place; N], Error> {
        let mut records = Records::default();
        let mut places: [Option<Place>; N] = [None; N];
        let mut fresh: [Option<(NodeId, Option<LabelBucketList>)>; N] = [None; N];
        for i in 0..N {
            let half = halves
                .get(i)
                .copied()
                .unwrap_or_else(|| corrupt("an admitted half"));
            let planned = self.admit_half(&mut records, half, cached.get(i))?;
            if let Target::Absent(_) = planned.target
                && let Some(slot) = fresh.get_mut(i)
            {
                *slot = Some((half.node, planned.record.list));
            }
            if let Some(slot) = places.get_mut(i) {
                *slot = Some(planned);
            }
        }
        check_room(&self.label_bucket_lists.pages(), &fresh)?;
        Ok(places.map(|p| p.unwrap_or_else(|| corrupt("an admitted half"))))
    }

    /// Where `half` goes, from its cached place or its node's record and list; fails for a node
    /// that is not live or a bucket without a free position.
    #[inline]
    fn admit_half(
        &self,
        records: &mut Records,
        half: Half,
        cached: Option<Place>,
    ) -> Result<Place, Error> {
        let planned = if let Some(place) = cached {
            live(place.record.state, half.node)?;
            place
        } else {
            self.place_checked(records, half)?
        };
        match planned.target {
            Target::Present(_, stored) if self.stored_positions(stored) == u32::MAX => {
                Err(Error::AddressSpaceExhausted)
            }
            Target::Present(..) | Target::Absent(_) => Ok(planned),
        }
    }

    fn place_checked(&self, records: &mut Records, half: Half) -> Result<Place, Error> {
        let read = |v| self.record(v);
        let own = records.get(half.node, read)?;
        records.get(half.target, read)?;
        Ok(self.place_in(own, half))
    }

    pub(super) fn place(&self, half: Half) -> Result<Place, Error> {
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

    /// How many positions a bucket stores, from its list entry.
    fn stored_positions(&self, stored: Stored) -> u32 {
        match stored {
            Stored::Plain(d) => d.stored(),
            Stored::Extended(_) => LabelBucket::resolve(&self.edges.pages(), stored)
                .slots
                .stored(),
        }
    }
}

/// Fails for a node that is being or already deleted.
fn live(state: NodeState, v: NodeId) -> Result<(), Error> {
    match state {
        NodeState::Live => Ok(()),
        NodeState::Detaching => Err(Error::Detaching(v)),
        NodeState::Deleted => Err(Error::Deleted(v)),
    }
}

/// The records of an insert's nodes (an edge touches at most two), each read once through the
/// reader it is given; a detaching or deleted node fails the insert.
#[derive(Default)]
struct Records([Option<(NodeId, Record)>; 2]);

impl Records {
    fn get(
        &mut self,
        v: NodeId,
        read: impl FnOnce(NodeId) -> Result<Record, Error>,
    ) -> Result<Record, Error> {
        if let Some((_, record)) = self.0.iter().flatten().find(|(w, _)| *w == v) {
            return Ok(*record);
        }
        let record = read(v)?;
        live(record.state, v)?;
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
