//! Admitting an insert: every check that can fail, before anything changes.

use super::{Half, Place, Target};
use crate::error::Error;
use crate::graph::Graph;
use crate::graph::GraphMemories;
use crate::id::NodeId;
use crate::node::label_bucket_list::{self, LabelBucketList, Lookup};
use crate::node::record::Record;
use crate::store::corrupt::corrupt;

impl<S: GraphMemories> Graph<S> {
    /// Fails as an insert of `halves` would, before anything changes; otherwise returns where
    /// each half goes. A cached half skips its node and list read.
    pub(super) fn admit_with<const N: usize>(
        &self,
        halves: &[Half; N],
        cached: &[Option<Place>; N],
    ) -> Result<[Place; N], Error> {
        let mut records: Records = [None; 2];
        let mut places: [Option<Place>; N] = [None; N];
        let mut fresh: [Option<(NodeId, Option<LabelBucketList>)>; N] = [None; N];
        for i in 0..N {
            let half = halves
                .get(i)
                .copied()
                .unwrap_or_else(|| corrupt("an admitted half"));
            let planned = self.admit_half(&mut records, half, cached.get(i).copied().flatten())?;
            if let Target::Absent(_) = planned.target
                && let Some(slot) = fresh.get_mut(i)
            {
                *slot = Some((half.node, planned.record.list));
            }
            if let Some(slot) = places.get_mut(i) {
                *slot = Some(planned);
            }
        }
        self.check_room(&fresh)?;
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
        if half.node.is_proxy() && half.target.is_proxy() {
            return Err(Error::Proxy(half.node));
        }
        if let Some(place) = cached {
            Error::unless_live(place.record.state, half.node)?;
            Ok(place)
        } else {
            self.place_checked(records, half)
        }
    }

    fn place_checked(&self, records: &mut Records, half: Half) -> Result<Place, Error> {
        let own = self.record_once(records, half.node)?;
        self.record_once(records, half.target)?;
        Ok(self.place_in(own, half))
    }

    pub(super) fn place(&self, half: Half) -> Result<Place, Error> {
        Ok(self.place_in(self.record(half.node)?, half))
    }

    fn place_in(&self, record: Record, half: Half) -> Place {
        let target =
            match label_bucket_list::lookup(&self.edges.store().pages(), record.list, half.key) {
                Lookup::Present { entry, stored } => Target::Present(entry, stored),
                Lookup::Absent { gap } => Target::Absent(gap),
            };
        Place { record, target }
    }

    /// `v`'s record, read once per insert; a deleting or deleted node fails the insert.
    fn record_once(&self, records: &mut Records, v: NodeId) -> Result<Record, Error> {
        if let Some((_, record)) = records.iter().flatten().find(|(w, _)| *w == v) {
            return Ok(*record);
        }
        let record = self.record(v)?;
        Error::unless_live(record.state, v)?;
        let free = records
            .iter_mut()
            .find(|slot| slot.is_none())
            .unwrap_or_else(|| corrupt("an edge with more than two nodes"));
        *free = Some((v, record));
        Ok(record)
    }

    /// Fails when new buckets would not fit a node's label bucket list. `fresh` lists each new
    /// bucket's node with its current list.
    fn check_room(&self, fresh: &[Option<(NodeId, Option<LabelBucketList>)>]) -> Result<(), Error> {
        let pages = self.edges.store().pages();
        for &(v, list) in fresh.iter().flatten() {
            let adding = fresh.iter().flatten().filter(|(w, _)| *w == v).count();
            if !label_bucket_list::has_room(&pages, list, adding) {
                return Err(Error::TooManyLabelBuckets(v));
            }
        }
        Ok(())
    }
}

/// The records of an insert's nodes: an edge touches at most two.
type Records = [Option<(NodeId, Record)>; 2];
