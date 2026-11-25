//! Building a graph from an edge list in one pass.

mod halves;

use crate::bucket::LabelBucketKey;
use crate::bucket::stored::Stored;
use crate::error::Error;
use crate::graph::Graph;
use crate::id::{Halves, NodeId, NodeState};
use crate::input::NewEdge;
use crate::node::label_bucket_list::{self, LabelBucketList, Lookup};
use crate::node::record::{RECORD_BYTES, Record};
use crate::store::memory::Memory;
use crate::store::page_store::PageStore;
use alloc::vec::Vec;
use halves::Half;

/// Single-bucket nodes whose label bucket lists are built in one batch.
const SINGLES_PER_BATCH: usize = 1024;

/// Nodes with one label bucket, waiting for their lists to be built in a batch.
type Singles = Vec<(NodeId, LabelBucketKey, Stored)>;

impl<M: Memory> Graph<M> {
    /// Builds the graph from `edges` in one pass. The graph must have its nodes and width
    /// declarations already, and no edges; otherwise this returns [`Error::NotEmpty`].
    ///
    /// Within a bucket, edges get positions in input order, like [`Placement::Append`](crate::input::Placement::Append). The
    /// `placement` of each [`NewEdge`] is ignored because there are no holes to reuse.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], [`Error::Detaching`], [`Error::Deleted`],
    /// [`Error::WrongValueLength`], [`Error::TooManyLabelBuckets`], [`Error::AddressSpaceExhausted`],
    /// [`Error::OutOfMemory`], or [`Error::NotEmpty`].
    pub fn bulk_load(&mut self, edges: &[NewEdge<'_>]) -> Result<Vec<Halves>, Error> {
        self.check_bulk_empty()?;
        for edge in edges {
            self.check_bulk_edge(edge)?;
        }
        if edges.is_empty() {
            return Ok(Vec::new());
        }
        let halves = halves::split(edges);
        let positions = halves::positions(&halves, edges.len());
        let records = self.build_lists(&halves, edges)?;
        self.nodes.write_all(&records);
        Ok(halves::edge_halves(edges, &positions))
    }

    /// Fails with [`Error::NotEmpty`] when any node has edges.
    fn check_bulk_empty(&self) -> Result<(), Error> {
        for v in 0..self.node_count() {
            if self.label_bucket_list(NodeId::new(v))?.is_some() {
                return Err(Error::NotEmpty);
            }
        }
        Ok(())
    }

    /// Checks that both ends of `edge` are live and its value fits its label.
    fn check_bulk_edge(&self, edge: &NewEdge<'_>) -> Result<(), Error> {
        self.check(edge.from())?;
        self.check(edge.to())?;
        for v in [edge.from(), edge.to()] {
            match self.record(v)?.state {
                NodeState::Live => {}
                NodeState::Detaching => return Err(Error::Detaching(v)),
                NodeState::Deleted => return Err(Error::Deleted(v)),
            }
        }
        self.value_width(edge.label(), edge.value())?;
        Ok(())
    }

    /// Builds every node's buckets and label bucket list, returning the encoded node records.
    fn build_lists(
        &mut self,
        halves: &[Half],
        edges: &[NewEdge<'_>],
    ) -> Result<Vec<[u8; RECORD_BYTES]>, Error> {
        let mut records =
            alloc::vec![[0u8; RECORD_BYTES]; usize::try_from(self.node_count()).unwrap_or(0)];
        let mut singles: Singles = Vec::with_capacity(SINGLES_PER_BATCH);
        for node_halves in halves.chunk_by(|a, b| a.node == b.node) {
            let (Some(head), Some(tail)) = (node_halves.first(), node_halves.last()) else {
                continue;
            };
            // Halves are sorted by key within a node, so equal ends mean one bucket.
            if head.key == tail.key {
                let stored = self.build_bucket(node_halves, edges)?;
                singles.push((head.node, head.key, stored));
                if singles.len() == SINGLES_PER_BATCH {
                    flush_singles(&mut self.label_bucket_lists, &mut singles, &mut records)?;
                }
            } else {
                let list = self.build_list(head.node, node_halves, edges)?;
                set_record(&mut records, head.node, list);
            }
        }
        flush_singles(&mut self.label_bucket_lists, &mut singles, &mut records)?;
        Ok(records)
    }

    /// Builds the buckets of a node with several and adds them to its label bucket list.
    fn build_list(
        &mut self,
        node: NodeId,
        node_halves: &[Half],
        edges: &[NewEdge<'_>],
    ) -> Result<Option<LabelBucketList>, Error> {
        let mut list: Option<LabelBucketList> = None;
        for bucket in node_halves.chunk_by(|a, b| a.key == b.key) {
            let Some(key) = bucket.first().map(|half| half.key) else {
                continue;
            };
            let stored = self.build_bucket(bucket, edges)?;
            let pages = self.label_bucket_lists.pages();
            let gap = match label_bucket_list::lookup(&pages, list, key) {
                Lookup::Present { .. } => return Err(Error::TooManyLabelBuckets(node)),
                Lookup::Absent { gap } => gap,
            };
            self.label_bucket_lists
                .reserve(label_bucket_list::PAGES_PER_ADD)?;
            list = Some(label_bucket_list::add(
                &mut self.label_bucket_lists,
                list,
                gap,
                key,
                &stored,
            )?);
        }
        Ok(list)
    }
}

fn flush_singles<M: Memory>(
    store: &mut PageStore<M>,
    singles: &mut Singles,
    records: &mut [[u8; RECORD_BYTES]],
) -> Result<(), Error> {
    if singles.is_empty() {
        return Ok(());
    }
    let entries: Vec<(LabelBucketKey, Stored)> = singles
        .iter()
        .map(|&(_, key, stored)| (key, stored))
        .collect();
    let built = label_bucket_list::one_batch(store, &entries)?;
    for (&(node, _, _), list) in singles.iter().zip(built) {
        set_record(records, node, Some(list));
    }
    singles.clear();
    Ok(())
}

/// Encodes `node`'s record with `list`; a node without a list keeps its empty record.
#[inline]
fn set_record(records: &mut [[u8; RECORD_BYTES]], node: NodeId, list: Option<LabelBucketList>) {
    if let Some(list) = list
        && let Some(record) = records.get_mut(usize::try_from(node.index()).unwrap_or(0))
    {
        *record = Record {
            list: Some(list),
            ..Record::NEW
        }
        .encode();
    }
}

impl<M: Memory> Graph<M> {
    /// Writes the bucket of `halves`, one label's halves at one node, with their edges' values.
    #[expect(clippy::indexing_slicing, reason = "a bucket's group is non-empty")]
    fn build_bucket(&mut self, halves: &[Half], edges: &[NewEdge<'_>]) -> Result<Stored, Error> {
        let width = self.widths.get(halves[0].key.label);
        self.edges.build_bucket(halves, width, |half: &Half| {
            edges
                .get(usize::try_from(half.edge).unwrap_or(usize::MAX))
                .map_or(&[], NewEdge::value)
        })
    }
}
