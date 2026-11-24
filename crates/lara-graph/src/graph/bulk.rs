//! Building a graph from an edge list in one pass.

mod bucket;
mod halves;

use super::{Error, Graph, LabelBucketKey, NewEdge};
use crate::id::{Halves, NodeId, NodeState};
use crate::label_bucket_list::{self, LabelBucketList, Lookup};
use crate::memory::Memory;
use crate::node::Record;
use crate::page_store::PageStore;
use crate::stored::Stored;
use alloc::vec::Vec;
use halves::Half;

/// Node records written per line while bulk load flushes them.
const RECORDS_PER_LINE_USIZE: usize = 8;

/// Single-bucket nodes whose label bucket lists are built in one batch.
const SINGLES_PER_BATCH: usize = 1024;

/// Nodes with one label bucket, waiting for their lists to be built in a batch.
type Singles = Vec<(NodeId, LabelBucketKey, Stored)>;

impl<M: Memory> Graph<M> {
    /// Builds the graph from `edges` in one pass. The graph must have its nodes and width
    /// declarations already, and no edges; otherwise this returns [`Error::NotEmpty`].
    ///
    /// Within a bucket, edges get positions in input order, like [`Placement::Append`]. The
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
        self.write_records(&records);
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
    ) -> Result<Vec<[u8; 8]>, Error> {
        let mut records = alloc::vec![[0u8; 8]; usize::try_from(self.node_count()).unwrap_or(0)];
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

    #[expect(clippy::indexing_slicing, reason = "each line covers records in range")]
    fn write_records(&self, records: &[[u8; 8]]) {
        for base in (0..records.len()).step_by(RECORDS_PER_LINE_USIZE) {
            let end = (base + RECORDS_PER_LINE_USIZE).min(records.len());
            let mut line = [0u8; 64];
            for (offset, record) in records[base..end].iter().enumerate() {
                let start = offset * 8;
                line[start..start + 8].copy_from_slice(record);
            }
            self.region().overwrite(
                super::record_at_index(u32::try_from(base).unwrap_or(u32::MAX)),
                &line[..(end - base) * 8],
            );
        }
    }
}

fn flush_singles<M: Memory>(
    store: &mut PageStore<M>,
    singles: &mut Singles,
    records: &mut [[u8; 8]],
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
fn set_record(records: &mut [[u8; 8]], node: NodeId, list: Option<LabelBucketList>) {
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
