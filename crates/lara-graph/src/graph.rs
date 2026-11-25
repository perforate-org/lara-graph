//! The graph: label buckets of edge halves per node and direction, over seven memories.
//!
//! The nodes memory is laid out in `node::store`, the node labels memory in `node::labels`.

use crate::error::Error;
use crate::id::{NodeId, NodeState};
use crate::node::label_bucket_list::LabelBucketList;
use crate::node::labels::LabelStore;
use crate::node::record::{RECORD_BYTES, Record};
use crate::node::store::{NodeStore, RecordLines};
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use crate::store::page_store::PageStore;

/// The layout version of every memory's header.
const VERSION: u8 = 1;

/// A multigraph with dense node ids and labeled edges, directed or undirected.
///
/// Every edge is stored at both ends (see [`EdgeRef`](crate::id::EdgeRef)), so traversal runs either way. Each
/// `(node, label, direction)` has a bucket addressed by [`Position`](crate::id::Position). Positions are stable:
/// removing an edge leaves a hole that a later insert may fill, and only [`Graph::compact`] moves
/// edges.
///
/// The two halves of an edge are matched by rank: the k-th live entry to `v` in `u`'s out
/// bucket pairs with the k-th live entry to `u` in `v`'s in bucket (undirected alike). Inserts
/// and compaction keep entries to one node in insertion order, so the ranks stay aligned
/// without links between the halves.
///
/// Reading one bucket costs a node read, a label bucket list read and one data read (a few more
/// past 1024 edges). An insert or removal copies at most one run of under 1024 edges or one page of
/// index.
pub struct Graph<M: Memory> {
    nodes: NodeStore<M>,
    labels: LabelStore<M>,
    edges: EdgeStore<M>,
    label_bucket_lists: PageStore<M>,
    /// Mirrors the label width table, which the label bucket list store holds.
    widths: Widths,
}

impl<M: Memory> Graph<M> {
    /// A graph over formatted memories, with its header mirrors loaded.
    fn assemble(
        (nodes, labels): (NodeStore<M>, LabelStore<M>),
        edges: PageStore<M>,
        label_bucket_lists: PageStore<M>,
    ) -> Self {
        let widths = Widths::load(nodes.widths_root(), &label_bucket_lists);
        Self {
            nodes,
            labels,
            edges: EdgeStore::new(edges),
            label_bucket_lists,
            widths,
        }
    }

    /// The edge store as reads see it.
    fn edge_view(&self) -> EdgeView<'_, M> {
        EdgeView::new(&self.edges, &self.widths)
    }

    fn check(&self, v: NodeId) -> Result<(), Error> {
        if self.nodes.contains(v) {
            Ok(())
        } else {
            Err(Error::UnknownNode(v))
        }
    }

    fn record(&self, v: NodeId) -> Result<Record, Error> {
        self.check(v)?;
        Ok(Record::decode(self.nodes.record_bytes(v)))
    }

    /// `v`'s state alone, without decoding its record's list.
    fn state(&self, v: NodeId) -> Result<NodeState, Error> {
        self.check(v)?;
        Ok(Record::state_of(self.nodes.record_bytes(v)))
    }

    /// A known node's record, undecoded: a whole record or the state alone decodes from it.
    fn record_bytes(&self, v: NodeId) -> [u8; RECORD_BYTES] {
        self.nodes.record_bytes(v)
    }

    /// [`Self::record_bytes`] through a batch's line of records.
    #[expect(clippy::inline_always, reason = "see RecordLines::bytes_of")]
    #[inline(always)]
    fn line_record_bytes(&self, lines: &mut RecordLines, v: NodeId) -> [u8; RECORD_BYTES] {
        lines.bytes_of(&self.nodes, v)
    }

    fn label_bucket_list(&self, v: NodeId) -> Result<Option<LabelBucketList>, Error> {
        Ok(self.record(v)?.list)
    }

    /// Rewrites an existing node's record. Cannot fail.
    fn write_record(&self, v: NodeId, record: Record) {
        self.nodes.write(v, record);
    }

    /// Publishes `v`'s new label bucket list in `record`, the record `v` has.
    fn set_label_bucket_list(&self, v: NodeId, record: Record, list: Option<LabelBucketList>) {
        self.write_record(v, Record { list, ..record });
    }
}

mod lifecycle;
mod locate;
mod memories;
mod node_labels;
mod read;
mod write;

pub use memories::{Memories, MemoryRole};
pub use read::{ReadBatch, TargetSet};
pub use write::DetachingSet;

#[cfg(test)]
use crate::bucket::pages_per_half;
use crate::edge::{EdgeStore, EdgeView, Widths};

#[cfg(test)]
mod tests;
