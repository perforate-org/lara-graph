//! The graph: label buckets of edge halves per node and direction, over five memories.
//!
//! The nodes memory is laid out in `node::store`, the node labels memory in `node::labels`.

use crate::error::Error;
use crate::id::{NodeId, NodeState};
use crate::node::label_bucket_list::LabelBucketList;
use crate::node::labels::LabelStore;
use crate::node::record::{RECORD_BYTES, Record};
use crate::node::store::{NodeStore, RecordLines};
use crate::store::corrupt::corrupt;
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
///
/// `S` is the memories' type: one [`Memory`](crate::Memory) type for all five, or a
/// [`Memories`] naming one per role.
///
/// # The everyday operations
///
/// - Opening: [`Graph::init`], or [`Graph::format`] and [`Graph::load`] apart.
/// - Nodes: [`Graph::add_node`], [`Graph::add_nodes`], [`Graph::node_state`],
///   [`Graph::begin_delete`] then [`Graph::delete_step`].
/// - Node labels: [`Graph::add_node_label`], [`Graph::remove_node_label`],
///   [`Graph::set_node_labels`], [`Graph::node_labels`], [`Graph::has_node_label`],
///   [`Graph::visit_nodes_with_label`].
/// - Edges: [`Graph::add_edge`], [`Graph::add_edges`], [`Graph::remove_edge`],
///   [`Graph::remove_edges_between`], [`Graph::set_edge_value`], [`Graph::edge_value`].
/// - Traversal: [`Graph::neighbors`], [`Graph::visit_neighbors`],
///   [`Graph::visit_neighbors_with_values`], [`Graph::visit_all_neighbors`], [`Graph::degree`],
///   [`Graph::contains_edge`], [`Graph::visit_edges_between`], and [`Graph::read_batch`] for many
///   nodes at once.
/// - Shared neighbours: [`Graph::sorted_neighbors`], [`Graph::common_neighbors`],
///   [`Graph::neighbor_set`] with [`Graph::common_with`].
///
/// # Positions, maintenance and value layouts
///
/// For callers that keep positions or spread work out themselves: [`Graph::add_edge_with`],
/// [`Graph::bulk_load`], [`Graph::visit_window`], [`Graph::nth_position`],
/// [`Graph::neighbors_at`], [`Graph::values_at`], [`Graph::visit_values`], [`Graph::counterpart`],
/// [`Graph::holes`], [`Graph::compact`], [`Graph::deleting_set`] with [`Graph::sweep`], and the
/// value width's evolution: [`Graph::declare_width`], [`Graph::change_width`],
/// [`Graph::migrate_values`], [`Graph::needs_migration`].
pub struct Graph<S: GraphMemories> {
    nodes: NodeStore<S::Nodes, S::Proxies>,
    labels: LabelStore<S::NodeLabels>,
    /// The page store: label buckets, and the label bucket lists beside them.
    edges: EdgeStore<S::Pages, S::Directory>,
    /// Mirrors the label width table, which the label bucket lists' pages hold.
    widths: Widths,
}

impl<S: GraphMemories> Graph<S> {
    /// A graph over formatted memories, with its header mirrors loaded.
    fn assemble(
        nodes: NodeStore<S::Nodes, S::Proxies>,
        labels: LabelStore<S::NodeLabels>,
        pages: PageStore<S::Pages, S::Directory>,
    ) -> Self {
        let widths = Widths::load(nodes.widths_root(), &pages);
        Self {
            nodes,
            labels,
            edges: EdgeStore::new(pages),
            widths,
        }
    }

    /// The edge store as reads see it.
    fn edge_view(&self) -> EdgeView<'_, S::Pages, S::Directory> {
        EdgeView::new(&self.edges, &self.widths)
    }

    fn check(&self, v: NodeId) -> Result<(), Error> {
        if self.nodes.contains(v) {
            Ok(())
        } else {
            Err(Error::UnknownNode(v))
        }
    }

    /// Inlined, as [`Self::state`] is: a node state read cost 7 % more as a call.
    #[inline]
    fn record(&self, v: NodeId) -> Result<Record, Error> {
        self.check(v)?;
        Ok(Record::decode(self.nodes.record_bytes(v)))
    }

    /// `v`'s state alone, without decoding its record's list.
    #[inline]
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

pub use memories::{GraphMemories, Memories, MemoriesOf, MemoryRole};
pub use read::{NeighborSet, ReadBatch};
pub use write::DeletingSet;

#[cfg(test)]
use crate::bucket::pages_per_half;
use crate::edge::{EdgeStore, EdgeView, Widths};

#[cfg(test)]
mod tests;
