//! Batched reads over many nodes or halves, sharing one batch's buffers.

mod halves;
mod list_page;

use super::super::pair::{Found, LastBucket};
use super::super::props::visit_joined;
use super::super::{Error, Graph, RecordLines};
use crate::descriptor::Descriptor;
use crate::id::{Direction, Edge, Label, NodeId, NodeState, Order, Position};
use crate::label_bucket::{self, Span, Window};
use crate::label_bucket_list::LabelBucketKey;
use crate::memory::Memory;
use crate::page_store::Pages;
use crate::stored::{LabelBucket, Stored};
use alloc::vec::Vec;
use core::ops::ControlFlow;
use list_page::{ListPageCache, Mode, for_each_list, mode, record, slots, stored};

/// A read-only batch over a list of nodes. The node-record line and flat-list page buffers
/// live as long as the batch, so several reads in one query share them. Drop it before mutating
/// the graph.
pub struct ReadBatch<'g, M: Memory> {
    graph: &'g Graph<M>,
    records: RecordLines,
    list_page: ListPageCache,
    last_bucket: LastBucket<Option<Descriptor>>,
    last_found: LastBucket<Option<Found>>,
    last_theirs: LastBucket<Found>,
}

impl<'g, M: Memory> ReadBatch<'g, M> {
    fn new(graph: &'g Graph<M>) -> Self {
        Self {
            graph,
            records: RecordLines::default(),
            list_page: ListPageCache::new(),
            last_bucket: LastBucket::default(),
            last_found: LastBucket::default(),
            last_theirs: LastBucket::default(),
        }
    }

    /// Calls `f` with `(node, edge)` for every live edge of each node's `(label, direction)`
    /// bucket, in `nodes` order and `order` within each bucket, until `f` breaks.
    ///
    /// The batch keeps an eight-record line of node records and, when a run of flat lists shares
    /// a 4 KiB page, that page. All nodes are checked before the first callback, so an unknown
    /// node leaves the callbacks untouched.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn visit_neighbors_each<B>(
        &mut self,
        nodes: &[NodeId],
        label: Label,
        direction: Direction,
        order: Order,
        f: impl FnMut(NodeId, Edge) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        self.visit_each(nodes, LabelBucketKey { label, direction }, order, f)
    }

    #[inline]
    fn visit_each<B>(
        &mut self,
        nodes: &[NodeId],
        key: LabelBucketKey,
        order: Order,
        mut f: impl FnMut(NodeId, Edge) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        let mode = mode(nodes);
        let lists = graph.label_bucket_lists.pages();
        let view = graph.edges.pages();
        for &v in nodes {
            let Some(stored) = self.stored_at(&lists, v, key, mode) else {
                continue;
            };
            let slots = slots(&view, stored);
            if let Some(found) =
                label_bucket::visit(&slots, &view, Span::all(&slots, order), Window::Page, |e| {
                    f(v, e)
                })
            {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// Calls `f` with `(node, edge, value)` for every live edge of each node's bucket, in
    /// `nodes` order and `order` within each bucket, until `f` breaks.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn visit_neighbors_with_values_each<B>(
        &mut self,
        nodes: &[NodeId],
        label: Label,
        direction: Direction,
        order: Order,
        mut f: impl FnMut(NodeId, Edge, &[u8]) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        let key = LabelBucketKey { label, direction };
        self.each_bucket(nodes, key, |v, view, bucket| {
            let width = graph.widths.get(label);
            visit_joined(view, bucket, width, order, &mut |e, value| f(v, e, value))
        })
    }

    /// Calls `f` with `(node, position, value)` for every value of each node's
    /// `(label, direction)` bucket, in `order`, until `f` breaks.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn visit_values_each<B>(
        &mut self,
        nodes: &[NodeId],
        label: Label,
        direction: Direction,
        order: Order,
        mut f: impl FnMut(NodeId, Position, &[u8]) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        let key = LabelBucketKey { label, direction };
        self.each_bucket(nodes, key, |v, _, bucket| {
            graph.visit_values_in(bucket, label, order, &mut |position, value| {
                f(v, position, value)
            })
        })
    }

    /// Calls `f` with each node that has a `key` bucket, and the bucket, in `nodes` order, until
    /// `f` breaks.
    #[inline]
    fn each_bucket<B>(
        &mut self,
        nodes: &[NodeId],
        key: LabelBucketKey,
        mut f: impl FnMut(NodeId, &Pages<'_, M>, &LabelBucket) -> Option<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        let mode = mode(nodes);
        for &v in nodes {
            let Some(stored) = self.stored_at(&graph.label_bucket_lists.pages(), v, key, mode)
            else {
                continue;
            };
            let view = graph.edges.pages();
            let bucket = LabelBucket::resolve(&view, stored);
            if let Some(done) = f(v, &view, &bucket) {
                return Ok(Some(done));
            }
        }
        Ok(None)
    }

    /// Replaces `out` with the live-edge counts of each node's `(label, direction)` bucket, one
    /// entry per input node and 0 when the node has no bucket. Validates all nodes before it
    /// writes `out`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn degree_each(
        &mut self,
        nodes: &[NodeId],
        label: Label,
        direction: Direction,
        out: &mut Vec<u32>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        out.clear();
        out.reserve(nodes.len());
        let mode = mode(nodes);
        let key = LabelBucketKey { label, direction };
        let lists = graph.label_bucket_lists.pages();
        let view = graph.edges.pages();
        for &v in nodes {
            let degree = self
                .stored_at(&lists, v, key, mode)
                .map_or(0, |stored| slots(&view, stored).live());
            out.push(degree);
        }
        Ok(())
    }

    /// Calls `f` with `(node, label)` for every bucket of each node on `direction`, in list
    /// order, until `f` breaks.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn labels_each<B>(
        &mut self,
        nodes: &[NodeId],
        direction: Direction,
        mut f: impl FnMut(NodeId, Label) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        let mode = mode(nodes);
        let records = &mut self.records;
        let list_page = &mut self.list_page;
        let lists = graph.label_bucket_lists.pages();
        for &v in nodes {
            let record = record(graph, records, v, mode);
            let Some(list) = record.list else { continue };
            if let Some(found) = for_each_list(&lists, list_page, list, mode, &mut |key, _| {
                if key.direction == direction {
                    f(v, key.label)
                } else {
                    ControlFlow::<B>::Continue(())
                }
            }) {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// Calls `f` with `(node, bool)` for each node: whether it has a live edge under any label,
    /// on any side. Within a node, stops at the first live bucket.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn has_edges_each<B>(
        &mut self,
        nodes: &[NodeId],
        mut f: impl FnMut(NodeId, bool) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        let mode = mode(nodes);
        let lists = graph.label_bucket_lists.pages();
        let view = graph.edges.pages();
        for &v in nodes {
            let has = self.has_live(&lists, &view, v, mode);
            if let ControlFlow::Break(found) = f(v, has) {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// Whether `v` has a bucket with a live edge; stops at the first.
    #[inline]
    fn has_live(
        &mut self,
        lists: &Pages<'_, M>,
        view: &Pages<'_, M>,
        v: NodeId,
        mode: Mode,
    ) -> bool {
        let record = record(self.graph, &mut self.records, v, mode);
        let Some(list) = record.list else {
            return false;
        };
        for_each_list(lists, &mut self.list_page, list, mode, &mut |_, listed| {
            if slots(view, listed.stored()).live() > 0 {
                ControlFlow::Break(())
            } else {
                ControlFlow::<()>::Continue(())
            }
        })
        .is_some()
    }

    /// Replaces `out` with the state of each input node. Validates all nodes before it writes
    /// `out`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn node_state_each(
        &mut self,
        nodes: &[NodeId],
        out: &mut Vec<NodeState>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        out.clear();
        out.reserve(nodes.len());
        let mode = mode(nodes);
        let records = &mut self.records;
        for &v in nodes {
            out.push(record(graph, records, v, mode).state);
        }
        Ok(())
    }

    /// The stored entry of `v`'s `key` bucket, read through the batch's buffers in `mode`.
    #[inline]
    fn stored_at(
        &mut self,
        lists: &Pages<'_, M>,
        v: NodeId,
        key: LabelBucketKey,
        mode: Mode,
    ) -> Option<Stored> {
        let list = record(self.graph, &mut self.records, v, mode).list?;
        stored(lists, &mut self.list_page, list, key, mode)
    }
}

/// Checks every node of a batch before its first read.
#[inline]
fn check_all<M: Memory>(
    graph: &Graph<M>,
    nodes: impl IntoIterator<Item = NodeId>,
) -> Result<(), Error> {
    for v in nodes {
        graph.check(v)?;
    }
    Ok(())
}

impl<M: Memory> Graph<M> {
    /// A read-only batch that keeps its node-record and list-page buffers across calls. Drop it
    /// before mutating the graph.
    #[must_use]
    pub fn read_batch(&self) -> ReadBatch<'_, M> {
        ReadBatch::new(self)
    }
}
