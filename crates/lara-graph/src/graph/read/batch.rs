//! Batched reads over many nodes or halves, sharing one batch's buffers.

mod buffers;
mod halves;
mod node_labels;
mod view;

use crate::bucket::LabelBucket;
use crate::bucket::LabelBucketKey;
use crate::bucket::descriptor::Descriptor;
use crate::bucket::stored::Stored;
use crate::error::Error;
use crate::graph::Graph;
use crate::graph::locate::Found;
use crate::graph::locate::LastBucket;
use crate::graph::read::TargetSet;
use crate::graph::read::sorted::first_past;
use crate::id::{Direction, Label, Neighbor, NodeId, NodeState, Position};
use crate::input::Order;
use crate::node::labels::CellLines;
use crate::node::store::RecordLines;
use crate::store::memory::Memory;
use alloc::vec::Vec;
use buffers::{ListPageCache, Mode, for_each_list, mode, record, state, stored};
use core::ops::ControlFlow;
use view::BatchView;

/// A read-only batch over a list of nodes. The node-record line and flat-list page buffers
/// live as long as the batch, so several reads in one query share them. Drop it before mutating
/// the graph.
pub struct ReadBatch<'g, M: Memory> {
    graph: &'g Graph<M>,
    records: RecordLines,
    label_cells: CellLines,
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
            label_cells: CellLines::default(),
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
        (label, direction): (Label, Direction),
        order: Order,
        f: impl FnMut(NodeId, Neighbor) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        self.visit_each(nodes, LabelBucketKey { label, direction }, order, f)
    }

    #[inline]
    fn visit_each<B>(
        &mut self,
        nodes: &[NodeId],
        key: LabelBucketKey,
        order: Order,
        mut f: impl FnMut(NodeId, Neighbor) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        let mode = mode(nodes);
        let view = graph.view();
        for &v in nodes {
            let Some(stored) = self.stored_at(&view, v, key, mode) else {
                continue;
            };
            if let Some(found) = view.visit(&view.slots(stored), order, |e| f(v, e)) {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// Calls `f` with `(node, label, edge)` for every live edge of each node on `direction`,
    /// whatever its label, in `nodes` order, until `f` breaks: [`Graph::visit_all_neighbors`]
    /// over many nodes, with the batch's record line and list page.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn visit_all_neighbors_each<B>(
        &mut self,
        nodes: &[NodeId],
        direction: Direction,
        order: Order,
        mut f: impl FnMut(NodeId, Label, Neighbor) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        let mode = mode(nodes);
        let view = graph.view();
        let mut buckets: Vec<(Label, Stored)> = Vec::new();
        for &v in nodes {
            self.buckets_on(&view, (v, direction), mode, &mut buckets);
            if order == Order::Descending {
                buckets.reverse();
            }
            let found = buckets.iter().find_map(|&(label, stored)| {
                view.visit(&view.slots(stored), order, |e| f(v, label, e))
            });
            if found.is_some() {
                return Ok(found);
            }
        }
        Ok(None)
    }

    /// Replaces `out` with `v`'s buckets on `direction`, in list order, through the batch's record
    /// line and list page.
    fn buckets_on(
        &mut self,
        view: &BatchView<'g, M>,
        (v, direction): (NodeId, Direction),
        mode: Mode,
        out: &mut Vec<(Label, Stored)>,
    ) {
        out.clear();
        let Some(list) = record(self.graph, &mut self.records, v, mode).list else {
            return;
        };
        for_each_list(view, &mut self.list_page, list, mode, &mut |key, listed| {
            if key.direction == direction {
                out.push((key.label, listed.stored()));
            }
            ControlFlow::<()>::Continue(())
        });
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
        (label, direction): (Label, Direction),
        order: Order,
        mut f: impl FnMut(NodeId, Neighbor, &[u8]) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let key = LabelBucketKey { label, direction };
        self.each_bucket(nodes, key, |v, view, bucket| {
            view.visit_with_values(bucket, label, order, &mut |e, value| f(v, e, value))
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
        (label, direction): (Label, Direction),
        order: Order,
        mut f: impl FnMut(NodeId, Position, &[u8]) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        let key = LabelBucketKey { label, direction };
        self.each_bucket(nodes, key, |v, _, bucket| {
            graph
                .edge_view()
                .visit_values_in(bucket, label, order, &mut |position, value| {
                    f(v, position, value)
                })
        })
    }

    /// Calls `f` with each node of `nodes` whose `(label, direction)` bucket shares targets with
    /// `set`, and those targets, ascending, each once, in `nodes` order, until `f` breaks:
    /// [`Graph::common_with`] over many nodes, through the batch's record line and list page, with
    /// one pair of buffers for all of them. `above` names, for each node, the target the shared
    /// ones must be past, if any: listing each triangle `u < w < x` once from `u`'s set passes
    /// `|w| Some(w)` over `u`'s targets past `u`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any of `nodes`, before the first callback.
    pub fn common_with_each<B>(
        &mut self,
        set: &mut TargetSet,
        nodes: &[NodeId],
        (label, direction): (Label, Direction),
        mut above: impl FnMut(NodeId) -> Option<NodeId>,
        mut f: impl FnMut(NodeId, &[NodeId]) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        if set.is_empty() {
            check_all(graph, nodes.iter().copied())?;
            return Ok(None);
        }
        let mut found = Vec::new();
        let mut shared = Vec::new();
        self.each_bucket(
            nodes,
            LabelBucketKey { label, direction },
            |w, _, bucket| {
                found.clear();
                graph.intersect_from(set, bucket, first_past(above(w)), &mut found);
                if found.is_empty() {
                    return None;
                }
                shared.clear();
                shared.extend(found.iter().map(|&t| NodeId::new(t)));
                f(w, &shared).break_value()
            },
        )
    }

    /// Calls `f` with each node that has a `key` bucket, and the bucket, in `nodes` order, until
    /// `f` breaks.
    #[inline]
    fn each_bucket<B>(
        &mut self,
        nodes: &[NodeId],
        key: LabelBucketKey,
        mut f: impl FnMut(NodeId, &BatchView<'_, M>, &LabelBucket) -> Option<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        let mode = mode(nodes);
        let view = graph.view();
        for &v in nodes {
            let Some(stored) = self.stored_at(&view, v, key, mode) else {
                continue;
            };
            let bucket = view.resolve(stored);
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
        (label, direction): (Label, Direction),
        out: &mut Vec<u32>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        out.clear();
        out.reserve(nodes.len());
        let mode = mode(nodes);
        let key = LabelBucketKey { label, direction };
        let view = graph.view();
        for &v in nodes {
            let degree = self
                .stored_at(&view, v, key, mode)
                .map_or(0, |stored| view.slots(stored).live());
            out.push(degree);
        }
        Ok(())
    }

    /// Calls `f` with `(node, label)` for the label of every bucket of each node on `direction`,
    /// in list order, until `f` breaks: [`Graph::edge_labels`] over many nodes.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn edge_labels_each<B>(
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
        let view = graph.view();
        for &v in nodes {
            let record = record(graph, records, v, mode);
            let Some(list) = record.list else { continue };
            if let Some(found) = for_each_list(&view, list_page, list, mode, &mut |key, _| {
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

    /// Replaces `out` with whether each node has a live edge under any label, on any side. Within
    /// a node, stops at the first live bucket. Validates all nodes before it writes `out`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn has_edges_each(&mut self, nodes: &[NodeId], out: &mut Vec<bool>) -> Result<(), Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        out.clear();
        out.reserve(nodes.len());
        let mode = mode(nodes);
        let view = graph.view();
        for &v in nodes {
            let has = self.has_live(&view, v, mode);
            out.push(has);
        }
        Ok(())
    }

    /// Whether `v` has a bucket with a live edge; stops at the first.
    #[inline]
    fn has_live(&mut self, view: &BatchView<'_, M>, v: NodeId, mode: Mode) -> bool {
        let record = record(self.graph, &mut self.records, v, mode);
        let Some(list) = record.list else {
            return false;
        };
        for_each_list(view, &mut self.list_page, list, mode, &mut |_, listed| {
            if view.slots(listed.stored()).live() > 0 {
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
            out.push(state(graph, records, v, mode));
        }
        Ok(())
    }

    /// The stored entry of `v`'s `key` bucket, read through the batch's buffers in `mode`.
    #[inline]
    fn stored_at(
        &mut self,
        view: &BatchView<'_, M>,
        v: NodeId,
        key: LabelBucketKey,
        mode: Mode,
    ) -> Option<Stored> {
        let list = record(self.graph, &mut self.records, v, mode).list?;
        stored(view, &mut self.list_page, list, key, mode)
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
