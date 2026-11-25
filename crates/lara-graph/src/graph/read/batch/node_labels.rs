//! Batched reads of node labels: each node's set, or whether it has one label. A dense batch
//! reads eight label cells at once.

use super::buffers::{label_cell, mode};
use super::{ReadBatch, check_all};
use crate::error::Error;
use crate::id::{Label, NodeId};
use crate::store::memory::Memory;
use alloc::vec::Vec;
use core::ops::ControlFlow;

impl<M: Memory> ReadBatch<'_, M> {
    /// Calls `f` with `(node, label)` for each label of each node, in `nodes` order and ascending
    /// within a node, until `f` breaks: [`Graph::node_labels`] over many nodes.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    ///
    /// [`Graph::node_labels`]: crate::Graph::node_labels
    /// [`Graph::node_count`]: crate::Graph::node_count
    pub fn node_labels_each<B>(
        &mut self,
        nodes: &[NodeId],
        mut f: impl FnMut(NodeId, Label) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        let mode = mode(nodes);
        let mut labels = Vec::new();
        for &v in nodes {
            labels.clear();
            graph.extend_labels(
                label_cell(graph, &mut self.label_cells, v, mode),
                &mut labels,
            );
            for &label in &labels {
                if let ControlFlow::Break(found) = f(v, label) {
                    return Ok(Some(found));
                }
            }
        }
        Ok(None)
    }

    /// Replaces `out` with whether each node has `label`: [`Graph::has_node_label`] over many
    /// nodes, one 5-byte cell each for a node with at most two labels. Validates all nodes before
    /// it writes `out`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    ///
    /// [`Graph::has_node_label`]: crate::Graph::has_node_label
    /// [`Graph::node_count`]: crate::Graph::node_count
    pub fn has_node_label_each(
        &mut self,
        nodes: &[NodeId],
        label: Label,
        out: &mut Vec<bool>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        check_all(graph, nodes.iter().copied())?;
        out.clear();
        out.reserve(nodes.len());
        let mode = mode(nodes);
        for &v in nodes {
            out.push(graph.set_has(label_cell(graph, &mut self.label_cells, v, mode), label));
        }
        Ok(())
    }
}
