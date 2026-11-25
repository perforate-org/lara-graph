//! A node's labels and each label's nodes, kept by `node::labels`.

use super::{Error, Graph};
use crate::id::{Label, NodeId, NodeState};
use crate::node::labels::CellLines;
use crate::node::labels::set::LabelSet;
use crate::store::memory::Memory;
use alloc::vec::Vec;
use core::ops::ControlFlow;

impl<M: Memory> Graph<M> {
    /// Replaces `v`'s labels with `labels` (in any order; repeats count once), and updates each
    /// label's nodes for the labels gained and lost. Up to two are kept in the node's cell; more
    /// go to the label bucket list store, so any set of labels fits.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], [`Error::Proxy`], [`Error::Deleted`], or [`Error::OutOfMemory`].
    /// On error the labels and postings are unchanged.
    pub fn set_node_labels(&mut self, v: NodeId, labels: &[Label]) -> Result<(), Error> {
        if self.record(v)?.state == NodeState::Deleted {
            return Err(Error::Deleted(v));
        }
        if v.is_proxy() {
            return Err(Error::Proxy(v));
        }
        let mut set: Vec<Label> = labels.to_vec();
        set.sort_unstable();
        set.dedup();
        Ok(self.labels.set(&mut self.label_bucket_lists, v, &set)?)
    }

    /// Appends `v`'s labels, ascending, to `out` after clearing it; a proxy has none.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn node_labels(&self, v: NodeId, out: &mut Vec<Label>) -> Result<(), Error> {
        self.check(v)?;
        out.clear();
        self.extend_labels(self.labels.cell(v), out);
        Ok(())
    }

    /// Whether `v` has `label`: one 5-byte read for a node with at most two labels; false for a
    /// proxy.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn has_node_label(&self, v: NodeId, label: Label) -> Result<bool, Error> {
        self.check(v)?;
        Ok(self.set_has(self.labels.cell(v), label))
    }

    /// Calls `f` with each node that has `label`, in ascending id order, starting after `after`
    /// (from the first node for `None`), until it breaks; returns the break value. A node keeps
    /// its labels while it is being detached and loses them once deleted.
    pub fn visit_nodes_with_label<B>(
        &self,
        label: Label,
        after: Option<NodeId>,
        mut f: impl FnMut(NodeId) -> ControlFlow<B>,
    ) -> Option<B> {
        let from = after.map_or(0, |v| v.index().saturating_add(1));
        self.labels.visit(
            &self.label_bucket_lists.pages(),
            (label, from, self.nodes.count()),
            &mut f,
        )
    }

    /// How many nodes have `label`.
    #[must_use]
    pub fn node_label_count(&self, label: Label) -> u32 {
        self.labels.count(&self.label_bucket_lists.pages(), label)
    }

    /// Drops `v`'s labels and postings as it is deleted. Cannot fail: removals never allocate.
    pub(super) fn clear_node_labels(&mut self, v: NodeId) {
        if v.is_proxy() {
            return;
        }
        self.labels.clear(&mut self.label_bucket_lists, v);
    }

    /// `v`'s label cell, through a batch's line of cells. `v` is known.
    pub(in crate::graph) fn label_cell_in(&self, lines: &mut CellLines, v: NodeId) -> LabelSet {
        lines.cell(&self.labels, v)
    }

    /// `v`'s label cell. `v` is known.
    pub(in crate::graph) fn label_cell(&self, v: NodeId) -> LabelSet {
        self.labels.cell(v)
    }

    /// Whether the set `cell` holds `label`.
    pub(in crate::graph) fn set_has(&self, cell: LabelSet, label: Label) -> bool {
        cell.contains(&self.label_bucket_lists.pages(), label)
    }

    /// Appends the labels of the set `cell`, ascending, to `out`.
    pub(in crate::graph) fn extend_labels(&self, cell: LabelSet, out: &mut Vec<Label>) {
        cell.extend_into(&self.label_bucket_lists.pages(), out);
    }
}
