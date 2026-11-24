//! A node's labels: a set in the node labels memory (see `node_labels`), read with one 5-byte
//! read when it has at most two, and the inverse, each label's nodes (see `label_postings`).
//! [`Graph::set_node_labels`] is the one writer of both, so the postings are always exactly the
//! inverse of the sets.

use super::{AT_POSTINGS, Error, Graph, label_cell_at, reserved};
use crate::id::{Label, NodeId, NodeState};
use crate::label_postings::{self, PAGES_PER_ADD};
use crate::memory::{Memory, Region};
use crate::node_labels::{CELL_BYTES, NodeLabels};
use crate::units::ByteOffset;
use alloc::vec::Vec;
use core::ops::ControlFlow;

impl<M: Memory> Graph<M> {
    /// Replaces `v`'s labels with `labels` (in any order; repeats count once), and updates each
    /// label's nodes for the labels gained and lost. Up to two are kept in the node's cell; more
    /// go to the label bucket list store, so any set of labels fits.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], [`Error::Deleted`], or [`Error::OutOfMemory`]. On error the
    /// labels and postings are unchanged.
    pub fn set_node_labels(&mut self, v: NodeId, labels: &[Label]) -> Result<(), Error> {
        if self.record(v)?.state == NodeState::Deleted {
            return Err(Error::Deleted(v));
        }
        let mut set: Vec<Label> = labels.to_vec();
        set.sort_unstable();
        set.dedup();
        let old_cell = self.node_labels_cell(v);
        let mut old = Vec::new();
        old_cell.extend_into(&self.label_bucket_lists.pages(), &mut old);
        let (gained, lost) = diff(&old, &set);
        let next = NodeLabels::store(&mut self.label_bucket_lists, &set)?;
        // Gains first, each with its pages reserved, undone on failure; losses never allocate.
        for (done, &label) in gained.iter().enumerate() {
            if let Err(error) = self.label_bucket_lists.reserve(PAGES_PER_ADD) {
                for &undo in gained.get(..done).unwrap_or(&[]) {
                    label_postings::remove(&mut self.label_bucket_lists, self.postings, undo, v);
                }
                next.release(&mut self.label_bucket_lists);
                self.write_postings_root();
                return Err(error.into());
            }
            self.postings = reserved(label_postings::add(
                &mut self.label_bucket_lists,
                self.postings,
                label,
                v,
            ));
        }
        for &label in &lost {
            label_postings::remove(&mut self.label_bucket_lists, self.postings, label, v);
        }
        self.write_postings_root();
        Region::new(&self.node_labels).overwrite(label_cell_at(v), &next.encode());
        old_cell.release(&mut self.label_bucket_lists);
        Ok(())
    }

    /// Appends `v`'s labels, ascending, to `out` after clearing it.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn node_labels(&self, v: NodeId, out: &mut Vec<Label>) -> Result<(), Error> {
        self.check(v)?;
        out.clear();
        self.node_labels_cell(v)
            .extend_into(&self.label_bucket_lists.pages(), out);
        Ok(())
    }

    /// Whether `v` has `label`: one 5-byte read for a node with at most two labels.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn has_node_label(&self, v: NodeId, label: Label) -> Result<bool, Error> {
        self.check(v)?;
        Ok(self
            .node_labels_cell(v)
            .contains(&self.label_bucket_lists.pages(), label))
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
        label_postings::visit(
            &self.label_bucket_lists.pages(),
            self.postings,
            (label, from, self.count),
            &mut f,
        )
    }

    /// How many nodes have `label`.
    #[must_use]
    pub fn node_label_count(&self, label: Label) -> u32 {
        label_postings::count(&self.label_bucket_lists.pages(), self.postings, label)
    }

    /// `v`'s label cell, decoded. `v` is known.
    pub(super) fn node_labels_cell(&self, v: NodeId) -> NodeLabels {
        let bytes: [u8; CELL_BYTES] = Region::new(&self.node_labels).read_array(label_cell_at(v));
        NodeLabels::decode(bytes)
    }

    /// Drops `v`'s labels and postings as it is deleted. Cannot fail: removals never allocate.
    pub(super) fn clear_node_labels(&mut self, v: NodeId) {
        let old = self.node_labels_cell(v);
        if old == NodeLabels::None {
            return;
        }
        let mut labels = Vec::new();
        old.extend_into(&self.label_bucket_lists.pages(), &mut labels);
        for label in labels {
            label_postings::remove(&mut self.label_bucket_lists, self.postings, label, v);
        }
        Region::new(&self.node_labels).overwrite(label_cell_at(v), &NodeLabels::None.encode());
        old.release(&mut self.label_bucket_lists);
    }

    fn write_postings_root(&self) {
        Region::new(&self.node_labels)
            .overwrite(ByteOffset::new(AT_POSTINGS), &self.postings.encode());
    }
}

/// The labels of `next` not in `old`, and of `old` not in `next`; both are sorted and distinct.
fn diff(old: &[Label], next: &[Label]) -> (Vec<Label>, Vec<Label>) {
    let gained = next
        .iter()
        .filter(|l| old.binary_search(l).is_err())
        .copied()
        .collect();
    let lost = old
        .iter()
        .filter(|l| next.binary_search(l).is_err())
        .copied()
        .collect();
    (gained, lost)
}
