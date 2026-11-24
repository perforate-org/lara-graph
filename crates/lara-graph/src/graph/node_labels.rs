//! A node's labels: a set in the node labels memory (see `node_labels`), read with one 5-byte
//! read when it has at most two.

use super::{Error, Graph, label_cell_at};
use crate::id::{Label, NodeId, NodeState};
use crate::memory::{Memory, Region};
use crate::node_labels::{CELL_BYTES, MAX_NODE_LABELS, NodeLabels, spill_bytes};
use crate::page_store::ByteClass;
use alloc::vec::Vec;

impl<M: Memory> Graph<M> {
    /// Replaces `v`'s labels with `labels` (in any order; repeats count once). Up to two are
    /// kept beside the node's record; more go to one extent.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], [`Error::Deleted`], [`Error::TooManyNodeLabels`] past 2047
    /// labels, or [`Error::OutOfMemory`]. On error the labels are unchanged.
    pub fn set_node_labels(&mut self, v: NodeId, labels: &[Label]) -> Result<(), Error> {
        if self.record(v)?.state == NodeState::Deleted {
            return Err(Error::Deleted(v));
        }
        let mut set: Vec<Label> = labels.to_vec();
        set.sort_unstable();
        set.dedup();
        let next = match *set.as_slice() {
            [] => NodeLabels::None,
            [a] => NodeLabels::One(a),
            [a, b] => NodeLabels::Two(a, b),
            _ if set.len() > MAX_NODE_LABELS => return Err(Error::TooManyNodeLabels(v)),
            _ => {
                let bytes = spill_bytes(&set);
                let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
                let class = ByteClass::for_bytes(len).unwrap_or(ByteClass::PAGE);
                let extent = self.label_bucket_lists.allocate(class)?;
                self.label_bucket_lists.write(extent.offset(), &bytes);
                NodeLabels::Spilled(extent)
            }
        };
        let old = self.node_labels_cell(v);
        Region::new(&self.node_labels).overwrite(label_cell_at(v), &next.encode());
        if let NodeLabels::Spilled(extent) = old {
            self.label_bucket_lists.release(extent);
        }
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

    /// `v`'s label cell, decoded. `v` is known.
    pub(super) fn node_labels_cell(&self, v: NodeId) -> NodeLabels {
        let bytes: [u8; CELL_BYTES] = Region::new(&self.node_labels).read_array(label_cell_at(v));
        NodeLabels::decode(bytes)
    }

    /// Drops `v`'s labels as it is deleted. Cannot fail.
    pub(super) fn clear_node_labels(&mut self, v: NodeId) {
        let old = self.node_labels_cell(v);
        if old == NodeLabels::None {
            return;
        }
        Region::new(&self.node_labels).overwrite(label_cell_at(v), &NodeLabels::None.encode());
        if let NodeLabels::Spilled(extent) = old {
            self.label_bucket_lists.release(extent);
        }
    }
}
