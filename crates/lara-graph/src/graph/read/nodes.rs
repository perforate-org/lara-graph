//! Reading a node: how many there are, its state, the labels it has buckets under, and whether
//! it has a live edge.

use crate::bucket::LabelBucket;
use crate::error::Error;
use crate::graph::Graph;
use crate::id::{Direction, Label, NodeId, NodeState};
use crate::node::label_bucket_list;
use crate::store::memory::Memory;
use alloc::vec::Vec;
use core::ops::ControlFlow;

impl<M: Memory> Graph<M> {
    /// Number of nodes; ids are `0..node_count`.
    #[must_use]
    pub const fn node_count(&self) -> u32 {
        self.nodes.count()
    }

    /// Whether `v` is live, being detached or deleted.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn node_state(&self, v: NodeId) -> Result<NodeState, Error> {
        self.state(v)
    }

    /// The labels of `v`'s edges on `direction`: those it has buckets under, in order. A node's own
    /// labels are [`Self::node_labels`].
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn edge_labels(
        &self,
        v: NodeId,
        direction: Direction,
        out: &mut Vec<Label>,
    ) -> Result<(), Error> {
        let Some(list) = self.label_bucket_list(v)? else {
            out.clear();
            return Ok(());
        };
        out.clear();
        label_bucket_list::for_each(&self.label_bucket_lists.pages(), Some(list), |key, _| {
            if key.direction == direction {
                out.push(key.label);
            }
            ControlFlow::<()>::Continue(())
        });
        Ok(())
    }

    /// Whether `v` has a live edge under any label, on any side. A bucket whose edges were all
    /// removed stays listed by [`Self::edge_labels`] until compaction drops it; this looks past it.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn has_edges(&self, v: NodeId) -> Result<bool, Error> {
        let Some(list) = self.label_bucket_list(v)? else {
            return Ok(false);
        };
        let view = self.edge_view().pages();
        Ok(label_bucket_list::for_each(
            &self.label_bucket_lists.pages(),
            Some(list),
            |_, listed| {
                if LabelBucket::resolve(&view, listed.stored()).slots.live() > 0 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::<()>::Continue(())
                }
            },
        )
        .is_some())
    }
}
