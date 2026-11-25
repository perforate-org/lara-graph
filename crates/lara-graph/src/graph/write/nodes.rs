//! Adding nodes: one, or a run of consecutive ids.

use crate::error::Error;
use crate::graph::Graph;
use crate::id::NodeId;
use crate::store::memory::Memory;
use core::num::NonZeroU32;

impl<M: Memory> Graph<M> {
    /// Adds a node with no edges and returns its id.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`], or [`Error::AddressSpaceExhausted`] past `u32::MAX - 1` nodes
    /// (the last id is reserved as the removed-edge marker).
    pub fn add_node(&mut self) -> Result<NodeId, Error> {
        self.add_nodes(NonZeroU32::MIN)
    }

    /// Adds `count` nodes and returns the first new id. Ids are consecutive: the new nodes are
    /// `first.index()..first.index() + count`.
    ///
    /// # Errors
    ///
    /// [`Error::AddressSpaceExhausted`] when the ids would not fit, or [`Error::OutOfMemory`]
    /// when the records cannot be grown. On error no node is visible.
    #[inline]
    pub fn add_nodes(&mut self, count: NonZeroU32) -> Result<NodeId, Error> {
        let first = self.node_count();
        let next = first
            .checked_add(count.get())
            .filter(|&n| n < u32::MAX)
            .ok_or(Error::AddressSpaceExhausted)?;
        // Labels first: a new node's cell is zeros (no labels), so growing the memory makes it.
        self.labels.reserve(next)?;
        self.nodes.extend(next)?;
        Ok(NodeId::new(first))
    }
}
