//! Adding nodes: one, or a run of consecutive ids; and proxies, likewise.

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
    /// [`Error::OutOfMemory`], or [`Error::AddressSpaceExhausted`] past 2^31 nodes (the ids above
    /// are proxies').
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
            .filter(|&n| n <= NodeId::FIRST_PROXY)
            .ok_or(Error::AddressSpaceExhausted)?;
        // Labels first: a new node's cell is zeros (no labels), so growing the memory makes it.
        self.labels.reserve(next)?;
        self.nodes.extend(next)?;
        Ok(NodeId::new(first))
    }

    /// Adds a proxy: a node standing for one another graph holds, so that an edge to it keeps
    /// both halves here and a remote node's deletion reaches every local edge to it through the
    /// proxy's buckets ([`Self::detach`]). A proxy has edges and a state like any node, but no
    /// labels; it is outside [`Self::node_count`] and [`Self::visit_nodes_with_label`]; and no edge
    /// joins two proxies. Its id has the top bit set ([`NodeId::is_proxy`]), so a read tells a
    /// proxy neighbour apart without reading anything.
    ///
    /// The proxies memory ([`crate::MemoryRole::Proxies`]) is written only from the first proxy.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`], or [`Error::AddressSpaceExhausted`] past `2^31 - 1` proxies.
    pub fn add_proxy(&mut self) -> Result<NodeId, Error> {
        self.add_proxies(NonZeroU32::MIN)
    }

    /// Adds `count` proxies and returns the first new id. Ids are consecutive: the new proxies
    /// are `NodeId::proxy(first..first + count)` for `first` the returned id's ordinal.
    ///
    /// # Errors
    ///
    /// [`Error::AddressSpaceExhausted`] when the ids would not fit, or [`Error::OutOfMemory`]
    /// when the records cannot be grown. On error no proxy is visible.
    pub fn add_proxies(&mut self, count: NonZeroU32) -> Result<NodeId, Error> {
        let first = self.nodes.proxy_count();
        let next = first
            .checked_add(count.get())
            .filter(|&n| n <= NodeId::PROXIES)
            .ok_or(Error::AddressSpaceExhausted)?;
        self.nodes.extend_proxies(next)?;
        Ok(NodeId::proxy(first))
    }
}
