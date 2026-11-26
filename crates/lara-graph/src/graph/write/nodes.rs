//! Adding nodes: one, or a run of consecutive ids; and proxies, likewise.

use crate::error::Error;
use crate::graph::Graph;
use crate::graph::GraphMemories;
use crate::id::{NodeId, NodeIds, NodeLabel};
use core::num::NonZeroU32;
use core::ops::Range;

impl<S: GraphMemories> Graph<S> {
    /// Adds a node with `labels` (in any order; repeats count once) and no edges, and returns its
    /// id: [`Self::add_nodes`] for one.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`], or [`Error::AddressSpaceExhausted`] past 2^31 nodes (the ids above
    /// are proxies'). On error no node is added.
    pub fn add_node(&mut self, labels: &[NodeLabel]) -> Result<NodeId, Error> {
        Ok(self.add_nodes(NonZeroU32::MIN, labels)?.first())
    }

    /// Adds `count` nodes, each with `labels` and no edges. Their ids are consecutive.
    ///
    /// # Errors
    ///
    /// [`Error::AddressSpaceExhausted`] when the ids would not fit, or [`Error::OutOfMemory`]
    /// when the records or labels cannot be grown. On error no node is added.
    #[inline]
    pub fn add_nodes(&mut self, count: NonZeroU32, labels: &[NodeLabel]) -> Result<NodeIds, Error> {
        let first = self.node_count();
        let next = first
            .checked_add(count.get())
            .filter(|&n| n <= NodeId::FIRST_PROXY)
            .ok_or(Error::AddressSpaceExhausted)?;
        // Labels first: a new node's cell is zeros (no labels), so growing the memory makes it.
        self.labels.reserve(next)?;
        if !labels.is_empty() {
            self.label_new_nodes(first..next, labels)?;
        }
        if let Err(error) = self.nodes.extend(next) {
            self.clear_new_labels(first..next);
            return Err(error.into());
        }
        Ok(NodeIds::new(NodeId::new(first), count))
    }

    /// Gives each of the nodes `range`, not yet visible, the set `labels`; on error none keeps
    /// any.
    #[inline(never)]
    fn label_new_nodes(&mut self, range: Range<u32>, labels: &[NodeLabel]) -> Result<(), Error> {
        let mut set = labels.to_vec();
        set.sort_unstable();
        set.dedup();
        for index in range.clone() {
            let v = NodeId::new(index);
            let labelled = self.labels.set(&mut self.edges.lists_mut(), v, &set);
            if let Err(error) = labelled {
                self.clear_new_labels(range.start..index);
                return Err(error.into());
            }
        }
        Ok(())
    }

    /// Drops the labels of the nodes `range`, not yet visible. Cannot fail.
    fn clear_new_labels(&mut self, range: Range<u32>) {
        for index in range {
            self.labels
                .clear(&mut self.edges.lists_mut(), NodeId::new(index));
        }
    }

    /// Adds a proxy: a node standing for one another graph holds, so that an edge to it keeps
    /// both halves here and a remote node's deletion reaches every local edge to it through the
    /// proxy's buckets ([`Self::begin_delete`]). A proxy has edges and a state like any node, but no
    /// labels; it is outside [`Self::node_count`] and [`Self::visit_nodes_with_label`]; and no edge
    /// joins two proxies. Its id has the top bit set ([`NodeId::is_proxy`]), so a read tells a
    /// proxy neighbour apart without reading anything.
    ///
    /// The proxies memory ([`crate::MemoryRole::Proxies`]) is written only from the first proxy.
    ///
    /// # Errors
    ///
    /// [`Error::NoProxiesMemory`], [`Error::OutOfMemory`], or [`Error::AddressSpaceExhausted`]
    /// past `2^31 - 1` proxies.
    pub fn add_proxy(&mut self) -> Result<NodeId, Error> {
        Ok(self.add_proxies(NonZeroU32::MIN)?.first())
    }

    /// Adds `count` proxies. Their ids are consecutive.
    ///
    /// # Errors
    ///
    /// [`Error::NoProxiesMemory`], [`Error::AddressSpaceExhausted`] when the ids would not fit,
    /// or [`Error::OutOfMemory`] when the records cannot be grown. On error no proxy is visible.
    pub fn add_proxies(&mut self, count: NonZeroU32) -> Result<NodeIds, Error> {
        if !self.nodes.has_proxies_memory() {
            return Err(Error::NoProxiesMemory);
        }
        let first = self.nodes.proxy_count();
        let next = first
            .checked_add(count.get())
            .filter(|&n| n <= NodeId::PROXIES)
            .ok_or(Error::AddressSpaceExhausted)?;
        self.nodes.extend_proxies(next)?;
        Ok(NodeIds::new(NodeId::proxy(first), count))
    }
}
