//! Making, opening and giving back a graph's memories, as ic-stable-structures' structures do:
//! `new` formats, `load` reopens, `init` does whichever the memories call for. Each reads or
//! writes headers only, so it costs the same on every message that might be the first after an
//! install or an upgrade.

use super::{Graph, VERSION};
use crate::error::InitError;
use crate::graph::memories::Memories;
use crate::node::labels::{LabelStore, MAGIC as LABELS_MAGIC};
use crate::node::store::{MAGIC as NODES_MAGIC, NodeStore};
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::{self, PageStore, header};
use crate::store::units::Overflow;

impl<M: Memory> Graph<M> {
    /// Formats `memories` as an empty graph, overwriting whatever they held.
    ///
    /// # Panics
    ///
    /// When a memory cannot grow; [`Self::try_new`] returns the error instead.
    #[must_use]
    pub fn new(memories: Memories<M>) -> Self {
        Self::try_new(memories).unwrap_or_else(|e| panic!("lara-graph: cannot make a graph: {e}"))
    }

    /// Opens the graph `memories` hold.
    ///
    /// # Panics
    ///
    /// When they hold no graph, or one of another layout version; [`Self::try_load`] returns
    /// the error instead.
    #[must_use]
    pub fn load(memories: Memories<M>) -> Self {
        Self::try_load(memories).unwrap_or_else(|e| panic!("lara-graph: cannot load a graph: {e}"))
    }

    /// Opens the graph `memories` hold, or formats an empty one when the nodes memory is empty
    /// or does not start with a graph's magic. The call for a canister's `thread_local!` state:
    /// the same code serves the first install and every upgrade.
    ///
    /// # Panics
    ///
    /// When the graph is of another layout version, or a memory cannot grow; [`Self::try_init`]
    /// returns the error instead.
    #[must_use]
    pub fn init(memories: Memories<M>) -> Self {
        Self::try_init(memories).unwrap_or_else(|e| panic!("lara-graph: cannot init a graph: {e}"))
    }

    /// [`Self::new`], returning the error.
    ///
    /// # Errors
    ///
    /// [`InitError::OutOfMemory`].
    pub fn try_new(memories: Memories<M>) -> Result<Self, InitError> {
        let (
            (nodes, node_labels),
            [edge_pages, edge_directory],
            [label_bucket_list_pages, label_bucket_list_directory],
        ) = memories.into_parts();
        let edges = PageStore::format(edge_pages, edge_directory).map_err(init_error)?;
        let label_bucket_lists =
            PageStore::format(label_bucket_list_pages, label_bucket_list_directory)
                .map_err(init_error)?;
        let nodes = NodeStore::format(nodes, VERSION).map_err(init_error)?;
        let labels = LabelStore::format(node_labels, VERSION).map_err(init_error)?;
        Ok(Self::assemble((nodes, labels), edges, label_bucket_lists))
    }

    /// [`Self::load`], returning the error.
    ///
    /// # Errors
    ///
    /// [`InitError::NotAGraph`] or [`InitError::UnsupportedVersion`].
    pub fn try_load(memories: Memories<M>) -> Result<Self, InitError> {
        let (
            (nodes, node_labels),
            [edge_pages, edge_directory],
            [label_bucket_list_pages, label_bucket_list_directory],
        ) = memories.into_parts();
        match header(&nodes, NODES_MAGIC) {
            None => return Err(InitError::NotAGraph),
            Some(VERSION) => {}
            Some(version) => return Err(InitError::UnsupportedVersion(version)),
        }
        match header(&node_labels, LABELS_MAGIC) {
            None => return Err(InitError::NotAGraph),
            Some(VERSION) => {}
            Some(version) => return Err(InitError::UnsupportedVersion(version)),
        }
        let edges = PageStore::open(edge_pages, edge_directory).map_err(open_error)?;
        let label_bucket_lists =
            PageStore::open(label_bucket_list_pages, label_bucket_list_directory)
                .map_err(open_error)?;
        Ok(Self::assemble(
            (NodeStore::open(nodes), LabelStore::open(node_labels)),
            edges,
            label_bucket_lists,
        ))
    }

    /// [`Self::init`], returning the error.
    ///
    /// # Errors
    ///
    /// [`InitError::UnsupportedVersion`], [`InitError::OutOfMemory`], or
    /// [`InitError::NotAGraph`] when the nodes memory holds a graph and the others do not.
    pub fn try_init(memories: Memories<M>) -> Result<Self, InitError> {
        if header(memories.nodes(), NODES_MAGIC).is_some() {
            Self::try_load(memories)
        } else {
            Self::try_new(memories)
        }
    }

    /// The memories back, as they were given: the graph is as it was left in them.
    #[must_use]
    pub fn into_memories(self) -> Memories<M> {
        let (edge_pages, edge_directory) = self.edges.into_memories();
        let (label_bucket_list_pages, label_bucket_list_directory) =
            self.label_bucket_lists.into_memories();
        Memories::from_parts(
            (self.nodes.into_memory(), self.labels.into_memory()),
            [edge_pages, edge_directory],
            [label_bucket_list_pages, label_bucket_list_directory],
        )
    }
}

fn init_error(error: WriteError) -> InitError {
    match error {
        WriteError::Grow(grow) => InitError::OutOfMemory(grow),
        WriteError::Overflow(Overflow) => {
            unreachable!("a header at the start of a memory cannot overflow the address space")
        }
    }
}

fn open_error(error: page_store::OpenError) -> InitError {
    match error {
        page_store::OpenError::NotAPageStore => InitError::NotAGraph,
        page_store::OpenError::UnsupportedVersion(v) => InitError::UnsupportedVersion(v),
    }
}
