//! Making, opening and giving back a graph's memories, as ic-stable-structures' structures do:
//! `format` formats, `load` reopens, `init` does whichever the memories call for. Each reads or
//! writes headers only, so it costs the same on every message that might be the first after an
//! install or an upgrade.

use super::{Graph, VERSION};
use crate::error::InitError;
use crate::graph::memories::{GraphMemories, Memories, MemoriesOf};
use crate::node::labels::{LabelStore, MAGIC as LABELS_MAGIC};
use crate::node::store::{MAGIC as NODES_MAGIC, NodeStore, PROXIES_MAGIC};
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::{self, PageStore, header};
use crate::store::units::Overflow;

impl<S: GraphMemories> Graph<S> {
    /// Formats `memories` as an empty graph, overwriting whatever they held.
    ///
    /// # Panics
    ///
    /// When a memory cannot grow; [`Self::try_format`] returns the error instead.
    #[must_use]
    pub fn format(memories: MemoriesOf<S>) -> Self {
        Self::try_format(memories)
            .unwrap_or_else(|e| panic!("lara-graph: cannot format a graph: {e}"))
    }

    /// Opens the graph `memories` hold.
    ///
    /// # Panics
    ///
    /// When they hold no graph, or one of another layout version; [`Self::try_load`] returns
    /// the error instead.
    #[must_use]
    pub fn load(memories: MemoriesOf<S>) -> Self {
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
    pub fn init(memories: MemoriesOf<S>) -> Self {
        Self::try_init(memories).unwrap_or_else(|e| panic!("lara-graph: cannot init a graph: {e}"))
    }

    /// [`Self::format`], returning the error.
    ///
    /// # Errors
    ///
    /// [`InitError::OutOfMemory`].
    pub fn try_format(memories: MemoriesOf<S>) -> Result<Self, InitError> {
        let Memories {
            nodes,
            pages,
            directory,
            node_labels,
            proxies,
        } = memories;
        let pages = PageStore::format(pages, directory).map_err(init_error)?;
        let nodes = NodeStore::format(nodes, proxies, VERSION).map_err(init_error)?;
        let labels = LabelStore::format(node_labels, VERSION).map_err(init_error)?;
        Ok(Self::assemble(nodes, labels, pages))
    }

    /// [`Self::load`], returning the error.
    ///
    /// # Errors
    ///
    /// [`InitError::NotAGraph`] or [`InitError::UnsupportedVersion`].
    pub fn try_load(memories: MemoriesOf<S>) -> Result<Self, InitError> {
        let Memories {
            nodes,
            pages,
            directory,
            node_labels,
            proxies,
        } = memories;
        // An empty proxies memory is a graph that never had a proxy.
        if let Some(proxies) = proxies.as_ref().filter(|p| p.size() > 0) {
            expect_header(proxies, PROXIES_MAGIC)?;
        }
        expect_header(&nodes, NODES_MAGIC)?;
        expect_header(&node_labels, LABELS_MAGIC)?;
        let pages = PageStore::open(pages, directory).map_err(open_error)?;
        Ok(Self::assemble(
            NodeStore::open(nodes, proxies, VERSION),
            LabelStore::open(node_labels),
            pages,
        ))
    }

    /// [`Self::init`], returning the error.
    ///
    /// # Errors
    ///
    /// [`InitError::UnsupportedVersion`], [`InitError::OutOfMemory`], or
    /// [`InitError::NotAGraph`] when the nodes memory holds a graph and the others do not.
    pub fn try_init(memories: MemoriesOf<S>) -> Result<Self, InitError> {
        if header(&memories.nodes, NODES_MAGIC).is_some() {
            Self::try_load(memories)
        } else {
            Self::try_format(memories)
        }
    }

    /// The memories back, as they were given: the graph is as it was left in them.
    #[must_use]
    pub fn into_memories(self) -> MemoriesOf<S> {
        let (pages, directory) = self.edges.into_memories();
        let (nodes, proxies) = self.nodes.into_memories();
        Memories {
            nodes,
            pages,
            directory,
            node_labels: self.labels.into_memory(),
            proxies,
        }
    }
}

/// Fails unless `memory` starts with `magic` and this layout version.
fn expect_header<M: Memory>(memory: &M, magic: [u8; 3]) -> Result<(), InitError> {
    match header(memory, magic) {
        None => Err(InitError::NotAGraph),
        Some(VERSION) => Ok(()),
        Some(version) => Err(InitError::UnsupportedVersion(version)),
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
