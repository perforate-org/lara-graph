//! The memories a graph lives in, each named by its role.

use crate::store::memory::Memory;

/// What one of a graph's memories holds. Each memory must be used by one graph only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryRole {
    /// One 6-byte record per node.
    Nodes,
    /// The pages: label buckets (edges, their values and target indexes), each node's label
    /// bucket list, the width table, larger node label sets and each label's nodes.
    Pages,
    /// One record per page: how it is allocated.
    Directory,
    /// One 5-byte label cell per node.
    NodeLabels,
    /// One 6-byte record per proxy ([`crate::NodeId::proxy`]). Nothing is written to it until
    /// the first proxy is added, so a graph without proxies leaves it empty.
    Proxies,
}

impl MemoryRole {
    /// The role's place in [`Self::ALL`], from 0: an offset for memory ids, say.
    #[must_use]
    pub const fn index(self) -> u8 {
        match self {
            Self::Nodes => 0,
            Self::Pages => 1,
            Self::Directory => 2,
            Self::NodeLabels => 3,
            Self::Proxies => 4,
        }
    }

    /// Every role, in order.
    pub const ALL: [Self; 5] = [
        Self::Nodes,
        Self::Pages,
        Self::Directory,
        Self::NodeLabels,
        Self::Proxies,
    ];
}

/// A graph's memories, one per [`MemoryRole`], each of its own type. A graph whose memories
/// are all one type `M` names that type alone (`Graph<M>`) and makes them with [`Self::from_fn`];
/// one that mixes them names this struct (`Graph<Memories<N, PG, ..>>`) and fills its fields.
///
/// `proxies` may be `None` for a graph that never adds a proxy: it saves a memory, and
/// [`crate::Graph::add_proxies`] then fails with [`crate::Error::NoProxiesMemory`].
#[derive(Debug)]
#[expect(
    clippy::exhaustive_structs,
    reason = "callers fill every role by name; a new role is a new layout version anyway"
)]
pub struct Memories<N, PG = N, DR = N, NL = N, P = N> {
    /// The [`MemoryRole::Nodes`] memory.
    pub nodes: N,
    /// The [`MemoryRole::Pages`] memory.
    pub pages: PG,
    /// The [`MemoryRole::Directory`] memory.
    pub directory: DR,
    /// The [`MemoryRole::NodeLabels`] memory.
    pub node_labels: NL,
    /// The [`MemoryRole::Proxies`] memory, if the graph is to hold proxies.
    pub proxies: Option<P>,
}

impl<M> Memories<M> {
    /// The memories `memory` gives for each role, asked in the order of [`MemoryRole::ALL`].
    /// With ic-stable-structures, for example:
    /// `Memories::from_fn(|role| manager.get(MemoryId::new(FIRST + role.index())))`, or a
    /// `match` on the role.
    pub fn from_fn(mut memory: impl FnMut(MemoryRole) -> M) -> Self {
        Self {
            nodes: memory(MemoryRole::Nodes),
            pages: memory(MemoryRole::Pages),
            directory: memory(MemoryRole::Directory),
            node_labels: memory(MemoryRole::NodeLabels),
            proxies: Some(memory(MemoryRole::Proxies)),
        }
    }
}

impl<M> IntoIterator for Memories<M> {
    type Item = (MemoryRole, M);
    type IntoIter = core::iter::Flatten<core::array::IntoIter<Option<(MemoryRole, M)>, 5>>;

    /// Each memory with its role, in the order of [`MemoryRole::ALL`]; proxies only if given.
    fn into_iter(self) -> Self::IntoIter {
        [
            Some((MemoryRole::Nodes, self.nodes)),
            Some((MemoryRole::Pages, self.pages)),
            Some((MemoryRole::Directory, self.directory)),
            Some((MemoryRole::NodeLabels, self.node_labels)),
            self.proxies.map(|p| (MemoryRole::Proxies, p)),
        ]
        .into_iter()
        .flatten()
    }
}

/// The memory types of a graph, one per [`MemoryRole`]: any [`Memory`] for all of them, or a
/// [`Memories`] naming each.
pub trait GraphMemories {
    /// The [`MemoryRole::Nodes`] memory's type.
    type Nodes: Memory;
    /// The [`MemoryRole::Pages`] memory's type.
    type Pages: Memory;
    /// The [`MemoryRole::Directory`] memory's type.
    type Directory: Memory;
    /// The [`MemoryRole::NodeLabels`] memory's type.
    type NodeLabels: Memory;
    /// The [`MemoryRole::Proxies`] memory's type.
    type Proxies: Memory;
}

impl<M: Memory> GraphMemories for M {
    type Nodes = M;
    type Pages = M;
    type Directory = M;
    type NodeLabels = M;
    type Proxies = M;
}

impl<N, PG, DR, NL, P> GraphMemories for Memories<N, PG, DR, NL, P>
where
    N: Memory,
    PG: Memory,
    DR: Memory,
    NL: Memory,
    P: Memory,
{
    type Nodes = N;
    type Pages = PG;
    type Directory = DR;
    type NodeLabels = NL;
    type Proxies = P;
}

/// The memories of a graph of memory types `S`.
pub type MemoriesOf<S> = Memories<
    <S as GraphMemories>::Nodes,
    <S as GraphMemories>::Pages,
    <S as GraphMemories>::Directory,
    <S as GraphMemories>::NodeLabels,
    <S as GraphMemories>::Proxies,
>;
