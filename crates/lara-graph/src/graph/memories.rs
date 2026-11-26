//! The memories a graph lives in, each named by its role.

use crate::store::memory::Memory;

/// What one of a graph's memories holds. Each memory must be used by one graph only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryRole {
    /// One 6-byte record per node.
    Nodes,
    /// The pages holding label buckets: edges, their values and target indexes.
    EdgePages,
    /// One record per edge page.
    EdgeDirectory,
    /// The pages holding each node's label bucket list, and the width table.
    LabelBucketListPages,
    /// One record per label bucket list page.
    LabelBucketListDirectory,
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
            Self::EdgePages => 1,
            Self::EdgeDirectory => 2,
            Self::LabelBucketListPages => 3,
            Self::LabelBucketListDirectory => 4,
            Self::NodeLabels => 5,
            Self::Proxies => 6,
        }
    }

    /// Every role, in order.
    pub const ALL: [Self; 7] = [
        Self::Nodes,
        Self::EdgePages,
        Self::EdgeDirectory,
        Self::LabelBucketListPages,
        Self::LabelBucketListDirectory,
        Self::NodeLabels,
        Self::Proxies,
    ];
}

/// A graph's memories, one per [`MemoryRole`], each of its own type. A graph whose memories
/// are all one type `M` names that type alone (`Graph<M>`) and makes them with [`Self::from_fn`];
/// one that mixes them names this struct (`Graph<Memories<N, EP, ..>>`) and fills its fields.
///
/// `proxies` may be `None` for a graph that never adds a proxy: it saves a memory, and
/// [`crate::Graph::add_proxies`] then fails with [`crate::Error::NoProxiesMemory`].
#[derive(Debug)]
#[expect(
    clippy::exhaustive_structs,
    reason = "callers fill every role by name; a new role is a new layout version anyway"
)]
pub struct Memories<N, EP = N, ED = N, LP = N, LD = N, NL = N, P = N> {
    /// The [`MemoryRole::Nodes`] memory.
    pub nodes: N,
    /// The [`MemoryRole::EdgePages`] memory.
    pub edge_pages: EP,
    /// The [`MemoryRole::EdgeDirectory`] memory.
    pub edge_directory: ED,
    /// The [`MemoryRole::LabelBucketListPages`] memory.
    pub label_bucket_list_pages: LP,
    /// The [`MemoryRole::LabelBucketListDirectory`] memory.
    pub label_bucket_list_directory: LD,
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
            edge_pages: memory(MemoryRole::EdgePages),
            edge_directory: memory(MemoryRole::EdgeDirectory),
            label_bucket_list_pages: memory(MemoryRole::LabelBucketListPages),
            label_bucket_list_directory: memory(MemoryRole::LabelBucketListDirectory),
            node_labels: memory(MemoryRole::NodeLabels),
            proxies: Some(memory(MemoryRole::Proxies)),
        }
    }
}

impl<M> IntoIterator for Memories<M> {
    type Item = (MemoryRole, M);
    type IntoIter = core::iter::Flatten<core::array::IntoIter<Option<(MemoryRole, M)>, 7>>;

    /// Each memory with its role, in the order of [`MemoryRole::ALL`]; proxies only if given.
    fn into_iter(self) -> Self::IntoIter {
        [
            Some((MemoryRole::Nodes, self.nodes)),
            Some((MemoryRole::EdgePages, self.edge_pages)),
            Some((MemoryRole::EdgeDirectory, self.edge_directory)),
            Some((
                MemoryRole::LabelBucketListPages,
                self.label_bucket_list_pages,
            )),
            Some((
                MemoryRole::LabelBucketListDirectory,
                self.label_bucket_list_directory,
            )),
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
    /// The [`MemoryRole::EdgePages`] memory's type.
    type EdgePages: Memory;
    /// The [`MemoryRole::EdgeDirectory`] memory's type.
    type EdgeDirectory: Memory;
    /// The [`MemoryRole::LabelBucketListPages`] memory's type.
    type LabelBucketListPages: Memory;
    /// The [`MemoryRole::LabelBucketListDirectory`] memory's type.
    type LabelBucketListDirectory: Memory;
    /// The [`MemoryRole::NodeLabels`] memory's type.
    type NodeLabels: Memory;
    /// The [`MemoryRole::Proxies`] memory's type.
    type Proxies: Memory;
}

impl<M: Memory> GraphMemories for M {
    type Nodes = M;
    type EdgePages = M;
    type EdgeDirectory = M;
    type LabelBucketListPages = M;
    type LabelBucketListDirectory = M;
    type NodeLabels = M;
    type Proxies = M;
}

impl<N, EP, ED, LP, LD, NL, P> GraphMemories for Memories<N, EP, ED, LP, LD, NL, P>
where
    N: Memory,
    EP: Memory,
    ED: Memory,
    LP: Memory,
    LD: Memory,
    NL: Memory,
    P: Memory,
{
    type Nodes = N;
    type EdgePages = EP;
    type EdgeDirectory = ED;
    type LabelBucketListPages = LP;
    type LabelBucketListDirectory = LD;
    type NodeLabels = NL;
    type Proxies = P;
}

/// The memories of a graph of memory types `S`.
pub type MemoriesOf<S> = Memories<
    <S as GraphMemories>::Nodes,
    <S as GraphMemories>::EdgePages,
    <S as GraphMemories>::EdgeDirectory,
    <S as GraphMemories>::LabelBucketListPages,
    <S as GraphMemories>::LabelBucketListDirectory,
    <S as GraphMemories>::NodeLabels,
    <S as GraphMemories>::Proxies,
>;
