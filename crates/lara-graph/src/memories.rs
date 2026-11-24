//! The memories a graph lives in, each named by its role.

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
        }
    }

    /// Every role, in order.
    pub const ALL: [Self; 6] = [
        Self::Nodes,
        Self::EdgePages,
        Self::EdgeDirectory,
        Self::LabelBucketListPages,
        Self::LabelBucketListDirectory,
        Self::NodeLabels,
    ];
}

/// A graph's memories, one per [`MemoryRole`].
#[derive(Debug)]
pub struct Memories<M> {
    nodes: M,
    edge_pages: M,
    edge_directory: M,
    label_bucket_list_pages: M,
    label_bucket_list_directory: M,
    node_labels: M,
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
        }
    }

    pub(crate) const fn nodes(&self) -> &M {
        &self.nodes
    }

    /// The nodes and node labels memories, then the edge store's pages and directory, then the
    /// label bucket list store's.
    pub(crate) fn into_parts(self) -> ((M, M), [M; 2], [M; 2]) {
        (
            (self.nodes, self.node_labels),
            [self.edge_pages, self.edge_directory],
            [
                self.label_bucket_list_pages,
                self.label_bucket_list_directory,
            ],
        )
    }

    pub(crate) fn from_parts(
        (nodes, node_labels): (M, M),
        [edge_pages, edge_directory]: [M; 2],
        [label_bucket_list_pages, label_bucket_list_directory]: [M; 2],
    ) -> Self {
        Self {
            nodes,
            edge_pages,
            edge_directory,
            label_bucket_list_pages,
            label_bucket_list_directory,
            node_labels,
        }
    }
}

impl<M> IntoIterator for Memories<M> {
    type Item = (MemoryRole, M);
    type IntoIter = core::array::IntoIter<(MemoryRole, M), 6>;

    /// Each memory with its role, in the order of [`MemoryRole::ALL`].
    fn into_iter(self) -> Self::IntoIter {
        [
            (MemoryRole::Nodes, self.nodes),
            (MemoryRole::EdgePages, self.edge_pages),
            (MemoryRole::EdgeDirectory, self.edge_directory),
            (
                MemoryRole::LabelBucketListPages,
                self.label_bucket_list_pages,
            ),
            (
                MemoryRole::LabelBucketListDirectory,
                self.label_bucket_list_directory,
            ),
            (MemoryRole::NodeLabels, self.node_labels),
        ]
        .into_iter()
    }
}
