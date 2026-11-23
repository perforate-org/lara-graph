//! Public identifiers.

/// A node. Nodes are dense: a graph with `n` nodes has ids `0..n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u32);

impl NodeId {
    /// The node with dense index `index`.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The dense index of this node.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// An edge label. Each `(node, label)` pair has its own bucket of edges; labels order
/// buckets by their raw value (callers may pack flags such as directedness into it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Label(u16);

impl Label {
    /// The label with raw value `raw`.
    #[must_use]
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    /// The raw value.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Which way a read goes through a bucket's positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Order {
    /// Lowest position first: insertion order under [`Placement::Append`].
    Ascending,
    /// Highest position first: newest first under [`Placement::Append`].
    Descending,
}

/// A window of a bucket's positions for [`Graph::visit_window`](crate::Graph::visit_window):
/// `span` positions from `from` on in `order`, holes included, so a window's cost is bounded
/// whatever was removed. Positions are stable, so a cursor stays valid until the bucket is
/// compacted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cursor {
    order: Order,
    from: Option<Position>,
    span: core::num::NonZeroU32,
}

impl Cursor {
    /// The first window: from the lowest position ascending, or the highest descending.
    #[must_use]
    pub const fn first(order: Order, span: core::num::NonZeroU32) -> Self {
        Self {
            order,
            from: None,
            span,
        }
    }

    /// The window starting at `from` (included).
    #[must_use]
    pub const fn at(order: Order, from: Position, span: core::num::NonZeroU32) -> Self {
        Self {
            order,
            from: Some(from),
            span,
        }
    }

    /// The direction.
    #[must_use]
    pub const fn order(&self) -> Order {
        self.order
    }

    /// The first position, or `None` for the bucket's first in [`Self::order`].
    #[must_use]
    pub const fn from(&self) -> Option<Position> {
        self.from
    }

    /// Positions per window.
    #[must_use]
    pub const fn span(&self) -> core::num::NonZeroU32 {
        self.span
    }
}

/// Bytes in each value of a label's edges: 1 to 4096.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Width(u16);

impl Width {
    /// The widest value: one per value page.
    pub const MAX: u16 = 4096;

    /// A width of `bytes`, if it is 1 to [`Self::MAX`].
    #[must_use]
    pub const fn new(bytes: u16) -> Option<Self> {
        if bytes >= 1 && bytes <= Self::MAX {
            Some(Self(bytes))
        } else {
            None
        }
    }

    /// The width in bytes.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Where an insert may put an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    /// Always after every stored position, so position order is insertion order. Holes stay
    /// until compaction, which keeps the order.
    Append,
    /// Into the first hole after the last live edge to the same node, anywhere in the bucket,
    /// else after every stored position. Fills holes without compaction, so positions stay
    /// stable; they do not follow insertion order (Gleaph's `Unordered`).
    ReuseHoles,
}

/// Where a node is in its life. Node ids are never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeState {
    /// Takes and holds edges.
    Live,
    /// Being deleted: its edges are removed step by step and it takes no new ones.
    Detaching,
    /// Deleted: it has no edges and takes none.
    Deleted,
}

/// Which of a node's adjacencies a bucket holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Direction {
    /// Directed edges leaving the node.
    Outgoing,
    /// Directed edges entering the node.
    Incoming,
    /// Undirected edges at the node.
    Undirected,
}

/// A position in one `(node, label)` bucket. Positions count removed edges, so an edge keeps
/// its position until the bucket is compacted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Position(u32);

impl Position {
    /// The position `index`.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The index of this position.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// One live edge of a bucket: where it is and where it goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Edge {
    position: Position,
    target: NodeId,
}

impl Edge {
    pub(crate) const fn new(position: Position, target: NodeId) -> Self {
        Self { position, target }
    }

    /// The edge's position in its bucket.
    #[must_use]
    pub const fn position(&self) -> Position {
        self.position
    }

    /// The node the edge points to.
    #[must_use]
    pub const fn target(&self) -> NodeId {
        self.target
    }
}

/// An edge that compaction moved to a new position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    from: Position,
    to: Position,
    target: NodeId,
}

impl Move {
    pub(crate) const fn new(from: Position, to: Position, target: NodeId) -> Self {
        Self { from, to, target }
    }

    /// Where the edge was.
    #[must_use]
    pub const fn from(&self) -> Position {
        self.from
    }

    /// Where the edge is now.
    #[must_use]
    pub const fn to(&self) -> Position {
        self.to
    }

    /// The node the edge points to.
    #[must_use]
    pub const fn target(&self) -> NodeId {
        self.target
    }
}

/// One half of an edge: its entry in one bucket. A directed edge `u → v` has an
/// [`Direction::Outgoing`] half at `u` and an [`Direction::Incoming`] half at `v`; an undirected
/// edge `u — v` has a [`Direction::Undirected`] half at each end, and a single one when `u = v`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeRef {
    node: NodeId,
    label: Label,
    direction: Direction,
    position: Position,
}

impl EdgeRef {
    /// The entry at `position` in `node`'s `(label, direction)` bucket.
    #[must_use]
    pub const fn new(node: NodeId, label: Label, direction: Direction, position: Position) -> Self {
        Self {
            node,
            label,
            direction,
            position,
        }
    }

    /// The node whose bucket holds this half.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// The edge's label.
    #[must_use]
    pub const fn label(&self) -> Label {
        self.label
    }

    /// Which bucket of the node.
    #[must_use]
    pub const fn direction(&self) -> Direction {
        self.direction
    }

    /// The position in that bucket.
    #[must_use]
    pub const fn position(&self) -> Position {
        self.position
    }
}

/// The halves of one edge, the canonical one first: where a caller keys what belongs to the
/// edge (properties, index postings). For a directed edge it is the out half; for an undirected
/// edge, the half at the larger node id. Ids grow in creation order and hubs tend to be old, so
/// this puts a hub's undirected edges' canonical halves in its newer neighbours' small buckets:
/// the keyed data spreads out, and compacting a hub moves few canonical halves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Halves {
    /// An edge stored at both ends.
    Pair {
        /// The out half, or the undirected half at the larger node.
        canonical: EdgeRef,
        /// The in half, or the undirected half at the smaller node.
        other: EdgeRef,
    },
    /// An undirected self-loop: its one entry is both halves.
    Single(EdgeRef),
}

impl Halves {
    pub(crate) fn new(canonical: EdgeRef, other: EdgeRef) -> Self {
        if canonical == other {
            Self::Single(canonical)
        } else {
            Self::Pair { canonical, other }
        }
    }

    /// The canonical half: the out half, or the undirected half at the larger node.
    #[must_use]
    pub const fn canonical(&self) -> EdgeRef {
        match *self {
            Self::Pair { canonical, .. } | Self::Single(canonical) => canonical,
        }
    }

    /// The other half (the canonical one again for an undirected self-loop).
    #[must_use]
    pub const fn other(&self) -> EdgeRef {
        match *self {
            Self::Pair { other, .. } | Self::Single(other) => other,
        }
    }
}

/// An edge that was removed: its halves, and its value as the canonical half held it (empty for
/// a label without a width).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RemovedEdge {
    halves: Halves,
    value: alloc::vec::Vec<u8>,
}

impl RemovedEdge {
    pub(crate) const fn new(halves: Halves, value: alloc::vec::Vec<u8>) -> Self {
        Self { halves, value }
    }

    /// The removed halves.
    #[must_use]
    pub const fn halves(&self) -> Halves {
        self.halves
    }

    /// The value the edge carried.
    #[must_use]
    pub fn value(&self) -> &[u8] {
        &self.value
    }
}

/// Whether a bucket has more compaction work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Progress {
    /// Nothing left to reclaim in this bucket.
    Done,
    /// Another step may reclaim more.
    Remaining,
}
