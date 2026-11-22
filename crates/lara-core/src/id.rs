//! Public identifiers.

/// A vertex. Vertices are dense: a graph with `n` vertices has ids `0..n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexId(u32);

impl VertexId {
    /// The vertex with dense index `index`.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The dense index of this vertex.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// An edge label. Each `(vertex, label)` pair has its own bucket of edges; labels order
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

/// Where an insert may put an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    /// Always after every stored position, so position order is insertion order (Gleaph's
    /// `Insertion` order). Holes stay until compaction, which keeps the order.
    Append,
    /// Into a hole when one is safe (no live edge to the same vertex after it), else after
    /// every stored position. Fills holes without compaction; positions do not follow insertion
    /// order (Gleaph's `Unordered`).
    ReuseHoles,
}

/// Where a vertex is in its life. Vertex ids are never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VertexState {
    /// Takes and holds edges.
    Live,
    /// Being deleted: its edges are removed step by step and it takes no new ones.
    Detaching,
    /// Deleted: it has no edges and takes none.
    Deleted,
}

/// Which of a vertex's adjacencies a bucket holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Side {
    /// Directed edges leaving the vertex.
    Out,
    /// Directed edges entering the vertex.
    In,
    /// Undirected edges at the vertex.
    Undirected,
}

/// A position in one `(vertex, label)` bucket. Positions count removed edges, so an edge keeps
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
    target: VertexId,
}

impl Edge {
    pub(crate) const fn new(position: Position, target: VertexId) -> Self {
        Self { position, target }
    }

    /// The edge's position in its bucket.
    #[must_use]
    pub const fn position(&self) -> Position {
        self.position
    }

    /// The vertex the edge points to.
    #[must_use]
    pub const fn target(&self) -> VertexId {
        self.target
    }
}

/// An edge that compaction moved to a new position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    from: Position,
    to: Position,
    target: VertexId,
}

impl Move {
    pub(crate) const fn new(from: Position, to: Position, target: VertexId) -> Self {
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

    /// The vertex the edge points to.
    #[must_use]
    pub const fn target(&self) -> VertexId {
        self.target
    }
}

/// One half of an edge: its entry in one bucket. A directed edge `u → v` has an [`Side::Out`]
/// half at `u` and an [`Side::In`] half at `v`; an undirected edge `u — v` has a
/// [`Side::Undirected`] half at each end, and a single one when `u = v`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeRef {
    vertex: VertexId,
    label: Label,
    side: Side,
    position: Position,
}

impl EdgeRef {
    /// The entry at `position` in `vertex`'s `(label, side)` bucket.
    #[must_use]
    pub const fn new(vertex: VertexId, label: Label, side: Side, position: Position) -> Self {
        Self {
            vertex,
            label,
            side,
            position,
        }
    }

    /// The vertex whose bucket holds this half.
    #[must_use]
    pub const fn vertex(&self) -> VertexId {
        self.vertex
    }

    /// The edge's label.
    #[must_use]
    pub const fn label(&self) -> Label {
        self.label
    }

    /// Which bucket of the vertex.
    #[must_use]
    pub const fn side(&self) -> Side {
        self.side
    }

    /// The position in that bucket.
    #[must_use]
    pub const fn position(&self) -> Position {
        self.position
    }
}

/// The halves of one edge, the canonical one first: where a caller keys what belongs to the
/// edge (properties, index postings). For a directed edge it is the out half; for an undirected
/// edge, the half at the larger vertex id. Ids grow in creation order and hubs tend to be old, so
/// this puts a hub's undirected edges' canonical halves in its newer neighbours' small buckets:
/// the keyed data spreads out, and compacting a hub moves few canonical halves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Halves {
    /// An edge stored at both ends.
    Pair {
        /// The out half, or the undirected half at the larger vertex.
        canonical: EdgeRef,
        /// The in half, or the undirected half at the smaller vertex.
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

    /// The canonical half: the out half, or the undirected half at the larger vertex.
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

/// Whether a bucket has more compaction work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Progress {
    /// Nothing left to reclaim in this bucket.
    Done,
    /// Another step may reclaim more.
    Remaining,
}
