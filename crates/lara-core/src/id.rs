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

/// The halves of one edge. For a directed edge, `source` is the out half and `target` the in
/// half; for an undirected edge, `source` is the half at the smaller vertex id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Halves {
    /// An edge stored at both ends.
    Pair {
        /// The out half, or the undirected half at the smaller vertex.
        source: EdgeRef,
        /// The in half, or the undirected half at the larger vertex.
        target: EdgeRef,
    },
    /// An undirected self-loop: its one entry is both halves.
    Single(EdgeRef),
}

impl Halves {
    pub(crate) fn new(source: EdgeRef, target: EdgeRef) -> Self {
        if source == target {
            Self::Single(source)
        } else {
            Self::Pair { source, target }
        }
    }

    /// The half at the source (directed) or smaller vertex (undirected).
    #[must_use]
    pub const fn source(&self) -> EdgeRef {
        match *self {
            Self::Pair { source, .. } | Self::Single(source) => source,
        }
    }

    /// The half at the other end.
    #[must_use]
    pub const fn target(&self) -> EdgeRef {
        match *self {
            Self::Pair { target, .. } | Self::Single(target) => target,
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
