//! What a graph's operations report: a removal, the edges a bounded step moved or removed, and
//! whether more remains.

#[cfg(doc)]
use crate::graph::Graph;
use crate::id::{Adjacency, EdgeRef, Halves, NodeId, Position, ValueVersion};
use alloc::vec::Vec;

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
    pub const fn node(&self) -> NodeId {
        self.target
    }

    /// The moved half before and after the move, in `node`'s `adjacency` bucket: the
    /// bucket [`Graph::compact`] was given. [`EdgeRef::is_canonical`] with [`Self::node`] tells
    /// whether keys held for the edge move with it.
    #[must_use]
    pub const fn edge_refs(&self, node: NodeId, adjacency: Adjacency) -> (EdgeRef, EdgeRef) {
        let (label, direction) = (adjacency.label(), adjacency.direction());
        (
            EdgeRef::of(node, label, direction, self.from),
            EdgeRef::of(node, label, direction, self.to),
        )
    }
}

/// An edge that was removed: its halves, its value as the canonical half held it (empty for a
/// label without a width), and how full each half left its bucket.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RemovedEdge {
    halves: Halves,
    value: Vec<u8>,
    /// The version `value` is in.
    version: ValueVersion,
    /// The canonical half's bucket's, then the other's (the same for an undirected self-loop).
    fills: [BucketFill; 2],
}

impl RemovedEdge {
    pub(crate) fn new(
        halves: Halves,
        (value, version): (Vec<u8>, ValueVersion),
        fills: [BucketFill; 2],
    ) -> Self {
        Self {
            halves,
            value,
            version,
            fills,
        }
    }

    /// The version of the label's value layout [`Self::value`] is in
    /// (see `Graph::change_width`); [`ValueVersion::FIRST`] for a label without a width.
    #[must_use]
    pub const fn value_version(&self) -> ValueVersion {
        self.version
    }

    /// How full the canonical half's bucket is after the removal.
    #[must_use]
    pub const fn canonical_fill(&self) -> BucketFill {
        self.fills[0]
    }

    /// How full the other half's bucket is after the removal (the canonical one's again for an
    /// undirected self-loop).
    #[must_use]
    pub const fn other_fill(&self) -> BucketFill {
        self.fills[1]
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

/// How full a bucket is: its live edges, and the holes among its positions that compaction can
/// reclaim. A removal reports it for both buckets it changed, so a caller deciding when to
/// compact needs no further read: compacting once the holes reach the live edges keeps a graph
/// under steady updates within about 1.1 to 1.3 times its size built fresh (`examples/churn.rs`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BucketFill {
    live: u32,
    holes: u32,
}

impl BucketFill {
    pub(crate) const fn new(live: u32, holes: u32) -> Self {
        Self { live, holes }
    }

    /// Live edges.
    #[must_use]
    pub const fn live(self) -> u32 {
        self.live
    }

    /// Removed edges whose positions the bucket still holds.
    #[must_use]
    pub const fn holes(self) -> u32 {
        self.holes
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

/// What one bounded step of a longer operation changed, and whether more remains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step<T> {
    changes: Vec<T>,
    progress: Progress,
}

impl<T> Step<T> {
    pub(crate) const fn new(changes: Vec<T>, progress: Progress) -> Self {
        Self { changes, progress }
    }

    /// What this step changed, in the order it changed it.
    #[must_use]
    pub fn changes(&self) -> &[T] {
        &self.changes
    }

    /// Whether another step has work to do.
    #[must_use]
    pub const fn progress(&self) -> Progress {
        self.progress
    }
}

/// One [`Graph::compact`] step: the edges that moved. Every other edge of the bucket kept its
/// position.
pub type Compaction = Step<Move>;

/// One [`Graph::sweep`]: the edges removed, both halves of each, with their values, and where the
/// sweep goes on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sweep {
    removed: Vec<RemovedEdge>,
    next: Option<Position>,
}

impl Sweep {
    pub(crate) const fn new(removed: Vec<RemovedEdge>, next: Option<Position>) -> Self {
        Self { removed, next }
    }

    /// The edges removed, in position order.
    #[must_use]
    pub fn changes(&self) -> &[RemovedEdge] {
        &self.removed
    }

    /// Where the next call goes on; `None` once the bucket's end was reached.
    #[must_use]
    pub const fn next(&self) -> Option<Position> {
        self.next
    }
}

/// One [`Graph::delete_step`]: the edges removed, both halves of each, with their values.
pub type Deletion = Step<RemovedEdge>;

/// What [`Graph::remove_edge`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Removal {
    /// The edge was removed; its halves' positions are now holes.
    Removed(RemovedEdge),
    /// No live edge was at that position.
    Absent,
}
