//! What a graph's operations report: a removal, the edges a bounded step moved or removed, and
//! whether more remains.

#[cfg(doc)]
use crate::graph::Graph;
use crate::id::{Direction, EdgeRef, Halves, Label, NodeId, Position};
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
    pub const fn target(&self) -> NodeId {
        self.target
    }

    /// The moved half before and after the move, in `node`'s `(label, direction)` bucket: the
    /// bucket [`Graph::compact`] was given. [`EdgeRef::is_canonical`] with [`Self::target`] tells
    /// whether keys held for the edge move with it.
    #[must_use]
    pub const fn edge_refs(
        &self,
        node: NodeId,
        (label, direction): (Label, Direction),
    ) -> (EdgeRef, EdgeRef) {
        (
            EdgeRef::new(node, label, direction, self.from),
            EdgeRef::new(node, label, direction, self.to),
        )
    }
}

/// An edge that was removed: its halves, and its value as the canonical half held it (empty for
/// a label without a width).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RemovedEdge {
    halves: Halves,
    value: Vec<u8>,
}

impl RemovedEdge {
    pub(crate) const fn new(halves: Halves, value: Vec<u8>) -> Self {
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

/// One [`Graph::detach_step`]: the edges removed, both halves of each, with their values.
pub type Detachment = Step<RemovedEdge>;

/// What [`Graph::remove_edge`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Removal {
    /// The edge was removed; its halves' positions are now holes.
    Removed(RemovedEdge),
    /// No live edge was at that position.
    Absent,
}
