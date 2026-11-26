//! What a graph's operations take beyond identifiers: the order a read walks a bucket, a window
//! of its positions, where an insert puts an edge, and the edges and values a batch writes.

#[cfg(doc)]
use crate::graph::Graph;
use crate::id::{EdgeLabel, EdgeRef, NodeId, Position};
use core::num::NonZeroU32;

/// Which way a read goes through a bucket's positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Order {
    /// Lowest position first: insertion order under [`Placement::Append`].
    Ascending,
    /// Highest position first: newest first under [`Placement::Append`].
    Descending,
}

/// A window of a bucket's positions for [`Graph::visit_window`](crate::Graph::visit_window):
/// `span` positions from `start` on in `order`, holes included, so a window's cost is bounded
/// whatever was removed. Positions are stable, so a cursor stays valid until the bucket is
/// compacted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cursor {
    order: Order,
    start: Option<Position>,
    span: NonZeroU32,
}

impl Cursor {
    /// The first window: from the lowest position ascending, or the highest descending.
    #[must_use]
    pub const fn first(order: Order, span: NonZeroU32) -> Self {
        Self {
            order,
            start: None,
            span,
        }
    }

    /// The window starting at `start` (included).
    #[must_use]
    pub const fn at(order: Order, start: Position, span: NonZeroU32) -> Self {
        Self {
            order,
            start: Some(start),
            span,
        }
    }

    /// The direction.
    #[must_use]
    pub const fn order(&self) -> Order {
        self.order
    }

    /// Where the window starts, or `None` for the bucket's first position in [`Self::order`].
    #[must_use]
    pub const fn start(&self) -> Option<Position> {
        self.start
    }

    /// Positions per window.
    #[must_use]
    pub const fn span(&self) -> NonZeroU32 {
        self.span
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

/// Whether [`Edge`] is directed or undirected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// `from -[label]-> to`.
    Directed,
    /// `a -[label]- b`, stored at both ends with one undirected half each.
    Undirected,
}

/// An edge to insert: its ends, label and kind, and its value when its label declares a width.
/// Where it goes in each bucket is the insert's choice ([`Graph::add_edge_with`]), not the
/// edge's.
#[derive(Clone, Copy, Debug)]
pub struct Edge<'a> {
    kind: EdgeKind,
    from: NodeId,
    label: EdgeLabel,
    to: NodeId,
    value: &'a [u8],
}

impl<'a> Edge<'a> {
    /// The directed edge `from -[label]-> to`, without a value.
    #[must_use]
    pub const fn directed(from: NodeId, label: EdgeLabel, to: NodeId) -> Self {
        Self {
            kind: EdgeKind::Directed,
            from,
            label,
            to,
            value: &[],
        }
    }

    /// The undirected edge `a -[label]- b`, without a value.
    #[must_use]
    pub const fn undirected(a: NodeId, label: EdgeLabel, b: NodeId) -> Self {
        Self {
            kind: EdgeKind::Undirected,
            from: a,
            label,
            to: b,
            value: &[],
        }
    }

    /// The edge with `value`, which must be as wide as its label declares.
    #[must_use]
    pub const fn value(self, value: &'a [u8]) -> Self {
        Self { value, ..self }
    }
}

impl Edge<'_> {
    pub(crate) const fn kind(&self) -> EdgeKind {
        self.kind
    }

    pub(crate) const fn from(&self) -> NodeId {
        self.from
    }

    pub(crate) const fn label(&self) -> EdgeLabel {
        self.label
    }

    pub(crate) const fn to(&self) -> NodeId {
        self.to
    }

    pub(crate) const fn bytes(&self) -> &[u8] {
        self.value
    }
}

/// One edge value to write in [`Graph::set_edge_values`].
#[derive(Clone, Copy, Debug)]
pub struct EdgeValue<'a> {
    half: EdgeRef,
    value: &'a [u8],
}

impl<'a> EdgeValue<'a> {
    /// Writes `value` to the edge `half` belongs to.
    #[must_use]
    pub const fn new(half: EdgeRef, value: &'a [u8]) -> Self {
        Self { half, value }
    }
}

impl EdgeValue<'_> {
    pub(crate) const fn half(&self) -> EdgeRef {
        self.half
    }

    pub(crate) const fn value(&self) -> &[u8] {
        self.value
    }
}
