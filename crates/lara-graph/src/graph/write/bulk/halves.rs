//! The halves of a bulk edge list: splitting edges into halves, and positions back into edges.

use crate::bucket::LabelBucketKey;
use crate::edge::BuiltHalf;
use crate::id::Direction;
use crate::id::{EdgeRef, Halves, NodeId, Position};
use crate::input::{Edge, EdgeKind};
use crate::store::corrupt::corrupt;
use alloc::vec::Vec;

/// Marks a half's position that bulk load has not assigned yet.
const UNASSIGNED: u32 = u32::MAX;

/// Which of an edge's two halves a [`Half`] is.
#[derive(Clone, Copy)]
pub(super) enum Side {
    /// The half [`Halves`] names first.
    Canonical,
    /// The other half.
    Other,
}

impl Side {
    /// The half's column in the positions table.
    const fn column(self) -> usize {
        match self {
            Self::Canonical => 0,
            Self::Other => 1,
        }
    }
}

/// One half of an input edge before positions are assigned.
pub(super) struct Half {
    pub(super) node: NodeId,
    pub(super) key: LabelBucketKey,
    pub(super) target: NodeId,
    pub(super) edge: u32,
    side: Side,
}

impl BuiltHalf for Half {
    fn target(&self) -> NodeId {
        self.target
    }
}

impl Half {
    /// The `side` half of input edge `edge`, held by `node` and pointing at `target`.
    const fn new(
        edge: u32,
        side: Side,
        key: LabelBucketKey,
        (node, target): (NodeId, NodeId),
    ) -> Self {
        Self {
            node,
            key,
            target,
            edge,
            side,
        }
    }

    /// The column of the half's position in the positions table.
    const fn column(&self) -> usize {
        self.side.column()
    }

    /// Whether two halves belong to the same label bucket of the same node.
    pub(super) fn same_bucket(&self, other: &Self) -> bool {
        self.node == other.node && self.key == other.key
    }
}

/// Splits `edges` into halves sorted by node and label bucket key.
pub(super) fn split(edges: &[Edge<'_>]) -> Vec<Half> {
    let mut halves: Vec<Half> = Vec::with_capacity(edges.len() * 2);
    for (edge_index, edge) in edges.iter().enumerate() {
        push_halves(
            &mut halves,
            edge,
            u32::try_from(edge_index).unwrap_or(u32::MAX),
        );
    }
    halves.sort_by_key(|half| (half.node, half.key));
    halves
}

/// Pushes the halves of one edge; an undirected self-loop has one.
fn push_halves(halves: &mut Vec<Half>, edge: &Edge<'_>, index: u32) {
    let key = |direction| LabelBucketKey {
        label: edge.label(),
        direction,
    };
    match edge.kind() {
        EdgeKind::Directed => {
            let out = Half::new(
                index,
                Side::Canonical,
                key(Direction::Outgoing),
                (edge.from(), edge.to()),
            );
            let into = Half::new(
                index,
                Side::Other,
                key(Direction::Incoming),
                (edge.to(), edge.from()),
            );
            halves.push(out);
            halves.push(into);
        }
        EdgeKind::Undirected => {
            let (low, high) = (edge.from().min(edge.to()), edge.from().max(edge.to()));
            halves.push(Half::new(
                index,
                Side::Canonical,
                key(Direction::Undirected),
                (high, low),
            ));
            if low != high {
                halves.push(Half::new(
                    index,
                    Side::Other,
                    key(Direction::Undirected),
                    (low, high),
                ));
            }
        }
    }
}

/// Gives each half its position within its bucket, in input order, as a table of
/// `[canonical, other]` per edge.
#[expect(clippy::indexing_slicing, reason = "every half names an input edge")]
pub(super) fn positions(halves: &[Half], edge_count: usize) -> Vec<[u32; 2]> {
    let mut positions = alloc::vec![[UNASSIGNED; 2]; edge_count];
    for bucket in halves.chunk_by(Half::same_bucket) {
        for (offset, half) in bucket.iter().enumerate() {
            let at = u32::try_from(offset).unwrap_or(u32::MAX);
            positions[usize::try_from(half.edge).unwrap_or(0)][half.column()] = at;
        }
    }
    positions
}

/// The [`Halves`] of every edge, from the positions table.
pub(super) fn edge_halves(edges: &[Edge<'_>], positions: &[[u32; 2]]) -> Vec<Halves> {
    let mut out = Vec::with_capacity(edges.len());
    for (edge, &[first, second]) in edges.iter().zip(positions) {
        out.push(edge_refs(edge, first, second));
    }
    out
}

/// The [`Halves`] of one edge; an undirected self-loop's two halves share one position.
fn edge_refs(edge: &Edge<'_>, first: u32, second: u32) -> Halves {
    let label = edge.label();
    match edge.kind() {
        EdgeKind::Directed => Halves::new(
            EdgeRef::of(edge.from(), label, Direction::Outgoing, canonical(first)),
            EdgeRef::of(edge.to(), label, Direction::Incoming, other(second)),
        ),
        EdgeKind::Undirected => {
            let (low, high) = (edge.from().min(edge.to()), edge.from().max(edge.to()));
            let second = if low == high { first } else { second };
            Halves::new(
                EdgeRef::of(high, label, Direction::Undirected, canonical(first)),
                EdgeRef::of(low, label, Direction::Undirected, other(second)),
            )
        }
    }
}

fn canonical(at: u32) -> Position {
    assigned(at, "a bulk edge without its canonical half")
}

fn other(at: u32) -> Position {
    assigned(at, "a bulk edge without its other half")
}

fn assigned(at: u32, missing: &str) -> Position {
    if at == UNASSIGNED {
        corrupt(missing);
    }
    Position::new(at)
}
