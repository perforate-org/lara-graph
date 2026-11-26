//! Inserting edges: every half or none, each with its value when the label has a width.
//!
//! [`admit`] checks an insert before anything changes, [`apply`] plans, reserves and writes its
//! halves, [`batch`] runs [`Graph::add_edges`] over the caches in [`cache`].

mod admit;
mod apply;
mod batch;
mod cache;

use crate::bucket::LabelBucketKey;
use crate::bucket::stored::Stored;
use crate::error::Error;
use crate::graph::Graph;
use crate::graph::GraphMemories;
use crate::id::{Adjacency, Halves, NodeId};
use crate::input::{Edge, EdgeKind, Placement};
use crate::node::label_bucket_list::EntryAt;
use crate::node::record::Record;

impl<S: GraphMemories> Graph<S> {
    /// Adds one edge, with its value in both halves: a [`Edge::directed`] edge
    /// `from -[label]-> to`, or a [`Edge::undirected`] one, which is normalized by node id. A
    /// directed self-loop has two halves; an undirected self-loop has one. Each half goes into the
    /// first hole its bucket can reuse, else after every position ([`Placement::ReuseHoles`]);
    /// [`Self::add_edge_with`] chooses otherwise, and [`Self::add_edges`] takes edges in a batch.
    /// Parallel edges keep their relative order either way.
    ///
    /// The value is exactly the label's width ([`Self::declare_width`]) long, or empty for a label
    /// without one.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for either endpoint, [`Error::Deleting`] or [`Error::Deleted`]
    /// for an endpoint being or already deleted, [`Error::WrongValueLength`],
    /// [`Error::TooManyLabelBuckets`], [`Error::OutOfMemory`] or [`Error::AddressSpaceExhausted`].
    /// On error the graph is unchanged.
    pub fn add_edge(&mut self, edge: Edge<'_>) -> Result<Halves, Error> {
        add_edge_reusing_holes(self, edge)
    }

    /// [`Self::add_edge`], placing each half by `placement`: [`Placement::Append`] keeps position
    /// order the order of insertion.
    ///
    /// # Errors
    ///
    /// As [`Self::add_edge`].
    pub fn add_edge_with(&mut self, edge: Edge<'_>, placement: Placement) -> Result<Halves, Error> {
        add_edge(self, edge, placement)
    }
}

/// [`Graph::add_edge`]: [`add_edge`] by the default placement.
fn add_edge_reusing_holes<S: GraphMemories>(
    graph: &mut Graph<S>,
    edge: Edge<'_>,
) -> Result<Halves, Error> {
    add_edge(graph, edge, Placement::ReuseHoles)
}

/// Inserts `edge`'s halves, every one or none.
fn add_edge<S: GraphMemories>(
    graph: &mut Graph<S>,
    edge: Edge<'_>,
    placement: Placement,
) -> Result<Halves, Error> {
    match edge.kind() {
        EdgeKind::Directed => {
            let ([first, second], _) = graph.insert_halves_with(
                Half::directed(&edge, placement),
                edge.bytes(),
                &[None; 2],
                None,
            )?;
            Ok(Halves::of(first, second))
        }
        EdgeKind::Undirected => match Half::undirected(&edge, placement) {
            (at_high, None) => {
                let ([half], _) =
                    graph.insert_halves_with([at_high], edge.bytes(), &[None], None)?;
                Ok(Halves::new(half, half))
            }
            (at_high, Some(at_low)) => {
                let ([first, second], _) =
                    graph.insert_halves_with([at_high, at_low], edge.bytes(), &[None; 2], None)?;
                Ok(Halves::of(first, second))
            }
        },
    }
}

/// Where a half goes: its node's record and its bucket, or where the bucket belongs.
#[derive(Clone, Copy)]
struct Place {
    /// The node's record when the half was admitted.
    record: Record,
    target: Target,
}

#[derive(Clone, Copy)]
enum Target {
    /// The bucket's list entry, resolved only where its header matters.
    Present(EntryAt, Stored),
    Absent(EntryAt),
}

/// One half to insert: an entry to `target` in `node`'s bucket `key`.
#[derive(Clone, Copy)]
struct Half {
    node: NodeId,
    key: LabelBucketKey,
    target: NodeId,
    placement: Placement,
}

impl Half {
    /// A directed edge's halves: outgoing at `from`, then incoming at `to`.
    const fn directed(edge: &Edge<'_>, placement: Placement) -> [Self; 2] {
        [
            Self::new(edge.from(), edge.label().outgoing(), edge.to(), placement),
            Self::new(edge.to(), edge.label().incoming(), edge.from(), placement),
        ]
    }

    /// An undirected edge's halves, normalized by node id: at the high end, then at the low end
    /// unless the edge is a self-loop, which has one half.
    fn undirected(edge: &Edge<'_>, placement: Placement) -> (Self, Option<Self>) {
        let (low, high) = (edge.from().min(edge.to()), edge.from().max(edge.to()));
        let at_high = Self::new(high, edge.label().undirected(), low, placement);
        if low == high {
            return (at_high, None);
        }
        let at_low = Self::new(low, edge.label().undirected(), high, placement);
        (at_high, Some(at_low))
    }

    const fn new(node: NodeId, adjacency: Adjacency, target: NodeId, placement: Placement) -> Self {
        let (label, direction) = (adjacency.label(), adjacency.direction());
        Self {
            node,
            key: LabelBucketKey { label, direction },
            target,
            placement,
        }
    }
}
