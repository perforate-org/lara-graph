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
use crate::id::{Direction, Halves, Label, NodeId};
use crate::input::Placement;
use crate::input::{EdgeKind, NewEdge};
use crate::node::label_bucket_list::EntryAt;
use crate::node::record::Record;
use crate::store::memory::Memory;

impl<M: Memory> Graph<M> {
    /// Adds one edge, placed by its placement in both buckets, with its value in both halves: a
    /// [`NewEdge::directed`] edge `from -[label]-> to`, or a [`NewEdge::undirected`] one, which is
    /// normalized by node id. A directed self-loop has two halves; an undirected self-loop has
    /// one. Parallel edges keep their relative order either way. [`Self::add_edges`] takes the
    /// same edges in a batch.
    ///
    /// The value is exactly the label's width ([`Self::declare_width`]) long, or empty for a label
    /// without one.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for either endpoint, [`Error::Detaching`] or [`Error::Deleted`]
    /// for an endpoint being or already deleted, [`Error::WrongValueLength`],
    /// [`Error::TooManyLabelBuckets`], [`Error::OutOfMemory`] or [`Error::AddressSpaceExhausted`].
    /// On error the graph is unchanged.
    pub fn add_edge(&mut self, edge: NewEdge<'_>) -> Result<Halves, Error> {
        add_edge(self, edge)
    }
}

/// Inserts `edge`'s halves, every one or none.
fn add_edge<M: Memory>(graph: &mut Graph<M>, edge: NewEdge<'_>) -> Result<Halves, Error> {
    match edge.kind() {
        EdgeKind::Directed => {
            let ([canonical, other], _) =
                graph.insert_halves_with(Half::directed(&edge), edge.value(), &[None; 2], None)?;
            Ok(Halves::new(canonical, other))
        }
        EdgeKind::Undirected => match Half::undirected(&edge) {
            (at_high, None) => {
                let ([half], _) =
                    graph.insert_halves_with([at_high], edge.value(), &[None], None)?;
                Ok(Halves::new(half, half))
            }
            (at_high, Some(at_low)) => {
                let ([canonical, other], _) =
                    graph.insert_halves_with([at_high, at_low], edge.value(), &[None; 2], None)?;
                Ok(Halves::new(canonical, other))
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
    const fn directed(edge: &NewEdge<'_>) -> [Self; 2] {
        [
            Self::new(
                edge.from(),
                (edge.label(), Direction::Outgoing),
                edge.to(),
                edge.placement(),
            ),
            Self::new(
                edge.to(),
                (edge.label(), Direction::Incoming),
                edge.from(),
                edge.placement(),
            ),
        ]
    }

    /// An undirected edge's halves, normalized by node id: at the high end, then at the low end
    /// unless the edge is a self-loop, which has one half.
    fn undirected(edge: &NewEdge<'_>) -> (Self, Option<Self>) {
        let (low, high) = (edge.from().min(edge.to()), edge.from().max(edge.to()));
        let at_high = Self::new(
            high,
            (edge.label(), Direction::Undirected),
            low,
            edge.placement(),
        );
        if low == high {
            return (at_high, None);
        }
        let at_low = Self::new(
            low,
            (edge.label(), Direction::Undirected),
            high,
            edge.placement(),
        );
        (at_high, Some(at_low))
    }

    const fn new(
        node: NodeId,
        (label, direction): (Label, Direction),
        target: NodeId,
        placement: Placement,
    ) -> Self {
        Self {
            node,
            key: LabelBucketKey { label, direction },
            target,
            placement,
        }
    }
}
