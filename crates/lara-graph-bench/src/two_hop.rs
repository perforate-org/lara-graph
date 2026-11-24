//! A two-hop traversal from a hub, built in layouts that differ only in where the second hop's
//! data lands: the measure of locality across nodes.
//!
//! The hub has an edge to each of [`SATELLITES`] satellites; each satellite has [`FANOUT`] edges
//! of its own (a run bucket). The traversal visits the hub's neighbours, then each neighbour's.

use core::ops::ControlFlow;
use lara_graph::{
    Direction, Graph, Label, Memory, NewEdge, NodeId, NodeState, Order, Placement, ReadBatch,
};

/// Satellites around the hub: the first hop.
pub const SATELLITES: u32 = 1024;
/// Edges of each satellite: the second hop.
pub const FANOUT: u32 = 8;
/// Nodes the scattered layout draws its satellites from.
const POOL: u32 = 65536;
const LABEL: Label = Label::new(2);

/// Where the second hop's data lands relative to the order the traversal reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Satellites have consecutive ids, and their buckets are built in traversal order.
    Ordered,
    /// Satellites have consecutive ids, but their buckets are built in a shuffled order.
    Shuffled,
    /// Satellites are random ids among [`POOL`] nodes, built in a shuffled order.
    Scattered,
}

impl Layout {
    /// Every layout, in the order reports list them.
    pub const ALL: [Self; 3] = [Self::Ordered, Self::Shuffled, Self::Scattered];

    /// A short name for reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ordered => "ordered",
            Self::Shuffled => "shuffled",
            Self::Scattered => "scattered",
        }
    }
}

/// splitmix64: deterministic shuffles without a dependency.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let n = u64::try_from(n).unwrap_or(u64::MAX).max(1);
        usize::try_from((z ^ (z >> 31)) % n).unwrap_or(0)
    }

    /// Fisher-Yates.
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }
}

fn node<M: Memory>(g: &mut Graph<M>) -> NodeId {
    g.add_node().unwrap_or_else(|e| panic!("node: {e}"))
}

fn edge<M: Memory>(g: &mut Graph<M>, from: NodeId, to: NodeId) {
    g.add_edge(NewEdge::directed(from, LABEL, to, Placement::Append, &[]))
        .unwrap_or_else(|e| panic!("insert: {e}"));
}

/// Builds the graph in `g` (empty) in `layout`; returns the hub.
///
/// # Panics
///
/// If the graph runs out of memory.
pub fn build<M: Memory>(g: &mut Graph<M>, layout: Layout) -> NodeId {
    let mut rng = Rng(7);
    let hub = node(g);
    let pool = match layout {
        Layout::Ordered | Layout::Shuffled => SATELLITES,
        Layout::Scattered => POOL,
    };
    let ids: Vec<NodeId> = (0..pool).map(|_| node(g)).collect();
    let mut satellites = ids;
    if layout == Layout::Scattered {
        rng.shuffle(&mut satellites);
    }
    satellites.truncate(usize::try_from(SATELLITES).unwrap_or(0));
    for &s in &satellites {
        edge(g, hub, s);
    }
    // The second hop's targets: fresh nodes, so only the satellites' out buckets are read.
    let targets: Vec<NodeId> = (0..FANOUT).map(|_| node(g)).collect();
    let mut build_order = satellites;
    if layout != Layout::Ordered {
        rng.shuffle(&mut build_order);
    }
    for &s in &build_order {
        for &t in &targets {
            edge(g, s, t);
        }
    }
    hub
}

/// Visits the hub's neighbours and each neighbour's; returns the second hop's edge count.
///
/// # Panics
///
/// If `hub` is not a node of `g`.
pub fn visit<M: Memory>(g: &Graph<M>, hub: NodeId) -> u64 {
    let mut seen = 0u64;
    let first = g.visit_neighbors(hub, LABEL, Direction::Outgoing, Order::Ascending, |e| {
        let second = g.visit_neighbors(
            e.target(),
            LABEL,
            Direction::Outgoing,
            Order::Ascending,
            |_| {
                seen += 1;
                ControlFlow::<()>::Continue(())
            },
        );
        second.unwrap_or_else(|err| panic!("second hop: {err}"));
        ControlFlow::<()>::Continue(())
    });
    first.unwrap_or_else(|err| panic!("first hop: {err}"));
    seen
}

/// The hub's neighbours: the satellites, in traversal order.
///
/// # Panics
///
/// If `hub` is not a node of `g`.
pub fn satellites<M: Memory>(g: &Graph<M>, hub: NodeId) -> Vec<NodeId> {
    let mut out = Vec::new();
    g.visit_neighbors(hub, LABEL, Direction::Outgoing, Order::Ascending, |e| {
        out.push(e.target());
        ControlFlow::<()>::Continue(())
    })
    .unwrap_or_else(|err| panic!("satellites: {err}"));
    out
}

/// Visits `v`'s edges in `direction`; returns how many it saw.
///
/// # Panics
///
/// If `v` is not a node of `g`.
pub fn visit_one<M: Memory>(g: &Graph<M>, v: NodeId, direction: Direction) -> u64 {
    let mut seen = 0u64;
    g.visit_neighbors(v, LABEL, direction, Order::Ascending, |_| {
        seen += 1;
        ControlFlow::<()>::Continue(())
    })
    .unwrap_or_else(|err| panic!("visit: {err}"));
    seen
}

/// `v`'s degree in `direction`.
///
/// # Panics
///
/// If `v` is not a node of `g`.
pub fn degree<M: Memory>(g: &Graph<M>, v: NodeId, direction: Direction) -> u32 {
    g.degree(v, LABEL, direction)
        .unwrap_or_else(|err| panic!("degree: {err}"))
}

/// Visits the `direction` bucket of every node in `nodes` in one batch; returns the edge count.
///
/// # Panics
///
/// If a node is not in `g`.
pub fn visit_many<M: Memory>(g: &Graph<M>, nodes: &[NodeId], direction: Direction) -> u64 {
    let mut seen = 0u64;
    let mut batch = g.read_batch();
    batch
        .visit_neighbors_each(nodes, LABEL, direction, Order::Ascending, |_, _| {
            seen += 1;
            ControlFlow::<()>::Continue(())
        })
        .unwrap_or_else(|err| panic!("batch: {err}"));
    seen
}

/// The same batch through a caller-held [`ReadBatch`].
///
/// # Panics
///
/// If a node is not in `g`.
pub fn visit_many_batch<M: Memory>(
    batch: &mut ReadBatch<'_, M>,
    nodes: &[NodeId],
    direction: Direction,
) -> u64 {
    let mut seen = 0u64;
    batch
        .visit_neighbors_each(nodes, LABEL, direction, Order::Ascending, |_, _| {
            seen += 1;
            ControlFlow::<()>::Continue(())
        })
        .unwrap_or_else(|err| panic!("batch: {err}"));
    seen
}

/// Sums the `direction` degree of every node in `nodes` through one batch.
///
/// # Panics
///
/// If a node is not in `g`.
pub fn degree_many<M: Memory>(g: &Graph<M>, nodes: &[NodeId], direction: Direction) -> u64 {
    let mut batch = g.read_batch();
    let mut degrees = Vec::new();
    batch
        .degree_each(nodes, LABEL, direction, &mut degrees)
        .unwrap_or_else(|err| panic!("degree batch: {err}"));
    degrees.into_iter().map(u64::from).sum()
}

/// Counts the live nodes in `nodes` through one batch.
///
/// # Panics
///
/// If a node is not in `g`.
pub fn node_state_many<M: Memory>(g: &Graph<M>, nodes: &[NodeId]) -> u64 {
    let mut batch = g.read_batch();
    let mut states = Vec::new();
    batch
        .node_state_each(nodes, &mut states)
        .unwrap_or_else(|err| panic!("state batch: {err}"));
    u64::try_from(
        states
            .iter()
            .filter(|state| **state == NodeState::Live)
            .count(),
    )
    .unwrap_or(u64::MAX)
}

/// Counts the `direction` labels of every node in `nodes` through one batch.
///
/// # Panics
///
/// If a node is not in `g`.
pub fn labels_many<M: Memory>(g: &Graph<M>, nodes: &[NodeId], direction: Direction) -> u64 {
    let mut batch = g.read_batch();
    let mut seen = 0u64;
    batch
        .labels_each(nodes, direction, |_, _| {
            seen += 1;
            ControlFlow::<()>::Continue(())
        })
        .unwrap_or_else(|err| panic!("labels batch: {err}"));
    seen
}

/// Counts the nodes in `nodes` that have a live edge, through one batch.
///
/// # Panics
///
/// If a node is not in `g`.
pub fn has_edges_many<M: Memory>(g: &Graph<M>, nodes: &[NodeId]) -> u64 {
    let mut batch = g.read_batch();
    let mut seen = 0u64;
    batch
        .has_edges_each(nodes, |_, has| {
            if has {
                seen += 1;
            }
            ControlFlow::<()>::Continue(())
        })
        .unwrap_or_else(|err| panic!("has_edges batch: {err}"));
    seen
}
