//! Space under a long run of updates: a sliding window of random directed edges, each step adding
//! one and removing the oldest, until the graph has turned over several times. Prints, at each
//! checkpoint, the memories' size (what stable memory would bill: it never shrinks), the holes
//! buckets keep, and at the end the size of a graph built fresh from the same live edges, so the
//! difference is what the updates left behind.
//!
//! Policies: holes refilled by inserts (`ReuseHoles`) or not (`Append`), and optionally a bucket
//! compacted once its holes reach its live edges.
//! Run with `cargo run --release --example churn -p lara-graph-bench`.

extern crate alloc;

use alloc::collections::VecDeque;
use core::fmt::Write as _;
use core::ops::ControlFlow;

use lara_graph::{
    Adjacency, Direction, Edge, EdgeLabel, Graph, Memories, Memory, MemoryRole, NodeId, Order,
    Placement, Progress, Removal, VecMemory,
};

type Failed = Box<dyn core::error::Error>;

/// Bytes per page memories grow by.
const WASM_PAGE: u64 = 65536;
const NODES: u32 = 100_000;
/// Live edges once the window is full: degree 8.
const WINDOW: usize = 800_000;
/// How many times the window turns over.
const TURNS: usize = 4;
/// Checkpoints per turn.
const PER_TURN: usize = 2;

/// A xorshift generator, so every run makes the same updates.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// When buckets are compacted.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Compaction {
    Never,
    /// Once a bucket's holes reach its live edges.
    AtHalfHoles,
}

#[derive(Clone, Copy)]
struct Policy {
    name: &'static str,
    labels: u16,
    placement: Placement,
    compact: Compaction,
}

const POLICIES: [Policy; 6] = [
    Policy {
        name: "1 label, reuse holes",
        labels: 1,
        placement: Placement::ReuseHoles,
        compact: Compaction::Never,
    },
    Policy {
        name: "1 label, append",
        labels: 1,
        placement: Placement::Append,
        compact: Compaction::Never,
    },
    Policy {
        name: "1 label, append + compact",
        labels: 1,
        placement: Placement::Append,
        compact: Compaction::AtHalfHoles,
    },
    Policy {
        name: "1 label, reuse holes + compact",
        labels: 1,
        placement: Placement::ReuseHoles,
        compact: Compaction::AtHalfHoles,
    },
    Policy {
        name: "8 labels, reuse holes",
        labels: 8,
        placement: Placement::ReuseHoles,
        compact: Compaction::Never,
    },
    Policy {
        name: "8 labels, reuse holes + compact",
        labels: 8,
        placement: Placement::ReuseHoles,
        compact: Compaction::AtHalfHoles,
    },
];

/// A graph over six heap memories, with handles kept to read their sizes as it runs.
struct Sized {
    graph: Graph<VecMemory>,
    memories: Vec<VecMemory>,
}

impl Sized {
    fn new() -> Self {
        let memories: Vec<VecMemory> = MemoryRole::ALL.iter().map(|_| VecMemory::new()).collect();
        let shared = memories.clone();
        let graph = Graph::format(Memories::from_fn(|role| {
            shared
                .get(usize::from(role.index()))
                .cloned()
                .unwrap_or_default()
        }));
        Self { graph, memories }
    }

    fn bytes(&self) -> u64 {
        self.memories.iter().map(|m| m.size() * WASM_PAGE).sum()
    }
}

type Link = (NodeId, EdgeLabel, NodeId);

fn add(
    g: &mut Graph<VecMemory>,
    (from, label, to): Link,
    placement: Placement,
) -> Result<(), Failed> {
    g.add_edge_with(Edge::directed(from, label, to), placement)?;
    Ok(())
}

/// Removes one edge `from -[label]-> to`, found from whichever end is cheaper; with `compact`,
/// compacts each end's bucket once its holes reach its live edges.
fn remove(
    g: &mut Graph<VecMemory>,
    (from, label, to): Link,
    compact: Compaction,
) -> Result<(), Failed> {
    let half = g.visit_edges_between(
        from,
        label.outgoing(),
        to,
        Order::Ascending,
        ControlFlow::Break,
    )?;
    let half = half.ok_or("a live edge of the window is missing")?;
    let Removal::Removed(edge) = g.remove_edge(half)? else {
        return Err("a found edge was not removed".into());
    };
    if compact == Compaction::Never {
        return Ok(());
    }
    let ends = [
        (from, Direction::Outgoing, edge.canonical_fill()),
        (to, Direction::Incoming, edge.other_fill()),
    ];
    for (v, direction, _) in ends
        .into_iter()
        .filter(|(_, _, fill)| fill.holes() >= fill.live().max(1))
    {
        while g.compact(v, Adjacency::new(label, direction))?.progress() == Progress::Remaining {}
    }
    Ok(())
}

/// Holes all buckets keep.
fn holes(g: &Graph<VecMemory>, labels: u16) -> Result<u64, Failed> {
    let mut sum = 0;
    for v in (0..NODES).map(NodeId::new) {
        for label in (0..labels).map(EdgeLabel::new) {
            for direction in [Direction::Outgoing, Direction::Incoming] {
                sum += u64::from(g.holes(v, Adjacency::new(label, direction))?);
            }
        }
    }
    Ok(sum)
}

/// `n` for a report: graphs here stay well below 4 GB.
fn float(n: u64) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

fn run(policy: Policy) -> Result<(), Failed> {
    let mut sized = Sized::new();
    for _ in 0..NODES {
        sized.graph.add_node(&[])?;
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let next = |rng: &mut Rng| -> Result<Link, Failed> {
        let node =
            |rng: &mut Rng| NodeId::new(u32::try_from(rng.below(u64::from(NODES))).unwrap_or(0));
        let label = EdgeLabel::new(u16::try_from(rng.below(u64::from(policy.labels)))?);
        Ok((node(rng), label, node(rng)))
    };
    let mut window = VecDeque::with_capacity(WINDOW);
    for _ in 0..WINDOW {
        let edge = next(&mut rng)?;
        add(&mut sized.graph, edge, policy.placement)?;
        window.push_back(edge);
    }
    println!("{}", policy.name);
    let report = |sized: &Sized, turn: f64| -> Result<(), Failed> {
        let bytes = sized.bytes();
        println!(
            "  turn {turn:>4.1}: {:>6.1} MB  {:>5.1} B/edge  holes {:>8}",
            float(bytes) / 1e6,
            float(bytes) / float(u64::try_from(WINDOW)?),
            holes(&sized.graph, policy.labels)?,
        );
        Ok(())
    };
    report(&sized, 0.0)?;
    let step = WINDOW / PER_TURN;
    for checkpoint in 1..=TURNS * PER_TURN {
        for _ in 0..step {
            let edge = next(&mut rng)?;
            add(&mut sized.graph, edge, policy.placement)?;
            window.push_back(edge);
            let old = window.pop_front().ok_or("empty window")?;
            remove(&mut sized.graph, old, policy.compact)?;
        }
        report(
            &sized,
            float(u64::try_from(checkpoint)?) / float(u64::try_from(PER_TURN)?),
        )?;
    }
    let mut fresh = Sized::new();
    for _ in 0..NODES {
        fresh.graph.add_node(&[])?;
    }
    for &edge in &window {
        add(&mut fresh.graph, edge, policy.placement)?;
    }
    let mut line = String::new();
    write!(
        line,
        "  fresh build of the same edges: {:.1} MB ({:.2}x)",
        float(fresh.bytes()) / 1e6,
        float(sized.bytes()) / float(fresh.bytes())
    )?;
    println!("{line}\n");
    Ok(())
}

fn main() -> Result<(), Failed> {
    for policy in POLICIES {
        run(policy)?;
    }
    Ok(())
}
