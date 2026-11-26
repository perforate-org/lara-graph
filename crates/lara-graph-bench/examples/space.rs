//! Space a graph takes, memory by memory, for random graphs of several shapes: bytes in all, per
//! node and per edge. Memories grow in 64 KiB pages, so a few megabytes of graph read exactly.
//! Run with `cargo run --release --example space -p lara-graph-bench`.

use core::fmt::Write as _;

use lara_graph::{Edge, EdgeLabel, Graph, Memories, Memory, NodeId, VecMemory, Width};

/// Bytes per page memories grow by.
const WASM_PAGE: u64 = 65536;

/// A xorshift generator, so every run builds the same graphs.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// A graph shape: `nodes` nodes, `edges` directed edges between uniform random endpoints, each
/// with one of `labels` labels, carrying values of `width` bytes.
struct Shape {
    name: &'static str,
    nodes: u64,
    edges: u64,
    labels: u16,
    width: u16,
}

const SHAPES: [Shape; 7] = [
    Shape {
        name: "1 label, degree 1",
        nodes: 100_000,
        edges: 100_000,
        labels: 1,
        width: 0,
    },
    Shape {
        name: "1 label, degree 8",
        nodes: 100_000,
        edges: 800_000,
        labels: 1,
        width: 0,
    },
    Shape {
        name: "1 label, degree 64",
        nodes: 100_000,
        edges: 6_400_000,
        labels: 1,
        width: 0,
    },
    Shape {
        name: "8 labels, degree 8",
        nodes: 100_000,
        edges: 800_000,
        labels: 8,
        width: 0,
    },
    Shape {
        name: "32 labels, degree 8",
        nodes: 100_000,
        edges: 800_000,
        labels: 32,
        width: 0,
    },
    Shape {
        name: "8 labels, degree 8, 8 B values",
        nodes: 100_000,
        edges: 800_000,
        labels: 8,
        width: 8,
    },
    Shape {
        name: "1 label, degree 8, 8 B values",
        nodes: 100_000,
        edges: 800_000,
        labels: 1,
        width: 8,
    },
];

type Failed = Box<dyn core::error::Error>;

fn build(shape: &Shape) -> Result<Graph<VecMemory>, Failed> {
    let mut g = Graph::format(Memories::from_fn(|_| VecMemory::new()));
    let nodes = (0..shape.nodes)
        .map(|_| g.add_node(&[]))
        .collect::<Result<Vec<NodeId>, _>>()?;
    let value = vec![0xAB; usize::from(shape.width)];
    if let Some(width) = Width::new(shape.width) {
        for label in 0..shape.labels {
            g.declare_width(EdgeLabel::new(label), width)?;
        }
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut node = || {
        usize::try_from(rng.below(shape.nodes))
            .ok()
            .and_then(|i| nodes.get(i).copied())
            .ok_or("a node out of range")
    };
    for i in 0..shape.edges {
        let (from, to) = (node()?, node()?);
        let label = EdgeLabel::new(u16::try_from(i % u64::from(shape.labels))?);
        g.add_edge(Edge::directed(from, label, to).value(&value))?;
    }
    Ok(g)
}

/// `n` for a report: graphs here stay well below 4 GB.
fn float(n: u64) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

fn main() -> Result<(), Failed> {
    for shape in &SHAPES {
        let mut total = 0;
        let mut parts = String::new();
        for (role, memory) in build(shape)?.into_memories() {
            let bytes = memory.size() * WASM_PAGE;
            total += bytes;
            write!(parts, " {role:?} {:.1}", float(bytes) / 1e6)?;
        }
        println!(
            "{:<32} {:>7.1} MB {:>6.1} B/node {:>5.1} B/edge |{parts}",
            shape.name,
            float(total) / 1e6,
            float(total) / float(shape.nodes),
            float(total) / float(shape.edges),
        );
    }
    Ok(())
}
