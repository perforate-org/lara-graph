//! Quickstart: build a graph, insert, remove, and scan.
//!
//! Run with: `cargo run --example quickstart -p lara-graph`

use lara_graph::graph::{LaraGraph, LaraMemories};
use lara_graph::memory::VecMemory;

fn main() {
    // One heap-backed memory per region (vertices, edges, logs, counts,
    // spans, free spans, blocks).
    let mem = LaraMemories {
        vertices: VecMemory::new(),
        edges: VecMemory::new(),
        logs: VecMemory::new(),
        counts: VecMemory::new(),
        spans: VecMemory::new(),
        free_spans: VecMemory::new(),
        blocks: VecMemory::new(),
        buckets: VecMemory::new(),
    };

    let mut g: LaraGraph<lara_graph::graph::Vertex, lara_graph::graph::Edge, VecMemory> =
        LaraGraph::new(mem, 4).unwrap();

    g.insert_edge(0, 1);
    g.insert_edge(0, 2);
    g.insert_edge(1, 3);
    g.remove_edge(0, 2); // logical delete (tombstone)

    for v in 0..4 {
        println!("neighbors of {v}: {:?}", g.out_neighbors_vec(v));
    }
    println!("edges: {}", g.num_edges());
}
