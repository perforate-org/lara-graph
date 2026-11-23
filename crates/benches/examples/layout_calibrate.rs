//! Calibration for `layout_compare`'s leaf model: the same streams through the real
//! `lara-graph`, reporting its rebalance count and slab footprint.
//!
//! Usage: cargo run -p benches --release --example layout_calibrate

use benches::{coo_power_law_stream, coo_uniform_stream};
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::memory::VecMemory;

fn run(workload: &str, n: u32, edges: &[(u32, u32)], segment: u32, max_log: usize) {
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
    let mut g: LaraGraph<Vertex, Edge, VecMemory> =
        LaraGraph::new_with_segments_and_max_log(mem, n, segment, max_log).expect("init");
    g.set_promote_threshold(Some(lara_graph::graph::T_PROMOTE));
    g.set_use_free_spans(true);
    for &(u, v) in edges {
        g.insert_edge(u, v);
    }
    println!(
        "| {workload} | lara-graph segment={} log={max_log} | {} | {:.2} |",
        g.segment_size(),
        g.rebalance_count(),
        g.slab_capacity() as f64 / g.num_edges() as f64,
    );
}

fn main() {
    println!("| workload | system | rebalances | slab capacity / all edges |");
    println!("|---|---|---|---|");
    let n = 1_000_000;
    let m = 10_000_000;
    let uniform = coo_uniform_stream(n, m, 7);
    let power = coo_power_law_stream(n, m, 99, 1.0);
    for segment in [16, 32] {
        for max_log in [0, 170] {
            run("uniform 1M/10M", n, &uniform, segment, max_log);
            run("power-law 1M/10M", n, &power, segment, max_log);
        }
    }
}
