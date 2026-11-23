//! Auxiliary tiny K probe (K=2/4/8). Rebuild per K value.
//! Reports a partial storage estimate, not total heap or resident memory.
//! Run with `cargo test -p benches --test tiny_k_probe -- --nocapture`.

use benches::coo_uniform_stream;
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::memory::VecMemory;
use std::collections::HashSet;
use std::time::Instant;

fn fresh_mem() -> LaraMemories<VecMemory> {
    LaraMemories {
        vertices: VecMemory::new(),
        edges: VecMemory::new(),
        logs: VecMemory::new(),
        counts: VecMemory::new(),
        spans: VecMemory::new(),
        free_spans: VecMemory::new(),
        blocks: VecMemory::new(),
        buckets: VecMemory::new(),
    }
}

fn report(tag: &str, g: &LaraGraph<Vertex, Edge, VecMemory>, insert_ms: f64, touched: u64) {
    let ne = g.num_edges();
    let total = g.memory_bytes_total();
    let tiny = g.tiny_memory_bytes();
    let logs = g.log_memory_bytes() - tiny;
    let slab = g.slab_capacity() * 4;
    let vertex = g.num_nodes() * 24;
    let tree = total.saturating_sub(slab + logs + tiny + vertex);
    let pe = |b: u64| b as f64 / ne as f64;
    let per_e = total as f64 / ne as f64;
    let tiny_rows = g.tiny_row_count();
    // Residency is not a spill count: folding may retire tiny rows and
    // rows in an already positioned leaf may never have used tiny storage.
    let non_tiny_touched = touched - tiny_rows;
    println!(
        "{tag}: insert={insert_ms:.1}ms state=after_insert partial_estimate={per_e:.1}B/e [slab={:.1} tree={:.1} logs={:.1} tiny={:.1} vtx={:.1}] tiny_rows={tiny_rows} non_tiny_touched={non_tiny_touched} pool={} log_chunks={} rb={} pr={}",
        pe(slab),
        pe(tree),
        pe(logs),
        pe(tiny),
        pe(vertex),
        g.tiny_pool_depth(),
        g.log_allocated_chunks(),
        g.rebalance_count(),
        g.promote_count(),
    );
}

fn build(stream: &[(u32, u32)], n: u32) -> (LaraGraph<Vertex, Edge, VecMemory>, f64, u64) {
    let touched = stream.iter().map(|&(a, _)| a).collect::<HashSet<_>>().len() as u64;
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), n).unwrap();
    let t0 = Instant::now();
    for &(a, b) in stream {
        g.insert_edge(a, b);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(g.num_edges(), stream.len() as u64);
    g.verify_invariants();
    (g, ms, touched)
}

#[test]
fn tiny_k_probe() {
    // 1. uniform strided n=10k/m=40k, degree exactly 4 (same as tiny_memory).
    let uniform: Vec<(u32, u32)> = (0..40_000u32)
        .map(|i| ((i * 7 + 3) % 10_000, (i * 13 + 5) % 10_000))
        .collect();
    let (g, ms, touched) = build(&uniform, 10_000);
    report("uniform10k", &g, ms, touched);

    // 2. hub-10k: single hot row, must fully spill at every K.
    let hub: Vec<(u32, u32)> = (0..10_000u32).map(|d| (1, d)).collect();
    let (g, ms, touched) = build(&hub, 20_480);
    report("hub10k", &g, ms, touched);

    // 3. dispersed uniform-random n=10k/m=40k (~Poisson(4) degrees).
    // Uniform random isolates the low-degree mass relevant to tiny K.
    // Rank-Zipf is a different source model, not a substitute for this one.
    let disp = coo_uniform_stream(10_000, 40_000, 1234);
    let (g, ms, touched) = build(&disp, 10_000);
    report("dispersed10k", &g, ms, touched);
}
