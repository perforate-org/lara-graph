//! Insert-breakdown probe (measurement only, lib untouched).
//! Run with `-- --nocapture`. Single-shot Instant values (trend use;
//! test profile is unoptimized so absolute ns are inflated — ratios matter).
use benches::{LaraAdapter, LaraConfig, coo_uniform_stream};
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::memory::VecMemory;
use lara_graph::traits::{CsrEdge, DynamicCsr};
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

fn bench_reps<T>(reps: u32, mut f: impl FnMut() -> T) -> f64 {
    let t0 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(f());
    }
    t0.elapsed().as_secs_f64() * 1e9 / reps as f64
}

#[test]
fn insert_probe() {
    let cfg = LaraConfig::default();
    // A. Uniform dispersed per-edge (rb-rare baseline: direct write + checks).
    let stream = coo_uniform_stream(10_000, 40_000, 1234);
    let t0 = Instant::now();
    let mut u = LaraAdapter::new(10_000, cfg).unwrap();
    for &(a, b) in &stream {
        u.insert_edge(a, b);
    }
    let u_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "uniform10k/insert: total={u_ms:.1}ms per_insert={:.0}ns rb={} pr={}",
        u_ms * 1e6 / 40_000.0,
        u.rebalance_count(),
        u.promote_count()
    );
    // F. Bulk build_pairs vs per-edge on the same stream.
    let t0 = Instant::now();
    let bulk = LaraAdapter::with_edges(10_000, &stream, cfg).unwrap();
    let bulk_ms = t0.elapsed().as_secs_f64() * 1000.0;
    std::hint::black_box(bulk.num_edges());
    println!(
        "uniform10k/bulk_vs_peredge: bulk={bulk_ms:.1}ms peredge={u_ms:.1}ms ratio={:.2}x",
        u_ms / bulk_ms
    );
    // B. Hub-10k growth (overflow + tombstone scans + rebalance + promote).
    let t0 = Instant::now();
    let mut h = LaraAdapter::new(20_480, cfg).unwrap();
    for t in 0..10_000u32 {
        h.insert_edge(1, t);
    }
    let h_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (hrb, hpr) = (h.rebalance_count(), h.promote_count());
    println!(
        "hub10k/insert: total={h_ms:.1}ms per_insert={:.0}ns rb={hrb} pr={hpr}",
        h_ms * 1e6 / 10_000.0
    );
    // C. Exact slab-room query, not insertion cost: dense prefixes make
    // room a metadata-only query. Disabling promotion also disables the
    // allocation cap, so this fixture may have extra tail room.
    {
        let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 4096).unwrap();
        g.set_promote_threshold(None);
        for d in 0..1500u32 {
            g.insert_edge(1, d);
        }
        let row = g.get_row_pub(1);
        use lara_graph::traits::CsrVertex as _;
        println!(
            "tombscan/row: live={} stored={} tree={} log_head={} rb={}",
            row.degree(),
            row.stored_degree(),
            row.tree_bucket(),
            row.log_head(),
            g.rebalance_count()
        );
        let room = g.row_has_slab_room(1);
        let ns = bench_reps(2000, || g.row_has_slab_room(1));
        println!(
            "tombscan/full_row stored={}: room={room} {ns:.0}ns total, {:.2}ns/slot",
            row.stored_degree(),
            ns / row.stored_degree().max(1) as f64
        );
    }
    // G. Row/edit micro-costs: get_row+set_row roundtrip, single edge write,
    // and post-promote tree_insert-only rate (hub row already tree).
    {
        let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 4096).unwrap();
        let ns = bench_reps(2000, || {
            let r = g.get_row_pub(1);
            g.set_row_pub(1, r);
        });
        println!("micro/row_read_write_roundtrip: {ns:.0}ns");
        let e = Edge::default().with_neighbor(7);
        let ns = bench_reps(2000, || g.write_edge_pub(0, &e));
        println!("micro/edge_write_slot0: {ns:.0}ns");
        let mut h: LaraGraph<Vertex, Edge, VecMemory> =
            LaraGraph::new(fresh_mem(), 20_480).unwrap();
        for t in 0..1500u32 {
            h.insert_edge(1, t);
        }
        assert!(h.promote_count() >= 1);
        let t0 = Instant::now();
        for t in 1500..6500u32 {
            h.insert_edge(1, t);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "treeonly/5000_postpromote: {ms:.2}ms per_insert={:.0}ns",
            ms * 1e6 / 5000.0
        );
    }
    // D. Density-check raw cost (f64 div + compare, no lib).
    let ns = bench_reps(1000, || {
        let mut x = 0u64;
        for i in 1..1024u64 {
            if i as f64 / 2048.0 >= 1.0 {
                x += i;
            }
        }
        std::hint::black_box(x);
    });
    println!(
        "densitycheck/raw_1024divs: {ns:.0}ns total, {:.2}ns/div",
        ns / 1024.0
    );
    // E. Single-rebalance cost on a representative leaf (uniform leaf state).
    let mut g2: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 10_000).unwrap();
    for &(a, b) in &stream {
        g2.insert_edge(a, b);
    }
    let rb0 = g2.rebalance_count();
    let t0 = Instant::now();
    g2.rebalance_leaf(0);
    let rb_us = t0.elapsed().as_secs_f64() * 1e6;
    println!("rebalance/single_leaf0: {rb_us:.1}us (prior rb count={rb0})");
    // Hub-leaf rebalance (dense leaf) for the hub amortization bound.
    let mut g3: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 20_480).unwrap();
    for t in 0..900u32 {
        g3.insert_edge(1, t);
    }
    let t0 = Instant::now();
    g3.rebalance_leaf(0);
    let hrb_us = t0.elapsed().as_secs_f64() * 1e6;
    println!("rebalance/hub_leaf_pre_promote: {hrb_us:.1}us");
}
