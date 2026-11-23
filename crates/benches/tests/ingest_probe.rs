//! Ingest vs per-edge probe (measurement only). `-- --nocapture`.
use benches::{LaraAdapter, LaraConfig, coo_uniform_stream};
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::ingest::IngestBuffer;
use lara_graph::memory::VecMemory;
use lara_graph::traits::DynamicCsr;
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
#[test]
fn ingest_probe() {
    let stream = coo_uniform_stream(10_000, 40_000, 1234);
    let cfg = LaraConfig::default();
    let t0 = Instant::now();
    let mut p = LaraAdapter::new(10_000, cfg).unwrap();
    for &(a, b) in &stream {
        p.insert_edge(a, b);
    }
    let per_ms = t0.elapsed().as_secs_f64() * 1000.0;
    // Ingest B=4096 with sort/flush split.
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 10_000).unwrap();
    let mut sort_ms = 0.0;
    let mut flush_ms = 0.0;
    let mut ing = IngestBuffer::new(10_000, 4096);
    let t_all = Instant::now();
    let mut pending: Vec<(u32, u32)> = Vec::new();
    for &(a, b) in &stream {
        ing.push(a, b);
        pending.push((a, b));
        if ing.len() >= 4096 {
            let t1 = Instant::now();
            pending.sort_unstable();
            sort_ms += t1.elapsed().as_secs_f64() * 1000.0;
            let t2 = Instant::now();
            ing.flush(&mut g);
            flush_ms += t2.elapsed().as_secs_f64() * 1000.0;
            pending.clear();
        }
    }
    if !pending.is_empty() {
        let t1 = Instant::now();
        pending.sort_unstable();
        sort_ms += t1.elapsed().as_secs_f64() * 1000.0;
        let t2 = Instant::now();
        ing.flush(&mut g);
        flush_ms += t2.elapsed().as_secs_f64() * 1000.0;
    }
    let all_ms = t_all.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(g.num_edges(), 40_000);
    println!(
        "uniform10k/ingest4096: peredge={per_ms:.1}ms ({:.0}ns/e) ingest={all_ms:.1}ms ({:.0}ns/e) ratio={:.2}x sort={sort_ms:.1}ms ({:.1}%) flush-write={flush_ms:.1}ms",
        per_ms * 1e6 / 40_000.0,
        all_ms * 1e6 / 40_000.0,
        per_ms / all_ms,
        sort_ms / all_ms * 100.0
    );
    // Hub-10k ingest for skew reference.
    let t0 = Instant::now();
    let mut h: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 20_480).unwrap();
    let mut ing2 = IngestBuffer::new(20_480, 4096);
    for t in 0..10_000u32 {
        ing2.push(1, t);
        if ing2.len() >= 4096 {
            ing2.flush(&mut h);
        }
    }
    ing2.flush(&mut h);
    let hub_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "hub10k/ingest4096: {hub_ms:.1}ms ({:.0}ns/e) rb={} pr={}",
        hub_ms * 1e6 / 10_000.0,
        h.rebalance_count(),
        h.promote_count()
    );
}
