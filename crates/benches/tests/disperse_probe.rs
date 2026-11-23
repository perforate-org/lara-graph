//! Auxiliary cap probe on rank-Zipf, uniform and hub workloads.
//! Storage estimates below are partial and post-settle, not total heap.
//! Use memory_probe for heap comparisons and Criterion for release timing.

use benches::{LaraAdapter, LaraConfig, coo_power_law_stream, coo_uniform_stream};
use lara_graph::traits::DynamicCsr;
use std::time::Instant;

fn report(tag: &str, cap: usize, g: &LaraAdapter, insert_ms: f64, scan_ms: f64, rb: u64, pr: u64) {
    let ne = g.num_edges();
    let (slab, tree, logs, tiny, vertex) = g.mem_breakdown();
    let total = g.memory_bytes_total();
    let pe = |b: u64| b as f64 / ne as f64;
    println!(
        "{tag} cap={cap}: insert={insert_ms:.1}ms insert_rb={rb} insert_pr={pr} state=post_settle partial_estimate={:.1}B/e [slab={:.1} tree={:.1} logs={:.1}({}B) tiny={:.1} vtx={:.1}] scan={scan_ms:.1}ms edges={ne}",
        total as f64 / ne as f64,
        pe(slab),
        pe(tree),
        pe(logs),
        logs,
        pe(tiny),
        pe(vertex),
    );
}

#[test]
fn disperse_probe() {
    // Rank-Zipf source IDs (same generator+seed as log_cap_sweep).
    let n = 100_000u32;
    let stream = coo_power_law_stream(n, 400_000, 99, 1.0);
    for &cap in &[64usize, 128, 170, 256, 512] {
        let cfg = LaraConfig {
            promote_threshold: Some(1024),
            max_log_entries: cap,
            ..LaraConfig::default()
        };
        let mut g = LaraAdapter::new(n, cfg).unwrap();
        let t0 = Instant::now();
        for &(u, v) in &stream {
            g.insert_edge(u, v);
        }
        let insert_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (rb, pr) = (g.rebalance_count(), g.promote_count());
        g.verify();
        g.settle();
        let t1 = Instant::now();
        let mut sum = 0u64;
        g.scan_all(|x| sum += x as u64);
        let scan_ms = t1.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(sum);
        report("rank_zipf", cap, &g, insert_ms, scan_ms, rb, pr);
    }
    // Uniform dispersed (same generator+seed as dgap_vs_log0 dispersed).
    let ustream = coo_uniform_stream(n, 400_000, 1234);
    for &cap in &[64usize, 128, 170] {
        let cfg = LaraConfig {
            promote_threshold: Some(1024),
            max_log_entries: cap,
            ..LaraConfig::default()
        };
        let mut g = LaraAdapter::new(n, cfg).unwrap();
        let t0 = Instant::now();
        for &(u, v) in &ustream {
            g.insert_edge(u, v);
        }
        let insert_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (rb, pr) = (g.rebalance_count(), g.promote_count());
        g.verify();
        g.settle();
        let t1 = Instant::now();
        let mut sum = 0u64;
        g.scan_all(|x| sum += x as u64);
        let scan_ms = t1.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(sum);
        report("uniform", cap, &g, insert_ms, scan_ms, rb, pr);
    }
    // Hub 32k (promote-absorption check).
    let hub_n = 65_536u32;
    for &cap in &[170usize, 512] {
        let cfg = LaraConfig {
            promote_threshold: Some(1024),
            max_log_entries: cap,
            ..LaraConfig::default()
        };
        let mut g = LaraAdapter::new(hub_n, cfg).unwrap();
        let t0 = Instant::now();
        for x in 0..32_768u32 {
            g.insert_edge(1, x);
        }
        let insert_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let (rb, pr) = (g.rebalance_count(), g.promote_count());
        g.verify();
        g.settle();
        let t1 = Instant::now();
        let mut sum = 0u64;
        g.for_each_neighbor(1, |x| sum += x as u64);
        let scan_ms = t1.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(sum);
        report("hub32k", cap, &g, insert_ms, scan_ms, rb, pr);
    }
}
