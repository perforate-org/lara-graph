//! Edges between two given nodes (`contains_edge`, `visit_edges_between`): the cost follows the
//! smaller end's bucket, read whole when it is small and through its target index when it is a
//! hub's.

use crate::{Bench, LABEL, directed, graph, node, nodes};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use core::ops::ControlFlow;
use lara_graph::{NodeId, Order};

const OUT: lara_graph::Adjacency = LABEL.outgoing();

/// 1024 nodes with 8 edges each to the next 8: every bucket small.
fn ring() -> (Bench, Vec<NodeId>) {
    let mut g = graph();
    let all = nodes(&mut g, 1024);
    for (i, &a) in all.iter().enumerate() {
        for k in 1..=8 {
            let b = all.get((i + k) % all.len()).copied().unwrap_or(a);
            directed(&mut g, a, b);
        }
    }
    (g, all)
}

fn contains(g: &Bench, u: NodeId, v: NodeId) -> bool {
    g.contains_edge(u, OUT, v)
        .unwrap_or_else(|e| panic!("contains: {e}"))
}

/// 1024 lookups between each node of [`ring`] and the node `ahead` of it.
fn ring_lookups(ahead: usize) -> BenchResult {
    let (g, all) = ring();
    let pairs: Vec<(NodeId, NodeId)> = all
        .iter()
        .copied()
        .zip(all.iter().copied().cycle().skip(ahead))
        .collect();
    bench_fn(|| {
        for &(u, v) in &pairs {
            black_box(contains(&g, u, v));
        }
    })
}

/// 1024 lookups of an edge that exists, between two small buckets.
#[bench(raw)]
fn contains_edge_small_1024() -> BenchResult {
    ring_lookups(4)
}

/// 1024 lookups of an edge that does not exist, between two small buckets that both exist.
#[bench(raw)]
fn contains_edge_absent_1024() -> BenchResult {
    ring_lookups(500)
}

/// From a hub of 1024 edges to each of its satellites: the satellite's tiny bucket is read.
#[bench(raw)]
fn contains_edge_hub_to_satellite_1024() -> BenchResult {
    let mut g = graph();
    let hub = node(&mut g);
    let satellites = nodes(&mut g, 1024);
    for &s in &satellites {
        directed(&mut g, hub, s);
    }
    bench_fn(|| {
        for &s in &satellites {
            black_box(contains(&g, hub, s));
        }
    })
}

/// All 1024 parallel edges from a hub of 65536 (to 64 targets in turn) to one target: the
/// target's bucket of 1024 is the smaller end, read whole.
#[bench(raw)]
fn edges_between_parallel_65536() -> BenchResult {
    let (g, last) = crate::lookup::parallel_hub();
    let (hub, target) = (last.canonical().node(), last.other().node());
    bench_fn(|| black_box(visit_all(&g, hub, target)))
}

/// Between two hubs: 4096 edges out of one, 256 of them to a node that takes in 65536. The
/// sender's indexed bucket is the smaller end; its index lists the 256 edges' blocks.
#[bench(raw)]
fn edges_between_hubs_4096() -> BenchResult {
    let mut g = graph();
    let (sender, sink) = (node(&mut g), node(&mut g));
    let others = nodes(&mut g, 4096);
    for (i, &o) in others.iter().enumerate() {
        directed(&mut g, sender, if i % 16 == 0 { sink } else { o });
    }
    for i in 0..65536 - 256 {
        let from = others.get(i % others.len()).copied().unwrap_or(sender);
        directed(&mut g, from, sink);
    }
    bench_fn(|| black_box(visit_all(&g, sender, sink)))
}

fn visit_all(g: &Bench, u: NodeId, v: NodeId) -> u32 {
    let mut n = 0;
    g.visit_edges_between(u, OUT, v, Order::Ascending, |_| {
        n += 1;
        ControlFlow::<()>::Continue(())
    })
    .unwrap_or_else(|e| panic!("between: {e}"));
    n
}
