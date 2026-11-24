//! Reads by handle: an edge's target and its counterpart half, one at a time or batched, in a
//! hub bucket and among parallel edges.

use crate::{
    LABEL, directed, first_halves, heap_graph, last, live_batched, live_each, node, nodes,
    seed_hub, seed_parallel,
};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use lara_graph::{Direction, EdgeRef, Graph, Halves, Position, ReadBatch, VecMemory};

/// Finds the out half of the hub's last edge from its in half at the leaf: a rank count over
/// the leaf's bucket and a selection in the hub's.
#[bench(raw)]
fn counterpart_into_hub_4096() -> BenchResult {
    let mut g = heap_graph();
    let (_, edges) = seed_hub(&mut g, 4096);
    let last = last(&edges);
    bench_fn(|| black_box(g.counterpart(last.other())))
}

/// Finds the in half from the hub's out half at its last position: the count scans the whole
/// hub bucket.
#[bench(raw)]
fn counterpart_from_hub_tail_4096() -> BenchResult {
    let mut g = heap_graph();
    let (_, edges) = seed_hub(&mut g, 4096);
    let last = last(&edges);
    bench_fn(|| black_box(g.counterpart(last.canonical())))
}

/// A hub of 65536 edges to 64 targets in turn: each target's 1024 parallel edges spread over
/// every block of the hub's bucket, 512 postings in its index. Returns the last edge.
pub(crate) fn parallel_hub() -> (Graph<VecMemory>, Halves) {
    let mut g = heap_graph();
    let hub = node(&mut g);
    let targets = nodes(&mut g, 64);
    let edges: Vec<Halves> = targets
        .iter()
        .cycle()
        .take(65536)
        .map(|&t| directed(&mut g, hub, t))
        .collect();
    let last = last(&edges);
    (g, last)
}

/// Pairs the hub's last edge from its out half: the rank of the edge among the hub's edges to
/// that target, through 512 postings of the hub's index.
#[bench(raw)]
fn counterpart_parallel_from_hub_65536() -> BenchResult {
    let (g, last) = parallel_hub();
    bench_fn(|| black_box(g.counterpart(last.canonical())))
}

/// Pairs the hub's last edge from the target's side: the 1024th edge to that target among the
/// hub's, through 512 postings of the hub's index.
#[bench(raw)]
fn counterpart_parallel_from_target_65536() -> BenchResult {
    let (g, last) = parallel_hub();
    bench_fn(|| black_box(g.counterpart(last.other())))
}

/// A 65536-edge hub and the out halves at its first 1024 positions.
fn hub_halves() -> (Graph<VecMemory>, Vec<EdgeRef>) {
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, 65536);
    (g, first_halves(hub, 1024))
}

/// One edge lookup by handle in a paged hub bucket.
#[bench(raw)]
fn edge_lookup_65536() -> BenchResult {
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, 65536);
    let half = EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(40_000));
    bench_fn(|| black_box(g.edge_target(half)))
}

/// 1024 edge lookups in one hub bucket, one call each.
#[bench(raw)]
fn edge_lookup_1024_loop() -> BenchResult {
    let (g, halves) = hub_halves();
    bench_fn(|| black_box(live_each(&halves, |half| g.edge_target(half))))
}

/// The same 1024 lookups through one `ReadBatch`, reusing the bucket descriptor.
#[bench(raw)]
fn edge_lookup_1024_batched() -> BenchResult {
    let (g, halves) = hub_halves();
    bench_fn(|| black_box(live_batched(&g, &halves, ReadBatch::edge_targets_each)))
}

/// 1024 counterparts in one hub bucket, one call each.
#[bench(raw)]
fn counterpart_1024_loop() -> BenchResult {
    let (g, halves) = hub_halves();
    bench_fn(|| black_box(live_each(&halves, |half| g.counterpart(half))))
}

/// The same 1024 counterparts through one `ReadBatch`, reusing the own bucket.
#[bench(raw)]
fn counterpart_1024_batched() -> BenchResult {
    let (g, halves) = hub_halves();
    bench_fn(|| black_box(live_batched(&g, &halves, ReadBatch::counterparts_each)))
}

/// 1024 counterparts of parallel edges, one call each.
#[bench(raw)]
fn counterpart_parallel_1024_loop() -> BenchResult {
    let mut g = heap_graph();
    let (_, halves) = seed_parallel(&mut g, 1024);
    bench_fn(|| black_box(live_each(&halves, |half| g.counterpart(half))))
}

/// The same 1024 counterparts through one `ReadBatch`, reusing both buckets.
#[bench(raw)]
fn counterpart_parallel_1024_batched() -> BenchResult {
    let mut g = heap_graph();
    let (_, halves) = seed_parallel(&mut g, 1024);
    bench_fn(|| black_box(live_batched(&g, &halves, ReadBatch::counterparts_each)))
}
