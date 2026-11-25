//! Removing a hub's edges one at a time or batched, compacting what is left, and refilling the
//! holes.

use crate::{LABEL, directed, heap_graph, nodes, seed_hub, seed_parallel, stable_graph};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use lara_graph::{
    Direction, EdgeRef, Graph, Halves, Memory, NewEdge, NodeId, Placement, Progress, Removal,
    VecMemory,
};

fn remove_hub_edges<M: Memory>(mut g: Graph<M>, degree: u32) -> (Graph<M>, Vec<EdgeRef>) {
    let (_, edges) = seed_hub(&mut g, degree);
    let halves = edges.iter().map(Halves::canonical).collect();
    (g, halves)
}

/// Removes `halves` one call each.
fn remove_each<M: Memory>(g: &mut Graph<M>, halves: &[EdgeRef]) {
    for &half in halves {
        black_box(
            g.remove_edge(half)
                .unwrap_or_else(|e| panic!("remove: {e}")),
        );
    }
}

/// Removes `halves` through one `remove_edges`.
fn remove_all<M: Memory>(g: &mut Graph<M>, halves: &[EdgeRef]) -> Vec<Removal> {
    g.remove_edges(halves)
        .unwrap_or_else(|e| panic!("remove batch: {e}"))
}

/// Compacts `hub`'s out bucket until done.
fn compact_fully(g: &mut Graph<VecMemory>, hub: NodeId) {
    while g
        .compact(hub, (LABEL, Direction::Outgoing))
        .unwrap_or_else(|e| panic!("compact: {e}"))
        .progress()
        == Progress::Remaining
    {}
}

#[bench(raw)]
fn hub_remove_edges_4096_batched_stable() -> BenchResult {
    let (mut g, halves) = remove_hub_edges(stable_graph(), 4096);
    bench_fn(|| black_box(remove_all(&mut g, &halves)))
}

#[bench(raw)]
fn hub_remove_edges_4096_loop_stable() -> BenchResult {
    let (mut g, halves) = remove_hub_edges(stable_graph(), 4096);
    bench_fn(|| remove_each(&mut g, &halves))
}

#[bench(raw)]
fn hub_remove_edges_parallel_4096_loop() -> BenchResult {
    let mut g = heap_graph();
    let (_, halves) = seed_parallel(&mut g, 4096);
    bench_fn(|| remove_each(&mut g, &halves))
}

#[bench(raw)]
fn hub_remove_edges_parallel_4096_batched() -> BenchResult {
    let mut g = heap_graph();
    let (_, halves) = seed_parallel(&mut g, 4096);
    bench_fn(|| black_box(remove_all(&mut g, &halves)))
}

/// Removes every other edge of a hub by handle, then compacts it until done (ic-stable-lara
/// `bench_labeled_stage2_hub_delete_half_by_slot_then_compact_<degree>`).
fn remove_half_then_compact(degree: u32) -> BenchResult {
    let mut g = heap_graph();
    let (hub, edges) = seed_hub(&mut g, degree);
    bench_fn(|| {
        {
            let _removals = canbench_rs::bench_scope("removals");
            for halves in edges.iter().step_by(2) {
                black_box(
                    g.remove_edge(halves.canonical())
                        .unwrap_or_else(|e| panic!("{e}")),
                );
            }
        }
        let _compaction = canbench_rs::bench_scope("compaction");
        compact_fully(&mut g, hub);
    })
}

#[bench(raw)]
fn hub_remove_half_compact_4096() -> BenchResult {
    remove_half_then_compact(4096)
}

#[bench(raw)]
fn hub_remove_half_compact_65536() -> BenchResult {
    remove_half_then_compact(65536)
}

/// The same removal and compaction through `remove_edges` for the removal half.
#[bench(raw)]
fn hub_remove_half_compact_65536_batched() -> BenchResult {
    let mut g = heap_graph();
    let (hub, edges) = seed_hub(&mut g, 65536);
    let halves: Vec<EdgeRef> = edges.iter().step_by(2).map(Halves::canonical).collect();
    bench_fn(|| {
        {
            let _removals = canbench_rs::bench_scope("removals");
            remove_all(&mut g, &halves);
        }
        let _compaction = canbench_rs::bench_scope("compaction");
        compact_fully(&mut g, hub);
    })
}

/// Whose edges refill a hub's holes.
#[derive(Clone, Copy)]
enum Refill {
    Fresh,
    Parallel,
}

/// A hub of `degree` edges with every other one removed, and the targets that refill its holes
/// (Gleaph's `Unordered`): fresh leaves, which take the holes from the front, or a second edge
/// to each remaining leaf, which takes the first hole after that leaf's edge.
fn holed_hub(degree: u32, to: Refill) -> (Graph<VecMemory>, NodeId, Vec<NodeId>) {
    let mut g = heap_graph();
    let (hub, edges) = seed_hub(&mut g, degree);
    for halves in edges.iter().step_by(2) {
        black_box(
            g.remove_edge(halves.canonical())
                .unwrap_or_else(|e| panic!("{e}")),
        );
    }
    let targets: Vec<NodeId> = match to {
        Refill::Fresh => nodes(&mut g, degree / 2),
        Refill::Parallel => edges
            .iter()
            .skip(1)
            .step_by(2)
            .map(|halves| halves.other().node())
            .collect(),
    };
    (g, hub, targets)
}

/// Refills a holed hub one insert at a time, reusing holes.
fn refill_hub(degree: u32, to: Refill) -> BenchResult {
    let (mut g, hub, targets) = holed_hub(degree, to);
    bench_fn(|| {
        for &leaf in &targets {
            black_box(directed(&mut g, hub, black_box(leaf)));
        }
    })
}

fn refill_hub_batched(degree: u32, to: Refill) -> BenchResult {
    let (mut g, hub, targets) = holed_hub(degree, to);
    let refill: Vec<NewEdge<'_>> = targets
        .iter()
        .map(|&leaf| NewEdge::directed(hub, LABEL, leaf, Placement::ReuseHoles, &[]))
        .collect();
    bench_fn(|| {
        black_box(
            g.add_edges(&refill)
                .unwrap_or_else(|e| panic!("batch: {e}")),
        )
    })
}

#[bench(raw)]
fn hub_refill_holes_65536_batched() -> BenchResult {
    refill_hub_batched(65536, Refill::Fresh)
}

#[bench(raw)]
fn hub_refill_holes_parallel_65536_batched() -> BenchResult {
    refill_hub_batched(65536, Refill::Parallel)
}

#[bench(raw)]
fn hub_refill_holes_65536() -> BenchResult {
    refill_hub(65536, Refill::Fresh)
}

#[bench(raw)]
fn hub_refill_holes_parallel_65536() -> BenchResult {
    refill_hub(65536, Refill::Parallel)
}

/// Removes one edge from the middle of a hub: the pair is matched through the hub's target index
/// and the hub's half leaves it.
#[bench(raw)]
fn hub_remove_one_65536() -> BenchResult {
    let mut g = heap_graph();
    let (_, edges) = seed_hub(&mut g, 65536);
    let middle = edges
        .get(30_000)
        .copied()
        .unwrap_or_else(|| panic!("no middle edge"));
    bench_fn(|| black_box(g.remove_edge(middle.other())))
}
