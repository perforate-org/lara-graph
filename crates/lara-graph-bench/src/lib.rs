//! Instruction counts of lara-graph operations on the Internet Computer, measured with canbench.
//!
//! Scenarios mirror ic-stable-lara's `canbench_results.yml` where one exists, so the two can be
//! compared. Every edge here is stored at both ends, as Gleaph needs; ic-stable-lara's
//! single-label hub benches store one side only.

extern crate alloc;

use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use core::num::NonZeroU32;
use core::ops::ControlFlow;
use ic_stable_structures::DefaultMemoryImpl;
use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
use lara_graph::{
    Cursor, Direction, EdgeKind, EdgeRef, EdgeValue, Graph, Halves, Label, Memories, Memory,
    NewEdge, NodeId, Order, Placement, Position, Progress, VecMemory, Width,
};

mod btree;
mod locality;
pub mod two_hop;

const LABEL: Label = Label::new(2);

/// A graph in five heap memories.
#[must_use]
pub fn heap_graph() -> Graph<VecMemory> {
    Graph::new(Memories::from_fn(|_| VecMemory::new()))
}

/// A graph in five virtual stable memories (ids 0..5), as a canister wires it.
fn stable_graph() -> Graph<VirtualMemory<DefaultMemoryImpl>> {
    let manager = MemoryManager::init(DefaultMemoryImpl::default());
    Graph::init(Memories::from_fn(|role| {
        manager.get(MemoryId::new(role.index()))
    }))
}

fn node<M: Memory>(g: &mut Graph<M>) -> NodeId {
    g.add_node().unwrap_or_else(|e| panic!("node: {e}"))
}

fn nodes<M: Memory>(g: &mut Graph<M>, n: u32) -> Vec<NodeId> {
    (0..n).map(|_| node(g)).collect()
}

/// The last edge a seed inserted.
fn last(edges: &[Halves]) -> Halves {
    edges.last().copied().unwrap_or_else(|| panic!("no edges"))
}

fn directed<M: Memory>(g: &mut Graph<M>, from: NodeId, to: NodeId) -> Halves {
    g.add_edge(
        EdgeKind::Directed,
        from,
        LABEL,
        to,
        Placement::ReuseHoles,
        &[],
    )
    .unwrap_or_else(|e| panic!("insert: {e}"))
}

/// A hub with an edge to each of `degree` fresh leaves; returns the hub and the edges.
fn seed_hub<M: Memory>(g: &mut Graph<M>, degree: u32) -> (NodeId, Vec<Halves>) {
    let hub = node(g);
    let leaves = nodes(g, degree);
    let edges = leaves
        .into_iter()
        .map(|leaf| directed(g, hub, leaf))
        .collect();
    (hub, edges)
}

fn grow_hub<M: Memory>(mut g: Graph<M>, degree: u32) -> BenchResult {
    let hub = node(&mut g);
    let leaves = nodes(&mut g, degree);
    bench_fn(|| {
        for &leaf in &leaves {
            black_box(directed(&mut g, hub, black_box(leaf)));
        }
    })
}

/// 1024 nodes added one at a time.
#[bench(raw)]
fn add_nodes_1024_loop() -> BenchResult {
    let mut g = heap_graph();
    bench_fn(|| {
        for _ in 0..1024 {
            black_box(g.add_node().unwrap_or_else(|e| panic!("node: {e}")));
        }
    })
}

/// The same 1024 nodes through one `add_nodes`.
#[bench(raw)]
fn add_nodes_1024_batched() -> BenchResult {
    let mut g = heap_graph();
    bench_fn(|| {
        black_box(
            g.add_nodes(NonZeroU32::new(1024).unwrap_or(NonZeroU32::MIN))
                .unwrap_or_else(|e| panic!("nodes: {e}")),
        )
    })
}

#[bench(raw)]
fn hub_insert_1024() -> BenchResult {
    grow_hub(heap_graph(), 1024)
}

#[bench(raw)]
fn hub_insert_4096() -> BenchResult {
    grow_hub(heap_graph(), 4096)
}

#[bench(raw)]
fn hub_insert_65536() -> BenchResult {
    grow_hub(heap_graph(), 65536)
}

#[bench(raw)]
fn hub_insert_4096_stable() -> BenchResult {
    grow_hub(stable_graph(), 4096)
}

fn grow_hub_batched(mut g: Graph<impl Memory>, degree: u32) -> BenchResult {
    let hub = node(&mut g);
    let leaves = nodes(&mut g, degree);
    let edges: Vec<NewEdge<'_>> = leaves
        .iter()
        .map(|&leaf| NewEdge::directed(hub, LABEL, leaf, Placement::ReuseHoles, &[]))
        .collect();
    bench_fn(|| black_box(g.add_edges(&edges).unwrap_or_else(|e| panic!("batch: {e}"))))
}

#[bench(raw)]
fn hub_insert_4096_batched() -> BenchResult {
    grow_hub_batched(heap_graph(), 4096)
}

#[bench(raw)]
fn hub_insert_4096_batched_stable() -> BenchResult {
    grow_hub_batched(stable_graph(), 4096)
}

fn remove_hub_edges<M: Memory>(mut g: Graph<M>, degree: u32) -> (Graph<M>, Vec<EdgeRef>) {
    let (_, edges) = seed_hub(&mut g, degree);
    let halves = edges.iter().map(Halves::canonical).collect();
    (g, halves)
}

#[bench(raw)]
fn hub_remove_edges_4096_batched_stable() -> BenchResult {
    let (mut g, halves) = remove_hub_edges(stable_graph(), 4096);
    bench_fn(|| {
        black_box(
            g.remove_edges(&halves)
                .unwrap_or_else(|e| panic!("remove batch: {e}")),
        )
    })
}

#[bench(raw)]
fn hub_remove_edges_4096_loop_stable() -> BenchResult {
    let (mut g, halves) = remove_hub_edges(stable_graph(), 4096);
    bench_fn(|| {
        for &half in &halves {
            black_box(
                g.remove_edge(half)
                    .unwrap_or_else(|e| panic!("remove: {e}")),
            );
        }
    })
}

fn grow_hub_undirected(mut g: Graph<impl Memory>, degree: u32) -> BenchResult {
    let hub = node(&mut g);
    let leaves = nodes(&mut g, degree);
    bench_fn(|| {
        for &leaf in &leaves {
            black_box(
                g.add_edge(
                    EdgeKind::Undirected,
                    hub,
                    LABEL,
                    black_box(leaf),
                    Placement::ReuseHoles,
                    &[],
                )
                .unwrap_or_else(|e| panic!("insert: {e}")),
            );
        }
    })
}

#[bench(raw)]
fn hub_insert_undirected_4096() -> BenchResult {
    grow_hub_undirected(heap_graph(), 4096)
}

#[bench(raw)]
fn hub_insert_undirected_4096_stable() -> BenchResult {
    grow_hub_undirected(stable_graph(), 4096)
}

fn grow_hub_undirected_batched(mut g: Graph<impl Memory>, degree: u32) -> BenchResult {
    let hub = node(&mut g);
    let leaves = nodes(&mut g, degree);
    let edges: Vec<NewEdge<'_>> = leaves
        .iter()
        .map(|&leaf| NewEdge::undirected(hub, LABEL, leaf, Placement::ReuseHoles, &[]))
        .collect();
    bench_fn(|| black_box(g.add_edges(&edges).unwrap_or_else(|e| panic!("batch: {e}"))))
}

#[bench(raw)]
fn hub_insert_undirected_4096_batched() -> BenchResult {
    grow_hub_undirected_batched(heap_graph(), 4096)
}

#[bench(raw)]
fn hub_insert_undirected_4096_batched_stable() -> BenchResult {
    grow_hub_undirected_batched(stable_graph(), 4096)
}

/// Parallel edges to a few targets: every insert also lands in a target's growing in bucket.
#[bench(raw)]
fn hub_insert_parallel_4096() -> BenchResult {
    let mut g = heap_graph();
    let hub = node(&mut g);
    let targets = nodes(&mut g, 8);
    bench_fn(|| {
        for &target in targets.iter().cycle().take(4096) {
            black_box(directed(&mut g, hub, target));
        }
    })
}

fn grow_hub_parallel_batched(mut g: Graph<impl Memory>) -> BenchResult {
    let hub = node(&mut g);
    let targets = nodes(&mut g, 8);
    let edges: Vec<NewEdge<'_>> = targets
        .iter()
        .cycle()
        .take(4096)
        .map(|&target| NewEdge::directed(hub, LABEL, target, Placement::ReuseHoles, &[]))
        .collect();
    bench_fn(|| black_box(g.add_edges(&edges).unwrap_or_else(|e| panic!("batch: {e}"))))
}

#[bench(raw)]
fn hub_insert_parallel_4096_batched() -> BenchResult {
    grow_hub_parallel_batched(heap_graph())
}

#[bench(raw)]
fn hub_insert_parallel_4096_batched_stable() -> BenchResult {
    grow_hub_parallel_batched(stable_graph())
}

#[derive(Clone, Copy)]
enum Mixed {
    Loop,
    Batch,
}

fn grow_hub_mixed(mut g: Graph<impl Memory>, degree: u32, mode: Mixed) -> BenchResult {
    let hub = node(&mut g);
    let leaves = nodes(&mut g, degree);
    let kinds: Vec<EdgeKind> = (0..degree)
        .map(|i| {
            if i % 2 == 0 {
                EdgeKind::Directed
            } else {
                EdgeKind::Undirected
            }
        })
        .collect();
    let edges: Vec<NewEdge<'_>> = leaves
        .iter()
        .zip(&kinds)
        .map(|(&leaf, &kind)| match kind {
            EdgeKind::Directed => NewEdge::directed(hub, LABEL, leaf, Placement::ReuseHoles, &[]),
            EdgeKind::Undirected => {
                NewEdge::undirected(hub, LABEL, leaf, Placement::ReuseHoles, &[])
            }
        })
        .collect();
    bench_fn(|| {
        if matches!(mode, Mixed::Batch) {
            black_box(
                g.add_edges(&edges)
                    .unwrap_or_else(|e| panic!("batch: {e}"))
                    .len(),
            )
        } else {
            for (&leaf, &kind) in leaves.iter().zip(&kinds) {
                black_box(
                    g.add_edge(kind, hub, LABEL, leaf, Placement::ReuseHoles, &[])
                        .unwrap_or_else(|e| panic!("insert: {e}")),
                );
            }
            black_box(leaves.len())
        }
    })
}

#[bench(raw)]
fn hub_insert_mixed_4096() -> BenchResult {
    grow_hub_mixed(heap_graph(), 4096, Mixed::Loop)
}

#[bench(raw)]
fn hub_insert_mixed_4096_batched() -> BenchResult {
    grow_hub_mixed(heap_graph(), 4096, Mixed::Batch)
}

#[bench(raw)]
fn hub_insert_mixed_4096_batched_stable() -> BenchResult {
    grow_hub_mixed(stable_graph(), 4096, Mixed::Batch)
}

/// 51 full scans of a 1024-edge hub bucket (ic-stable-lara `bench_l_s2_h_scn_d_1024`).
fn scan_hub<M: Memory>(mut g: Graph<M>, degree: u32) -> BenchResult {
    let (hub, _) = seed_hub(&mut g, degree);
    let mut out = Vec::new();
    bench_fn(|| {
        for _ in 0..51 {
            g.neighbors_directed(hub, LABEL, Direction::Outgoing, &mut out)
                .unwrap_or_else(|e| panic!("scan: {e}"));
            black_box(out.len());
        }
    })
}

/// The same scans through the visitor, counting edges without collecting them.
fn visit_hub<M: Memory>(g: Graph<M>, degree: u32) -> BenchResult {
    visit_hub_in(g, degree, Order::Ascending)
}

fn visit_hub_in<M: Memory>(mut g: Graph<M>, degree: u32, order: Order) -> BenchResult {
    let (hub, _) = seed_hub(&mut g, degree);
    bench_fn(|| {
        for _ in 0..51 {
            let mut count = 0u32;
            g.visit_neighbors(hub, LABEL, Direction::Outgoing, order, |_| {
                count += 1;
                ControlFlow::<()>::Continue(())
            })
            .unwrap_or_else(|e| panic!("visit: {e}"));
            black_box(count);
        }
    })
}

#[bench(raw)]
fn hub_visit_1024() -> BenchResult {
    visit_hub(heap_graph(), 1024)
}

#[bench(raw)]
fn hub_visit_65536() -> BenchResult {
    visit_hub(heap_graph(), 65536)
}

#[bench(raw)]
fn hub_visit_descending_65536() -> BenchResult {
    visit_hub_in(heap_graph(), 65536, Order::Descending)
}

/// A whole 65536-edge hub read in windows of 256 positions, newest first.
#[bench(raw)]
fn hub_windows_descending_65536() -> BenchResult {
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, 65536);
    let span = NonZeroU32::new(256).unwrap_or(NonZeroU32::MIN);
    bench_fn(|| {
        let mut cursor = Some(Cursor::first(Order::Descending, span));
        let mut count = 0u32;
        while let Some(at) = cursor {
            cursor = g
                .visit_window(hub, LABEL, Direction::Outgoing, at, |_| count += 1)
                .unwrap_or_else(|e| panic!("window: {e}"));
        }
        black_box(count);
    })
}

/// 256 scattered positions of a 65536-edge hub read in one call.
#[bench(raw)]
fn hub_targets_at_256_of_65536() -> BenchResult {
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, 65536);
    let positions: Vec<Position> = (0..256u32)
        .map(|i| Position::new(i * 251 % 65536))
        .collect();
    let mut out = Vec::new();
    bench_fn(|| {
        g.targets_at(hub, LABEL, Direction::Outgoing, &positions, &mut out)
            .unwrap_or_else(|e| panic!("targets: {e}"));
        black_box(out.len());
    })
}

/// A hub whose label carries 8-byte values, each edge given one.
fn seed_valued_hub(g: &mut Graph<VecMemory>, degree: u32) -> NodeId {
    g.declare_width(LABEL, Width::new(8).unwrap_or_else(|| panic!("width")))
        .unwrap_or_else(|e| panic!("width: {e}"));
    let hub = node(g);
    for leaf in nodes(g, degree) {
        let value = u64::from(leaf.index()).to_le_bytes();
        g.add_edge(
            EdgeKind::Directed,
            hub,
            LABEL,
            leaf,
            Placement::Append,
            &value,
        )
        .unwrap_or_else(|e| panic!("insert: {e}"));
    }
    hub
}

/// Growing a hub to 4096 edges with 8-byte values: both halves of each.
#[bench(raw)]
fn hub_insert_valued_4096() -> BenchResult {
    let mut g = heap_graph();
    g.declare_width(LABEL, Width::new(8).unwrap_or_else(|| panic!("width")))
        .unwrap_or_else(|e| panic!("width: {e}"));
    let hub = node(&mut g);
    let leaves = nodes(&mut g, 4096);
    bench_fn(|| {
        for &leaf in &leaves {
            let value = u64::from(leaf.index()).to_le_bytes();
            black_box(
                g.add_edge(
                    EdgeKind::Directed,
                    hub,
                    LABEL,
                    leaf,
                    Placement::Append,
                    &value,
                )
                .unwrap_or_else(|e| panic!("insert: {e}")),
            );
        }
    })
}

/// 51 values-first scans of a 65536-edge hub (values only).
#[bench(raw)]
fn hub_visit_values_65536() -> BenchResult {
    let mut g = heap_graph();
    let hub = seed_valued_hub(&mut g, 65536);
    bench_fn(|| {
        for _ in 0..51 {
            let mut sum = 0u64;
            g.visit_values(
                hub,
                (LABEL, Direction::Outgoing),
                Order::Ascending,
                |_, v| {
                    sum = sum.wrapping_add(u64::from(v.first().copied().unwrap_or(0)));
                    ControlFlow::<()>::Continue(())
                },
            )
            .unwrap_or_else(|e| panic!("values: {e}"));
            black_box(sum);
        }
    })
}

/// 51 combined scans (edge and value) of a 1024-edge hub.
#[bench(raw)]
fn hub_visit_with_values_1024() -> BenchResult {
    let mut g = heap_graph();
    let hub = seed_valued_hub(&mut g, 1024);
    bench_fn(|| {
        for _ in 0..51 {
            let mut count = 0u32;
            g.visit_neighbors_with_values(
                hub,
                (LABEL, Direction::Outgoing),
                Order::Ascending,
                |_, _| {
                    count += 1;
                    ControlFlow::<()>::Continue(())
                },
            )
            .unwrap_or_else(|e| panic!("visit: {e}"));
            black_box(count);
        }
    })
}

/// 1024 sources, each with one valued out-edge, one node at a time.
#[bench(raw)]
fn visit_with_values_many_1024_loop() -> BenchResult {
    let mut g = heap_graph();
    g.declare_width(LABEL, Width::new(8).unwrap_or_else(|| panic!("width")))
        .unwrap_or_else(|e| panic!("width: {e}"));
    let sources: Vec<NodeId> = (0..1024)
        .map(|i| {
            let source = node(&mut g);
            let target = node(&mut g);
            let value = [u8::try_from(i % 251).unwrap_or_else(|_| panic!("byte")); 8];
            g.add_edge(
                EdgeKind::Directed,
                source,
                LABEL,
                target,
                Placement::Append,
                &value,
            )
            .unwrap_or_else(|e| panic!("insert: {e}"));
            source
        })
        .collect();
    bench_fn(|| {
        let mut count = 0u32;
        for &source in &sources {
            g.visit_neighbors_with_values(
                source,
                (LABEL, Direction::Outgoing),
                Order::Ascending,
                |_, _| {
                    count += 1;
                    ControlFlow::<()>::Continue(())
                },
            )
            .unwrap_or_else(|e| panic!("visit: {e}"));
        }
        black_box(count)
    })
}

/// The same 1024 sources through one `ReadBatch`.
#[bench(raw)]
fn visit_with_values_many_1024_batched() -> BenchResult {
    let mut g = heap_graph();
    g.declare_width(LABEL, Width::new(8).unwrap_or_else(|| panic!("width")))
        .unwrap_or_else(|e| panic!("width: {e}"));
    let sources: Vec<NodeId> = (0..1024)
        .map(|i| {
            let source = node(&mut g);
            let target = node(&mut g);
            let value = [u8::try_from(i % 251).unwrap_or_else(|_| panic!("byte")); 8];
            g.add_edge(
                EdgeKind::Directed,
                source,
                LABEL,
                target,
                Placement::Append,
                &value,
            )
            .unwrap_or_else(|e| panic!("insert: {e}"));
            source
        })
        .collect();
    bench_fn(|| {
        let mut batch = g.read_batch();
        let mut count = 0u32;
        batch
            .visit_neighbors_with_values_each(
                &sources,
                LABEL,
                Direction::Outgoing,
                Order::Ascending,
                |_, _, _| {
                    count += 1;
                    ControlFlow::<()>::Continue(())
                },
            )
            .unwrap_or_else(|e| panic!("batch: {e}"));
        black_box(count)
    })
}

fn one_valued_edge_per_source() -> (Graph<VecMemory>, Vec<NodeId>) {
    let mut g = heap_graph();
    g.declare_width(LABEL, Width::new(8).unwrap_or_else(|| panic!("width")))
        .unwrap_or_else(|e| panic!("width: {e}"));
    let sources: Vec<NodeId> = (0..1024)
        .map(|i| {
            let source = node(&mut g);
            let target = node(&mut g);
            let value = [u8::try_from(i % 251).unwrap_or_else(|_| panic!("byte")); 8];
            g.add_edge(
                EdgeKind::Directed,
                source,
                LABEL,
                target,
                Placement::Append,
                &value,
            )
            .unwrap_or_else(|e| panic!("insert: {e}"));
            source
        })
        .collect();
    (g, sources)
}

#[bench(raw)]
fn visit_values_many_1024_loop() -> BenchResult {
    let (g, sources) = one_valued_edge_per_source();
    bench_fn(|| {
        let mut count = 0u32;
        for &source in &sources {
            g.visit_values(
                source,
                (LABEL, Direction::Outgoing),
                Order::Ascending,
                |_, _| {
                    count += 1;
                    ControlFlow::<()>::Continue(())
                },
            )
            .unwrap_or_else(|e| panic!("visit: {e}"));
        }
        black_box(count)
    })
}

#[bench(raw)]
fn visit_values_many_1024_batched() -> BenchResult {
    let (g, sources) = one_valued_edge_per_source();
    bench_fn(|| {
        let mut batch = g.read_batch();
        let mut count = 0u32;
        batch
            .visit_values_each(
                &sources,
                LABEL,
                Direction::Outgoing,
                Order::Ascending,
                |_, _, _| {
                    count += 1;
                    ControlFlow::<()>::Continue(())
                },
            )
            .unwrap_or_else(|e| panic!("batch: {e}"));
        black_box(count)
    })
}

/// Overwriting one edge's value in both halves, in a 4096-edge hub.
#[bench(raw)]
fn set_value_4096() -> BenchResult {
    let mut g = heap_graph();
    let hub = seed_valued_hub(&mut g, 4096);
    let half = EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(4000));
    bench_fn(|| black_box(g.set_edge_value(half, &[7; 8])))
}

#[bench(raw)]
fn hub_scan_1024() -> BenchResult {
    scan_hub(heap_graph(), 1024)
}

#[bench(raw)]
fn hub_scan_1024_stable() -> BenchResult {
    scan_hub(stable_graph(), 1024)
}

#[bench(raw)]
fn hub_scan_65536() -> BenchResult {
    scan_hub(heap_graph(), 65536)
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
        while g
            .compact(hub, LABEL, Direction::Outgoing)
            .unwrap_or_else(|e| panic!("compact: {e}"))
            .progress()
            == Progress::Remaining
        {}
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

/// Removes every other edge of a hub, then inserts as many edges reusing holes (Gleaph's
/// `Unordered`): to fresh leaves, which take the holes from the front, or a second edge to each
/// remaining leaf, which takes the first hole after that leaf's edge.
fn refill_hub(degree: u32, to: Refill) -> BenchResult {
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
    bench_fn(|| {
        for &leaf in &targets {
            black_box(directed(&mut g, hub, black_box(leaf)));
        }
    })
}

/// Whose edges refill a hub's holes.
#[derive(Clone, Copy)]
enum Refill {
    Fresh,
    Parallel,
}

#[bench(raw)]
fn hub_refill_holes_65536() -> BenchResult {
    refill_hub(65536, Refill::Fresh)
}

#[bench(raw)]
fn hub_refill_holes_parallel_65536() -> BenchResult {
    refill_hub(65536, Refill::Parallel)
}

/// Inserts the `nth` (1-based) out-edge of each of 512 nodes, whose earlier edges are in
/// place, to a fresh target each: the cost of a small bucket's growth. The third fills a tiny
/// bucket's last inline slot; the fourth moves it to a run.
fn grow_small(nth: u32) -> BenchResult {
    let mut g = heap_graph();
    let sources = nodes(&mut g, 512);
    for _ in 1..nth {
        for &v in &sources {
            let target = node(&mut g);
            directed(&mut g, v, target);
        }
    }
    let targets = nodes(&mut g, 512);
    bench_fn(|| {
        for (&v, &target) in sources.iter().zip(&targets) {
            black_box(directed(&mut g, v, target));
        }
    })
}

#[bench(raw)]
fn small_insert_3rd_512() -> BenchResult {
    grow_small(3)
}

#[bench(raw)]
fn small_insert_4th_512() -> BenchResult {
    grow_small(4)
}

#[bench(raw)]
fn small_insert_5th_512() -> BenchResult {
    grow_small(5)
}

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
fn parallel_hub() -> (Graph<VecMemory>, Halves) {
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

fn budget(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap_or(NonZeroU32::MIN)
}

/// Detaches a hub of `degree` leaves in steps of `step` edges (ic-stable-lara
/// `bench_l_s2_det_hub_<degree>` and its stepped variant).
/// A hub with 1024 parallel edges to one target: every step re-enters the same opposite bucket.
#[bench(raw)]
fn detach_parallel_1024() -> BenchResult {
    let mut g = heap_graph();
    let (hub, _) = seed_parallel(&mut g, 1024);
    bench_fn(|| {
        g.detach(hub).unwrap_or_else(|e| panic!("detach: {e}"));
        black_box(
            g.detach_step(hub, budget(u32::MAX))
                .unwrap_or_else(|e| panic!("detach step: {e}")),
        )
    })
}

fn detach_hub(degree: u32, step: u32) -> BenchResult {
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, degree);
    bench_fn(|| {
        g.detach(hub).unwrap_or_else(|e| panic!("detach: {e}"));
        while g
            .detach_step(hub, budget(step))
            .unwrap_or_else(|e| panic!("step: {e}"))
            .progress()
            == Progress::Remaining
        {}
    })
}

#[bench(raw)]
fn detach_hub_1024() -> BenchResult {
    detach_hub(1024, u32::MAX)
}

#[bench(raw)]
fn detach_hub_4096() -> BenchResult {
    detach_hub(4096, u32::MAX)
}

#[bench(raw)]
fn detach_hub_4096_step_64() -> BenchResult {
    detach_hub(4096, 64)
}

#[bench(raw)]
fn detach_hub_65536() -> BenchResult {
    detach_hub(65536, u32::MAX)
}

/// Detaches a satellite whose one edge is the last into a hub of in-degree `degree`: the other
/// half is found by scanning the hub's in bucket (ic-stable-lara `bench_l_s2_det_sat_<degree>`).
fn detach_satellite(degree: u32) -> BenchResult {
    let mut g = heap_graph();
    let hub = node(&mut g);
    for source in nodes(&mut g, degree) {
        directed(&mut g, source, hub);
    }
    let satellite = node(&mut g);
    directed(&mut g, satellite, hub);
    bench_fn(|| {
        g.detach(satellite)
            .unwrap_or_else(|e| panic!("detach: {e}"));
        black_box(g.detach_step(satellite, budget(u32::MAX)))
    })
}

#[bench(raw)]
fn detach_satellite_1024() -> BenchResult {
    detach_satellite(1024)
}

#[bench(raw)]
fn detach_satellite_4096() -> BenchResult {
    detach_satellite(4096)
}

#[bench(raw)]
fn detach_satellite_65536() -> BenchResult {
    detach_satellite(65536)
}

/// How a bench detaches many nodes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Detaching {
    /// Each with its own detach steps.
    OneByOne,
    /// A sweep of the bucket they share first.
    Swept,
}

/// Detaches `n` of a hub's 65536 in-neighbours, spread evenly.
fn detach_many(n: u32, how: Detaching) -> BenchResult {
    let mut g = heap_graph();
    let hub = node(&mut g);
    let sources = nodes(&mut g, 65536);
    for &s in &sources {
        directed(&mut g, s, hub);
    }
    let step = usize::try_from(65536 / n).unwrap_or(1);
    let gone: Vec<NodeId> = sources.iter().step_by(step).copied().collect();
    bench_fn(|| {
        for &d in &gone {
            g.detach(d).unwrap_or_else(|e| panic!("detach: {e}"));
        }
        if how == Detaching::Swept {
            let detaching = g
                .detaching_set(gone.iter().copied())
                .unwrap_or_else(|e| panic!("detaching set: {e}"));
            sweep_all(&mut g, hub, &detaching);
        }
        for &d in &gone {
            detach_rest(&mut g, d);
        }
    })
}

/// Sweeps the hub's in-bucket of edges from `detaching`, to its end.
fn sweep_all(g: &mut Graph<VecMemory>, hub: NodeId, detaching: &lara_graph::DetachingSet) {
    let mut from = Some(Position::new(0));
    while let Some(at) = from {
        let sweep = g
            .sweep(
                hub,
                (LABEL, Direction::Incoming),
                (at, budget(4096)),
                detaching,
            )
            .unwrap_or_else(|e| panic!("sweep: {e}"));
        from = sweep.next();
    }
}

/// Detaches `v`'s remaining edges until it is deleted.
fn detach_rest(g: &mut Graph<VecMemory>, v: NodeId) {
    while g
        .detach_step(v, budget(u32::MAX))
        .unwrap_or_else(|e| panic!("step: {e}"))
        .progress()
        == Progress::Remaining
    {}
}

#[bench(raw)]
fn detach_1024_of_hub_65536() -> BenchResult {
    detach_many(1024, Detaching::OneByOne)
}

#[bench(raw)]
fn detach_1024_of_hub_65536_swept() -> BenchResult {
    detach_many(1024, Detaching::Swept)
}

#[bench(raw)]
fn detach_4096_of_hub_65536() -> BenchResult {
    detach_many(4096, Detaching::OneByOne)
}

#[bench(raw)]
fn detach_4096_of_hub_65536_swept() -> BenchResult {
    detach_many(4096, Detaching::Swept)
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
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, 65536);
    let halves: Vec<EdgeRef> = (0..1024)
        .map(|i| EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(i)))
        .collect();
    bench_fn(|| {
        let mut live = 0u64;
        for &half in &halves {
            live += u64::from(
                g.edge_target(half)
                    .unwrap_or_else(|err| panic!("edge: {err}"))
                    .is_some(),
            );
        }
        black_box(live)
    })
}

/// The same 1024 lookups through one `ReadBatch`, reusing the bucket descriptor.
#[bench(raw)]
fn edge_lookup_1024_batched() -> BenchResult {
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, 65536);
    let halves: Vec<EdgeRef> = (0..1024)
        .map(|i| EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(i)))
        .collect();
    bench_fn(|| {
        let mut batch = g.read_batch();
        let mut out = Vec::new();
        batch
            .edge_targets_each(&halves, &mut out)
            .unwrap_or_else(|err| panic!("edge batch: {err}"));
        let live: u64 = out.iter().map(|target| u64::from(target.is_some())).sum();
        black_box(live)
    })
}

/// 1024 edge values in one hub bucket, one call each.
#[bench(raw)]
fn edge_value_1024_loop() -> BenchResult {
    let mut g = heap_graph();
    let hub = seed_valued_hub(&mut g, 65536);
    let halves: Vec<EdgeRef> = (0..1024)
        .map(|i| EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(i)))
        .collect();
    bench_fn(|| {
        let mut live = 0u64;
        for &half in &halves {
            live += u64::from(
                g.edge_value(half)
                    .unwrap_or_else(|err| panic!("value: {err}"))
                    .is_some(),
            );
        }
        black_box(live)
    })
}

/// The same 1024 edge values through one `ReadBatch`, reusing the bucket lookup.
#[bench(raw)]
fn edge_value_1024_batched() -> BenchResult {
    let mut g = heap_graph();
    let hub = seed_valued_hub(&mut g, 65536);
    let halves: Vec<EdgeRef> = (0..1024)
        .map(|i| EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(i)))
        .collect();
    bench_fn(|| {
        let mut batch = g.read_batch();
        let mut out = Vec::new();
        batch
            .edge_values_each(&halves, &mut out)
            .unwrap_or_else(|err| panic!("value batch: {err}"));
        let live: u64 = out.iter().map(|value| u64::from(value.is_some())).sum();
        black_box(live)
    })
}

/// 1024 edge values in one hub bucket, one write each.
#[bench(raw)]
fn set_edge_value_1024_loop() -> BenchResult {
    let mut g = heap_graph();
    let hub = seed_valued_hub(&mut g, 65536);
    let halves: Vec<EdgeRef> = (0..1024)
        .map(|i| EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(i)))
        .collect();
    let value = [7u8; 8];
    bench_fn(|| {
        for &half in &halves {
            black_box(
                g.set_edge_value(half, &value)
                    .unwrap_or_else(|err| panic!("value: {err}")),
            );
        }
    })
}

/// The same 1024 writes through one batch.
#[bench(raw)]
fn set_edge_value_1024_batched() -> BenchResult {
    let mut g = heap_graph();
    let hub = seed_valued_hub(&mut g, 65536);
    let value = [7u8; 8];
    let writes: Vec<EdgeValue<'_>> = (0..1024)
        .map(|i| {
            EdgeValue::new(
                EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(i)),
                &value,
            )
        })
        .collect();
    bench_fn(|| {
        black_box(
            g.set_edge_values(&writes)
                .unwrap_or_else(|err| panic!("value batch: {err}")),
        )
    })
}

/// 1024 counterparts in one hub bucket, one call each.
#[bench(raw)]
fn counterpart_1024_loop() -> BenchResult {
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, 65536);
    let halves: Vec<EdgeRef> = (0..1024)
        .map(|i| EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(i)))
        .collect();
    bench_fn(|| {
        let mut live = 0u64;
        for &half in &halves {
            live += u64::from(
                g.counterpart(half)
                    .unwrap_or_else(|err| panic!("counterpart: {err}"))
                    .is_some(),
            );
        }
        black_box(live)
    })
}

/// The same 1024 counterparts through one `ReadBatch`, reusing the own bucket.
#[bench(raw)]
fn counterpart_1024_batched() -> BenchResult {
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, 65536);
    let halves: Vec<EdgeRef> = (0..1024)
        .map(|i| EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(i)))
        .collect();
    bench_fn(|| {
        let mut batch = g.read_batch();
        let mut out = Vec::new();
        batch
            .counterparts_each(&halves, &mut out)
            .unwrap_or_else(|err| panic!("counterpart batch: {err}"));
        let live: u64 = out.iter().map(|other| u64::from(other.is_some())).sum();
        black_box(live)
    })
}

/// A hub with `n` parallel edges to one target.
fn seed_parallel<M: Memory>(g: &mut Graph<M>, n: u32) -> (NodeId, Vec<EdgeRef>) {
    let hub = node(g);
    let target = node(g);
    for _ in 0..n {
        black_box(directed(g, hub, target));
    }
    let halves = (0..n)
        .map(|i| EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(i)))
        .collect();
    (hub, halves)
}

/// 1024 counterparts of parallel edges, one call each.
#[bench(raw)]
fn counterpart_parallel_1024_loop() -> BenchResult {
    let mut g = heap_graph();
    let (_, halves) = seed_parallel(&mut g, 1024);
    bench_fn(|| {
        let mut live = 0u64;
        for &half in &halves {
            live += u64::from(
                g.counterpart(half)
                    .unwrap_or_else(|err| panic!("counterpart: {err}"))
                    .is_some(),
            );
        }
        black_box(live)
    })
}

/// The same 1024 counterparts through one `ReadBatch`, reusing both buckets.
#[bench(raw)]
fn counterpart_parallel_1024_batched() -> BenchResult {
    let mut g = heap_graph();
    let (_, halves) = seed_parallel(&mut g, 1024);
    bench_fn(|| {
        let mut batch = g.read_batch();
        let mut out = Vec::new();
        batch
            .counterparts_each(&halves, &mut out)
            .unwrap_or_else(|err| panic!("counterpart batch: {err}"));
        let live: u64 = out.iter().map(|other| u64::from(other.is_some())).sum();
        black_box(live)
    })
}
