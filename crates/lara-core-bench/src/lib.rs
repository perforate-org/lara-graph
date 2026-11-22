//! Instruction counts of lara-core operations on the Internet Computer, measured with canbench.
//!
//! Scenarios mirror ic-stable-lara's `canbench_results.yml` where one exists, so the two can be
//! compared. Every edge here is stored at both ends, as Gleaph needs; ic-stable-lara's
//! single-label hub benches store one side only.

use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use core::num::NonZeroU32;
use core::ops::ControlFlow;
use ic_stable_structures::DefaultMemoryImpl;
use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
use lara_core::{
    EdgeRef, Graph, GrowFailed, Halves, Label, Memories, Memory, Position, Progress, Side,
    VecMemory, VertexId,
};

const LABEL: Label = Label::new(2);

/// One of the canister's stable memories, as a lara-core [`Memory`].
struct Stable(VirtualMemory<DefaultMemoryImpl>);

impl Memory for Stable {
    fn size(&self) -> u64 {
        ic_stable_structures::Memory::size(&self.0)
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        let previous = ic_stable_structures::Memory::grow(&self.0, pages);
        u64::try_from(previous).map_err(|_| GrowFailed::new(self.size(), pages))
    }

    fn read(&self, offset: u64, buf: &mut [u8]) {
        ic_stable_structures::Memory::read(&self.0, offset, buf);
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        ic_stable_structures::Memory::write(&self.0, offset, buf);
    }
}

fn heap_graph() -> Graph<VecMemory> {
    let memories = Memories::new(
        VecMemory::new(),
        VecMemory::new(),
        VecMemory::new(),
        VecMemory::new(),
        VecMemory::new(),
    );
    Graph::create(memories).unwrap_or_else(|e| panic!("create: {e}"))
}

/// A graph in five virtual stable memories (ids 0..5), as Gleaph wires it.
fn stable_graph() -> Graph<Stable> {
    let manager = MemoryManager::init(DefaultMemoryImpl::default());
    let memory = |id| Stable(manager.get(MemoryId::new(id)));
    let memories = Memories::new(memory(0), memory(1), memory(2), memory(3), memory(4));
    Graph::create(memories).unwrap_or_else(|e| panic!("create: {e}"))
}

fn vertex<M: Memory>(g: &mut Graph<M>) -> VertexId {
    g.add_vertex().unwrap_or_else(|e| panic!("vertex: {e}"))
}

fn vertices<M: Memory>(g: &mut Graph<M>, n: u32) -> Vec<VertexId> {
    (0..n).map(|_| vertex(g)).collect()
}

/// The last edge a seed inserted.
fn last(edges: &[Halves]) -> Halves {
    edges.last().copied().unwrap_or_else(|| panic!("no edges"))
}

fn directed<M: Memory>(g: &mut Graph<M>, from: VertexId, to: VertexId) -> Halves {
    g.insert_directed(from, LABEL, to)
        .unwrap_or_else(|e| panic!("insert: {e}"))
}

/// A hub with an edge to each of `degree` fresh leaves; returns the hub and the edges.
fn seed_hub<M: Memory>(g: &mut Graph<M>, degree: u32) -> (VertexId, Vec<Halves>) {
    let hub = vertex(g);
    let leaves = vertices(g, degree);
    let edges = leaves
        .into_iter()
        .map(|leaf| directed(g, hub, leaf))
        .collect();
    (hub, edges)
}

fn grow_hub<M: Memory>(mut g: Graph<M>, degree: u32) -> BenchResult {
    let hub = vertex(&mut g);
    let leaves = vertices(&mut g, degree);
    bench_fn(|| {
        for &leaf in &leaves {
            black_box(directed(&mut g, hub, black_box(leaf)));
        }
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

/// Parallel edges to a few targets: every insert also lands in a target's growing in bucket.
#[bench(raw)]
fn hub_insert_parallel_4096() -> BenchResult {
    let mut g = heap_graph();
    let hub = vertex(&mut g);
    let targets = vertices(&mut g, 8);
    bench_fn(|| {
        for &target in targets.iter().cycle().take(4096) {
            black_box(directed(&mut g, hub, target));
        }
    })
}

/// 51 full scans of a 1024-edge hub bucket (ic-stable-lara `bench_l_s2_h_scn_d_1024`).
fn scan_hub<M: Memory>(mut g: Graph<M>, degree: u32) -> BenchResult {
    let (hub, _) = seed_hub(&mut g, degree);
    let mut out = Vec::new();
    bench_fn(|| {
        for _ in 0..51 {
            g.neighbors(hub, LABEL, Side::Out, &mut out)
                .unwrap_or_else(|e| panic!("scan: {e}"));
            black_box(out.len());
        }
    })
}

/// The same scans through the visitor, counting edges without collecting them.
fn visit_hub<M: Memory>(mut g: Graph<M>, degree: u32) -> BenchResult {
    let (hub, _) = seed_hub(&mut g, degree);
    bench_fn(|| {
        for _ in 0..51 {
            let mut count = 0u32;
            g.visit_neighbors(hub, LABEL, Side::Out, |_| {
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
        for halves in edges.iter().step_by(2) {
            black_box(
                g.remove_edge(halves.source())
                    .unwrap_or_else(|e| panic!("{e}")),
            );
        }
        while g
            .compact(hub, LABEL, Side::Out)
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

/// Finds the out half of the hub's last edge from its in half at the leaf: a rank count over
/// the leaf's bucket and a selection in the hub's.
#[bench(raw)]
fn counterpart_into_hub_4096() -> BenchResult {
    let mut g = heap_graph();
    let (_, edges) = seed_hub(&mut g, 4096);
    let last = last(&edges);
    bench_fn(|| black_box(g.counterpart(last.target())))
}

/// Finds the in half from the hub's out half at its last position: the count scans the whole
/// hub bucket.
#[bench(raw)]
fn counterpart_from_hub_tail_4096() -> BenchResult {
    let mut g = heap_graph();
    let (_, edges) = seed_hub(&mut g, 4096);
    let last = last(&edges);
    bench_fn(|| black_box(g.counterpart(last.source())))
}

fn budget(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap_or(NonZeroU32::MIN)
}

/// Detaches a hub of `degree` leaves in steps of `step` edges (ic-stable-lara
/// `bench_l_s2_det_hub_<degree>` and its stepped variant).
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
    let hub = vertex(&mut g);
    for source in vertices(&mut g, degree) {
        directed(&mut g, source, hub);
    }
    let satellite = vertex(&mut g);
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

/// One edge lookup by handle in a paged hub bucket.
#[bench(raw)]
fn edge_lookup_65536() -> BenchResult {
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, 65536);
    let half = EdgeRef::new(hub, LABEL, Side::Out, Position::new(40_000));
    bench_fn(|| black_box(g.edge(half)))
}
