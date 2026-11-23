//! Locality across nodes and the parts of one node visit, over the two-hop graph of
//! [`crate::two_hop`].
//!
//! The `two_hop_*` benches traverse the same graph in three layouts. The others take one node
//! visit apart, each over the same 1024 satellites in stable memory: a raw read, then the node
//! record, the label bucket list, an inline bucket, and a run bucket's slots.

use crate::two_hop::{self, Layout};
use crate::{heap_graph, stable_graph};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use ic_stable_structures::DefaultMemoryImpl;
use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
use lara_graph::{Direction, Graph, Memory, NodeId};

/// Two hops from the hub, in `layout`.
fn two_hop<M: Memory>(mut g: Graph<M>, layout: Layout) -> BenchResult {
    let hub = two_hop::build(&mut g, layout);
    bench_fn(|| black_box(two_hop::visit(&g, hub)))
}

#[bench(raw)]
fn two_hop_ordered() -> BenchResult {
    two_hop(heap_graph(), Layout::Ordered)
}

#[bench(raw)]
fn two_hop_shuffled() -> BenchResult {
    two_hop(heap_graph(), Layout::Shuffled)
}

#[bench(raw)]
fn two_hop_scattered() -> BenchResult {
    two_hop(heap_graph(), Layout::Scattered)
}

#[bench(raw)]
fn two_hop_ordered_stable() -> BenchResult {
    two_hop(stable_graph(), Layout::Ordered)
}

#[bench(raw)]
fn two_hop_shuffled_stable() -> BenchResult {
    two_hop(stable_graph(), Layout::Shuffled)
}

#[bench(raw)]
fn two_hop_scattered_stable() -> BenchResult {
    two_hop(stable_graph(), Layout::Scattered)
}

/// Calls `f` on each satellite of the ordered two-hop graph in stable memory, inside the bench.
fn per_satellite<R>(
    f: impl Fn(&Graph<VirtualMemory<DefaultMemoryImpl>>, NodeId) -> R,
) -> BenchResult {
    let mut g = stable_graph();
    let hub = two_hop::build(&mut g, Layout::Ordered);
    let satellites = two_hop::satellites(&g, hub);
    bench_fn(|| {
        for &s in &satellites {
            black_box(f(&g, black_box(s)));
        }
    })
}

/// 1024 reads of 8 bytes from a virtual stable memory, `stride` bytes apart, over 4 buckets of
/// the memory manager (128 wasm pages each). A read in the bucket of the one before hits the
/// manager's cache of the last bucket; one in another bucket takes its slow path.
fn read_8_bytes(stride: u64) -> BenchResult {
    let manager = MemoryManager::init(DefaultMemoryImpl::default());
    let memory = manager.get(MemoryId::new(0));
    Memory::grow(&memory, 4 * 128).unwrap_or_else(|e| panic!("grow: {e}"));
    let span = 4 * 128 * 65536;
    let mut buf = [0u8; 8];
    bench_fn(|| {
        for i in 0..1024u64 {
            Memory::read(&memory, black_box(i * stride % span), &mut buf);
            black_box(&buf);
        }
    })
}

/// Every read in the same bucket: 4096 bytes apart.
#[bench(raw)]
fn read_8_bytes_1024() -> BenchResult {
    read_8_bytes(4096)
}

/// Every read in another bucket than the one before: one bucket (8 MiB) and 4096 bytes apart.
#[bench(raw)]
fn read_8_bytes_1024_other_bucket() -> BenchResult {
    read_8_bytes(128 * 65536 + 4096)
}

/// Reads each satellite's node record.
#[bench(raw)]
fn node_state_1024() -> BenchResult {
    per_satellite(|g, s| g.node_state(s).unwrap_or_else(|e| panic!("{e}")))
}

/// Reads each satellite's record and looks its out bucket up in its list (no slots).
#[bench(raw)]
fn degree_1024() -> BenchResult {
    per_satellite(|g, s| two_hop::degree(g, s, Direction::Outgoing))
}

/// Visits each satellite's in bucket: one edge, held inline in its list entry.
#[bench(raw)]
fn visit_inline_1024() -> BenchResult {
    per_satellite(|g, s| two_hop::visit_one(g, s, Direction::Incoming))
}

/// Visits each satellite's out bucket: eight edges in a run.
#[bench(raw)]
fn visit_run_1024() -> BenchResult {
    per_satellite(|g, s| two_hop::visit_one(g, s, Direction::Outgoing))
}

/// The same out-bucket visits, batched over the satellite list: one record line per eight nodes.
#[bench(raw)]
fn visit_run_1024_batched() -> BenchResult {
    let mut g = stable_graph();
    let hub = two_hop::build(&mut g, Layout::Ordered);
    let satellites = two_hop::satellites(&g, hub);
    bench_fn(|| black_box(two_hop::visit_many(&g, &satellites, Direction::Outgoing)))
}

/// The in-bucket visits, batched over the satellite list.
#[bench(raw)]
fn visit_inline_1024_batched() -> BenchResult {
    let mut g = stable_graph();
    let hub = two_hop::build(&mut g, Layout::Ordered);
    let satellites = two_hop::satellites(&g, hub);
    bench_fn(|| black_box(two_hop::visit_many(&g, &satellites, Direction::Incoming)))
}

/// The out-bucket visits in the shuffled layout, where ids are dense but list pages are not.
#[bench(raw)]
fn visit_run_1024_batched_shuffled() -> BenchResult {
    let mut g = stable_graph();
    let hub = two_hop::build(&mut g, Layout::Shuffled);
    let satellites = two_hop::satellites(&g, hub);
    bench_fn(|| black_box(two_hop::visit_many(&g, &satellites, Direction::Outgoing)))
}

/// The out-bucket visits in the scattered layout, where consecutive ids do not share a line.
#[bench(raw)]
fn visit_run_1024_batched_scattered() -> BenchResult {
    let mut g = stable_graph();
    let hub = two_hop::build(&mut g, Layout::Scattered);
    let satellites = two_hop::satellites(&g, hub);
    bench_fn(|| black_box(two_hop::visit_many(&g, &satellites, Direction::Outgoing)))
}

/// The out-bucket visits through an explicit `ReadBatch` context.
#[bench(raw)]
fn visit_run_1024_batch_context() -> BenchResult {
    let mut g = stable_graph();
    let hub = two_hop::build(&mut g, Layout::Ordered);
    let satellites = two_hop::satellites(&g, hub);
    let mut batch = g.read_batch();
    bench_fn(|| {
        black_box(two_hop::visit_many_batch(
            &mut batch,
            &satellites,
            Direction::Outgoing,
        ))
    })
}

/// The out-degree of every satellite through one `ReadBatch`.
#[bench(raw)]
fn degree_1024_batched() -> BenchResult {
    let mut g = stable_graph();
    let hub = two_hop::build(&mut g, Layout::Ordered);
    let satellites = two_hop::satellites(&g, hub);
    bench_fn(|| black_box(two_hop::degree_many(&g, &satellites, Direction::Outgoing)))
}

/// The state of every satellite through one `ReadBatch`.
#[bench(raw)]
fn node_state_1024_batched() -> BenchResult {
    let mut g = stable_graph();
    let hub = two_hop::build(&mut g, Layout::Ordered);
    let satellites = two_hop::satellites(&g, hub);
    bench_fn(|| black_box(two_hop::node_state_many(&g, &satellites)))
}

/// Reads each satellite's outgoing labels one node at a time.
#[bench(raw)]
fn labels_1024() -> BenchResult {
    per_satellite(|g, s| {
        let mut labels = Vec::new();
        g.labels(s, Direction::Outgoing, &mut labels)
            .unwrap_or_else(|err| panic!("labels: {err}"));
        labels.len()
    })
}

/// The same label reads through one `ReadBatch`.
#[bench(raw)]
fn labels_1024_batched() -> BenchResult {
    let mut g = stable_graph();
    let hub = two_hop::build(&mut g, Layout::Ordered);
    let satellites = two_hop::satellites(&g, hub);
    bench_fn(|| black_box(two_hop::labels_many(&g, &satellites, Direction::Outgoing)))
}

/// Checks each satellite for live edges one node at a time.
#[bench(raw)]
fn has_edges_1024() -> BenchResult {
    per_satellite(|g, s| {
        g.has_edges(s)
            .unwrap_or_else(|err| panic!("has_edges: {err}"))
    })
}

/// The same edge checks through one `ReadBatch`.
#[bench(raw)]
fn has_edges_1024_batched() -> BenchResult {
    let mut g = stable_graph();
    let hub = two_hop::build(&mut g, Layout::Ordered);
    let satellites = two_hop::satellites(&g, hub);
    bench_fn(|| black_box(two_hop::has_edges_many(&g, &satellites)))
}
