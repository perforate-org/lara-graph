//! Edges with 8-byte values: growing a valued hub, scanning values with or without their edges,
//! one node at a time or batched, and reading or overwriting values by handle.

use crate::{LABEL, first_halves, heap_graph, live_batched, live_each, node, nodes};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use core::ops::ControlFlow;
use lara_graph::{
    Direction, EdgeRef, EdgeValue, Error, Graph, NewEdge, NodeId, Order, Placement, Position,
    ReadBatch, VecMemory, Width,
};

const OUT: (lara_graph::Label, Direction) = (LABEL, Direction::Outgoing);

/// Declares `LABEL`'s values 8 bytes wide.
fn declare_width(g: &mut Graph<VecMemory>) {
    g.declare_width(LABEL, Width::new(8).unwrap_or_else(|| panic!("width")))
        .unwrap_or_else(|e| panic!("width: {e}"));
}

/// Inserts `from -[LABEL]-> to` carrying `value`.
fn valued(g: &mut Graph<VecMemory>, from: NodeId, to: NodeId, value: &[u8]) {
    g.add_edge(NewEdge::directed(from, LABEL, to, Placement::Append, value))
        .unwrap_or_else(|e| panic!("insert: {e}"));
}

/// A hub whose label carries 8-byte values, each edge given one.
fn seed_valued_hub(g: &mut Graph<VecMemory>, degree: u32) -> NodeId {
    declare_width(g);
    let hub = node(g);
    for leaf in nodes(g, degree) {
        valued(g, hub, leaf, &u64::from(leaf.index()).to_le_bytes());
    }
    hub
}

/// 1024 sources, each with one valued out-edge.
fn one_valued_edge_per_source() -> (Graph<VecMemory>, Vec<NodeId>) {
    let mut g = heap_graph();
    declare_width(&mut g);
    let sources: Vec<NodeId> = (0..1024)
        .map(|i| {
            let source = node(&mut g);
            let target = node(&mut g);
            let value = [u8::try_from(i % 251).unwrap_or_else(|_| panic!("byte")); 8];
            valued(&mut g, source, target, &value);
            source
        })
        .collect();
    (g, sources)
}

/// Counts one visited edge.
fn tick(count: &mut u32) -> ControlFlow<()> {
    *count += 1;
    ControlFlow::Continue(())
}

/// Visits each of `sources` with `visit`, which ticks the count it is given; returns the count.
fn count_each(
    sources: &[NodeId],
    mut visit: impl FnMut(NodeId, &mut u32) -> Result<Option<()>, Error>,
) -> u32 {
    let mut count = 0u32;
    for &source in sources {
        visit(source, &mut count).unwrap_or_else(|e| panic!("visit: {e}"));
    }
    count
}

/// Growing a hub to 4096 edges with 8-byte values: both halves of each.
#[bench(raw)]
fn hub_insert_valued_4096() -> BenchResult {
    let mut g = heap_graph();
    declare_width(&mut g);
    let hub = node(&mut g);
    let leaves = nodes(&mut g, 4096);
    bench_fn(|| {
        for &leaf in &leaves {
            let value = u64::from(leaf.index()).to_le_bytes();
            black_box(
                g.add_edge(NewEdge::directed(
                    hub,
                    LABEL,
                    leaf,
                    Placement::Append,
                    &value,
                ))
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
            g.visit_values(hub, (OUT.0, OUT.1), Order::Ascending, |_, v| {
                sum = sum.wrapping_add(u64::from(v.first().copied().unwrap_or(0)));
                ControlFlow::<()>::Continue(())
            })
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
            g.visit_neighbors_with_values(hub, (OUT.0, OUT.1), Order::Ascending, |_, _| {
                tick(&mut count)
            })
            .unwrap_or_else(|e| panic!("visit: {e}"));
            black_box(count);
        }
    })
}

/// 1024 sources, each with one valued out-edge, one node at a time.
#[bench(raw)]
fn visit_with_values_many_1024_loop() -> BenchResult {
    let (g, sources) = one_valued_edge_per_source();
    bench_fn(|| {
        black_box(count_each(&sources, |source, count| {
            g.visit_neighbors_with_values(source, (OUT.0, OUT.1), Order::Ascending, |_, _| {
                tick(count)
            })
        }))
    })
}

/// The same 1024 sources through one `ReadBatch`.
#[bench(raw)]
fn visit_with_values_many_1024_batched() -> BenchResult {
    let (g, sources) = one_valued_edge_per_source();
    bench_fn(|| {
        let mut batch = g.read_batch();
        let mut count = 0u32;
        batch
            .visit_neighbors_with_values_each(
                &sources,
                (OUT.0, OUT.1),
                Order::Ascending,
                |_, _, _| tick(&mut count),
            )
            .unwrap_or_else(|e| panic!("batch: {e}"));
        black_box(count)
    })
}

#[bench(raw)]
fn visit_values_many_1024_loop() -> BenchResult {
    let (g, sources) = one_valued_edge_per_source();
    bench_fn(|| {
        black_box(count_each(&sources, |source, count| {
            g.visit_values(source, (OUT.0, OUT.1), Order::Ascending, |_, _| tick(count))
        }))
    })
}

#[bench(raw)]
fn visit_values_many_1024_batched() -> BenchResult {
    let (g, sources) = one_valued_edge_per_source();
    bench_fn(|| {
        let mut batch = g.read_batch();
        let mut count = 0u32;
        batch
            .visit_values_each(&sources, (OUT.0, OUT.1), Order::Ascending, |_, _, _| {
                tick(&mut count)
            })
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

/// 1024 edge values in one hub bucket, one call each.
#[bench(raw)]
fn edge_value_1024_loop() -> BenchResult {
    let mut g = heap_graph();
    let halves = first_halves(seed_valued_hub(&mut g, 65536), 1024);
    bench_fn(|| black_box(live_each(&halves, |half| g.edge_value(half))))
}

/// The same 1024 edge values through one `ReadBatch`, reusing the bucket lookup.
#[bench(raw)]
fn edge_value_1024_batched() -> BenchResult {
    let mut g = heap_graph();
    let halves = first_halves(seed_valued_hub(&mut g, 65536), 1024);
    bench_fn(|| black_box(live_batched(&g, &halves, ReadBatch::edge_values_each)))
}

/// 1024 edge values in one hub bucket, one write each.
#[bench(raw)]
fn set_edge_value_1024_loop() -> BenchResult {
    let mut g = heap_graph();
    let halves = first_halves(seed_valued_hub(&mut g, 65536), 1024);
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
    let halves = first_halves(hub, 1024);
    let writes: Vec<EdgeValue<'_>> = halves
        .iter()
        .map(|&half| EdgeValue::new(half, &value))
        .collect();
    bench_fn(|| {
        black_box(
            g.set_edge_values(&writes)
                .unwrap_or_else(|err| panic!("value batch: {err}")),
        )
    })
}
