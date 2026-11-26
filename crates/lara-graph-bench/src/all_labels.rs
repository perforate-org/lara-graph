//! Reading each node's edges under every label on one direction: a loop of `visit_neighbors` over
//! `edge_labels`, against `visit_all_neighbors`, one list walk a node, and its batched form.
//! 1024 nodes with 8 labels of 4 out-edges each, or 1 label of 32.

use crate::{Bench, graph, nodes};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use core::ops::ControlFlow;
use lara_graph::{Direction, Edge, EdgeLabel, NodeId, Order};

const NODES: u32 = 1024;
const EDGES_PER_NODE: u32 = 32;

/// 1024 nodes, each with `EDGES_PER_NODE` out-edges spread over `labels` labels.
fn build(labels: u16) -> (Bench, Vec<NodeId>) {
    let mut g = graph();
    let all = nodes(&mut g, NODES);
    for (i, &from) in all.iter().enumerate() {
        for k in 0..EDGES_PER_NODE {
            let at = (i + usize::try_from(k).unwrap_or(0) * 31 + 1) % all.len();
            let to = all.get(at).copied().unwrap_or(from);
            let label = EdgeLabel::new(u16::try_from(k % u32::from(labels)).unwrap_or(0));
            g.add_edge(Edge::directed(from, label, to))
                .unwrap_or_else(|e| panic!("insert: {e}"));
        }
    }
    (g, all)
}

fn by_labels(labels: u16) -> BenchResult {
    let (g, all) = build(labels);
    let mut found = Vec::new();
    bench_fn(|| {
        let mut count = 0u32;
        for &v in &all {
            g.edge_labels(v, Direction::Outgoing, &mut found)
                .unwrap_or_else(|e| panic!("labels: {e}"));
            for &label in &found {
                g.visit_neighbors(v, label.outgoing(), Order::Ascending, |_| {
                    count += 1;
                    ControlFlow::<()>::Continue(())
                })
                .unwrap_or_else(|e| panic!("visit: {e}"));
            }
        }
        black_box(count);
    })
}

fn all_at_once(labels: u16) -> BenchResult {
    let (g, all) = build(labels);
    bench_fn(|| {
        let mut count = 0u32;
        for &v in &all {
            g.visit_all_neighbors(v, Direction::Outgoing, Order::Ascending, |_, _| {
                count += 1;
                ControlFlow::<()>::Continue(())
            })
            .unwrap_or_else(|e| panic!("visit: {e}"));
        }
        black_box(count);
    })
}

fn batched(labels: u16) -> BenchResult {
    let (g, all) = build(labels);
    bench_fn(|| {
        let mut count = 0u32;
        g.read_batch()
            .visit_all_neighbors_each(&all, Direction::Outgoing, Order::Ascending, |_, _, _| {
                count += 1;
                ControlFlow::<()>::Continue(())
            })
            .unwrap_or_else(|e| panic!("visit: {e}"));
        black_box(count);
    })
}

#[bench(raw)]
fn all_labels_by_label_8() -> BenchResult {
    by_labels(8)
}

#[bench(raw)]
fn all_labels_visit_8() -> BenchResult {
    all_at_once(8)
}

#[bench(raw)]
fn all_labels_batched_8() -> BenchResult {
    batched(8)
}

#[bench(raw)]
fn all_labels_by_label_1() -> BenchResult {
    by_labels(1)
}

#[bench(raw)]
fn all_labels_visit_1() -> BenchResult {
    all_at_once(1)
}

#[bench(raw)]
fn all_labels_batched_1() -> BenchResult {
    batched(1)
}
