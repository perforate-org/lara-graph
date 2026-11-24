//! Node labels: a far node's label checked during an expansion, as `(a)-[]->(b:L)` does for
//! every neighbour, one node at a time.

use crate::{LABEL, directed, heap_graph, node, nodes, stable_graph};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use core::ops::ControlFlow;
use lara_graph::{Direction, Graph, Label, Memory, NodeId, Order};

/// `n` nodes, each labelled with `per_node` labels from a pool of 16, the one checked among them
/// for every other node.
fn labelled<M: Memory>(g: &mut Graph<M>, n: u32, per_node: u16) -> Vec<NodeId> {
    let all = nodes(g, n);
    for (i, &x) in (0u16..).zip(&all) {
        let first = i % 2;
        let labels: Vec<Label> = (0..per_node).map(|k| Label::new(first + 2 * k)).collect();
        g.set_node_labels(x, &labels)
            .unwrap_or_else(|e| panic!("labels: {e}"));
    }
    all
}

fn check_each<M: Memory>(g: &Graph<M>, all: &[NodeId]) -> u32 {
    let mut hits = 0;
    for &x in all {
        if g.has_node_label(x, Label::new(0))
            .unwrap_or_else(|e| panic!("label: {e}"))
        {
            hits += 1;
        }
    }
    hits
}

/// 1024 label checks on nodes with one label, in heap memory.
#[bench(raw)]
fn has_node_label_1024() -> BenchResult {
    let mut g = heap_graph();
    let all = labelled(&mut g, 1024, 1);
    bench_fn(|| black_box(check_each(&g, &all)))
}

/// The same in virtual stable memory, as Gleaph's `label_checks` runs.
#[bench(raw)]
fn has_node_label_1024_stable() -> BenchResult {
    let mut g = stable_graph();
    let all = labelled(&mut g, 1024, 1);
    bench_fn(|| black_box(check_each(&g, &all)))
}

/// 1024 label checks on nodes with eight labels each: a binary search of a spilled set.
#[bench(raw)]
fn has_node_label_spilled_1024_stable() -> BenchResult {
    let mut g = stable_graph();
    let all = labelled(&mut g, 1024, 8);
    bench_fn(|| black_box(check_each(&g, &all)))
}

/// An expansion from a hub of 1024 edges keeping the neighbours with a label:
/// `(hub)-[]->(b:L)`.
#[bench(raw)]
fn expand_label_filter_1024_stable() -> BenchResult {
    let mut g = stable_graph();
    let hub = node(&mut g);
    let all = labelled(&mut g, 1024, 1);
    for &x in &all {
        directed(&mut g, hub, x);
    }
    bench_fn(|| {
        let mut kept = 0u32;
        g.visit_neighbors(hub, LABEL, Direction::Outgoing, Order::Ascending, |e| {
            if g.has_node_label(e.target(), Label::new(0))
                .unwrap_or_else(|err| panic!("label: {err}"))
            {
                kept += 1;
            }
            ControlFlow::<()>::Continue(())
        })
        .unwrap_or_else(|e| panic!("visit: {e}"));
        black_box(kept)
    })
}
