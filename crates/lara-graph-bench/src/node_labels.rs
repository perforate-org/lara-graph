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

/// The 1024 checks of `has_node_label_1024_stable` as one batch: eight cells read at once.
#[bench(raw)]
fn has_node_label_1024_batched_stable() -> BenchResult {
    let mut g = stable_graph();
    let all = labelled(&mut g, 1024, 1);
    bench_fn(|| {
        let mut has = Vec::new();
        g.read_batch()
            .has_node_label_each(&all, Label::new(0), &mut has)
            .unwrap_or_else(|e| panic!("labels: {e}"));
        black_box(has.iter().filter(|&&h| h).count())
    })
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
        g.visit_neighbors(hub, (LABEL, Direction::Outgoing), Order::Ascending, |e| {
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

/// Labels 1024 nodes with one label each, in stable memory.
#[bench(raw)]
fn set_node_labels_1024_stable() -> BenchResult {
    let mut g = stable_graph();
    let all = nodes(&mut g, 1024);
    bench_fn(|| {
        for &x in &all {
            g.set_node_labels(x, &[Label::new(3)])
                .unwrap_or_else(|e| panic!("labels: {e}"));
        }
    })
}

/// Labels 1024 nodes with eight labels each: a spilled set per node.
#[bench(raw)]
fn set_node_labels_spilled_1024_stable() -> BenchResult {
    let mut g = stable_graph();
    let all = nodes(&mut g, 1024);
    let labels: Vec<Label> = (0..8).map(Label::new).collect();
    bench_fn(|| {
        for &x in &all {
            g.set_node_labels(x, &labels)
                .unwrap_or_else(|e| panic!("labels: {e}"));
        }
    })
}

/// 10,000 nodes, the first `rare` with label 1 and the rest with label 0, as Gleaph's
/// `node_scan` benches lay them out.
fn scan_fixture<M: Memory>(g: &mut Graph<M>, rare: u32) {
    for (i, x) in (0..).zip(nodes(g, 10_000)) {
        let label = Label::new(u16::from(i < rare));
        g.set_node_labels(x, &[label])
            .unwrap_or_else(|e| panic!("labels: {e}"));
    }
}

fn scan(g: &Graph<impl Memory>, label: Label) -> u32 {
    let mut n = 0;
    g.visit_nodes_with_label(label, None, |_| {
        n += 1;
        ControlFlow::<()>::Continue(())
    });
    n
}

/// The nodes of a label 16 of 10,000 nodes have, in stable memory: Gleaph's rare-label scan.
#[bench(raw)]
fn nodes_with_label_rare_10k_stable() -> BenchResult {
    let mut g = stable_graph();
    scan_fixture(&mut g, 16);
    bench_fn(|| black_box(scan(&g, Label::new(1))))
}

/// The nodes of a label 9,984 of 10,000 nodes have: Gleaph's common-label scan.
#[bench(raw)]
fn nodes_with_label_common_10k_stable() -> BenchResult {
    let mut g = stable_graph();
    scan_fixture(&mut g, 16);
    bench_fn(|| black_box(scan(&g, Label::new(0))))
}
