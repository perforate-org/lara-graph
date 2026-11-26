//! Growing a graph: nodes one at a time or in a batch, a hub's edges one at a time, batched or
//! bulk loaded, and small buckets changing form.

use crate::{LABEL, directed, graph, node, nodes};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use core::num::NonZeroU32;
use lara_graph::{Edge, EdgeKind, Graph, Memory};

/// 1024 nodes added one at a time.
#[bench(raw)]
fn add_nodes_1024_loop() -> BenchResult {
    let mut g = graph();
    bench_fn(|| {
        for _ in 0..1024 {
            black_box(g.add_node(&[]).unwrap_or_else(|e| panic!("node: {e}")));
        }
    })
}

/// The same 1024 nodes through one `add_nodes`.
#[bench(raw)]
fn add_nodes_1024_batched() -> BenchResult {
    let mut g = graph();
    bench_fn(|| {
        black_box(
            g.add_nodes(NonZeroU32::new(1024).unwrap_or(NonZeroU32::MIN), &[])
                .unwrap_or_else(|e| panic!("nodes: {e}")),
        )
    })
}

pub(crate) fn grow_hub<M: Memory>(mut g: Graph<M>, degree: u32) -> BenchResult {
    let hub = node(&mut g);
    let leaves = nodes(&mut g, degree);
    bench_fn(|| {
        for &leaf in &leaves {
            black_box(directed(&mut g, hub, black_box(leaf)));
        }
    })
}

fn bulk_hub<M: Memory>(mut g: Graph<M>, degree: u32) -> BenchResult {
    let hub = node(&mut g);
    let leaves = nodes(&mut g, degree);
    let edges: Vec<Edge<'_>> = leaves
        .iter()
        .map(|&leaf| Edge::directed(hub, LABEL, leaf))
        .collect();
    bench_fn(|| {
        black_box(g.bulk_load(&edges).unwrap_or_else(|e| panic!("bulk: {e}")));
    })
}

#[bench(raw)]
fn bulk_hub_4096() -> BenchResult {
    bulk_hub(graph(), 4096)
}

#[bench(raw)]
fn bulk_hub_65536() -> BenchResult {
    bulk_hub(graph(), 65536)
}

#[bench(raw)]
fn hub_insert_1024() -> BenchResult {
    grow_hub(graph(), 1024)
}

#[bench(raw)]
fn hub_insert_4096() -> BenchResult {
    grow_hub(graph(), 4096)
}

#[bench(raw)]
fn hub_insert_65536() -> BenchResult {
    grow_hub(graph(), 65536)
}

fn grow_hub_batched(mut g: Graph<impl Memory>, degree: u32) -> BenchResult {
    let hub = node(&mut g);
    let leaves = nodes(&mut g, degree);
    let edges: Vec<Edge<'_>> = leaves
        .iter()
        .map(|&leaf| Edge::directed(hub, LABEL, leaf))
        .collect();
    bench_fn(|| black_box(g.add_edges(&edges).unwrap_or_else(|e| panic!("batch: {e}"))))
}

#[bench(raw)]
fn hub_insert_4096_batched() -> BenchResult {
    grow_hub_batched(graph(), 4096)
}

fn grow_hub_undirected(mut g: Graph<impl Memory>, degree: u32) -> BenchResult {
    let hub = node(&mut g);
    let leaves = nodes(&mut g, degree);
    bench_fn(|| {
        for &leaf in &leaves {
            black_box(
                g.add_edge(Edge::undirected(hub, LABEL, black_box(leaf)))
                    .unwrap_or_else(|e| panic!("insert: {e}")),
            );
        }
    })
}

#[bench(raw)]
fn hub_insert_undirected_4096() -> BenchResult {
    grow_hub_undirected(graph(), 4096)
}

fn grow_hub_undirected_batched(mut g: Graph<impl Memory>, degree: u32) -> BenchResult {
    let hub = node(&mut g);
    let leaves = nodes(&mut g, degree);
    let edges: Vec<Edge<'_>> = leaves
        .iter()
        .map(|&leaf| Edge::undirected(hub, LABEL, leaf))
        .collect();
    bench_fn(|| black_box(g.add_edges(&edges).unwrap_or_else(|e| panic!("batch: {e}"))))
}

#[bench(raw)]
fn hub_insert_undirected_4096_batched() -> BenchResult {
    grow_hub_undirected_batched(graph(), 4096)
}

/// Parallel edges to a few targets: every insert also lands in a target's growing in bucket.
#[bench(raw)]
fn hub_insert_parallel_4096() -> BenchResult {
    let mut g = graph();
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
    let edges: Vec<Edge<'_>> = targets
        .iter()
        .cycle()
        .take(4096)
        .map(|&target| Edge::directed(hub, LABEL, target))
        .collect();
    bench_fn(|| black_box(g.add_edges(&edges).unwrap_or_else(|e| panic!("batch: {e}"))))
}

#[bench(raw)]
fn hub_insert_parallel_4096_batched() -> BenchResult {
    grow_hub_parallel_batched(graph())
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
    let edges: Vec<Edge<'_>> = leaves
        .iter()
        .zip(&kinds)
        .map(|(&leaf, &kind)| match kind {
            EdgeKind::Directed => Edge::directed(hub, LABEL, leaf),
            EdgeKind::Undirected => Edge::undirected(hub, LABEL, leaf),
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
            for &edge in &edges {
                black_box(g.add_edge(edge).unwrap_or_else(|e| panic!("insert: {e}")));
            }
            black_box(edges.len())
        }
    })
}

#[bench(raw)]
fn hub_insert_mixed_4096() -> BenchResult {
    grow_hub_mixed(graph(), 4096, Mixed::Loop)
}

#[bench(raw)]
fn hub_insert_mixed_4096_batched() -> BenchResult {
    grow_hub_mixed(graph(), 4096, Mixed::Batch)
}

/// Inserts the `nth` (1-based) out-edge of each of 512 nodes, whose earlier edges are in
/// place, to a fresh target each: the cost of a small bucket's growth. The third fills a tiny
/// bucket's last inline slot; the fourth moves it to a run.
fn grow_small(nth: u32) -> BenchResult {
    let mut g = graph();
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
