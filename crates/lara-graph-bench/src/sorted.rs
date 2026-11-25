//! Sorted views and shared targets (`sorted_targets`, `common_targets`, `target_set` with
//! `common_with`) by bucket size: a bucket is read and sorted in the heap. Each sorted read has an
//! unsorted one beside it, collecting the same targets in position order, so the difference is
//! the cost of the order; intersections with many nodes are measured a pair at a time and with
//! one set reused.

use crate::{LABEL, directed, heap_graph, node, nodes};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use core::ops::ControlFlow;
use lara_graph::{Direction, Graph, NodeId, Order, VecMemory};

const OUT: (lara_graph::Label, Direction) = (LABEL, Direction::Outgoing);

/// A xorshift generator, so every build is the same.
struct Rng(u64);

impl Rng {
    fn pick(&mut self, among: &[NodeId]) -> NodeId {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let len = u64::try_from(among.len()).unwrap_or(1).max(1);
        let at = usize::try_from(self.0 % len).unwrap_or(0);
        among
            .get(at)
            .copied()
            .unwrap_or_else(|| panic!("empty pool"))
    }
}

/// A node with `degree` edges to random nodes of `pool`.
fn spread(g: &mut Graph<VecMemory>, degree: u32, pool: &[NodeId], rng: &mut Rng) -> NodeId {
    let from = node(g);
    for _ in 0..degree {
        let to = rng.pick(pool);
        black_box(directed(g, from, to));
    }
    from
}

/// A node of `degree` edges to random targets among `2 × degree` nodes.
fn one(degree: u32) -> (Graph<VecMemory>, NodeId) {
    let mut g = heap_graph();
    let pool = nodes(&mut g, 2 * degree);
    let v = spread(&mut g, degree, &pool, &mut Rng(0x9E37_79B9_7F4A_7C15));
    (g, v)
}

fn unsorted(degree: u32) -> BenchResult {
    let (g, v) = one(degree);
    let mut out: Vec<NodeId> = Vec::new();
    bench_fn(|| {
        out.clear();
        g.visit_neighbors(v, OUT, Order::Ascending, |e| {
            out.push(e.target());
            ControlFlow::<()>::Continue(())
        })
        .unwrap_or_else(|e| panic!("visit: {e}"));
        black_box(out.len());
    })
}

fn sorted(degree: u32) -> BenchResult {
    let (g, v) = one(degree);
    let mut out = Vec::new();
    bench_fn(|| {
        g.sorted_targets(v, OUT, &mut out)
            .unwrap_or_else(|e| panic!("sorted: {e}"));
        black_box(out.len());
    })
}

/// Two nodes of `first` and `second` edges to random targets among the same
/// `2 × max(first, second)` nodes.
fn common(first: u32, second: u32) -> BenchResult {
    let mut g = heap_graph();
    let pool = nodes(&mut g, 2 * first.max(second));
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let one_end = spread(&mut g, first, &pool, &mut rng);
    let other_end = spread(&mut g, second, &pool, &mut rng);
    let mut out = Vec::new();
    bench_fn(|| {
        g.common_targets(one_end, other_end, OUT, &mut out)
            .unwrap_or_else(|e| panic!("common: {e}"));
        black_box(out.len());
    })
}

/// Nodes of the given degrees, each with edges to random targets among the same
/// `2 × max degree` nodes, intersected together.
fn common_of(degrees: &[u32]) -> BenchResult {
    let mut g = heap_graph();
    let pool = nodes(&mut g, 2 * degrees.iter().copied().max().unwrap_or(1));
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let ends: Vec<NodeId> = degrees
        .iter()
        .map(|&d| spread(&mut g, d, &pool, &mut rng))
        .collect();
    let mut out = Vec::new();
    bench_fn(|| {
        g.common_targets_of(&ends, OUT, &mut out)
            .unwrap_or_else(|e| panic!("common: {e}"));
        black_box(out.len());
    })
}

#[bench(raw)]
fn unsorted_16() -> BenchResult {
    unsorted(16)
}

#[bench(raw)]
fn sorted_16() -> BenchResult {
    sorted(16)
}

#[bench(raw)]
fn unsorted_64() -> BenchResult {
    unsorted(64)
}

#[bench(raw)]
fn sorted_64() -> BenchResult {
    sorted(64)
}

#[bench(raw)]
fn unsorted_256() -> BenchResult {
    unsorted(256)
}

#[bench(raw)]
fn sorted_256() -> BenchResult {
    sorted(256)
}

#[bench(raw)]
fn unsorted_1024() -> BenchResult {
    unsorted(1024)
}

#[bench(raw)]
fn sorted_1024() -> BenchResult {
    sorted(1024)
}

/// Paged: read whole and sorted like a run.
#[bench(raw)]
fn unsorted_4096() -> BenchResult {
    unsorted(4096)
}

#[bench(raw)]
fn sorted_4096() -> BenchResult {
    sorted(4096)
}

#[bench(raw)]
fn unsorted_65536() -> BenchResult {
    unsorted(65536)
}

#[bench(raw)]
fn sorted_65536() -> BenchResult {
    sorted(65536)
}

#[bench(raw)]
fn common_16() -> BenchResult {
    common(16, 16)
}

#[bench(raw)]
fn common_64() -> BenchResult {
    common(64, 64)
}

#[bench(raw)]
fn common_256() -> BenchResult {
    common(256, 256)
}

#[bench(raw)]
fn common_1024() -> BenchResult {
    common(1024, 1024)
}

/// Two hubs, both read whole: the first as a bitmap, the second looked up in it.
#[bench(raw)]
fn common_4096_4096() -> BenchResult {
    common(4096, 4096)
}

/// A hub and a small node: the small node's targets looked up in the hub's index.
#[bench(raw)]
fn common_65536_64() -> BenchResult {
    common(65536, 64)
}

#[bench(raw)]
fn common_65536_256() -> BenchResult {
    common(65536, 256)
}

/// Past the lookup ratio: the hub is read whole and looked up in the small one's bitmap.
#[bench(raw)]
fn common_65536_1024() -> BenchResult {
    common(65536, 1024)
}

/// Three nodes of 64 edges.
#[bench(raw)]
fn common3_64() -> BenchResult {
    common_of(&[64, 64, 64])
}

#[bench(raw)]
fn common3_256() -> BenchResult {
    common_of(&[256, 256, 256])
}

#[bench(raw)]
fn common3_1024() -> BenchResult {
    common_of(&[1024, 1024, 1024])
}

/// A hub, a node of 1024 edges and one of 64: the hub is looked up only for what the two others
/// share.
#[bench(raw)]
fn common3_65536_1024_64() -> BenchResult {
    common_of(&[65536, 1024, 64])
}

/// `nodes` nodes of `degree` edges each to random nodes among them.
fn random_graph(nodes_n: u32, degree: u32) -> (Graph<VecMemory>, Vec<NodeId>) {
    let mut g = heap_graph();
    let pool = nodes(&mut g, nodes_n);
    let mut rng = Rng(0x5851_F42D_4C95_7F2D);
    for &from in &pool {
        for _ in 0..degree {
            let to = rng.pick(&pool);
            black_box(directed(&mut g, from, to));
        }
    }
    (g, pool)
}

/// How a node is intersected with many others.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pairing {
    /// A `common_targets` per pair.
    Pairs,
    /// One `target_set`, intersected with each through `common_with`.
    Reused,
    /// One `target_set`, intersected with all through `ReadBatch::common_with_each`.
    Batched,
    /// As `Batched`, each triangle listed once: `u < w < x`.
    Ordered,
}

/// Lists the triangles of a random graph: for each edge `u -> w`, the targets `u` and `w` share.
fn triangles(nodes_n: u32, degree: u32, pairing: Pairing) -> BenchResult {
    let (g, pool) = random_graph(nodes_n, degree);
    let mut targets = Vec::new();
    let mut out = Vec::new();
    let mut batch = g.read_batch();
    bench_fn(|| {
        let mut found = 0usize;
        for &u in &pool {
            g.sorted_targets(u, OUT, &mut targets)
                .unwrap_or_else(|e| panic!("targets: {e}"));
            match pairing {
                Pairing::Pairs => {
                    for &w in &targets {
                        g.common_targets(u, w, OUT, &mut out)
                            .unwrap_or_else(|e| panic!("common: {e}"));
                        found += out.len();
                    }
                }
                Pairing::Reused => {
                    let mut set = g.target_set(u, OUT).unwrap_or_else(|e| panic!("set: {e}"));
                    for &w in &targets {
                        g.common_with(&mut set, (w, None), OUT, &mut out)
                            .unwrap_or_else(|e| panic!("common: {e}"));
                        found += out.len();
                    }
                }
                Pairing::Batched | Pairing::Ordered => {
                    let mut set = g.target_set(u, OUT).unwrap_or_else(|e| panic!("set: {e}"));
                    let ordered = pairing == Pairing::Ordered;
                    let from = if ordered {
                        targets.partition_point(|&w| w <= u)
                    } else {
                        0
                    };
                    batch
                        .common_with_each(
                            &mut set,
                            targets.get(from..).unwrap_or(&[]),
                            OUT,
                            |w| ordered.then_some(w),
                            |_, shared| {
                                found += shared.len();
                                ControlFlow::<()>::Continue(())
                            },
                        )
                        .unwrap_or_else(|e| panic!("common: {e}"));
                }
            }
        }
        black_box(found);
    })
}

/// Triangles of 1024 nodes of degree 16, a pair at a time.
#[bench(raw)]
fn triangles_1024_16_pairs() -> BenchResult {
    triangles(1024, 16, Pairing::Pairs)
}

/// The same, each node's set built once for its edges.
#[bench(raw)]
fn triangles_1024_16_reused() -> BenchResult {
    triangles(1024, 16, Pairing::Reused)
}

#[bench(raw)]
fn triangles_1024_16_batched() -> BenchResult {
    triangles(1024, 16, Pairing::Batched)
}

/// Each triangle once, as `u < w < x`.
#[bench(raw)]
fn triangles_1024_16_ordered() -> BenchResult {
    triangles(1024, 16, Pairing::Ordered)
}

#[bench(raw)]
fn triangles_256_64_pairs() -> BenchResult {
    triangles(256, 64, Pairing::Pairs)
}

#[bench(raw)]
fn triangles_256_64_reused() -> BenchResult {
    triangles(256, 64, Pairing::Reused)
}

/// One node of 1024 edges against 64 nodes of 64 edges, all among the same 2048 nodes:
/// `common_targets` per pair, or one set intersected with each.
fn one_against_many(pairing: Pairing) -> BenchResult {
    let mut g = heap_graph();
    let pool = nodes(&mut g, 2048);
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let big = spread(&mut g, 1024, &pool, &mut rng);
    let small: Vec<NodeId> = (0..64)
        .map(|_| spread(&mut g, 64, &pool, &mut rng))
        .collect();
    let mut out = Vec::new();
    bench_fn(|| {
        let mut found = 0usize;
        if pairing == Pairing::Reused {
            let mut set = g
                .target_set(big, OUT)
                .unwrap_or_else(|e| panic!("set: {e}"));
            for &w in &small {
                g.common_with(&mut set, (w, None), OUT, &mut out)
                    .unwrap_or_else(|e| panic!("common: {e}"));
                found += out.len();
            }
        } else {
            for &w in &small {
                g.common_targets(big, w, OUT, &mut out)
                    .unwrap_or_else(|e| panic!("common: {e}"));
                found += out.len();
            }
        }
        black_box(found);
    })
}

#[bench(raw)]
fn common_1024_x64_pairs() -> BenchResult {
    one_against_many(Pairing::Pairs)
}

#[bench(raw)]
fn common_1024_x64_reused() -> BenchResult {
    one_against_many(Pairing::Reused)
}

#[bench(raw)]
fn triangles_256_64_batched() -> BenchResult {
    triangles(256, 64, Pairing::Batched)
}

#[bench(raw)]
fn triangles_256_64_ordered() -> BenchResult {
    triangles(256, 64, Pairing::Ordered)
}
