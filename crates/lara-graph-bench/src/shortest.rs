//! Single-source shortest paths (Dijkstra) over a weighted 32 × 32 grid: a sparse, road-like
//! graph whose nodes have at most four out-edges, each carrying an 8-byte weight. The visit
//! with values is the read Dijkstra makes for every node it settles, so its per-node cost sets
//! the cost of the whole search. A breadth-first search over the same grid, without values,
//! separates that read from the queue's work.

use crate::{LABEL, heap_graph, nodes, stable_graph};
use alloc::collections::{BinaryHeap, VecDeque};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::cmp::Reverse;
use core::hint::black_box;
use core::ops::ControlFlow;
use lara_graph::{Direction, Graph, Memory, NewEdge, NodeId, Order, Placement, Width};

const SIDE: u32 = 32;
const OUT: (lara_graph::Label, Direction) = (LABEL, Direction::Outgoing);

/// A `SIDE` × `SIDE` grid with an edge each way between neighbours, weighted 1..=100 by a fixed
/// sequence; returns the nodes in row-major order.
fn grid<M: Memory>(g: &mut Graph<M>) -> Vec<NodeId> {
    g.declare_width(LABEL, Width::new(8).unwrap_or_else(|| panic!("width")))
        .unwrap_or_else(|e| panic!("width: {e}"));
    let all = nodes(g, SIDE * SIDE);
    let mut seed = 0x2545_f491_u64;
    let mut weight = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) % 100 + 1
    };
    let at = |row: u32, col: u32| {
        all.get(usize::try_from(row * SIDE + col).unwrap_or(usize::MAX))
            .copied()
            .unwrap_or_else(|| panic!("grid node"))
    };
    let pairs = (0..SIDE).flat_map(|row| (0..SIDE).map(move |col| (row, col)));
    let links: Vec<(NodeId, NodeId)> = pairs
        .flat_map(|(row, col)| {
            let here = at(row, col);
            let right = (col + 1 < SIDE).then(|| at(row, col + 1));
            let down = (row + 1 < SIDE).then(|| at(row + 1, col));
            [right, down]
                .into_iter()
                .flatten()
                .flat_map(move |there| [(here, there), (there, here)])
        })
        .collect();
    for (from, to) in links {
        let w = weight().to_le_bytes();
        g.add_edge(NewEdge::directed(from, LABEL, to, Placement::Append, &w))
            .unwrap_or_else(|e| panic!("insert: {e}"));
    }
    all
}

/// The index of `v` in the distance table.
fn slot(v: NodeId) -> usize {
    usize::try_from(v.index()).unwrap_or(usize::MAX)
}

/// Dijkstra from `source`: the sum of every reachable node's distance.
fn dijkstra<M: Memory>(g: &Graph<M>, all: &[NodeId], source: NodeId) -> u64 {
    let mut dist = vec![u64::MAX; all.len()];
    let mut queue = BinaryHeap::new();
    if let Some(d) = dist.get_mut(slot(source)) {
        *d = 0;
    }
    queue.push(Reverse((0u64, source.index())));
    while let Some(Reverse((d, index))) = queue.pop() {
        let Some(&v) = all.get(usize::try_from(index).unwrap_or(usize::MAX)) else {
            continue;
        };
        if dist.get(slot(v)).is_some_and(|&best| d > best) {
            continue;
        }
        g.visit_neighbors_with_values(v, (OUT.0, OUT.1), Order::Ascending, |e, value| {
            let w = value
                .first_chunk::<8>()
                .map_or(u64::MAX, |bytes| u64::from_le_bytes(*bytes));
            let next = d.saturating_add(w);
            if let Some(best) = dist.get_mut(slot(e.target()))
                && next < *best
            {
                *best = next;
                queue.push(Reverse((next, e.target().index())));
            }
            ControlFlow::<()>::Continue(())
        })
        .unwrap_or_else(|e| panic!("visit: {e}"));
    }
    dist.iter().filter(|&&d| d != u64::MAX).sum()
}

/// Breadth-first search from `source`, reading edges without values: how many nodes it reaches.
fn bfs<M: Memory>(g: &Graph<M>, all: &[NodeId], source: NodeId) -> u32 {
    let mut seen = vec![false; all.len()];
    let mut queue = VecDeque::from([source]);
    if let Some(s) = seen.get_mut(slot(source)) {
        *s = true;
    }
    let mut reached = 0;
    while let Some(v) = queue.pop_front() {
        reached += 1;
        g.visit_neighbors(v, (OUT.0, OUT.1), Order::Ascending, |e| {
            if let Some(s) = seen.get_mut(slot(e.target()))
                && !*s
            {
                *s = true;
                queue.push_back(e.target());
            }
            ControlFlow::<()>::Continue(())
        })
        .unwrap_or_else(|e| panic!("visit: {e}"));
    }
    reached
}

fn first(all: &[NodeId]) -> NodeId {
    all.first().copied().unwrap_or_else(|| panic!("empty grid"))
}

/// Dijkstra over the grid in heap memory, from a corner: 1024 nodes settled, 3968 edges read.
#[bench(raw)]
fn shortest_paths_grid_1024() -> BenchResult {
    let mut g = heap_graph();
    let all = grid(&mut g);
    bench_fn(|| black_box(dijkstra(&g, &all, first(&all))))
}

/// The same in virtual stable memory.
#[bench(raw)]
fn shortest_paths_grid_1024_stable() -> BenchResult {
    let mut g = stable_graph();
    let all = grid(&mut g);
    bench_fn(|| black_box(dijkstra(&g, &all, first(&all))))
}

/// Breadth-first search over the same grid, without values or a priority queue.
#[bench(raw)]
fn bfs_grid_1024() -> BenchResult {
    let mut g = heap_graph();
    let all = grid(&mut g);
    bench_fn(|| black_box(bfs(&g, &all, first(&all))))
}
