//! Updates at steady state: a sliding window of random directed edges among 8192 nodes, each
//! update adding one edge and removing the oldest. 1024 updates are measured right after the
//! window is built, and again after it has turned over once, with holes refilled and optionally
//! a bucket compacted once its holes reach its live edges (see `examples/churn.rs` for space).

use crate::{Bench, LABEL, graph, nodes};
use alloc::collections::VecDeque;
use canbench_rs::{BenchResult, bench, bench_fn};
use core::ops::ControlFlow;
use lara_graph::{Adjacency, Direction, Edge, NodeId, Order, Progress, Removal};

const NODES: u32 = 8192;
/// Live edges: degree 8.
const WINDOW: usize = 65536;
const MEASURED: usize = 1024;

/// When buckets are compacted.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Compaction {
    Never,
    AtHalfHoles,
}

/// A xorshift generator, so every build is the same.
struct Rng(u64);

impl Rng {
    fn edge(&mut self, among: &[NodeId]) -> (NodeId, NodeId) {
        let mut pick = || {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            let at = usize::try_from(self.0 % u64::try_from(among.len()).unwrap_or(1)).unwrap_or(0);
            among
                .get(at)
                .copied()
                .unwrap_or_else(|| panic!("empty pool"))
        };
        (pick(), pick())
    }
}

struct Window {
    graph: Bench,
    nodes: Vec<NodeId>,
    live: VecDeque<(NodeId, NodeId)>,
    rng: Rng,
    compaction: Compaction,
}

impl Window {
    fn new(compaction: Compaction) -> Self {
        let mut graph = graph();
        let nodes = nodes(&mut graph, NODES);
        let mut window = Self {
            graph,
            nodes,
            live: VecDeque::with_capacity(WINDOW + 1),
            rng: Rng(0x9E37_79B9_7F4A_7C15),
            compaction,
        };
        for _ in 0..WINDOW {
            window.add();
        }
        window
    }

    fn add(&mut self) {
        let (from, to) = self.rng.edge(&self.nodes);
        self.graph
            .add_edge(Edge::directed(from, LABEL, to))
            .unwrap_or_else(|e| panic!("insert: {e}"));
        self.live.push_back((from, to));
    }

    /// One update: an edge in, the oldest out.
    fn update(&mut self) {
        self.add();
        let Some((from, to)) = self.live.pop_front() else {
            return;
        };
        let out = LABEL.outgoing();
        let half = self
            .graph
            .visit_edges_between(from, out, to, Order::Ascending, ControlFlow::Break)
            .unwrap_or_else(|e| panic!("find: {e}"))
            .unwrap_or_else(|| panic!("a live edge of the window is missing"));
        let removal = self
            .graph
            .remove_edge(half)
            .unwrap_or_else(|e| panic!("remove: {e}"));
        if self.compaction == Compaction::AtHalfHoles
            && let Removal::Removed(edge) = removal
        {
            let ends = [
                (from, Direction::Outgoing, edge.canonical_fill()),
                (to, Direction::Incoming, edge.other_fill()),
            ];
            for (v, direction, fill) in ends {
                if fill.holes() >= fill.live().max(1) {
                    self.compact(v, direction);
                }
            }
        }
    }

    fn compact(&mut self, v: NodeId, direction: Direction) {
        while self
            .graph
            .compact(v, Adjacency::new(LABEL, direction))
            .unwrap_or_else(|e| panic!("compact: {e}"))
            .progress()
            == Progress::Remaining
        {}
    }
}

fn measure(compaction: Compaction, turns: usize) -> BenchResult {
    let mut window = Window::new(compaction);
    for _ in 0..turns * WINDOW {
        window.update();
    }
    bench_fn(|| {
        for _ in 0..MEASURED {
            window.update();
        }
    })
}

/// 1024 updates right after the build.
#[bench(raw)]
fn churn_fresh_1024() -> BenchResult {
    measure(Compaction::Never, 0)
}

/// The same after a turn of the window, holes refilled but never compacted.
#[bench(raw)]
fn churn_turned_1024() -> BenchResult {
    measure(Compaction::Never, 1)
}

/// Right after the build, compacting a bucket once its holes reach its live edges.
#[bench(raw)]
fn churn_fresh_compacting_1024() -> BenchResult {
    measure(Compaction::AtHalfHoles, 0)
}

/// After a turn, compacting as above.
#[bench(raw)]
fn churn_turned_compacting_1024() -> BenchResult {
    measure(Compaction::AtHalfHoles, 1)
}
