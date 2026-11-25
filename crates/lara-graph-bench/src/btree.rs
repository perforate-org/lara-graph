//! A `StableBTreeMap` baseline for adjacency scans, built to be the best fair version of that
//! design rather than a strawman.
//!
//! One edge is one entry per half, keyed by `(source, label, direction, target)`: 11 bytes, no
//! value. The key is a fixed byte string, so [`Storable::to_bytes`] borrows it and the scan
//! decodes only the target. Both halves are stored, as lara-graph stores them. The map is one
//! global `StableBTreeMap`, the crate's canonical use: the entries for one
//! `(node, label, direction)` are a contiguous range.
//!
//! The workload matches the lara-graph hub benches: a hub with an edge to each of `degree`
//! fresh leaves, one label, scans repeated 51 times. Targets are distinct here, so the key needs
//! no duplicate tie-breaker; a multigraph needs four more bytes for a half id.
//!
//! The map lives in a `MemoryManager` virtual memory over `DefaultMemoryImpl` (on wasm32 the
//! canister's stable memory), as lara-graph's benches do. Its lara-graph counterparts:
//! `btree_visit_*` against `hub_visit_*`, `btree_scan_*` against `hub_scan_*`, `btree_insert_*`
//! against `hub_insert_*` and `btree_two_hop` against `two_hop_ordered`.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use ic_stable_structures::{
    DefaultMemoryImpl, StableBTreeMap, Storable,
    memory_manager::{MemoryId, MemoryManager, VirtualMemory},
    storable::Bound,
};
use lara_graph::NodeId;

use crate::two_hop::{FANOUT, SATELLITES};

/// The label the hub benches use, as in the rest of this crate.
const LABEL: u16 = 2;
/// A directed edge's `Outgoing` half.
const OUT: u8 = 0;
/// A directed edge's `Incoming` half.
const IN: u8 = 1;
/// Bytes in an edge key: source 4, label 2, direction 1, target 4.
const KEY_LEN: usize = 11;

/// `(source, label, direction, target)` in big-endian byte order, so lexicographic order is
/// numeric order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeKey([u8; KEY_LEN]);

impl EdgeKey {
    fn new(source: u32, target: u32, direction: u8) -> Self {
        let mut bytes = [0u8; KEY_LEN];
        let (source_bytes, rest) = bytes.split_at_mut(4);
        source_bytes.copy_from_slice(&source.to_be_bytes());
        let (label_bytes, rest) = rest.split_at_mut(2);
        label_bytes.copy_from_slice(&LABEL.to_be_bytes());
        let (direction_bytes, rest) = rest.split_at_mut(1);
        direction_bytes.copy_from_slice(&[direction]);
        let (target_bytes, _) = rest.split_at_mut(4);
        target_bytes.copy_from_slice(&target.to_be_bytes());
        Self(bytes)
    }

    fn target(&self) -> u32 {
        let (_, rest) = self.0.split_at(7);
        let (target, _) = rest.split_at(4);
        u32::from_be_bytes(
            target
                .try_into()
                .unwrap_or_else(|_| panic!("edge key target")),
        )
    }
}

impl Storable for EdgeKey {
    fn to_bytes(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.0)
    }

    fn into_bytes(self) -> Vec<u8> {
        self.0.to_vec()
    }

    fn from_bytes(bytes: Cow<'_, [u8]>) -> Self {
        let bytes = bytes.as_ref();
        assert_eq!(bytes.len(), KEY_LEN, "an edge key is 11 bytes");
        let mut key = [0u8; KEY_LEN];
        key.copy_from_slice(bytes);
        Self(key)
    }

    const BOUND: Bound = Bound::Bounded {
        max_size: 11,
        is_fixed_size: true,
    };
}

/// The `StableBTreeMap` adjacency store: both halves of every directed edge.
struct BTreeGraph<M: ic_stable_structures::Memory> {
    edges: StableBTreeMap<EdgeKey, (), M>,
}

impl<M: ic_stable_structures::Memory> BTreeGraph<M> {
    fn new(memory: M) -> Self {
        Self {
            edges: StableBTreeMap::init(memory),
        }
    }

    fn add_edge(&mut self, from: NodeId, to: NodeId) {
        self.edges
            .insert(EdgeKey::new(from.index(), to.index(), OUT), ());
        self.edges
            .insert(EdgeKey::new(to.index(), from.index(), IN), ());
    }

    /// Visits `node`'s `direction` half, in target order.
    fn scan(&self, node: NodeId, direction: u8, mut visit: impl FnMut(NodeId)) {
        let start = EdgeKey::new(node.index(), 0, direction);
        let end = EdgeKey::new(node.index(), u32::MAX, direction);
        for key in self.edges.keys_range(start..=end) {
            visit(NodeId::new(key.target()));
        }
    }
}

/// A hub and an edge to each of `degree` fresh leaves.
fn hub<M: ic_stable_structures::Memory>(memory: M, degree: u32) -> (BTreeGraph<M>, NodeId) {
    let mut graph = BTreeGraph::new(memory);
    let hub = NodeId::new(0);
    for leaf in 1..=degree {
        graph.add_edge(hub, NodeId::new(leaf));
    }
    (graph, hub)
}

/// 51 scans of the hub, counting edges without collecting them.
fn visit<M: ic_stable_structures::Memory>(memory: M, degree: u32) -> BenchResult {
    let (graph, hub) = hub(memory, degree);
    bench_fn(|| {
        for _ in 0..51 {
            let mut count = 0u32;
            graph.scan(hub, OUT, |_| count += 1);
            black_box(count);
        }
    })
}

/// 51 scans of the hub, collecting the targets into a reusable `Vec`.
fn scan<M: ic_stable_structures::Memory>(memory: M, degree: u32) -> BenchResult {
    let (graph, hub) = hub(memory, degree);
    let mut out = Vec::new();
    bench_fn(|| {
        for _ in 0..51 {
            out.clear();
            graph.scan(hub, OUT, |target| out.push(target.index()));
            black_box(out.len());
        }
    })
}

/// Growing the hub one edge at a time, both halves of each.
fn insert<M: ic_stable_structures::Memory>(memory: M, degree: u32) -> BenchResult {
    let mut graph = BTreeGraph::new(memory);
    let hub = NodeId::new(0);
    bench_fn(|| {
        for leaf in 1..=degree {
            graph.add_edge(hub, NodeId::new(leaf));
        }
    })
}

/// The two-hop traversal of `two_hop`: the hub's satellites, then each satellite's edges.
fn two_hop<M: ic_stable_structures::Memory>(memory: M) -> BenchResult {
    let mut graph = BTreeGraph::new(memory);
    let hub = NodeId::new(0);
    for satellite in 0..SATELLITES {
        graph.add_edge(hub, NodeId::new(1 + satellite));
    }
    let first_target = 1 + SATELLITES;
    for satellite in 0..SATELLITES {
        let from = NodeId::new(1 + satellite);
        for target in 0..FANOUT {
            graph.add_edge(from, NodeId::new(first_target + target));
        }
    }
    bench_fn(|| {
        let mut count = 0u32;
        graph.scan(hub, OUT, |satellite| {
            graph.scan(satellite, OUT, |_| count += 1);
        });
        black_box(count);
    })
}

/// A `MemoryManager` virtual memory, as a canister wires one.
fn stable_memory() -> VirtualMemory<DefaultMemoryImpl> {
    let manager = MemoryManager::init(DefaultMemoryImpl::default());
    manager.get(MemoryId::new(0))
}

#[bench(raw)]
fn btree_visit_1024() -> BenchResult {
    visit(stable_memory(), 1024)
}

#[bench(raw)]
fn btree_visit_65536() -> BenchResult {
    visit(stable_memory(), 65536)
}

#[bench(raw)]
fn btree_scan_1024() -> BenchResult {
    scan(stable_memory(), 1024)
}

#[bench(raw)]
fn btree_scan_65536() -> BenchResult {
    scan(stable_memory(), 65536)
}

#[bench(raw)]
fn btree_insert_1024() -> BenchResult {
    insert(stable_memory(), 1024)
}

#[bench(raw)]
fn btree_insert_4096() -> BenchResult {
    insert(stable_memory(), 4096)
}

#[bench(raw)]
fn btree_insert_65536() -> BenchResult {
    insert(stable_memory(), 65536)
}

#[bench(raw)]
fn btree_two_hop() -> BenchResult {
    two_hop(stable_memory())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ic_stable_structures::VectorMemory;

    #[test]
    fn scans_only_the_hub_out() {
        let (graph, hub) = hub(VectorMemory::default(), 1024);
        let mut seen = 0;
        graph.scan(hub, OUT, |_| seen += 1);
        assert_eq!(seen, 1024);
    }
}
