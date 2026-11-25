//! Instruction counts of lara-graph operations on the Internet Computer, measured with canbench.
//!
//! Scenarios mirror ic-stable-lara's `canbench_results.yml` where one exists, so the two can be
//! compared. Every edge here is stored at both ends, as Gleaph needs; ic-stable-lara's
//! single-label hub benches store one side only.

extern crate alloc;

use core::hint::black_box;
use ic_stable_structures::DefaultMemoryImpl;
use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
use lara_graph::{
    Direction, EdgeRef, Error, Graph, Halves, Label, Memories, Memory, NewEdge, NodeId, Placement,
    Position, ReadBatch, VecMemory,
};

mod all_labels;
mod between;
mod btree;
mod churn;
mod detach;
mod insert;
mod locality;
mod lookup;
mod node_labels;
mod remove;
mod scan;
mod shortest;
mod sorted;
pub mod two_hop;
mod values;

const LABEL: Label = Label::new(2);

/// A graph in six heap memories.
#[must_use]
pub fn heap_graph() -> Graph<VecMemory> {
    Graph::new(Memories::from_fn(|_| VecMemory::new()))
}

/// A graph in seven virtual stable memories (ids 0..7), as a canister wires it.
fn stable_graph() -> Graph<VirtualMemory<DefaultMemoryImpl>> {
    let manager = MemoryManager::init(DefaultMemoryImpl::default());
    Graph::init(Memories::from_fn(|role| {
        manager.get(MemoryId::new(role.index()))
    }))
}

fn node<M: Memory>(g: &mut Graph<M>) -> NodeId {
    g.add_node().unwrap_or_else(|e| panic!("node: {e}"))
}

fn nodes<M: Memory>(g: &mut Graph<M>, n: u32) -> Vec<NodeId> {
    (0..n).map(|_| node(g)).collect()
}

/// The last edge a seed inserted.
fn last(edges: &[Halves]) -> Halves {
    edges.last().copied().unwrap_or_else(|| panic!("no edges"))
}

fn directed<M: Memory>(g: &mut Graph<M>, from: NodeId, to: NodeId) -> Halves {
    g.add_edge(NewEdge::directed(
        from,
        LABEL,
        to,
        Placement::ReuseHoles,
        &[],
    ))
    .unwrap_or_else(|e| panic!("insert: {e}"))
}

/// A hub with an edge to each of `degree` fresh leaves; returns the hub and the edges.
fn seed_hub<M: Memory>(g: &mut Graph<M>, degree: u32) -> (NodeId, Vec<Halves>) {
    let hub = node(g);
    let leaves = nodes(g, degree);
    let edges = leaves
        .into_iter()
        .map(|leaf| directed(g, hub, leaf))
        .collect();
    (hub, edges)
}

/// A hub with `n` parallel edges to one target.
fn seed_parallel<M: Memory>(g: &mut Graph<M>, n: u32) -> (NodeId, Vec<EdgeRef>) {
    let hub = node(g);
    let target = node(g);
    for _ in 0..n {
        black_box(directed(g, hub, target));
    }
    (hub, first_halves(hub, n))
}

/// The out halves at `hub`'s first `n` positions.
fn first_halves(hub: NodeId, n: u32) -> Vec<EdgeRef> {
    (0..n)
        .map(|i| EdgeRef::new(hub, LABEL, Direction::Outgoing, Position::new(i)))
        .collect()
}

/// Reads each of `halves` one call at a time; returns how many found their edge.
fn live_each<T>(
    halves: &[EdgeRef],
    mut read: impl FnMut(EdgeRef) -> Result<Option<T>, Error>,
) -> u64 {
    let mut live = 0u64;
    for &half in halves {
        live += u64::from(
            read(half)
                .unwrap_or_else(|err| panic!("read: {err}"))
                .is_some(),
        );
    }
    live
}

/// Reads all of `halves` through one `ReadBatch`; returns how many found their edge.
fn live_batched<'g, M: Memory, T>(
    g: &'g Graph<M>,
    halves: &[EdgeRef],
    read: impl FnOnce(&mut ReadBatch<'g, M>, &[EdgeRef], &mut Vec<Option<T>>) -> Result<(), Error>,
) -> u64 {
    let mut batch = g.read_batch();
    let mut out = Vec::new();
    read(&mut batch, halves, &mut out).unwrap_or_else(|err| panic!("batch: {err}"));
    out.iter().map(|found| u64::from(found.is_some())).sum()
}
