use super::*;
use crate::descriptor::{PAGE_SLOTS, Paged, Slot, TINY_SLOTS};
use crate::id::{Cursor, Halves, Order, Placement, Position, Width};
use crate::memories::Memories;
use crate::memory::{PAGE_SIZE, VecMemory};
use crate::page_store::{ByteClass, Extent, Pages};
use crate::paged;
use crate::stored::{Index, LabelBucket};
use crate::targets;
use crate::values;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use core::num::NonZeroU32;
use core::ops::ControlFlow;

/// The backing memories of a test graph; graphs borrow them so they can be reopened.
#[derive(Default)]
struct Backing {
    memories: [VecMemory; 5],
}

impl Backing {
    fn memories(&self) -> Memories<VecMemory> {
        Memories::from_fn(|role| self.memories[usize::from(role.index())].clone())
    }

    fn create(&self, n: u32) -> Graph<VecMemory> {
        let mut g = Graph::new(self.memories());
        for _ in 0..n {
            g.add_node().unwrap();
        }
        g
    }

    fn open(&self) -> Graph<VecMemory> {
        Graph::load(self.memories())
    }
}

fn v(i: u32) -> NodeId {
    NodeId::new(i)
}

fn l(i: u16) -> Label {
    Label::new(i)
}

fn p(i: u32) -> Position {
    Position::new(i)
}

/// The live edges of `x`'s `(label, direction)` bucket as `(position, target)`.
fn edges_on<M: Memory>(
    g: &Graph<M>,
    x: NodeId,
    label: Label,
    direction: Direction,
) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    g.neighbors_directed(x, label, direction, &mut out).unwrap();
    out.into_iter()
        .map(|e| (e.position().index(), e.target().index()))
        .collect()
}

/// The live out-edges of `x` under `label`.
fn edges<M: Memory>(g: &Graph<M>, x: NodeId, label: Label) -> Vec<(u32, u32)> {
    edges_on(g, x, label, Direction::Outgoing)
}

fn out_ref(x: NodeId, label: Label, at: Position) -> EdgeRef {
    EdgeRef::new(x, label, Direction::Outgoing, at)
}

/// Inserts `from -[label]-> to` and returns the out half's position.
fn insert<M: Memory>(g: &mut Graph<M>, from: NodeId, label: Label, to: NodeId) -> Position {
    g.add_edge(NewEdge::directed(
        from,
        label,
        to,
        Placement::ReuseHoles,
        &[],
    ))
    .unwrap()
    .canonical()
    .position()
}

/// The report of removing the edge with `halves` and no value.
fn removed(halves: Halves) -> Removal {
    Removal::Removed(RemovedEdge::new(halves, Vec::new()))
}

/// Removes the directed edge whose out half is at `at`; returns its target if it was live.
fn remove<M: Memory>(g: &mut Graph<M>, from: NodeId, label: Label, at: Position) -> Option<u32> {
    match g.remove_edge(out_ref(from, label, at)).unwrap() {
        Removal::Removed(r) => Some(r.halves().other().node().index()),
        Removal::Absent => None,
    }
}

fn out_bucket<M: Memory>(g: &Graph<M>, x: NodeId, label: Label) -> Option<Descriptor> {
    g.bucket(
        x,
        LabelBucketKey {
            label,
            direction: Direction::Outgoing,
        },
    )
    .unwrap()
}

mod batch_reads;
mod batch_writes;
mod between;
mod bucket_forms;
mod compaction;
mod detach;
mod edge_values;
mod edges;
mod index_pages;
mod invariants;
mod lifecycle;
mod model;
mod reads;

use invariants::check;
