use super::*;
use crate::bucket::LabelBucketKey;
use crate::bucket::descriptor::{Descriptor, PAGE_SLOTS, Paged, Slot, TINY_SLOTS};
use crate::bucket::paged;
use crate::bucket::slots;
use crate::bucket::target_index;
use crate::bucket::values;
use crate::bucket::{Index, LabelBucket};
use crate::error::{BatchError, InitError};
use crate::graph::memories::Memories;
use crate::id::{Direction, EdgeRef, Halves, Label, NodeId, NodeState, Position, Width};
use crate::input::{Cursor, Order, Placement};
use crate::input::{EdgeValue, NewEdge};
use crate::node::label_bucket_list;
use crate::outcome::{BucketFill, Progress, Removal, RemovedEdge};
use crate::store::memory::{PAGE_SIZE, VecMemory};
use crate::store::page_store::{ByteClass, Extent, Pages};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroU32;
use core::ops::ControlFlow;

/// The backing memories of a test graph; graphs borrow them so they can be reopened.
#[derive(Default)]
struct Backing {
    memories: [VecMemory; 6],
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
    g.neighbors(x, (label, direction), &mut out).unwrap();
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

/// The halves a removal reports, if it removed an edge.
fn removed(removal: Result<Removal, Error>) -> Option<Halves> {
    match removal {
        Ok(Removal::Removed(edge)) => Some(edge.halves()),
        Ok(Removal::Absent) | Err(_) => None,
    }
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

mod all_labels;
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
mod node_labels;
mod reads;

use invariants::check;
