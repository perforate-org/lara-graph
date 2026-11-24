//! LARA (Localized Adjacency Relocation Array): a labeled multigraph store on paged memory.
//!
//! # What LARA guarantees, in priority order
//!
//! 1. **CSR-level scans.** A `(node, label)` label bucket is read as contiguous slots: a node read,
//!    a label bucket list read and one data read. A read cannot consult allocation metadata.
//! 2. **Bounded update cost.** An insert or removal copies at most one run below a page or one
//!    page of index entries; a full one-page run becomes the first data page without copying.
//! 3. Supporting mechanisms: a few edges live inline in the descriptor, edges keep stable
//!    positions (removal leaves a hole), and freed pages serve any use.
//!
//! Every piece of state lives behind the memory seam, so a graph can be reopened. Behaviour is
//! carried by types rather than by raw integers or flags.
//!
//! The crate is `no_std` + `alloc`.

#![no_std]

extern crate alloc;

// Runs the README's examples as doctests.
#[cfg(all(doctest, feature = "ic-stable-structures"))]
#[doc = include_str!("../../../README.md")]
struct ReadmeDoctests;

mod compact;
mod corrupt;
mod descriptor;
mod graph;
mod id;
mod label_bucket;
mod label_bucket_list;
mod memories;
mod memory;
mod page_store;
mod paged;
mod scale;
mod stored;
mod targets;

mod detaching_set;
mod node;
mod txn;
mod units;
mod values;

pub use detaching_set::DetachingSet;
pub use graph::{
    Compaction, Detachment, EdgeKind, EdgeValue, Error, Graph, InitError, NewEdge, ReadBatch,
    Removal, Step, Sweep,
};
pub use id::{
    Cursor, Direction, Edge, EdgeRef, Halves, Label, Move, NodeId, NodeState, Order, Placement,
    Position, Progress, RemovedEdge, Width,
};
pub use memories::{Memories, MemoryRole};
pub use memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
