//! LARA (Labeled Adjacency Rooted Array): a labeled multigraph store on paged memory.
//!
//! # What LARA guarantees, in priority order
//!
//! 1. **CSR-level scans.** A `(node, label)` label bucket is read as contiguous slots: a node read,
//!    a label bucket list read and one data read. A read cannot consult allocation metadata.
//! 2. **Bounded update cost.** An insert or removal copies at most one run below a page or one
//!    page of index entries; a full one-page run becomes the first data page without copying.
//!    Every update finishes within its call: there is no deferred maintenance. Work a caller may
//!    spread out (compaction, detaching a node) is offered as bounded steps it calls.
//! 3. Supporting mechanisms: a few edges live inline in the descriptor, edges keep stable
//!    positions (removal leaves a hole), and freed pages serve any use.
//!
//! Every piece of state lives behind the memory seam, so a graph can be reopened. Behaviour is
//! carried by types rather than by raw integers or flags.
//!
//! The code is the specification: each module documents its persistent layout and the reasons
//! for its choices, `tests/public-api.txt` pins the public surface, and
//! `crates/lara-graph-bench/canbench_results.yml` holds the measured costs.
//!
//! The crate is `no_std` + `alloc`.

#![no_std]

extern crate alloc;

// Runs the README's examples as doctests.
#[cfg(all(doctest, feature = "ic-stable-structures"))]
#[doc = include_str!("../../../README.md")]
struct ReadmeDoctests;

// Layers, each using only those before it: `store`, `bucket`, then `node` and `edge` side by
// side, then `graph`.
mod bucket;
mod edge;
mod error;
mod graph;
mod id;
mod input;
mod node;
mod outcome;
mod store;

pub use error::{BatchError, Error, InitError};
pub use graph::{DetachingSet, Graph, Graph as StableGraph, Memories, MemoryRole, ReadBatch};
pub use id::{Direction, EdgeRef, Halves, Label, Neighbor, NodeId, NodeState, Position, Width};
pub use input::{Cursor, EdgeKind, EdgeValue, NewEdge, Order, Placement};

pub use outcome::{Compaction, Detachment, Move, Progress, Removal, RemovedEdge, Step, Sweep};
pub use store::memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
