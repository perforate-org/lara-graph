//! LARA core: a mutable CSR-style adjacency store.
//!
//! # What LARA guarantees, in priority order
//!
//! 1. **CSR-level scans.** A `(vertex, label)` bucket is read as contiguous slots: a vertex read,
//!    a list read and one data read. A read cannot consult allocation metadata.
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

mod bucket;
mod compact;
mod corrupt;
mod descriptor;
mod graph;
mod id;
mod labels;
mod memory;
mod page_store;
mod paged;
/// The TLSF span allocator: kept for the storage comparisons in `crates/benches`.
#[cfg(feature = "bench-internals")]
mod span_alloc;
mod stored;
mod targets;

#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod bench_internals;
mod txn;
mod units;
mod values;
mod vertex;
mod vertex_set;

pub use graph::{
    Compaction, CreateError, Detachment, Error, Graph, Memories, OpenError, Removal, Step, Sweep,
};
pub use id::{
    Cursor, Edge, EdgeRef, Halves, Label, Move, Order, Placement, Position, Progress, RemovedEdge,
    Side, VertexId, VertexState, Width,
};
pub use memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
pub use vertex_set::VertexSet;
