//! Rust port of DGAP ([arXiv:2403.02665]), a PMA-based dynamic CSR with
//! per-segment edge logs. PMDK persistence is abstracted behind
//! [`memory::Memory`]; the graph algorithms are a faithful port of
//! `graph.h` intended for verification and benchmarking against LARA.
//!
//! [arXiv:2403.02665]: https://arxiv.org/abs/2403.02665

pub mod graph;
pub mod memory;
pub mod util;

pub use graph::{
    DgapGraph, DgapNeighborhood, LOW_0, LOW_H, LogEntry, MAX_LOG_ENTRIES, RawEdge, SegMeta, UP_0,
    UP_H, VertexElement,
};
pub use memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
pub use util::{ceil_div, ceil_log2, floor_log2, hyperceil, hyperfloor};
