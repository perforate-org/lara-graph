//! Sorted-run batching ingest: buffer + sort + per-source run flush.
//!
//! [`IngestBuffer::flush`] groups buffered pairs by source and applies
//! each run via [`crate::graph::LaraGraph::insert_sorted_run`]. Unordered
//! sorts `(src, dst)` as before; Insertion stably groups only by source.
//! Eligible runs batch writes; other runs use per-edge insertion.
//!
//! The default graph preserves edge multiplicity, not adjacency order.
//! Buffered edges are invisible until explicit `flush`, which may follow
//! direct insertions. `push` does not auto-flush at the capacity hint.
//! In Insertion mode, push(A), direct insert(B), flush yields B,A for that
//! source. Flush applies buffered relative order, not destination order or
//! global chronology. Mutation and flush require only [`CsrEdge`]; explicit
//! graph configuration, not trait presence, selects ordering.

use crate::graph::{AdjacencyOrder, LaraGraph};
use crate::memory::Memory;
use crate::traits::{CsrEdge, CsrVertex};
use alloc::vec::Vec;

/// Default caller-visible flush-granularity hint; not an enforced limit.
pub const DEFAULT_INGEST_BUFFER_CAP: usize = 4096;

/// Buffered sorted-run ingest for [`LaraGraph`].
///
/// Buffer `(src, dst)` pairs with [`IngestBuffer::push`], apply them with
/// [`IngestBuffer::flush`]. See the module docs for the contract.
#[derive(Debug, Clone)]
pub struct IngestBuffer {
    buf: Vec<(u32, u32)>,
    cap: usize,
}

impl Default for IngestBuffer {
    /// Empty buffer with [`DEFAULT_INGEST_BUFFER_CAP`].
    fn default() -> Self {
        Self {
            buf: Vec::new(),
            cap: DEFAULT_INGEST_BUFFER_CAP,
        }
    }
}

impl IngestBuffer {
    /// Creates an empty buffer with a `buffer_cap` flush-granularity hint.
    ///
    /// `graph_capacity_hint` reserves the backing vec (vertex count
    /// estimate; purely an allocation hint, never a correctness bound).
    pub fn new(graph_capacity_hint: u32, buffer_cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(graph_capacity_hint as usize),
            cap: buffer_cap.max(1),
        }
    }
    /// Number of buffered (not yet flushed) edges.
    pub fn len(&self) -> usize {
        self.buf.len()
    }
    /// `true` when no edges are buffered.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
    /// Configured buffer capacity (flush granularity hint).
    pub fn buffer_cap(&self) -> usize {
        self.cap
    }
    /// Changes the buffer capacity (applies to future flushes).
    pub fn set_buffer_cap(&mut self, cap: usize) {
        self.cap = cap.max(1);
    }
    /// Buffers one edge (invisible until [`IngestBuffer::flush`]).
    pub fn push(&mut self, src: u32, dst: u32) {
        self.buf.push((src, dst));
    }
    /// Groups the buffer by source and applies each run via
    /// [`LaraGraph::insert_sorted_run`]. A no-op on an empty buffer.
    /// Insertion preserves buffered relative order per source, after earlier
    /// direct applications; Unordered sorts by `(src, dst)`.
    ///
    /// Ordered records/sources are checked before sorting or graph writes.
    /// A marker (including its neighbor-only reconstruction) or invalid source
    /// panics with the buffer retained. Other failures, including a later
    /// source run's depth limit or a backend panic, are not transactional.
    pub fn flush<V: CsrVertex, E: CsrEdge, M: Memory>(&mut self, graph: &mut LaraGraph<V, E, M>) {
        if self.buf.is_empty() {
            return;
        }
        if graph.order() == AdjacencyOrder::Insertion {
            for &(src, dst) in &self.buf {
                graph.check_insert_record(src, &E::with_neighbor(E::default(), dst));
            }
            self.buf.sort_by_key(|&(src, _)| src);
        } else {
            self.buf.sort_unstable();
        }
        let mut dsts: Vec<u32> = Vec::new();
        let mut i = 0;
        while i < self.buf.len() {
            let src = self.buf[i].0;
            let mut j = i;
            while j < self.buf.len() && self.buf[j].0 == src {
                j += 1;
            }
            dsts.clear();
            dsts.extend(self.buf[i..j].iter().map(|&(_, d)| d));
            graph.insert_sorted_run(src, &dsts);
            i = j;
        }
        self.buf.clear();
    }
}
