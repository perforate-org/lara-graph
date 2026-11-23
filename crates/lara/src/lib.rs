//! Generic LARA: localized adjacency relocation array over any [`memory::Memory`].
//!
//! Ported from `ic-stable-lara` with IC-specific concerns removed (stable
//! structures, canbench, bidirectional scans, inline property bytes, batch
//! reservation preflight).
//!
//! Core pieces:
//! - [`free_span::FreeSpanStore`] — retired slab ranges, best-fit reuse with
//!   coalescing;
//! - [`span_meta::SegmentSpanMetaStore`] — explicit `physical_start` per PMA
//!   leaf (relocated segments may sit anywhere);
//! - [`tree_csr::TreeCsrBucket`] — 4 KiB block-store mode for high-degree
//!   vertices (derived depth, promote/deepen/flatten);
//! - [`graph::LaraGraph`] — unlabeled mutable CSR wiring those together;
//! - [`labeled::LabeledLaraGraph`] — multi-label adjacency extension.
//!
//! Graphs default to [`AdjacencyOrder::Unordered`]. The nonlabeled core can
//! explicitly select per-source [`AdjacencyOrder::Insertion`] while empty;
//! ordinary operations still require only [`CsrEdge`]. [`LabeledLaraGraph`]
//! selects the same policy per label, across all vertices, while that label is
//! empty. Ordering is local to a source (and label), not stable edge IDs,
//! ordinals or global cross-label chronology.
//!
//! The crate is `no_std`-compatible: the core (traits, [`memory::Memory`],
//! [`memory::GrowFailed`], [`memory::safe_write`]) compiles with only
//! `alloc`; the heap-backed [`memory::VecMemory`] backend and the
//! `std::error::Error` impls are gated behind the default `std` feature.
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod block_store;
pub mod bucket;
#[cfg(feature = "std")]
pub mod catalog;
pub mod free_span;
pub mod graph;
pub mod ingest;
pub mod labeled;
pub mod memory;
#[cfg(feature = "mmap")]
pub mod mmap_memory;
pub mod span_meta;
pub mod traits;
pub mod tree_csr;
pub mod util;

pub use block_store::{BLOCK_PAYLOAD_BYTES, BlockError, BlockKind, LtbRawBlockStore};
pub use bucket::{BucketDescriptor, BucketStore};
#[cfg(feature = "std")]
pub use catalog::{LabelCatalog, LabelLimit};
pub use free_span::{FreeSpan, FreeSpanError, FreeSpanStore};
pub use graph::{AdjacencyOrder, Edge, LaraGraph, OrderChangeError, Vertex};
pub use ingest::{DEFAULT_INGEST_BUFFER_CAP, IngestBuffer};
pub use labeled::{Bucket, Label, LabelId, LabeledLaraGraph};
#[cfg(feature = "std")]
pub use memory::VecMemory;
pub use memory::{GrowFailed, Memory, PAGE_SIZE, PersistCounterMemory, PersistCounters};
#[cfg(feature = "mmap")]
pub use mmap_memory::MmapMemory;
pub use span_meta::{SegmentSpanMeta, SegmentSpanMetaStore};
pub use traits::{
    CsrEdge, CsrEdgeTombstone, CsrVertex, CsrVertexTombstone, DynamicCsr, FixedWidth, RemovableCsr,
};
pub use tree_csr::{B, R_MAX, TreeCsrBucket, derive_depth};
