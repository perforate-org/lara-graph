//! [`LaraGraph`]: generalized LARA — mutable CSR with free-span store,
//! segment span metadata, per-leaf overflow logs, and Tree-CSR mode.
//!
//! Ported from `ic-stable-lara` with IC-specific concerns removed (no stable
//! structures, no canbench, no bidirectional scans, no inline property
//! bytes, no IC batch reservation preflight). Vertex rows and PMA leaf counts are
//! authoritative heap vectors, not synchronized backend columns. There
//! is no graph reopen/recovery API. Backend records use fixed-width,
//! little-endian codecs through [`Memory`]:
//!
//! - **Vertex/count handles**: retained for interface compatibility;
//!   the graph does not read, write or grow these backend regions.
//! - **Edge slab**: each PMA leaf segment owns a physical span (explicit
//!   `physical_start` in [`SegmentSpanMetaStore`]) that it relocates as a
//!   unit — the localized-relocation primitive DGAP lacks. Retired spans
//!   are recycled best-fit via [`FreeSpanStore`].
//! - **Per-leaf overflow logs**: fixed [`MAX_LOG_ENTRIES`] slots per leaf,
//!   per-vertex chains (DGAP-compatible).
//! - **Tree-CSR buckets**: vertices whose live degree reaches [`T_PROMOTE`]
//!   move into an ordinal-addressed 4 KiB block tree managed through an
//!   [`LtbRawBlockStore`] (layout constants in [`crate::tree_csr`]).
//!
//! Unordered scans use a dense live slab prefix + log chain, or a tree
//! bucket. Insertion-ordered slab scans skip configured typed markers. Update paths use CSR geometry (the next
//! row's base inside the leaf), PMA leaf counts, span metadata, and
//! free-span reuse. PMA thresholds are height-interpolated like DGAP
//! (leaves 1.0/0.25, root 0.75/0.50); rebalances are leaf-local.
//!
//! Ordinary graph operations require only [`CsrEdge`]. The default
//! [`AdjacencyOrder::Unordered`] uses dense swap removal or log unlinking.
//! Marker capability alone does not select a policy. Explicitly enabling
//! [`AdjacencyOrder::Insertion`] through [`LaraGraph::set_order`] requires
//! [`CsrEdgeTombstone`]; it preserves per-source application order through
//! typed slab markers, tiny/tree shifts and oldest-first logs. Labels are
//! not configured here. Raw tree and standalone ordinal operations remain distinct.

use crate::block_store::{BLOCK_PAYLOAD_BYTES, LtbRawBlockStore};
use crate::free_span::FreeSpanStore;
use crate::memory::{GrowFailed, Memory, safe_write};
use crate::span_meta::{SPAN_UNASSIGNED, SegmentSpanMeta, SegmentSpanMetaStore};
use crate::traits::{
    CsrEdge, CsrEdgeTombstone, CsrVertex, CsrVertexTombstone, DynamicCsr, FixedWidth, NO_BUCKET,
    RemovableCsr,
};
use crate::tree_csr::{B, R_MAX, assert_depth_one, derive_depth, root_len};
use crate::util::{ceil_div as ceil_div_u, ceil_log2 as ceil_log2_u};
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

/// Log capacity per leaf segment (matches DGAP).
pub const MAX_LOG_ENTRIES: usize = 170;
/// Tree-CSR root-array cap (= [`R_MAX`]). Promotion and insertion enforce
/// the depth-1 limit of `B * R_MAX` neighbors before mutation.
pub const CAP_TREE: u32 = 1024;
/// Slab → tree promotion trigger on stored slab width, including ordered markers.
/// Log entries contribute to live degree but not this width. Doubles as the slab-mode allocated-space cap:
/// `allocated_space(x) = max(x, min(max(2*x, 1), T))` per row after
/// rebalance. The cap limits headroom, never live data. Bulk build uses a
/// separate layout and does not eagerly promote. `None` disables promotion
/// and the cap. Default T is 1024; configured demotion uses T/2.
pub const T_PROMOTE: u32 = 1024;
/// Tree → slab demotion trigger on live degree (hysteresis band T/2).
pub const T_DEMOTE: u32 = T_PROMOTE / 2;
/// Stack buffer width for hot-path edge I/O (mirrors the
/// `INLINE_EDGE_BYTES` pattern of the reference implementation).
pub const INLINE_EDGE_BYTES: usize = 64;
/// Buffer width in bytes for small slab scans and record collection.
/// Larger neighbor scans use [`SLAB_SCAN_BUF_BYTES`].
pub const SCAN_CHUNK_BYTES: usize = 512;

/// Scratch size for [`LaraGraph::for_each_in_slab`]'s bulk-read chunks.
/// Each chunk contains whole records, up to 32 KiB, clipped to the stored
/// extent. Records wider than this buffer retain per-record reads.
pub const SLAB_SCAN_BUF_BYTES: usize = 32 * 1024;
/// Upper bound for the single-read fast path in
/// [`LaraGraph::for_each_slab_record`]. Spans at or below this size are
/// read in one call; larger spans use the existing chunked path.
pub const BULK_SCAN_BYTES: usize = 64 * 1024;

// Allocation rule (fixed): `allocated_space(x) = min(max(2*x, 1), T)`.
//
// Sweep result (see paper §5): at equal totals the gap formula is
// irrelevant — only the total (G) and the cap (T) move insert/memory.
// The single formula keeps `2x` headroom capped at the promote
// threshold `T`, so a bucket reaching the cap promotes and the tree
// becomes the escape hatch.

// PMA density thresholds (leaf-level; height interpolation is a no-op at a
// single leaf, kept for DGAP parity in docs/tests).
/// Upper density threshold (leaf full → rebalance).
pub const UP_0: f64 = 1.00;
/// Lower density threshold (leaf sparse → compact).
pub const LOW_0: f64 = 0.25;

// -- Wire rows ------------------------------------------------------------

/// Unlabeled vertex row: 24 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vertex {
    base: u64,
    degree: u32,
    stored: u32,
    log_head: i32,
    bucket: u32,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            base: 0,
            degree: 0,
            stored: 0,
            log_head: -1,
            bucket: NO_BUCKET,
        }
    }
}

impl FixedWidth for Vertex {
    const BYTES: usize = 24;

    fn read_from(bytes: &[u8]) -> Self {
        Self {
            base: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            degree: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            stored: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            log_head: i32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            bucket: u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
        }
    }

    fn write_to(&self, bytes: &mut [u8]) {
        bytes[0..8].copy_from_slice(&self.base.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.degree.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.stored.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.log_head.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.bucket.to_le_bytes());
    }
}

impl CsrVertex for Vertex {
    fn base_slot_start(&self) -> u64 {
        self.base
    }
    fn degree(&self) -> u32 {
        self.degree
    }
    fn stored_degree(&self) -> u32 {
        self.stored
    }
    fn log_head(&self) -> i32 {
        self.log_head
    }
    fn tree_bucket(&self) -> u32 {
        self.bucket
    }

    fn with_base_slot_start(mut self, start: u64) -> Self {
        self.base = start;
        self
    }
    fn with_degree(mut self, n: u32) -> Self {
        self.degree = n;
        self.stored = n;
        self
    }
    fn with_live_degree(mut self, n: u32) -> Self {
        self.degree = n;
        self
    }
    fn with_stored_degree(mut self, n: u32) -> Self {
        self.stored = n;
        self
    }
    fn with_log_head(mut self, head: i32) -> Self {
        self.log_head = head;
        self
    }
    fn with_tree_bucket(mut self, bucket: u32) -> Self {
        self.bucket = bucket;
        self
    }
}

impl CsrVertexTombstone for Vertex {
    fn is_row_tombstone(&self) -> bool {
        false
    }
    fn with_row_tombstone(self, _t: bool) -> Self {
        self
    }
}

/// Marker used by Edge's optional tombstone capability, not by the
/// default graph's dense-prefix liveness contract.
pub const EDGE_TOMBSTONE: u32 = u32::MAX;

/// Unlabeled edge row: 4 bytes (neighbor id).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge(u32);

impl Default for Edge {
    fn default() -> Self {
        Self(EDGE_TOMBSTONE)
    }
}

impl FixedWidth for Edge {
    const BYTES: usize = 4;

    fn read_from(bytes: &[u8]) -> Self {
        Self(u32::from_le_bytes(bytes[0..4].try_into().unwrap()))
    }
    fn write_to(&self, bytes: &mut [u8]) {
        bytes[0..4].copy_from_slice(&self.0.to_le_bytes());
    }
}

impl CsrEdge for Edge {
    fn neighbor(&self) -> u32 {
        self.0
    }
    fn with_neighbor(mut self, neighbor: u32) -> Self {
        self.0 = neighbor;
        self
    }
}

impl CsrEdgeTombstone for Edge {
    fn tombstone_edge() -> Self {
        Self(EDGE_TOMBSTONE)
    }
    fn is_tombstone(&self) -> bool {
        self.0 == EDGE_TOMBSTONE
    }
}

/// Per-leaf overflow-log entry: `(src, dst, label, prev)` (16 bytes).
/// `label` identifies the labeled bucket (unlabeled rows set it to
/// `u16::MAX`); the two bytes after it are reserved padding for alignment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LogEntry {
    pub(crate) src: u32,
    pub(crate) dst: u32,
    pub(crate) label: u16,
    pub(crate) _pad: u16,
    pub(crate) prev: i32,
}

const LOG_ENTRY_BYTES: u64 = 16;

/// Log-directory sentinel: the leaf holds no log entries and no chunk is
/// backed for it (mirrors the `-1` empty-log idiom used by `log_head`).
const LOG_UNALLOCATED: u64 = u64::MAX;

/// Pooled-tiny row flag: top bit of the `stored` row field is reserved.
/// Slab widths must fit in the remaining 31 bits (not merely T_PROMOTE).
/// All slab readers must use [`stored_of`], never the raw field.
const TINY_FLAG: u32 = 1 << 31;
/// Fixed tiny-inline capacity: first `K` neighbors live in a `K × 4`-byte
/// chunk referenced by the row (`log_head` = chunk byte offset; the log
/// is unused while tiny-active, so the field is free; `base` stays 0).
// K-tuning verdict (tiny_k_probe): K=8 wins dispersed (25.9 vs 43.7 B/e,
// 8.2 vs 19.5ms); uniform-strided pays +4 B/e for a deg-9 spill cliff
// instead of deg-5; K=2 eliminated (spill storm on every fixture).
const TINY_K: u32 = 8;
const TINY_CHUNK_BYTES: u64 = TINY_K as u64 * 4;
/// Sweep ceiling for [`TINY_K`] (probe range K=2/4/8); the chunk reader
/// sizes its stack buffer for this max so only `TINY_K` moves per point.
const TINY_K_MAX: usize = 8;

/// `true` when the row is tiny-inline active (first `degree <= K`
/// neighbors in a `K × 4`-byte chunk, no slab/log/tree state).
fn is_tiny_row<V: CsrVertex>(row: &V) -> bool {
    row.stored_degree() & TINY_FLAG != 0
}

/// Slab-visible stored width: the raw `stored` field with the tiny flag
/// masked off. Uniform entry point for every slab-path read.
fn stored_of<V: CsrVertex>(row: &V) -> u32 {
    row.stored_degree() & !TINY_FLAG
}

/// Per-region memory bundle (mirrors ic-stable-lara's one-memory-per-store
/// layout; backends are cheap handles such as `Rc<RefCell<Vec<u8>>>`).
pub struct LaraMemories<M: Memory> {
    /// Unused compatibility handle; vertex rows are held on the heap.
    /// The graph leaves this backend region untouched.
    pub vertices: M,
    /// Edge slab (grows).
    pub edges: M,
    /// Per-leaf overflow logs.
    pub logs: M,
    /// Unused compatibility handle; PMA leaf counts are held on the heap.
    /// The graph leaves this backend region untouched.
    pub counts: M,
    /// Segment span metadata ([`SegmentSpanMetaStore`] region).
    pub spans: M,
    /// Free-span store ([`FreeSpanStore`] region).
    pub free_spans: M,
    /// Tree-CSR block store ([`LtbRawBlockStore`] region).
    pub blocks: M,
    /// Labeled-bucket descriptor store (opaque to the unlabeled core;
    /// owned by the labeled layer).
    pub buckets: M,
}

/// Heap state of one promoted Tree-CSR bucket. Payload bytes live in the
/// shared [`LtbRawBlockStore`], but the root is not persisted; byte backing
/// alone does not make a recoverable graph.
struct TreeState {
    root: Vec<u32>,
    stored: u32,
}

/// Per-source adjacency ordering for a core graph, not global edge chronology.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AdjacencyOrder {
    /// No survivor-order guarantee; dense prefixes use swap removal.
    #[default]
    Unordered,
    /// Preserve successful graph-application order within each source.
    Insertion,
}

/// An actual ordering change was rejected because a canonical row has live edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderChangeError;

impl core::fmt::Display for OrderChangeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("cannot change adjacency order while the selected scope has live edges")
    }
}
#[cfg(feature = "std")]
impl std::error::Error for OrderChangeError {}

// Installed only by the capability-bounded setter, never by trait detection.
struct InsertionOps<E> {
    tombstone: fn() -> E,
    is_tombstone: fn(&E) -> bool,
}

/// Generalized LARA graph.
pub struct LaraGraph<V: CsrVertex, E: CsrEdge, M: Memory> {
    vertices_mem: M,
    edges_mem: M,
    logs_mem: M,
    counts_mem: M,
    vertices_off: u64,
    logs_entries_off: u64,

    vertices: Vec<V>,
    trees: Vec<TreeState>,

    segment_actual: Vec<u64>,
    segment_total: Vec<u64>,
    log_idx: Vec<u32>,
    /// Lazy log directory: per-leaf chunk base in `logs_mem`, or
    /// [`LOG_UNALLOCATED`] when the leaf holds no log entries. Directory
    /// tax is 8 B/leaf (0.3% of one `cap × 16 B` chunk at cap=170).
    log_base: Vec<u64>,
    /// Free stack of retired fixed-size log chunks (all chunks share
    /// `log_capacity` entries, so reuse never fragments).
    log_pool: Vec<u64>,
    /// Bump cursor for fresh log chunks (starts at `logs_entries_off`).
    /// Shared with tiny-inline chunks (both bump 1:1 byte space; both
    /// counted in the backed total, split by `tiny_bumped`).
    log_next: u64,
    /// Free stack of retired `K × 4`-byte tiny-inline chunks (byte offsets
    /// into `logs_mem`; same backend as log chunks, so persist/mmap
    /// coverage follows the existing delegation).
    tiny_pool: Vec<u64>,
    /// Bytes ever bumped for tiny-inline chunks (live + pooled; the
    /// reported tiny term — never decreases, like the log term).
    tiny_bumped: u64,

    free_spans: FreeSpanStore<M>,
    span_meta: SegmentSpanMetaStore<M>,
    blocks: LtbRawBlockStore<M>,
    /// Labeled-bucket descriptor memory (opaque to the core; the labeled
    /// layer drives it through [`LaraGraph::buckets_mem`]).
    buckets_mem: M,

    segment_size: u32,
    segment_count: u64,
    num_edges: u64,
    edge_capacity: u64,
    /// Shared scratch for cell moves (rebalances, compaction).
    scratch: Vec<u8>,
    /// Promotion threshold knob (`None` disables Tree-CSR mode — the
    /// "no-tree" ablation arm).
    promote_threshold: Option<u32>,
    /// Effective per-leaf overflow-log cap (entries). Defaults to
    /// [`MAX_LOG_ENTRIES`]; see [`LaraGraph::set_max_log_entries`].
    max_log_entries: usize,
    /// Allocated per-leaf log capacity (entries). Fixed at construction;
    /// bounds how far [`LaraGraph::set_max_log_entries`] may grow.
    log_capacity: usize,
    /// Free-span reuse knob (`false` = bump-allocate slab tail only — the
    /// "no-free-span" ablation arm).
    use_free_spans: bool,
    /// Perf instrument: completed `rebalance_leaf` calls (measurement
    /// only — never read by insert/scan logic).
    rebalance_count: u64,
    /// Perf instrument: completed `promote` calls (measurement only).
    promote_count: u64,
    insertion: Option<InsertionOps<E>>,
    _edge: PhantomData<E>,
}

// =========================================================================
// Core impl: construction, bulk build, tree I/O, scans and introspection.
// Ordinary operations keep E: CsrEdge even when ordering is configured.
// =========================================================================

impl<V: CsrVertex, E: CsrEdgeTombstone, M: Memory> LaraGraph<V, E, M> {
    /// Selects per-source ordering for the entire core graph.
    ///
    /// An actual change requires every canonical row's live degree to be
    /// zero; same-mode calls are no-ops even on populated graphs. Empty
    /// slab/log extents are retired before publishing the new mode. This
    /// does not erase unused bytes or recover an earlier insertion history.
    /// Configure before population with [`Self::build_pairs`].
    ///
    /// The setter needs marker capability, but ordinary graph operations do
    /// not. Insertion uses typed slab markers, dense tiny/tree shifts and
    /// oldest-first logs. No raw neighbor ID is implicitly reserved. Ordered
    /// insertion rejects a marker record or a neighbor-only reconstruction
    /// recognized as a marker; see [`Self::insert_edge_record`].
    ///
    /// The guarantee is per source, not global chronology or ordinal/edge-ID
    /// stability. Buffered push is not application; see [`crate::IngestBuffer`].
    /// This heap setting does not introduce persistence or reopen support.
    ///
    /// ```compile_fail
    /// use lara_graph::{AdjacencyOrder, CsrEdge, CsrVertex, LaraGraph, Memory};
    /// fn requires_marker<V: CsrVertex, E: CsrEdge, M: Memory>(g: &mut LaraGraph<V, E, M>) {
    ///     g.set_order(AdjacencyOrder::Insertion).unwrap();
    /// }
    /// ```
    pub fn set_order(&mut self, order: AdjacencyOrder) -> Result<(), OrderChangeError> {
        if self.order() == order {
            return Ok(());
        }
        if self.vertices.iter().any(|row| row.degree() != 0) {
            return Err(OrderChangeError);
        }
        // Every row is empty, so no sibling owns a live log entry. Retire
        // old marker extents before changing the predicate interpretation.
        for row in &mut self.vertices {
            if !is_tiny_row(row) && (stored_of(row) != 0 || row.log_head() != -1) {
                *row = row.with_stored_degree(0).with_log_head(-1);
            }
        }
        for leaf in 0..self.segment_count {
            self.log_idx[leaf as usize] = 0;
            self.free_log_leaf(leaf);
        }
        self.insertion = match order {
            AdjacencyOrder::Unordered => None,
            AdjacencyOrder::Insertion => Some(InsertionOps {
                tombstone: E::tombstone_edge,
                is_tombstone: E::is_tombstone,
            }),
        };
        Ok(())
    }
}

impl<V: CsrVertex, E: CsrEdge, M: Memory> LaraGraph<V, E, M> {
    /// Current per-source adjacency ordering, shared by all core vertices.
    pub fn order(&self) -> AdjacencyOrder {
        if self.insertion.is_some() {
            AdjacencyOrder::Insertion
        } else {
            AdjacencyOrder::Unordered
        }
    }

    #[inline]
    fn is_live(&self, e: &E) -> bool {
        self.insertion
            .as_ref()
            .is_none_or(|ops| !(ops.is_tombstone)(e))
    }

    pub(crate) fn check_insert_record(&self, src: u32, e: &E) {
        assert!((src as usize) < self.vertices.len(), "src out of range");
        if self.insertion.is_some() {
            // Validate before routing: neighbor-only paths must not create
            // a record that a later slab fold would interpret as a marker.
            assert!(
                self.is_live(e) && self.is_live(&E::with_neighbor(E::default(), e.neighbor())),
                "ordered insertion rejects tombstone records or neighbor-only reconstructions"
            );
        }
    }

    /// Segment geometry for `n` vertices (DGAP `compute_capacity` parity:
    /// ideal size `ceil_log2(n)`, count hyperfloored to a power of two).
    fn compute_segments(n: u64) -> (u32, u64) {
        let ideal = u64::from(ceil_log2_u(n.max(1)).max(1));
        let mut count = ceil_div_u(n.max(1), ideal).max(1);
        count = 1u64 << (63 - count.leading_zeros());
        let size = ceil_div_u(n.max(1), count) as u32;
        (size, count)
    }

    /// Creates an empty graph with `n` vertices.
    ///
    /// Vertex/count metadata lives on the heap; the corresponding handles
    /// in `mem` are retained without backend access. Other regions are
    /// initialized as needed and can return [`GrowFailed`].
    pub fn new(mem: LaraMemories<M>, n: u32) -> Result<Self, GrowFailed> {
        let (segment_size, segment_count) = Self::compute_segments(n as u64);
        Self::new_impl(mem, n as u64, segment_size, segment_count, MAX_LOG_ENTRIES)
    }

    /// Creates an empty graph with an explicit per-leaf overflow-log cap.
    /// Each lazily-backed chunk holds `max_log_entries` entries, so
    /// growing beyond the construction value later is impossible — see
    /// [`LaraGraph::set_max_log_entries`]. No log memory is backed until
    /// the first overflow.
    pub fn new_with_max_log_entries(
        mem: LaraMemories<M>,
        n: u32,
        max_log_entries: usize,
    ) -> Result<Self, GrowFailed> {
        let (segment_size, segment_count) = Self::compute_segments(n as u64);
        Self::new_impl(mem, n as u64, segment_size, segment_count, max_log_entries)
    }

    fn new_impl(
        mem: LaraMemories<M>,
        _n: u64,
        segment_size: u32,
        segment_count: u64,
        max_log_entries: usize,
    ) -> Result<Self, GrowFailed> {
        let LaraMemories {
            vertices: mem_vertices,
            edges: mem_edges,
            logs: mem_logs,
            counts: mem_counts,
            spans: mem_spans,
            free_spans: mem_free_spans,
            blocks: mem_blocks,
            buckets: mem_buckets,
        } = mem;
        let n_padded = segment_size as u64 * segment_count;

        // Legacy offset exposed by region_offsets; no vertex backend is
        // backed or synchronized. Rows and counts are heap vectors below.
        let vertices_off = 8u64;

        // Logs (lazy): only the 4096-byte header region is backed at
        // construction. Per-leaf `cap × 16 B` chunks are bump-allocated
        // from `logs_entries_off` on first overflow and recycled through
        // a free stack on fold — an empty graph pays the directory only.
        safe_write(&mem_logs, 4096 - 1, &[0u8])?;

        // Initial slab: one 4 KiB-span reservation so reads are safe.
        let edge_capacity: u64 = 1024;
        safe_write(&mem_edges, edge_capacity * E::BYTES as u64 - 1, &[0u8])?;

        let free_spans = FreeSpanStore::new(mem_free_spans)?;
        let mut span_meta = SegmentSpanMetaStore::new(mem_spans)?;
        span_meta.grow_to(segment_count)?;
        let blocks = LtbRawBlockStore::new(mem_blocks)?;

        let mut g = Self {
            vertices_mem: mem_vertices,
            edges_mem: mem_edges,
            logs_mem: mem_logs,
            counts_mem: mem_counts,
            vertices_off,
            logs_entries_off: 4096,
            vertices: vec![V::default(); n_padded as usize],
            trees: Vec::new(),
            segment_actual: vec![0; segment_count as usize],
            segment_total: vec![0; segment_count as usize],
            log_idx: vec![0; segment_count as usize],
            log_base: vec![LOG_UNALLOCATED; segment_count as usize],
            log_pool: Vec::new(),
            log_next: 4096,
            tiny_pool: Vec::new(),
            tiny_bumped: 0,
            free_spans,
            span_meta,
            blocks,
            buckets_mem: mem_buckets,
            segment_size,
            segment_count,
            num_edges: 0,
            edge_capacity,
            scratch: Vec::new(),
            promote_threshold: Some(T_PROMOTE),
            use_free_spans: true,
            rebalance_count: 0,
            promote_count: 0,
            max_log_entries,
            log_capacity: max_log_entries,
            insertion: None,
            _edge: PhantomData,
        };
        // Default spans: unassigned; set lazily on first build/relocate.
        for leaf in 0..segment_count {
            g.span_meta.set(
                leaf,
                SegmentSpanMeta {
                    physical_start: SPAN_UNASSIGNED,
                },
            )?;
        }
        Ok(g)
    }

    /// Creates an empty graph with an explicit segment size (sweep knob).
    pub fn new_with_segments(
        mem: LaraMemories<M>,
        n: u32,
        segment_size: u32,
    ) -> Result<Self, GrowFailed> {
        // Ideal count from the requested size, hyperfloored to a power of
        // two; the segment size is then recomputed (DGAP compute_capacity
        // parity) so the padded vertex count covers `n`.
        let count = ceil_div_u(n.max(1) as u64, segment_size as u64).max(1);
        let count_pow2 = 1u64 << (63 - count.leading_zeros());
        let recomputed = ceil_div_u(n.max(1) as u64, count_pow2).max(1) as u32;
        Self::new_impl(mem, n as u64, recomputed, count_pow2, MAX_LOG_ENTRIES)
    }

    /// Creates an empty graph with an explicit segment size and log cap.
    pub fn new_with_segments_and_max_log(
        mem: LaraMemories<M>,
        n: u32,
        segment_size: u32,
        max_log_entries: usize,
    ) -> Result<Self, GrowFailed> {
        let count = ceil_div_u(n.max(1) as u64, segment_size as u64).max(1);
        let count_pow2 = 1u64 << (63 - count.leading_zeros());
        let recomputed = ceil_div_u(n.max(1) as u64, count_pow2).max(1) as u32;
        Self::new_impl(mem, n as u64, recomputed, count_pow2, max_log_entries)
    }

    /// Sets the promotion threshold (`None` = never promote).
    /// Demotion uses half this threshold, or the default [`T_DEMOTE`]
    /// if promotion is disabled after trees have already been created.
    pub fn set_promote_threshold(&mut self, threshold: Option<u32>) {
        self.promote_threshold = threshold;
    }

    pub(crate) fn demote_threshold(&self) -> u32 {
        self.promote_threshold.map_or(T_DEMOTE, |t| t / 2)
    }

    /// Enables or disables free-span reuse (ablation knob).
    pub fn set_use_free_spans(&mut self, enabled: bool) {
        self.use_free_spans = enabled;
    }

    /// Sets the effective per-leaf overflow-log cap.
    ///
    /// May only shrink (or hold) the construction capacity: values above
    /// the allocated [`LaraGraph::new_with_max_log_entries`] size panic,
    /// since the log region is fixed at construction. Shrinking takes
    /// effect on the next insert (a fuller-than-cap log folds via
    /// rebalance first). There is no builder form: the cap must be fixed
    /// before [`LaraGraph::new`] lays out memory — use
    /// [`LaraGraph::new_with_max_log_entries`] (or
    /// [`LaraGraph::new_with_segments_and_max_log`]) instead.
    pub fn set_max_log_entries(&mut self, max: usize) {
        assert!(
            max <= self.log_capacity,
            "max_log_entries {max} exceeds log region capacity {}",
            self.log_capacity
        );
        self.max_log_entries = max;
    }

    /// Effective per-leaf overflow-log cap (default [`MAX_LOG_ENTRIES`]).
    pub fn max_log_entries(&self) -> usize {
        self.max_log_entries
    }

    /// Current promote threshold (`None` = promotion disabled). Doubles as
    /// the slab allocated-space cap (see [`T_PROMOTE`]).
    pub fn promote_threshold(&self) -> Option<u32> {
        self.promote_threshold
    }

    /// Effective slab allocated-space cap: the promote threshold, or no
    /// cap (`u32::MAX`) when promotion is disabled.
    pub fn alloc_cap(&self) -> u64 {
        self.promote_threshold
            .map(|t| t as u64)
            .unwrap_or(u64::from(u32::MAX))
    }

    /// Allocation for live width `x`: `max(x, min(max(2*x, 1), T))`.
    /// The `1` keeps one headroom slot for empty buckets so the first
    /// post-rebalance insert lands in the slab (for `x >= 1`, `2x >= x+1`
    /// already guarantees a free slot).
    fn alloc_for(&self, x: u64) -> u64 {
        (2u64.saturating_mul(x).max(1)).min(self.alloc_cap()).max(x)
    }

    /// Bulk build from COO edges in the default Unordered mode (Builder
    /// layout parity with DGAP). Lays out contiguous leaf spans with weighted
    /// per-vertex slack, mirroring DGAP's `spread_weighted` at leaf
    /// granularity.
    pub fn with_edge_list(
        mem: LaraMemories<M>,
        edge_list: &[(u32, u32)],
        n: u32,
    ) -> Result<Self, GrowFailed> {
        let mut g = Self::new(mem, n)?;
        g.build_pairs(edge_list)?;
        Ok(g)
    }

    /// Replaces adjacency from COO pairs, preserving the configured mode.
    ///
    /// In Insertion mode, each source follows its relative input order;
    /// sources need not be grouped and destinations need not be sorted.
    /// Configure an empty graph first with [`Self::set_order`]. Replacement
    /// starts new adjacency sequences and retires all old storage owners.
    /// Ordered records/sources are checked before any replacement writes;
    /// arbitrary codec/backend failures are not transactional.
    pub fn build_pairs(&mut self, edge_list: &[(u32, u32)]) -> Result<(), GrowFailed> {
        let nv = self.vertices.len() as u32;
        let ss = self.segment_size as usize;

        // Degree counts.
        let mut deg = vec![0u32; nv as usize];
        for &(src, dst) in edge_list {
            if self.insertion.is_some() {
                self.check_insert_record(src, &E::with_neighbor(E::default(), dst));
            }
            deg[src as usize] += 1;
        }

        // Per-leaf spans: (deg_sum + ss) * 4 contiguous.
        let mut leaf_deg = vec![0u64; self.segment_count as usize];
        for (v, &d) in deg.iter().enumerate() {
            leaf_deg[v / ss] += d as u64;
        }
        let mut starts = vec![0u64; self.segment_count as usize];
        let mut totals = vec![0u64; self.segment_count as usize];
        let mut cursor = 0u64;
        for leaf in 0..self.segment_count as usize {
            let span = (leaf_deg[leaf] + ss as u64) * 4;
            starts[leaf] = cursor;
            totals[leaf] = span;
            cursor += span;
        }
        self.ensure_slab(cursor)?;

        // This is a replacement, not an append: retire every old owner.
        for tree in self.trees.drain(..) {
            for id in tree.root {
                self.blocks.release(id).expect("rebuild tree release");
            }
        }
        for v in 0..self.vertices.len() {
            let row = self.vertices[v];
            if is_tiny_row(&row) {
                self.tiny_free(row.log_head() as u64);
            }
        }
        for leaf in 0..self.segment_count {
            self.free_log_leaf(leaf);
        }
        self.log_idx.fill(0);
        self.free_spans.clear();
        self.release_span(cursor, self.edge_capacity - cursor);

        // Weighted spread within each leaf; write rows and edges.
        for leaf in 0..self.segment_count as usize {
            let v0 = leaf * ss;
            let v1 = ((v0 + ss) as u64).min(nv as u64) as usize;
            let deg_sum: u64 = deg[v0..v1].iter().map(|&d| d as u64).sum();
            let span = totals[leaf];
            let slack = span.saturating_sub(deg_sum) as f64;
            let total_deg = deg_sum + (v1 - v0) as u64;
            let mut step = slack / total_deg as f64;
            let precision = 100_000_000.0;
            step = ((step * precision) as i64) as f64 / precision;

            let mut base = starts[leaf];
            for (v, &d) in deg.iter().enumerate().skip(v0).take(v1 - v0) {
                self.vertices[v] = V::default().with_base_slot_start(base).with_degree(d);
                base += d as u64 + (step * (d as f64 + 1.0)) as u64;
            }
            self.segment_actual[leaf] = deg_sum;
            self.segment_total[leaf] = span;
            self.span_meta.set(
                leaf as u64,
                SegmentSpanMeta {
                    physical_start: starts[leaf],
                },
            )?;
        }

        // Per-vertex cursors preserve input-relative order even for interleaved sources.
        let mut cursors: Vec<u64> = self
            .vertices
            .iter()
            .map(|row| row.base_slot_start())
            .collect();
        for &(src, dst) in edge_list {
            let w = cursors[src as usize];
            self.write_edge(w, &E::with_neighbor(E::default(), dst));
            cursors[src as usize] = w + 1;
        }

        self.num_edges = edge_list.len() as u64;
        Ok(())
    }

    // -- slab I/O ----------------------------------------------------------

    #[inline]
    fn read_edge(&self, slot: u64) -> E {
        if E::BYTES <= INLINE_EDGE_BYTES {
            let mut buf = [0u8; INLINE_EDGE_BYTES];
            self.edges_mem
                .read(slot * E::BYTES as u64, &mut buf[..E::BYTES]);
            E::read_from(&buf[..E::BYTES])
        } else {
            let mut buf = vec![0u8; E::BYTES];
            self.edges_mem.read(slot * E::BYTES as u64, &mut buf);
            E::read_from(&buf)
        }
    }

    #[inline]
    fn write_edge(&mut self, slot: u64, e: &E) {
        if E::BYTES <= INLINE_EDGE_BYTES {
            let mut buf = [0u8; INLINE_EDGE_BYTES];
            e.write_to(&mut buf[..E::BYTES]);
            self.edges_mem
                .write(slot * E::BYTES as u64, &buf[..E::BYTES]);
        } else {
            let mut buf = vec![0u8; E::BYTES];
            e.write_to(&mut buf);
            self.edges_mem.write(slot * E::BYTES as u64, &buf);
        }
    }

    /// Grows the slab to at least `slots` slots (idempotent).
    pub fn ensure_slab(&mut self, slots: u64) -> Result<(), GrowFailed> {
        if slots > self.edge_capacity {
            safe_write(
                &self.edges_mem,
                slots
                    .checked_mul(E::BYTES as u64)
                    .expect("slab address overflow")
                    - 1,
                &[0u8],
            )?;
            self.edge_capacity = slots;
        }
        Ok(())
    }

    // -- vertex rows -------------------------------------------------------

    #[inline]
    fn get_row(&self, v: u32) -> V {
        self.vertices[v as usize]
    }

    #[inline]
    fn set_row(&mut self, v: u32, row: V) {
        self.vertices[v as usize] = row;
    }

    // -- log I/O -----------------------------------------------------------

    /// Backed bytes of one log chunk (`log_capacity × 16`).
    fn log_chunk_bytes(&self) -> u64 {
        self.log_capacity as u64 * LOG_ENTRY_BYTES
    }

    /// Chunk base for `leaf`, allocating (pool reuse, else bump) on first
    /// use. Called only on the overflow path — the slab fast path never
    /// touches the log, so its cost is unchanged.
    fn ensure_log_base(&mut self, leaf: u64) -> u64 {
        let base = self.log_base[leaf as usize];
        if base != LOG_UNALLOCATED {
            return base;
        }
        let chunk = self.log_chunk_bytes();
        if chunk == 0 {
            return LOG_UNALLOCATED;
        }
        let fresh = match self.log_pool.pop() {
            Some(reused) => reused,
            None => {
                let at = self.log_next;
                self.log_next += chunk;
                safe_write(&self.logs_mem, at + chunk - 1, &[0u8]).expect("log chunk grow");
                at
            }
        };
        self.log_base[leaf as usize] = fresh;
        fresh
    }

    /// Returns `leaf`'s chunk to the free stack (fold point; the bumped
    /// backing stays reserved — memory accounting counts bumped bytes).
    fn free_log_leaf(&mut self, leaf: u64) {
        let base = self.log_base[leaf as usize];
        if base != LOG_UNALLOCATED {
            self.log_base[leaf as usize] = LOG_UNALLOCATED;
            self.log_pool.push(base);
        }
    }

    /// Backed log memory in bytes: header + all bumped chunks (live and
    /// pooled) + the DRAM directory. Pooled chunks stay reserved so the
    /// total never decreases within a graph's lifetime.
    pub fn log_memory_bytes(&self) -> u64 {
        4096 + (self.log_next - self.logs_entries_off) + self.segment_count * 8
    }

    /// Number of leaves owning a backed log chunk. A chunk may retain
    /// no live entries after promotion/deletion; a whole-leaf fold frees it.
    pub fn log_allocated_chunks(&self) -> u64 {
        self.log_base
            .iter()
            .filter(|&&b| b != LOG_UNALLOCATED)
            .count() as u64
    }

    /// Retired-chunk free-stack depth (measurement only).
    pub fn log_pool_depth(&self) -> usize {
        self.log_pool.len()
    }

    /// Backs one pooled `K × 4`-byte tiny chunk. Fresh offsets must fit
    /// the signed 32-bit row field; reject before advancing the cursor.
    fn tiny_alloc(&mut self) -> u64 {
        if let Some(reused) = self.tiny_pool.pop() {
            return reused;
        }
        let at = self.log_next;
        assert!(
            i32::try_from(at).is_ok(),
            "tiny chunk offset exceeds signed 32-bit row field"
        );
        safe_write(&self.logs_mem, at + TINY_CHUNK_BYTES - 1, &[0u8]).expect("tiny chunk grow");
        self.log_next += TINY_CHUNK_BYTES;
        self.tiny_bumped += TINY_CHUNK_BYTES;
        at
    }

    /// Returns a tiny-inline chunk to the free stack.
    fn tiny_free(&mut self, off: u64) {
        self.tiny_pool.push(off);
    }

    /// Reads only the `n * 4` live bytes of a tiny-active row, with no
    /// backend read when empty. The buffer holds [`TINY_K_MAX`] entries;
    /// only its first `n` entries are valid.
    fn read_tiny(&self, off: u64, n: u32) -> [u32; TINY_K_MAX] {
        debug_assert!(n <= TINY_K);
        let mut buf = [0u8; TINY_K_MAX * 4];
        if n > 0 {
            self.logs_mem.read(off, &mut buf[..n as usize * 4]);
        }
        let mut out = [0u32; TINY_K_MAX];
        for k in 0..n as usize {
            out[k] = u32::from_le_bytes(buf[k * 4..k * 4 + 4].try_into().unwrap());
        }
        out
    }

    /// Backed tiny-inline memory in bytes (live + pooled; never
    /// decreases). Split out of the log term for reporting.
    pub fn tiny_memory_bytes(&self) -> u64 {
        self.tiny_bumped
    }

    /// Retired tiny-chunk free-stack depth (measurement only).
    pub fn tiny_pool_depth(&self) -> usize {
        self.tiny_pool.len()
    }

    /// Number of tiny-inline active rows (measurement only).
    pub fn tiny_row_count(&self) -> u64 {
        self.vertices.iter().filter(|r| is_tiny_row(*r)).count() as u64
    }

    /// Whether `v`'s row is tiny-inline active (measurement only).
    pub fn row_is_tiny(&self, v: u32) -> bool {
        is_tiny_row(&self.vertices[v as usize])
    }

    fn log_region(&self, leaf: u64) -> u64 {
        self.log_base[leaf as usize]
    }

    /// Scan-only log read: `dst` + `prev` (8 bytes in two reads).
    /// `src`/`label` are context-reconstructible during scans, so the
    /// remaining 8 bytes are skipped. Rebalance/deletion keep `read_log`.
    #[inline]
    fn read_log_scan(&self, leaf: u64, idx: u32) -> (u32, i32) {
        let base = self.log_region(leaf) + idx as u64 * LOG_ENTRY_BYTES;
        let mut b4 = [0u8; 4];
        self.logs_mem.read(base + 4, &mut b4);
        let dst = u32::from_le_bytes(b4);
        self.logs_mem.read(base + 12, &mut b4);
        (dst, i32::from_le_bytes(b4))
    }

    #[inline]
    fn read_log(&self, leaf: u64, idx: u32) -> LogEntry {
        let off = self.log_region(leaf) + idx as u64 * LOG_ENTRY_BYTES;
        let mut buf = [0u8; 16];
        self.logs_mem.read(off, &mut buf);
        LogEntry {
            src: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            dst: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            label: u16::from_le_bytes(buf[8..10].try_into().unwrap()),
            _pad: u16::from_le_bytes(buf[10..12].try_into().unwrap()),
            prev: i32::from_le_bytes(buf[12..16].try_into().unwrap()),
        }
    }

    #[inline]
    fn write_log(&mut self, leaf: u64, idx: u32, e: &LogEntry) {
        let _ = self.ensure_log_base(leaf);
        let off = self.log_region(leaf) + idx as u64 * LOG_ENTRY_BYTES;
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&e.src.to_le_bytes());
        buf[4..8].copy_from_slice(&e.dst.to_le_bytes());
        buf[8..10].copy_from_slice(&e.label.to_le_bytes());
        buf[10..12].copy_from_slice(&e._pad.to_le_bytes());
        buf[12..16].copy_from_slice(&e.prev.to_le_bytes());
        self.logs_mem.write(off, &buf);
    }

    // -- geometry ----------------------------------------------------------

    /// Exclusive slab boundary: the next row's base in the same leaf,
    /// otherwise the leaf end. Repacking preserves nondecreasing bases,
    /// including zero-width tree rows.
    fn slab_boundary(&self, v: u32, row: &V) -> u64 {
        // Hot path first: non-final rows are bounded by the next row's
        // base. `self.vertices` is a DRAM vec (not a Memory backend), so
        // this is one indexed load, not a backend round-trip — kept
        // because no spare row bytes exist to cache the end (see Part 2
        // row-layout note). Only leaf-final rows pay the span-meta read.
        let within = v as u64 % self.segment_size as u64;
        if within + 1 < self.segment_size as u64 && (v as usize + 1) < self.vertices.len() {
            let next = self.vertices[v as usize + 1].base_slot_start();
            if next >= row.base_slot_start() {
                return next;
            }
        }
        self.leaf_span_end(v as u64 / self.segment_size as u64)
    }

    fn leaf_span_end(&self, leaf: u64) -> u64 {
        let start = self.span_meta.get(leaf).physical_start;
        if start == SPAN_UNASSIGNED {
            return 0;
        }
        start + self.segment_total[leaf as usize]
    }

    // -- tree-CSR I/O ------------------------------------------------------

    /// Appends to the tree bucket for `v` (reserve/commit).
    ///
    /// Tree buckets are depth-1 only (root → leaf blocks); exceeding the
    /// depth-1 coverage of `B * R_MAX` slots fails closed rather than
    /// mis-address (depth-2 addressing is a future slice).
    fn tree_insert(&mut self, tree_idx: usize, target: u32) {
        let state = &mut self.trees[tree_idx];
        let new_stored = state.stored + 1;
        assert_depth_one(u64::from(new_stored));
        let new_root_len = root_len(new_stored);
        while state.root.len() < new_root_len {
            let id = self.blocks.mint_zeroed().expect("tree mint");
            state.root.push(id);
        }
        let slot = new_stored - 1;
        let in_block = slot as usize % B;
        let block_id = state.root[slot as usize / B];
        self.blocks
            .write_payload_partial(block_id, in_block * 4, &target.to_le_bytes())
            .expect("tree write");
        state.stored = new_stored;
    }

    fn tree_read(&self, tree_idx: usize, slot: u32) -> u32 {
        let state = &self.trees[tree_idx];
        let in_block = slot as usize % B;
        let block_id = state.root[slot as usize / B];
        let mut buf = [0u8; 4];
        self.blocks
            .read_payload_partial(block_id, in_block * 4, &mut buf)
            .expect("tree read");
        u32::from_le_bytes(buf)
    }

    // -- tree-state accessors (shared with the labeled layer) ------------
    //
    // The labeled layer promotes/demotes per-bucket Tree-CSR buckets using
    // the same `trees` vector and block store as the unlabeled core. These
    // accessors expose the minimal surface it needs; the core's own
    // promote/demote keep operating on `trees` directly.

    /// Number of live Tree-CSR buckets.
    pub fn tree_state_len(&self) -> usize {
        self.trees.len()
    }

    /// Pushes a new tree bucket (root block ids + stored count) and returns
    /// its id.
    pub fn tree_state_push(&mut self, root: Vec<u32>, stored: u32) -> u32 {
        assert_depth_one(u64::from(stored));
        assert_eq!(
            root.len(),
            root_len(stored),
            "tree root does not cover stored slots"
        );
        self.trees.push(TreeState { root, stored });
        (self.trees.len() - 1) as u32
    }

    /// Stored (live) count of tree bucket `idx`.
    pub fn tree_state_stored(&self, idx: usize) -> u32 {
        self.trees[idx].stored
    }

    /// Appends `target` to tree bucket `idx` (O(1) reserve/commit).
    pub fn tree_state_insert(&mut self, idx: usize, target: u32) {
        self.tree_insert(idx, target);
    }

    /// Reads slot `slot` of tree bucket `idx`.
    pub fn tree_state_read(&self, idx: usize, slot: u32) -> u32 {
        self.tree_read(idx, slot)
    }

    /// Scans the stored slot extent, not retained root capacity, in order.
    /// Reads each occupied block once; a partial tail reads only its used
    /// prefix through the checked partial reader. Full blocks retain the
    /// mapped-read fast path. Order and multiplicity match per-slot reads;
    /// callbacks run after the backend read has returned.
    pub fn for_each_in_tree<F: FnMut(u32)>(&self, idx: usize, mut f: F) {
        let stored = self.trees[idx].stored;
        let mut payload = [0u8; BLOCK_PAYLOAD_BYTES];
        let used_blocks = stored.div_ceil(B as u32) as usize;
        for (root_index, &block_id) in self.trees[idx].root.iter().take(used_blocks).enumerate() {
            let first = root_index * B;
            let remaining = stored.saturating_sub(first as u32).min(B as u32);
            if remaining == B as u32 {
                self.blocks.read_payload_mapped(block_id, &mut payload)
            } else {
                self.blocks.read_payload_partial(
                    block_id,
                    0,
                    &mut payload[..remaining as usize * 4],
                )
            }
            .expect("tree scan read");
            for s in 0..remaining {
                let off = s as usize * 4;
                f(u32::from_le_bytes(
                    payload[off..off + 4].try_into().unwrap(),
                ));
            }
        }
    }

    /// Removes one dense tree slot without preserving survivor order.
    /// Retained roots and unused tail bytes remain owned and unchanged.
    pub(crate) fn tree_remove_swap(&mut self, idx: usize, slot: u32) {
        let last = self.trees[idx].stored - 1;
        if slot != last {
            let target = self.tree_read(idx, last);
            let block_id = self.trees[idx].root[slot as usize / B];
            self.blocks
                .write_payload_partial(block_id, slot as usize % B * 4, &target.to_le_bytes())
                .expect("tree swap");
        }
        self.trees[idx].stored = last;
    }

    /// Removes slot `slot` of tree bucket `idx`, shifting the tail left to
    /// keep the bucket left-packed.
    pub fn tree_state_remove_shift(&mut self, idx: usize, slot: u32) {
        let stored = self.trees[idx].stored;
        for s in slot..stored - 1 {
            let nxt = self.tree_read(idx, s + 1);
            let in_block = s as usize % B;
            let block_id = self.trees[idx].root[s as usize / B];
            self.blocks
                .write_payload_partial(block_id, in_block * 4, &nxt.to_le_bytes())
                .expect("tree shift");
        }
        self.trees[idx].stored -= 1;
    }

    /// Releases tree bucket `idx`'s blocks and removes it from `trees`.
    /// Callers must fix up any `tree_bucket` ids greater than `idx`.
    pub fn tree_state_release(&mut self, idx: usize) {
        for &block_id in &self.trees[idx].root {
            self.blocks.release(block_id).expect("tree release");
        }
        self.trees.remove(idx);
    }

    /// Mints a zeroed tree block.
    pub fn mint_block(&mut self) -> Result<u32, crate::block_store::BlockError> {
        self.blocks.mint_zeroed()
    }

    /// Writes a full tree block payload.
    pub fn write_block_payload(
        &mut self,
        id: u32,
        src: &[u8; crate::block_store::BLOCK_PAYLOAD_BYTES],
    ) -> Result<(), crate::block_store::BlockError> {
        self.blocks.write_payload(id, src)
    }

    /// Reads a partial tree block payload.
    pub fn read_block_payload_partial(
        &mut self,
        id: u32,
        offset: usize,
        dst: &mut [u8],
    ) -> Result<(), crate::block_store::BlockError> {
        self.blocks.read_payload_partial(id, offset, dst)
    }

    /// Writes a partial tree block payload.
    pub fn write_block_payload_partial(
        &mut self,
        id: u32,
        offset: usize,
        src: &[u8],
    ) -> Result<(), crate::block_store::BlockError> {
        self.blocks.write_payload_partial(id, offset, src)
    }

    /// Releases a tree block back to the store.
    pub fn release_block(&mut self, id: u32) -> Result<(), crate::block_store::BlockError> {
        self.blocks.release(id)
    }

    /// Current slab capacity in slots.
    pub fn edge_capacity(&self) -> u64 {
        self.edge_capacity
    }

    // -- raw record scan ---------------------------------------------------

    /// Chunk-batched record reader: yields every record in the supplied
    /// backed slab prefix as `&E`, with no marker interpretation.
    /// Ordered graph prefixes may contain markers; this raw reader includes them.
    #[inline]
    pub fn for_each_record_in_row<F: FnMut(&E)>(&self, base: u64, stored: u32, mut f: F) {
        if stored == 0 {
            return;
        }
        if E::BYTES > SCAN_CHUNK_BYTES {
            for slot in 0..stored {
                let e = self.read_edge(base + slot as u64);
                f(&e);
            }
            return;
        }
        let slots_per_chunk = (SCAN_CHUNK_BYTES / E::BYTES) as u32;
        let mut buf = [0u8; SCAN_CHUNK_BYTES];
        let mut start = 0u32;
        while start < stored {
            let n = (stored - start).min(slots_per_chunk) as usize;
            let bytes = n * E::BYTES;
            self.edges_mem
                .read((base + start as u64) * E::BYTES as u64, &mut buf[..bytes]);
            for k in 0..n {
                f(&E::read_from(&buf[k * E::BYTES..(k + 1) * E::BYTES]));
            }
            start += n as u32;
        }
    }

    // -- introspection -----------------------------------------------------

    /// Number of vertices (including rounded-up padding rows).
    pub fn num_nodes(&self) -> u64 {
        self.vertices.len() as u64
    }

    /// Number of live edges.
    pub fn num_edges(&self) -> u64 {
        self.num_edges
    }

    /// Logical out-degree of `v`.
    pub fn out_degree(&self, v: u32) -> u64 {
        self.get_row(v).degree() as u64
    }

    /// Allocated slab width of `v`'s row (boundary − base), capped at
    /// the promote threshold by the rebalance layout.
    pub fn row_allocated(&self, v: u32) -> u64 {
        let row = self.get_row(v);
        let boundary = self.slab_boundary(v, &row);
        boundary.saturating_sub(row.base_slot_start())
    }

    /// Segment size (vertices per leaf).
    pub fn segment_size(&self) -> u32 {
        self.segment_size
    }

    /// Number of leaf segments.
    pub fn segment_count(&self) -> u64 {
        self.segment_count
    }

    /// Free-span store statistics: `(span_count, free_slots, largest)`.
    pub fn free_span_stats(&self) -> (u32, u64, u64) {
        self.free_spans.stats()
    }

    /// Tree-CSR bucket count (promoted vertices).
    pub fn tree_bucket_count(&self) -> usize {
        self.trees.len()
    }

    /// Slab capacity in slots (grows on relocation).
    pub fn slab_capacity(&self) -> u64 {
        self.edge_capacity
    }

    /// Total edge-slot footprint for memory-overhead comparisons: slab
    /// high-water plus tree data blocks × [`B`] slots each. Root
    /// descriptors (small DRAM `Vec`s) are excluded. Accurate for
    /// insert-only workloads (block IDs are never recycled mid-run).
    ///
    /// Note: excludes the per-leaf overflow-log region — see
    /// [`LaraGraph::memory_bytes_total`] for the log-inclusive total.
    pub fn memory_slots(&self) -> u64 {
        self.edge_capacity + self.blocks.minted() as u64 * B as u64
    }

    /// Completed `rebalance_leaf` calls since construction or the last
    /// [`LaraGraph::reset_perf_counters`]. Measurement instrument only —
    /// insert/scan behavior never reads these counters.
    pub fn rebalance_count(&self) -> u64 {
        self.rebalance_count
    }

    /// Completed `promote` calls since construction or the last
    /// [`LaraGraph::reset_perf_counters`]. Measurement instrument only.
    pub fn promote_count(&self) -> u64 {
        self.promote_count
    }

    /// Zeroes [`LaraGraph::rebalance_count`] and
    /// [`LaraGraph::promote_count`]. Use after bulk build to count only
    /// the dynamic-insert phase.
    pub fn reset_perf_counters(&mut self) {
        self.rebalance_count = 0;
        self.promote_count = 0;
    }

    /// Legacy **partial storage estimate**, not total heap, mapped backing,
    /// or resident memory: slab (`edge_capacity × E::BYTES`) + tree block
    /// payloads + bumped log/tiny chunks and log directory + vertex wire
    /// rows. Omits page rounding, spare Vec capacity, tree headers/roots,
    /// counts, span/free-span metadata, allocator indexes, and other heap
    /// allocations. Do not compare this number with another graph's total.
    /// See [`LaraGraph::memory_slots`] for the narrower slot-only metric.
    ///
    /// The log term counts actually bumped chunks, not the configured
    /// cap — an overflow-free graph pays the 4096-byte header plus the
    /// directory only. Shrinking via [`LaraGraph::set_max_log_entries`]
    /// does not release bumped memory (retired chunks park on the free
    /// stack for reuse). The tiny term ([`LaraGraph::tiny_memory_bytes`])
    /// is backed inside the same region; use it to split the log term
    /// for reporting (pure log = `log_memory_bytes() - tiny_memory_bytes()`).
    pub fn memory_bytes_total(&self) -> u64 {
        let slab = self.edge_capacity * E::BYTES as u64;
        let tree = self.blocks.minted() as u64 * crate::block_store::BLOCK_PAYLOAD_BYTES as u64;
        let logs = self.log_memory_bytes();
        let verts = self.vertices.len() as u64 * V::BYTES as u64;
        slab + tree + logs + verts
    }

    /// Storage checks shared with the labeled layer; independent of row
    /// interpretation. These are diagnostics, not hot-path validation.
    pub(crate) fn verify_storage_invariants(&self) {
        let check_ranges = |mut ranges: Vec<(u64, u64)>, limit: u64| {
            ranges.sort_unstable();
            let mut end = 0;
            for (start, next) in ranges {
                assert!(
                    end <= start && start < next && next <= limit,
                    "overlapping or out-of-bounds allocation"
                );
                end = next;
            }
        };
        let mut slabs: Vec<_> = self
            .free_spans
            .spans()
            .iter()
            .map(|s| (s.start, s.start + s.len))
            .collect();
        let mut chunks = Vec::new();
        for leaf in 0..self.segment_count as usize {
            let start = self.span_meta.get(leaf as u64).physical_start;
            let total = self.segment_total[leaf];
            if start == SPAN_UNASSIGNED {
                assert_eq!(total, 0);
            } else if total > 0 {
                slabs.push((start, start + total));
            }
            let base = self.log_base[leaf];
            assert!(self.log_idx[leaf] as usize <= self.log_capacity);
            if base == LOG_UNALLOCATED {
                assert_eq!(self.log_idx[leaf], 0);
            } else {
                chunks.push((base, base + self.log_chunk_bytes()));
            }
        }
        check_ranges(slabs, self.edge_capacity);
        for &base in &self.log_pool {
            chunks.push((base, base + self.log_chunk_bytes()));
        }
        let mut tiny_count = 0u64;
        for row in &self.vertices {
            if is_tiny_row(row) {
                assert!(row.log_head() >= 0, "missing tiny chunk");
                let base = row.log_head() as u64;
                chunks.push((base, base + TINY_CHUNK_BYTES));
                tiny_count += 1;
            }
        }
        for &base in &self.tiny_pool {
            assert!(i32::try_from(base).is_ok());
            chunks.push((base, base + TINY_CHUNK_BYTES));
        }
        assert_eq!(
            (tiny_count + self.tiny_pool.len() as u64) * TINY_CHUNK_BYTES,
            self.tiny_bumped,
            "unowned tiny chunk"
        );
        assert!(
            chunks
                .iter()
                .all(|&(start, _)| start >= self.logs_entries_off)
        );
        assert_eq!(
            chunks.iter().map(|&(start, end)| end - start).sum::<u64>(),
            self.log_next - self.logs_entries_off,
            "unowned log-region bytes"
        );
        check_ranges(chunks, self.log_next);
        let mut block_ids = Vec::new();
        for tree in &self.trees {
            assert_depth_one(u64::from(tree.stored));
            assert!(
                tree.root.len() >= root_len(tree.stored) && tree.root.len() <= R_MAX,
                "tree root coverage"
            );
            assert!(tree.root.iter().all(|&id| id < self.blocks.minted()));
            block_ids.extend_from_slice(&tree.root);
        }
        block_ids.sort_unstable();
        assert!(
            block_ids.windows(2).all(|w| w[0] != w[1]),
            "shared tree block"
        );
    }

    // -- accessors shared with the labeled layer --------------------------

    /// Reads one vertex row (crate-internal contract for the labeled layer).
    pub fn get_row_pub(&self, v: u32) -> V {
        self.get_row(v)
    }

    /// Replaces one vertex row.
    pub fn set_row_pub(&mut self, v: u32, row: V) {
        self.set_row(v, row)
    }

    /// Reads one slab slot.
    pub fn read_edge_pub(&self, slot: u64) -> E {
        self.read_edge(slot)
    }

    /// Writes one slab slot (no geometry checks; the labeled layer keeps
    /// its own per-bucket cursors inside the row's stored width).
    pub fn write_edge_pub(&mut self, slot: u64, e: &E) {
        self.write_edge(slot, e)
    }

    /// Legacy vertex-column offset and the overflow-log entries offset.
    /// The first value is retained for compatibility only: there is no
    /// backed vertex column. Vertex rows and leaf counts live on the heap.
    pub fn region_offsets(&self) -> (u64, u64) {
        (self.vertices_off, self.logs_entries_off)
    }

    // -- accessors for the labeled layer ------------------------------------

    /// Mutable access to the opaque bucket-descriptor region.
    pub fn buckets_mem(&self) -> &M {
        &self.buckets_mem
    }

    /// Mutable access to the opaque bucket-descriptor region.
    pub fn buckets_mem_mut(&mut self) -> &mut M {
        &mut self.buckets_mem
    }

    /// Reads `(physical_start, segment_total, segment_actual)` for one leaf.
    pub fn leaf_geometry(&self, leaf: u64) -> (u64, u64, u64) {
        let start = self.span_meta.get(leaf).physical_start;
        (
            start,
            self.segment_total[leaf as usize],
            self.segment_actual[leaf as usize],
        )
    }

    /// Publishes `(physical_start, segment_total, segment_actual)` for one
    /// leaf (used by the labeled rebalance to commit a new layout).
    pub fn set_leaf_geometry(&mut self, leaf: u64, start: u64, total: u64, actual: u64) {
        let _ = self.span_meta.set(
            leaf,
            SegmentSpanMeta {
                physical_start: start,
            },
        );
        self.segment_total[leaf as usize] = total;
        self.segment_actual[leaf as usize] = actual;
    }

    /// Allocates `slots` of slab, returning the absolute slot start
    /// (best-fit free-span reuse when enabled, else bump the tail).
    pub fn allocate_span(&mut self, slots: u64) -> u64 {
        if self.use_free_spans {
            if let Some(start) = self.free_spans.alloc(slots) {
                return start;
            }
        }
        let start = self.edge_capacity;
        self.ensure_slab(start.checked_add(slots).expect("slab capacity overflow"))
            .expect("slab grow");
        start
    }

    /// Releases a retired slab region back to the free-span store (best
    /// effort; the no-free-span ablation arm books nothing).
    pub fn release_span(&mut self, start: u64, slots: u64) {
        if self.use_free_spans && start != SPAN_UNASSIGNED && slots > 0 {
            let _ = self.free_spans.release(start, slots);
        }
    }

    /// Reads one leaf-log entry (crate-internal; the labeled layer folds
    /// bucket logs).
    pub(crate) fn read_log_pub(&self, leaf: u64, idx: u32) -> LogEntry {
        self.read_log(leaf, idx)
    }

    /// Writes one leaf-log entry (crate-internal).
    pub(crate) fn write_log_pub(&mut self, leaf: u64, idx: u32, e: &LogEntry) {
        self.write_log(leaf, idx, e);
    }

    /// Retires one leaf's log chunk to the free stack (crate-internal;
    /// the labeled layer folds bucket logs without touching the core
    /// rebalance, so it must release through here).
    pub(crate) fn free_log_pub(&mut self, leaf: u64) {
        self.free_log_leaf(leaf)
    }

    pub(crate) fn reset_edge_count(&mut self) {
        self.num_edges = 0;
    }

    /// Adjusts the live-edge count by `delta` (can be negative).
    pub fn add_edges(&mut self, delta: i64) {
        self.num_edges = (self.num_edges as i64 + delta).max(0) as u64;
    }

    /// Returns the leaf-log insertion cursor for a leaf.
    pub fn log_idx_pub(&self, leaf: u64) -> u32 {
        self.log_idx[leaf as usize]
    }

    /// Resets the leaf-log insertion cursor.
    pub fn set_log_idx_pub(&mut self, leaf: u64, idx: u32) {
        self.log_idx[leaf as usize] = idx;
    }
}

// =========================================================================
// Graph mutation, scans, rebalance, promotion and invariant checks.
// Explicitly installed marker operations do not add ordinary trait bounds.
// =========================================================================

impl<V: CsrVertex, E: CsrEdge, M: Memory> LaraGraph<V, E, M> {
    /// Promotes `v` to Tree-CSR mode: snapshots neighbors, builds the
    /// bucket block-batched, retires the slab extent and clears the log head.
    ///
    /// Only reachable for 4-byte neighbor-only rows (`E::BYTES == 4`); the
    /// caller guards promotion so wide edge types never enter tree mode.
    fn promote(&mut self, v: u32) {
        assert_depth_one(u64::from(self.get_row(v).degree()));
        let mut neighbors = Vec::new();
        self.for_each_neighbor(v, |n| neighbors.push(n));
        assert_depth_one(neighbors.len() as u64);
        let live = neighbors.len() as u32;
        self.promote_count += 1;

        let mut root = Vec::new();
        let new_root_len = root_len(live);
        for _ in 0..new_root_len {
            let id = self.blocks.mint_zeroed().expect("promote mint");
            root.push(id);
        }
        for (root_index, &block_id) in root.iter().enumerate() {
            let first = root_index * B;
            let last = (first + B).min(neighbors.len());
            let mut payload = [0u8; crate::block_store::BLOCK_PAYLOAD_BYTES];
            for (s, t) in neighbors[first..last].iter().enumerate() {
                let off = s * 4;
                payload[off..off + 4].copy_from_slice(&t.to_le_bytes());
            }
            self.blocks
                .write_payload(block_id, &payload)
                .expect("promote write");
        }
        let tree_idx = self.trees.len();
        self.trees.push(TreeState { root, stored: live });

        // Retire the old extent and chain through metadata. Their unused
        // backing bytes need no marker writes and are not physically erased.
        let mut row = self.get_row(v);
        let leaf = v as u64 / self.segment_size as u64;
        // segment_actual includes both slab and log edges, not trees.
        self.segment_actual[leaf as usize] -= live as u64;
        row = row
            .with_live_degree(live)
            .with_stored_degree(0)
            .with_log_head(-1);
        // Sibling chains still own their indices. Only a whole-leaf fold
        // may reset this shared cursor and recycle its chunk.
        self.set_row(v, row.with_tree_bucket(tree_idx as u32));
    }

    /// Demotes `v` from Tree-CSR back to slab mode (ADR 0088 §7 / Plan
    /// 0319 §Step 2 parity). Collects the bucket's live records, stages
    /// them in the slab, releases the tree blocks, and lets
    /// [`LaraGraph::rebalance_leaf`] do the weighted layout and span
    /// publication.
    ///
    /// Only reachable for 4-byte neighbor-only rows (`E::BYTES == 4`), so
    /// records are reconstructed exactly with `E::with_neighbor`.
    ///
    /// `trees.remove(idx)` shifts later bucket indices; every vertex row
    /// with `tree_bucket > idx` is decremented (O(n), demotion is rare).
    ///
    /// The fresh slab row has `stored = degree <= T/2`, where T is the
    /// configured threshold (default T/2 when promotion is disabled).
    fn demote_tree_to_slab(&mut self, v: u32, row: V) {
        let t = row.tree_bucket() as usize;
        let leaf = v as u64 / self.segment_size as u64;

        // 1. Collect live records from the tree bucket (E::BYTES == 4).
        let mut live: Vec<u32> = Vec::with_capacity(self.trees[t].stored as usize);
        for slot in 0..self.trees[t].stored {
            live.push(self.tree_read(t, slot));
        }
        let live_count = live.len() as u32;

        // 2. Stage the records at the slab tail (fresh, no overlap with
        //    the leaf's other rows). rebalance_leaf re-reads them here.
        let staging = self.edge_capacity;
        self.ensure_slab(self.edge_capacity + live_count as u64)
            .expect("demote slab");
        for (k, &n) in live.iter().enumerate() {
            let e = E::with_neighbor(E::default(), n);
            self.write_edge(staging + k as u64, &e);
        }

        // 3. Convert the row to a slab row pointing at the staged records.
        let new_row = row
            .with_base_slot_start(staging)
            .with_degree(live_count)
            .with_log_head(-1)
            .with_tree_bucket(NO_BUCKET);
        self.set_row(v, new_row);

        // 4. Release tree blocks and remove the bucket; fix up indices.
        for &block_id in &self.trees[t].root {
            self.blocks.release(block_id).expect("demote release");
        }
        self.trees.remove(t);
        for u in 0..self.vertices.len() as u32 {
            let r = self.get_row(u);
            let b = r.tree_bucket();
            if b != NO_BUCKET && b as usize > t {
                self.set_row(u, r.with_tree_bucket(b - 1));
            }
        }

        // 5. Layout + span publication (recomputes segment_actual).
        self.rebalance_leaf(leaf);
        self.release_span(staging, u64::from(live_count));
    }

    /// Inserts one directed edge (no dedup). Sugar over
    /// [`LaraGraph::insert_edge_record`] with a default-payload record.
    pub fn insert_edge(&mut self, src: u32, dst: u32) {
        self.insert_edge_record(src, E::with_neighbor(E::default(), dst));
    }

    /// Inserts one typed edge record for `src`. Handles CSR boundary
    /// checks, overflow log, leaf rebalance, tree
    /// promotion, and PMA accounting. Record-preserving: any payload in
    /// `E` beyond the neighbor survives slab residency until rebalance
    /// (which also preserves records verbatim).
    ///
    /// Unordered deletion shortens a dense prefix. Insertion-ordered deletion
    /// leaves typed slab markers until repacking; new records never fill old
    /// holes or precede pending logs. Tiny/tree/log storage keeps only the
    /// neighbor, not arbitrary payload fields.
    ///
    /// In Insertion mode this panics before routing if `e`, or
    /// `E::with_neighbor(E::default(), e.neighbor())`, is recognized by the
    /// configured marker predicate. The second check prevents a live record
    /// from turning into a marker after a neighbor-only transition. This is
    /// not a comparison with a marker's neighbor or encoded wire word.
    ///
    /// With promotion enabled, 4-byte rows are limited to `B * R_MAX`
    /// live neighbors; a larger insertion panics before changing the row.
    pub fn insert_edge_record(&mut self, src: u32, e: E) {
        self.check_insert_record(src, &e);
        let row = self.get_row(src);
        if E::BYTES == 4 && self.promote_threshold.is_some() {
            assert_depth_one(u64::from(row.degree()) + 1);
        }
        if is_tiny_row(&row) {
            self.tiny_insert(src, row, e.neighbor());
            return;
        }
        self.insert_record_cold(src, e, true);
    }

    /// Appends to a tiny-active row, or spills it once full.
    ///
    /// Below [`TINY_K`] the neighbor lands in the row's `K × 4`-byte chunk
    /// (one write, no slab/log touch, no density accounting — the chunk
    /// is outside both). At capacity the chunk replays through the cold
    /// path with tiny routing disabled, so replayed edges land in the
    /// log/slab exactly as fresh inserts do today (no re-tiny loop).
    /// A later deletion may leave the row stateless and eligible for tiny
    /// storage again. There is no lifetime spill-once flag.
    fn tiny_insert(&mut self, src: u32, row: V, dst: u32) {
        let n = row.degree();
        let off = row.log_head() as u64;
        if n < TINY_K {
            self.logs_mem.write(off + n as u64 * 4, &dst.to_le_bytes());
            self.set_row(
                src,
                row.with_live_degree(n + 1)
                    .with_stored_degree(TINY_FLAG | (n + 1)),
            );
            self.num_edges += 1;
            return;
        }
        let chunk = self.read_tiny(off, n);
        self.tiny_free(off);
        // Reset to a stateless row so replay rebuilds counters exactly:
        // the K chunk edges are already counted in `num_edges`, and
        // replay re-adds each — compensate upfront so the net effect is
        // +1 (the newcomer). `segment_actual` needs no compensation:
        // tiny rows never contributed to it, and replayed log/slab
        // writes must join it like any fresh insert.
        self.set_row(src, row.with_degree(0).with_log_head(-1));
        self.num_edges -= n as u64;
        for &replayed in chunk.iter().take(n as usize) {
            self.insert_record_cold(src, E::with_neighbor(E::default(), replayed), false);
        }
        self.insert_record_cold(src, E::with_neighbor(E::default(), dst), false);
    }

    /// Cold insert path (slab / log / rebalance / promote).
    /// `tiny_ok` allows fresh-row tiny-inline conversion at the log-write
    /// point; spill replay passes `false` so replayed edges reproduce
    /// today's log/slab routing exactly.
    fn insert_record_cold(&mut self, src: u32, e: E, tiny_ok: bool) {
        assert!((src as usize) < self.vertices.len(), "src out of range");
        let dst = e.neighbor();
        let mut row = self.get_row(src);
        debug_assert!(!is_tiny_row(&row), "tiny rows enter via tiny_insert");
        let leaf = src as u64 / self.segment_size as u64;

        if row.tree_bucket() != NO_BUCKET {
            let t = row.tree_bucket() as usize;
            self.tree_insert(t, dst);
            row = row.with_live_degree(row.degree() + 1);
            self.set_row(src, row);
            self.num_edges += 1;
            return;
        }

        let boundary = self.slab_boundary(src, &row);
        let loc = row.base_slot_start() + stored_of(&row) as u64;
        if loc < boundary && (self.insertion.is_none() || row.log_head() == -1) {
            self.write_edge(loc, &e);
            row = row
                .with_live_degree(row.degree() + 1)
                .with_stored_degree(stored_of(&row) + 1);
            self.segment_actual[leaf as usize] += 1;
        } else if tiny_ok
            && E::BYTES == 4
            && row.degree() == 0
            && stored_of(&row) == 0
            && row.log_head() == -1
        {
            // Stateless small rows use pooled chunks independently of the
            // overflow-log cap. Rows with slab room stay on the slab path.
            let off = self.tiny_alloc();
            self.logs_mem.write(off, &dst.to_le_bytes());
            row = row
                .with_live_degree(1)
                .with_stored_degree(TINY_FLAG | 1)
                .with_log_head(off as i32);
        } else {
            // Overflow log (fold first when the leaf log is full).
            if self.log_idx[leaf as usize] as usize >= self.max_log_entries {
                self.rebalance_leaf(leaf);
                row = self.get_row(src);
            }
            let boundary = self.slab_boundary(src, &row);
            let loc = row.base_slot_start() + stored_of(&row) as u64;
            if loc < boundary && (self.insertion.is_none() || row.log_head() == -1) {
                self.write_edge(loc, &e);
                row = row
                    .with_live_degree(row.degree() + 1)
                    .with_stored_degree(stored_of(&row) + 1);
                self.segment_actual[leaf as usize] += 1;
            } else if (self.log_idx[leaf as usize] as usize) < self.max_log_entries {
                let idx = self.log_idx[leaf as usize];
                let entry = LogEntry {
                    src,
                    dst,
                    label: u16::MAX,
                    _pad: 0,
                    prev: row.log_head(),
                };
                self.write_log(leaf, idx, &entry);
                row = row
                    .with_live_degree(row.degree() + 1)
                    .with_log_head(idx as i32);
                self.log_idx[leaf as usize] = idx + 1;
                self.segment_actual[leaf as usize] += 1;
            } else {
                // No log room even after the fold above. Below the cap a
                // rebalance always frees slab room (`alloc = min(max(2x,1),T)
                // > stored` for `x < T`), so only the capped boundary
                // (`x >= T`, no slab room by design) reaches here — and
                // there the stored width already trips the promote
                // trigger. Promote and insert into the tree (no log write,
                // so cap=0 never touches the zero-length log region).
                // No-loop guarantee: promotion evacuates the slab, so the
                // tree path below always makes progress.
                let t = self.promote_threshold;
                assert!(
                    E::BYTES == 4
                        && t.is_some()
                        && row.tree_bucket() == NO_BUCKET
                        && stored_of(&row) >= t.unwrap(),
                    "log cap exhausted with no slab room and promotion ineligible (no-log mode needs promotion at the capped boundary)"
                );
                self.promote(src);
                let mut row2 = self.get_row(src);
                debug_assert!(row2.tree_bucket() != NO_BUCKET);
                let ti = row2.tree_bucket() as usize;
                self.tree_insert(ti, dst);
                row2 = row2.with_live_degree(row2.degree() + 1);
                self.set_row(src, row2);
                self.num_edges += 1;
                return;
            }
        }
        // Row reuse: `row` already holds the post-write state, so the
        // promote check below reuses it instead of re-reading the row
        // backend. Exception: a density rebalance re-lays the leaf and
        // rewrites this row — then the row must be re-read.
        let row_after = row;
        self.set_row(src, row);
        self.num_edges += 1;

        // Leaf-density check (leaf-local rebalance/relocation).
        // Integer form: UP_0 == 1.0 exactly, so `actual >= total` is the
        // same predicate as the old `actual/total >= UP_0` f64 division
        // with no float cost and no rounding question. Checked every
        // insert (a skip-counter would shift rebalance timing and change
        // observable rebalance counts; the compare itself is ~1ns, so
        // there is nothing worth thinning — see `density_thinning` doc).
        let actual = self.segment_actual[leaf as usize];
        let total = self.segment_total[leaf as usize];
        let rebalanced = if total > 0 && actual >= total {
            self.rebalance_leaf(leaf);
            true
        } else {
            false
        };

        // Tree-CSR promotion check. Tree mode is 4-byte neighbor-only
        // (unlabeled [`Edge`]); wide edge types (e.g. labeled records) are
        // never promoted — the tree I/O path assumes `E::BYTES == 4`.
        //
        // Keep the stored-slab-width trigger; live degree also includes
        // logs. Ordered markers remain in this width until stable repacking.
        let row = if rebalanced {
            self.get_row(src)
        } else {
            row_after
        };
        if let Some(t) = self.promote_threshold {
            if E::BYTES == 4
                && row.tree_bucket() == NO_BUCKET
                && !is_tiny_row(&row)
                && stored_of(&row) >= t
            {
                self.promote(src);
            }
        }
    }

    /// Inserts a same-source run in one row pass (ingest fast path).
    ///
    /// The historical method name does not require sorted destinations.
    /// In Insertion mode, `dsts` is applied in slice order and all records
    /// pass the marker/reconstruction check before writes. Unordered keeps
    /// multiplicity without an ordering guarantee. Already-tiny rows
    /// batch a run that fits in their owned chunk. Already-tree rows batch
    /// a run wholly inside one existing block, using at most 64 bytes per
    /// write. Both store raw LE neighbor IDs after applying the edge's
    /// default/neighbor builder, without allocating or updating slab density.
    /// Fresh tiny allocation, spill, missing tree blocks and block-crossing
    /// runs retain the entire per-edge path.
    ///
    /// The slab fast path checks room and density for the whole run, then
    /// updates the row/counts and checks promotion after writing the run.
    /// All must hold: slab row, whole run fits in the slab
    /// room (`loc + len <= boundary`), no pending ordered log is overtaken,
    /// and the leaf stays below the
    /// density trigger (`actual + len < total`, or `total == 0`). If these
    /// checks fail, uses per-edge [`LaraGraph::insert_edge_record`] for the
    /// entire run (log, rebalance, promotion may occur
    /// mid-run there).
    ///
    /// Eligible slab writes group positive record widths up to 64 bytes
    /// into bounded, whole-record chunks; each codec receives a zeroed
    /// record slice. Other widths retain per-record writes. Batched paths
    /// publish counts after the payload run. They are not transactional:
    /// a builder/codec/backend panic can leave partial payload with old
    /// counts, not necessarily the same prefix as per-edge insertion.
    /// A run exceeding the promotion-enabled depth-1 limit is rejected
    /// before any of its edges are inserted.
    pub fn insert_sorted_run(&mut self, src: u32, dsts: &[u32]) {
        if dsts.is_empty() {
            return;
        }
        assert!((src as usize) < self.vertices.len(), "src out of range");
        let len = dsts.len() as u64;
        let leaf = src as u64 / self.segment_size as u64;
        let row = self.get_row(src);
        if E::BYTES == 4 && (self.promote_threshold.is_some() || row.tree_bucket() != NO_BUCKET) {
            assert_depth_one(u64::from(row.degree()) + len);
        }
        if self.insertion.is_some() {
            for &dst in dsts {
                self.check_insert_record(src, &E::with_neighbor(E::default(), dst));
            }
        }
        if E::BYTES == 4 && is_tiny_row(&row) && u64::from(row.degree()) + len <= u64::from(TINY_K)
        {
            let mut buf = [0u8; TINY_K_MAX * 4];
            for (k, &dst) in dsts.iter().enumerate() {
                let target = E::with_neighbor(E::default(), dst).neighbor();
                buf[k * 4..(k + 1) * 4].copy_from_slice(&target.to_le_bytes());
            }
            let off = row.log_head() as u64 + u64::from(row.degree()) * 4;
            self.logs_mem.write(off, &buf[..dsts.len() * 4]);
            let n = row.degree() + dsts.len() as u32;
            self.set_row(
                src,
                row.with_live_degree(n).with_stored_degree(TINY_FLAG | n),
            );
            self.num_edges += len;
            return;
        }
        if E::BYTES == 4 && row.tree_bucket() != NO_BUCKET {
            let state = &mut self.trees[row.tree_bucket() as usize];
            let slot = state.stored as usize;
            let in_block = slot % B;
            // Use retained roots too, but never mint or cross a block here.
            if len <= (B - in_block) as u64 && slot / B < state.root.len() {
                assert_depth_one(u64::from(state.stored) + len);
                let block_id = state.root[slot / B];
                let slots_per_chunk = INLINE_EDGE_BYTES / 4;
                for (chunk_index, chunk) in dsts.chunks(slots_per_chunk).enumerate() {
                    let mut buf = [0u8; INLINE_EDGE_BYTES];
                    for (k, &dst) in chunk.iter().enumerate() {
                        let target = E::with_neighbor(E::default(), dst).neighbor();
                        buf[k * 4..(k + 1) * 4].copy_from_slice(&target.to_le_bytes());
                    }
                    self.blocks
                        .write_payload_partial(
                            block_id,
                            (in_block + chunk_index * slots_per_chunk) * 4,
                            &buf[..chunk.len() * 4],
                        )
                        .expect("tree run write");
                }
                state.stored += dsts.len() as u32;
                self.set_row(src, row.with_live_degree(row.degree() + dsts.len() as u32));
                self.num_edges += len;
                return;
            }
        }
        let fast = row.tree_bucket() == NO_BUCKET
            && !is_tiny_row(&row)
            && (self.insertion.is_none() || row.log_head() == -1)
            && {
                let boundary = self.slab_boundary(src, &row);
                let loc = row.base_slot_start() + stored_of(&row) as u64;
                loc + len <= boundary
            }
            && {
                let actual = self.segment_actual[leaf as usize];
                let total = self.segment_total[leaf as usize];
                total == 0 || actual + len < total
            };
        if !fast {
            for &dst in dsts {
                self.insert_edge_record(src, E::with_neighbor(E::default(), dst));
            }
            return;
        }
        let mut row = row;
        let base = row.base_slot_start() + stored_of(&row) as u64;
        if E::BYTES > 0 && E::BYTES <= INLINE_EDGE_BYTES {
            let slots_per_chunk = INLINE_EDGE_BYTES / E::BYTES;
            for (chunk_index, chunk) in dsts.chunks(slots_per_chunk).enumerate() {
                // Match write_edge's zeroed input for every record, even
                // when a codec leaves some padding bytes untouched.
                let mut buf = [0u8; INLINE_EDGE_BYTES];
                for (k, &dst) in chunk.iter().enumerate() {
                    E::with_neighbor(E::default(), dst)
                        .write_to(&mut buf[k * E::BYTES..(k + 1) * E::BYTES]);
                }
                let slot = base + (chunk_index * slots_per_chunk) as u64;
                self.edges_mem
                    .write(slot * E::BYTES as u64, &buf[..chunk.len() * E::BYTES]);
            }
        } else {
            for (k, &dst) in dsts.iter().enumerate() {
                self.write_edge(base + k as u64, &E::with_neighbor(E::default(), dst));
            }
        }
        row = row
            .with_live_degree(row.degree() + dsts.len() as u32)
            .with_stored_degree(stored_of(&row) + dsts.len() as u32);
        self.segment_actual[leaf as usize] += len;
        let row_after = row;
        self.set_row(src, row);
        self.num_edges += len;
        // Density re-check (cannot fire: guaranteed `actual + len < total`
        // above, but keep the predicate identical to the per-edge path).
        let actual = self.segment_actual[leaf as usize];
        let total = self.segment_total[leaf as usize];
        let rebalanced = if total > 0 && actual >= total {
            self.rebalance_leaf(leaf);
            true
        } else {
            false
        };
        let row = if rebalanced {
            self.get_row(src)
        } else {
            row_after
        };
        if let Some(t) = self.promote_threshold {
            if E::BYTES == 4 && row.tree_bucket() == NO_BUCKET && stored_of(&row) >= t {
                self.promote(src);
            }
        }
    }

    /// Whether `v`'s row has physical slab tail room beyond its stored extent.
    /// Ordered marker holes do not count as tail room. No payload search is
    /// needed. Pending ordered logs may prevent using available tail room;
    /// this is not a prediction of insertion's placement or maintenance.
    pub fn row_has_slab_room(&self, v: u32) -> bool {
        let row = self.get_row(v);
        // Tiny-active rows own no slab range; their appends go through
        // the tiny path (or spill), never the slab.
        if is_tiny_row(&row) {
            return false;
        }
        let boundary = self.slab_boundary(v, &row);
        (row.base_slot_start() + stored_of(&row) as u64) < boundary
    }

    /// Removes one directed edge; `false` when absent. Sugar over
    /// [`LaraGraph::remove_edge_matching`] matching on the neighbor.
    pub fn remove_edge(&mut self, src: u32, dst: u32) -> bool {
        self.remove_edge_matching(src, |e| e.neighbor() == dst)
    }

    /// Removes the first live record of `src` matching `pred` (neighbor +
    /// payload equality decided by the caller), in the selected traversal order.
    /// Unordered replaces a non-tail match with that prefix's last entry.
    /// Insertion uses a typed slab marker or shifts a tiny/tree suffix left.
    /// Logs unlink one entry (oldest matching entry in Insertion mode).
    /// Ordered tree deletion remains linear in the suffix length. Search,
    /// PMA accounting, demotion and low-density compaction are separate work.
    ///
    /// Tree and log entries carry only a neighbor, so `pred` sees a record
    /// reconstructed from that neighbor (default payload) in those paths.
    pub fn remove_edge_matching(&mut self, src: u32, pred: impl Fn(&E) -> bool) -> bool {
        let mut row = self.get_row(src);
        let leaf = src as u64 / self.segment_size as u64;

        // Tiny-active delete: shift or replace from the tail, retaining
        // the owned chunk even at zero live neighbors.
        if is_tiny_row(&row) {
            let n = row.degree();
            let off = row.log_head() as u64;
            let chunk = self.read_tiny(off, n);
            for k in 0..n as usize {
                if pred(&E::with_neighbor(E::default(), chunk[k])) {
                    if self.insertion.is_some() {
                        // At most K-1 raw neighbors; keep the owned chunk.
                        for j in k..n as usize - 1 {
                            self.logs_mem
                                .write(off + j as u64 * 4, &chunk[j + 1].to_le_bytes());
                        }
                    } else if k != n as usize - 1 {
                        self.logs_mem
                            .write(off + k as u64 * 4, &chunk[n as usize - 1].to_le_bytes());
                    }
                    self.set_row(
                        src,
                        row.with_live_degree(n - 1)
                            .with_stored_degree(TINY_FLAG | (n - 1)),
                    );
                    self.num_edges -= 1;
                    return true;
                }
            }
            return false;
        }

        if row.tree_bucket() != NO_BUCKET {
            let t = row.tree_bucket() as usize;
            // Linear value search; the selected policy controls suffix movement.
            for slot in 0..self.trees[t].stored {
                let n = self.tree_read(t, slot);
                let rec = E::with_neighbor(E::default(), n);
                if pred(&rec) {
                    if self.insertion.is_some() {
                        // ponytail: linear suffix movement preserves order without
                        // tree marker/bitmap state; redesign only if measured costs justify it.
                        self.tree_state_remove_shift(t, slot);
                    } else {
                        self.tree_remove_swap(t, slot);
                    }
                    row = row.with_live_degree(row.degree() - 1);
                    self.set_row(src, row);
                    self.num_edges -= 1;
                    // Configured hysteresis band T/2.
                    if row.degree() <= self.demote_threshold() {
                        self.demote_tree_to_slab(src, row);
                    }
                    return true;
                }
            }
            return false;
        }

        // Slab scan.
        for slot in 0..stored_of(&row) {
            let loc = row.base_slot_start() + slot as u64;
            let e = self.read_edge(loc);
            if self.is_live(&e) && pred(&e) {
                if let Some(ops) = &self.insertion {
                    self.write_edge(loc, &(ops.tombstone)());
                } else {
                    let last = stored_of(&row) - 1;
                    if slot != last {
                        let tail = self.read_edge(row.base_slot_start() + u64::from(last));
                        self.write_edge(loc, &tail);
                    }
                    row = row.with_stored_degree(last);
                }
                row = row.with_live_degree(row.degree() - 1);
                self.set_row(src, row);
                self.segment_actual[leaf as usize] -= 1;
                self.num_edges -= 1;
                self.maybe_compact_leaf(leaf);
                return true;
            }
        }

        // Find in the selected traversal order, then use the same unlink.
        if row.log_head() != -1 {
            let mut found = None;
            if self.insertion.is_some() {
                let mut chain = Vec::new();
                let mut cur = row.log_head();
                while cur != -1 {
                    let entry = self.read_log(leaf, cur as u32);
                    chain.push((cur, entry));
                    cur = entry.prev;
                }
                for k in (0..chain.len()).rev() {
                    let (idx, entry) = chain[k];
                    if pred(&E::with_neighbor(E::default(), entry.dst)) {
                        let prev_idx = if k == 0 { -1 } else { chain[k - 1].0 };
                        found = Some((prev_idx, idx, entry));
                        break;
                    }
                }
            } else {
                let mut prev_idx = -1;
                let mut cur = row.log_head();
                while cur != -1 {
                    let entry = self.read_log(leaf, cur as u32);
                    if pred(&E::with_neighbor(E::default(), entry.dst)) {
                        found = Some((prev_idx, cur, entry));
                        break;
                    }
                    prev_idx = cur;
                    cur = entry.prev;
                }
            }
            if let Some((prev_idx, cur, entry)) = found {
                if prev_idx == -1 {
                    row = row.with_log_head(entry.prev);
                } else {
                    let mut p = self.read_log(leaf, prev_idx as u32);
                    p.prev = entry.prev;
                    self.write_log(leaf, prev_idx as u32, &p);
                }
                self.write_log(leaf, cur as u32, &LogEntry::default());
                row = row.with_live_degree(row.degree() - 1);
                self.set_row(src, row);
                self.segment_actual[leaf as usize] -= 1;
                self.num_edges -= 1;
                return true;
            }
        }
        false
    }

    fn maybe_compact_leaf(&mut self, leaf: u64) {
        let actual = self.segment_actual[leaf as usize];
        let total = self.segment_total[leaf as usize];
        if total > 0 && (actual as f64 / total as f64) < LOW_0 {
            self.rebalance_leaf(leaf);
        }
    }

    // -- rebalance / relocation --------------------------------------------

    /// Leaf-local rebalance: fold logs and relocate the
    /// leaf span when its target size differs from the current span
    /// (grow on density, shrink after deletes). The old span is released
    /// to the free-span store; the new one is acquired best-fit.
    pub fn rebalance_leaf(&mut self, leaf: u64) {
        self.rebalance_count += 1;
        let leaf_us = leaf as usize;
        let v0 = (leaf * self.segment_size as u64) as usize;
        let v1 = ((v0 as u64 + self.segment_size as u64) as usize).min(self.vertices.len());

        // Collect live adjacency per row (slab ascending, then log oldest
        // to newest). Slab records are copied verbatim (label + neighbor)
        // so rebalance preserves wide edge types. Ordered slab markers
        // are skipped. Log entries carry only a neighbor, so they are
        // reconstructed from `dst` as before.
        let mut live: Vec<Vec<E>> = Vec::with_capacity(v1 - v0);
        let mut sum_live = 0u64;
        for v in v0..v1 {
            let row = self.get_row(v as u32);
            let mut edges = Vec::new();
            if row.tree_bucket() == NO_BUCKET {
                if is_tiny_row(&row) {
                    // Tiny rows ride the rebalance into the slab: chunk
                    // neighbors become ordinary live records (flag cleared
                    // by the pack below).
                    let chunk = self.read_tiny(row.log_head() as u64, row.degree());
                    edges.extend(
                        chunk[..row.degree() as usize]
                            .iter()
                            .map(|&dst| E::with_neighbor(E::default(), dst)),
                    );
                } else {
                    self.for_each_record_in_row(row.base_slot_start(), stored_of(&row), |e| {
                        if self.is_live(e) {
                            edges.push(*e);
                        }
                    });
                    // Log chain newest → oldest; reverse for insertion order.
                    // (Tiny rows take the branch above and never reach here,
                    // so `log_head` is a genuine chain head or -1.)
                    let mut chain = Vec::new();
                    let mut cur = row.log_head();
                    while cur != -1 {
                        let entry = self.read_log(leaf, cur as u32);
                        chain.push(entry.dst);
                        cur = entry.prev;
                    }
                    chain.reverse();
                    edges.extend(chain.iter().map(|&dst| E::with_neighbor(E::default(), dst)));
                }
            }
            sum_live += edges.len() as u64;
            live.push(edges);
        }

        // Fixed single formula: alloc(x) = min(max(2*x, 1), T); the leaf
        // span is the sum of per-row allocations.
        let old_start = self.span_meta.get(leaf).physical_start;
        let old_total = self.segment_total[leaf_us];

        // Always re-pack (no in-place fast path): a demoted tree row's
        // records may be staged outside the current span, so the leaf must
        // be re-laid out to bring every row's base inside the span.
        {
            // Per-row allocation: alloc(x) = min(max(2*x, 1), T).
            // Tree rows hold no slab data (alloc 0).
            let mut allocs: Vec<u64> = Vec::with_capacity(live.len());
            let mut total_alloc = 0u64;
            for (i, edges) in live.iter().enumerate() {
                let v = (v0 + i) as u32;
                let old_row = self.get_row(v);
                if old_row.tree_bucket() != NO_BUCKET {
                    allocs.push(0);
                    continue;
                }
                let x = edges.len() as u64;
                // The cap limits headroom, never the live data itself: when
                // `x` already exceeds the cap the row keeps exactly `x`
                // slots (zero headroom; further inserts spill to the log).
                // Bulk-built or just-folded rows may exceed T even with
                // promotion enabled; rebalance itself does not promote.
                let alloc = self.alloc_for(x);
                allocs.push(alloc);
                total_alloc += alloc;
            }
            // Acquire the new span: best-fit free-span reuse (ablation arm
            // `use_free_spans == false` bumps the slab tail only).
            let new_start = self.allocate_span(total_alloc);
            // Pack rows contiguously with their capped allocations.
            let mut base = new_start;
            for (i, edges) in live.iter().enumerate() {
                let v = (v0 + i) as u32;
                let old_row = self.get_row(v);
                let is_tree = old_row.tree_bucket() != NO_BUCKET;
                let new_row = if is_tree {
                    // Zero-width rows must preserve monotone bases so
                    // the preceding slab row still sees its boundary.
                    old_row.with_base_slot_start(base).with_stored_degree(0)
                } else {
                    old_row
                        .with_base_slot_start(base)
                        .with_stored_degree(edges.len() as u32)
                        .with_degree(edges.len() as u32)
                        .with_log_head(-1)
                };
                for (k, e) in edges.iter().enumerate() {
                    self.write_edge(base + k as u64, e);
                }
                base += allocs[i];
                self.set_row(v, new_row);
                if is_tiny_row(&old_row) {
                    self.tiny_free(old_row.log_head() as u64);
                }
            }

            // Release the old span; publish the new geometry. The
            // no-free-span ablation arm never books retired ranges (its
            // slab is bump-allocated, so release would charge the arm for
            // bookkeeping it deliberately opts out of).
            if self.use_free_spans && old_start != SPAN_UNASSIGNED && old_total > 0 {
                self.free_spans
                    .release(old_start, old_total)
                    .expect("release old span");
            }
            self.span_meta
                .set(
                    leaf,
                    SegmentSpanMeta {
                        physical_start: new_start,
                    },
                )
                .expect("span meta");
            self.segment_total[leaf_us] = total_alloc;
        }

        // Logs are folded; reset the leaf log and retire its chunk.
        self.log_idx[leaf_us] = 0;
        self.free_log_leaf(leaf);
        self.segment_actual[leaf_us] = sum_live;
    }

    // -- neighbor scan ------------------------------------------------------

    /// Visits live slab neighbors in ascending slot order, preserving
    /// multiplicity. The bound is the stored extent, including ordered markers,
    /// not live degree (which may include logs) or allocation.
    /// All widths decode through `E::read_from`; only explicitly configured
    /// markers are filtered. Four bytes do not imply the built-in encoding.
    ///
    /// Spans above [`SCAN_CHUNK_BYTES`] use whole-record chunks of at most
    /// [`SLAB_SCAN_BUF_BYTES`] with `read_mapped`, including the short tail.
    /// Smaller spans keep ordinary reads; records wider than the bulk
    /// buffer use per-record reads. Core and labeled neighbor scans share
    /// this reader; record collection and invariant checks are separate.
    ///
    /// The caller supplies a fully backed span. Graph mutation requires
    /// `&mut self`, and each read copies into a local buffer before any
    /// callback, so nested read-only scans do not retain a backend borrow.
    #[inline]
    pub fn for_each_in_slab<F: FnMut(u32)>(&self, base: u64, stored: u32, mut f: F) {
        self.for_each_slab_record(base, stored, |e| {
            if self.is_live(&e) {
                f(e.neighbor());
            }
        });
    }

    // Decode once; the caller owns the liveness policy (core or one label).
    #[inline]
    pub(crate) fn for_each_slab_record<F: FnMut(E)>(&self, base: u64, stored: u32, mut f: F) {
        if stored == 0 {
            return;
        }
        let bytes = stored as u64 * E::BYTES as u64;
        // Single-read fast path for fully determined spans at or below the
        // bulk bound (gleaph `read_slots_contiguous` equivalent). Larger
        // spans fall through to the existing chunked path.
        if bytes <= BULK_SCAN_BYTES as u64 && E::BYTES <= BULK_SCAN_BYTES {
            let mut buf = vec![0u8; bytes as usize];
            self.edges_mem.read(base * E::BYTES as u64, &mut buf);
            for k in 0..stored as usize {
                f(E::read_from(&buf[k * E::BYTES..(k + 1) * E::BYTES]));
            }
            return;
        }
        // Keep the small buffer for small spans; reuse the existing bulk
        // buffer across larger spans instead of reverting to tiny chunks.
        if bytes > SCAN_CHUNK_BYTES as u64 && E::BYTES <= SLAB_SCAN_BUF_BYTES {
            let slots_per_chunk = (SLAB_SCAN_BUF_BYTES / E::BYTES) as u32;
            let mut buf = [0u8; SLAB_SCAN_BUF_BYTES];
            let mut start = 0u32;
            while start < stored {
                let n = (stored - start).min(slots_per_chunk) as usize;
                let bytes = n * E::BYTES;
                self.edges_mem
                    .read_mapped((base + start as u64) * E::BYTES as u64, &mut buf[..bytes]);
                for k in 0..n {
                    f(E::read_from(&buf[k * E::BYTES..(k + 1) * E::BYTES]));
                }
                start += n as u32;
            }
            return;
        }
        if E::BYTES > SCAN_CHUNK_BYTES {
            // Degenerate row width: fall back to per-slot reads.
            for slot in 0..stored {
                f(self.read_edge(base + slot as u64));
            }
            return;
        }
        let slots_per_chunk = (SCAN_CHUNK_BYTES / E::BYTES) as u32;
        let mut buf = [0u8; SCAN_CHUNK_BYTES];
        let mut start = 0u32;
        while start < stored {
            let n = (stored - start).min(slots_per_chunk) as usize;
            let bytes = n * E::BYTES;
            self.edges_mem
                .read((base + start as u64) * E::BYTES as u64, &mut buf[..bytes]);
            for k in 0..n {
                f(E::read_from(&buf[k * E::BYTES..(k + 1) * E::BYTES]));
            }
            start += n as u32;
        }
    }

    /// Visits `v`'s live neighbors under the selected ordering contract.
    /// Insertion visits live slab slots then logs oldest-first; tiny/tree
    /// prefixes are ascending. This preserves successful application order
    /// within `v`, not ordinal stability or chronology across sources.
    /// Unordered visits logs newest-first and makes no survivor-order promise.
    pub fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, mut f: F) {
        let row = self.get_row(v);
        if row.tree_bucket() != NO_BUCKET {
            let t = row.tree_bucket() as usize;
            self.for_each_in_tree(t, f);
            return;
        }
        // Tiny-active scan: one `degree × 4`-byte read, none when empty.
        if is_tiny_row(&row) {
            let chunk = self.read_tiny(row.log_head() as u64, row.degree());
            for &dst in &chunk[..row.degree() as usize] {
                f(dst);
            }
            return;
        }
        self.for_each_in_slab(row.base_slot_start(), stored_of(&row), &mut f);
        let leaf = v as u64 / self.segment_size as u64;
        let mut cur = row.log_head();
        if self.insertion.is_some() && cur != -1 {
            let mut chain = Vec::new();
            while cur != -1 {
                let (dst, prev) = self.read_log_scan(leaf, cur as u32);
                chain.push(dst);
                cur = prev;
            }
            for dst in chain.into_iter().rev() {
                f(dst);
            }
        } else {
            while cur != -1 {
                let (dst, prev) = self.read_log_scan(leaf, cur as u32);
                f(dst);
                cur = prev;
            }
        }
    }

    /// Collects neighbors (tests/benches).
    pub fn out_neighbors_vec(&self, v: u32) -> Vec<u32> {
        let mut out = Vec::new();
        self.for_each_neighbor(v, |n| out.push(n));
        out
    }

    /// Full-graph scan: sources ascending, each using its selected local order.
    /// Visits every live edge once, not in global insertion chronology.
    pub fn scan_all<F: FnMut(u32)>(&self, mut f: F) {
        for v in 0..self.vertices.len() as u32 {
            self.for_each_neighbor(v, &mut f);
        }
    }

    // -- invariant checks --------------------------------------------------

    /// Checks row counts, geometry, chunk ownership and log chains,
    /// including leaves whose slab span has not yet been placed.
    pub fn verify_invariants(&self) {
        let _ = (&self.scratch, &self.vertices_mem, &self.counts_mem);
        self.verify_storage_invariants();
        let mut total_live = 0u64;
        let mut tree_owners = vec![false; self.trees.len()];
        for leaf in 0..self.segment_count {
            let start = self.span_meta.get(leaf).physical_start;
            let end = if start == SPAN_UNASSIGNED {
                0
            } else {
                start + self.segment_total[leaf as usize]
            };
            let v0 = (leaf * self.segment_size as u64) as usize;
            let v1 = (v0 + self.segment_size as usize).min(self.vertices.len());
            let mut occupied = 0u64;
            let mut seen = vec![false; self.log_idx[leaf as usize] as usize];
            for v in v0..v1 {
                let row = self.get_row(v as u32);
                total_live += u64::from(row.degree());
                if row.tree_bucket() != NO_BUCKET {
                    let t = row.tree_bucket() as usize;
                    assert!(!tree_owners[t], "shared tree bucket");
                    tree_owners[t] = true;
                    assert_eq!(row.degree(), self.trees[t].stored);
                    assert_eq!(row.stored_degree(), 0, "tree row keeps no slab cells");
                    assert_eq!(row.log_head(), -1);
                    continue;
                }
                if is_tiny_row(&row) {
                    assert_eq!(stored_of(&row), row.degree(), "tiny count mismatch");
                    assert!(row.degree() <= TINY_K, "tiny count overflow");
                    continue;
                }
                if start == SPAN_UNASSIGNED {
                    assert_eq!(stored_of(&row), 0, "slab cells in unplaced leaf");
                } else {
                    let boundary = if v + 1 < v1 {
                        self.vertices[v + 1].base_slot_start()
                    } else {
                        end
                    };
                    assert!(
                        row.base_slot_start() >= start
                            && boundary <= end
                            && row.base_slot_start() + u64::from(stored_of(&row)) <= boundary,
                        "row crosses its slab boundary"
                    );
                }
                let mut live = u64::from(stored_of(&row));
                if self.insertion.is_some() {
                    live = 0;
                    self.for_each_record_in_row(row.base_slot_start(), stored_of(&row), |e| {
                        if self.is_live(e) {
                            live += 1;
                        }
                    });
                }
                let mut cur = row.log_head();
                while cur != -1 {
                    assert!(
                        cur >= 0 && (cur as usize) < seen.len(),
                        "log index outside allocated prefix"
                    );
                    assert!(!seen[cur as usize], "shared or cyclic log chain");
                    seen[cur as usize] = true;
                    let entry = self.read_log(leaf, cur as u32);
                    assert_eq!(entry.src, v as u32, "log entry belongs to another row");
                    assert_eq!(entry.label, u16::MAX);
                    live += 1;
                    cur = entry.prev;
                }
                assert_eq!(live, u64::from(row.degree()), "row {v} live count mismatch");
                occupied += live;
            }
            assert_eq!(
                occupied, self.segment_actual[leaf as usize],
                "leaf {leaf} actual mismatch"
            );
        }
        assert!(
            tree_owners.iter().all(|&owned| owned),
            "unowned tree bucket"
        );
        assert_eq!(self.num_edges, total_live);
    }
}

// =========================================================================
// Both graph interfaces work with CsrEdge and honor the configured policy.
// =========================================================================

impl<V: CsrVertex, E: CsrEdge, M: Memory> DynamicCsr for LaraGraph<V, E, M> {
    fn num_nodes(&self) -> u64 {
        LaraGraph::num_nodes(self)
    }
    fn num_edges(&self) -> u64 {
        LaraGraph::num_edges(self)
    }
    fn out_degree(&self, v: u32) -> u64 {
        LaraGraph::out_degree(self, v)
    }
    fn insert_edge(&mut self, src: u32, dst: u32) {
        LaraGraph::insert_edge(self, src, dst)
    }
    fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, f: F) {
        LaraGraph::for_each_neighbor(self, v, f)
    }
    fn scan_all<F: FnMut(u32)>(&self, f: F) {
        LaraGraph::scan_all(self, f)
    }
}

impl<V: CsrVertex, E: CsrEdge, M: Memory> RemovableCsr for LaraGraph<V, E, M> {
    fn remove_edge(&mut self, src: u32, dst: u32) -> bool {
        LaraGraph::remove_edge(self, src, dst)
    }
}

// Keep derive_depth referenced for the depth-boundary contract (the graph's
// tree buckets use root_len/derive_depth via tree_csr helpers).
#[allow(dead_code)]
fn _depth_contract(stored: u32) -> u32 {
    derive_depth(stored)
}

#[cfg(all(test, feature = "std"))]
mod audit_tests {
    use super::*;
    use crate::memory::VecMemory;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn graph() -> LaraGraph<Vertex, Edge, VecMemory> {
        LaraGraph::new(
            LaraMemories {
                vertices: VecMemory::new(),
                edges: VecMemory::new(),
                logs: VecMemory::new(),
                counts: VecMemory::new(),
                spans: VecMemory::new(),
                free_spans: VecMemory::new(),
                blocks: VecMemory::new(),
                buckets: VecMemory::new(),
            },
            64,
        )
        .unwrap()
    }

    #[test]
    fn empty_and_singleton_segment_geometry() {
        type Graph = LaraGraph<Vertex, Edge, VecMemory>;
        assert_eq!(Graph::compute_segments(0), (1, 1));
        assert_eq!(Graph::compute_segments(1), (1, 1));
    }

    #[test]
    fn tiny_address_limit_fails_before_allocation() {
        let mut g = graph();
        // Synthetic cursor: exercise narrowing without reserving 2 GiB.
        let limit = i32::MAX as u64 + 1;
        g.log_next = limit;
        let pages = g.logs_mem.size();
        assert!(catch_unwind(AssertUnwindSafe(|| g.tiny_alloc())).is_err());
        assert_eq!(g.log_next, limit);
        assert_eq!(g.tiny_bumped, 0);
        assert_eq!(g.logs_mem.size(), pages);
    }

    #[test]
    fn unplaced_tiny_count_is_checked() {
        let mut g = graph();
        for dst in 0..4 {
            g.insert_edge(0, dst);
        }
        g.verify_invariants();
        let row = g.get_row(0).with_stored_degree(TINY_FLAG | 1);
        g.set_row(0, row);
        assert!(catch_unwind(AssertUnwindSafe(|| g.verify_invariants())).is_err());
    }

    #[test]
    fn cyclic_log_is_checked_before_following_it() {
        let mut g = graph();
        for dst in 0..9 {
            g.insert_edge(0, dst);
        }
        g.verify_invariants();
        let head = g.get_row(0).log_head();
        assert!(head >= 0);
        let mut entry = g.read_log(0, head as u32);
        entry.prev = head;
        g.write_log(0, head as u32, &entry);
        assert!(catch_unwind(AssertUnwindSafe(|| g.verify_invariants())).is_err());
    }
}
