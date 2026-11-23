//! [`LabeledLaraGraph`]: multi-label adjacency extension of LARA.
//!
//! Slab records are plain 4-byte neighbors (the shared [`crate::Edge`]
//! wire); the label → sub-span mapping lives **out of band** in a per-vertex
//! **bucket descriptor run** ([`BucketDescriptor`] / [`BucketStore`], ADR
//! 0001 parity). Keeping the label out of the record lets a hot bucket
//! promote to per-bucket Tree-CSR mode naturally (`E::BYTES == 4`).
//!
//! Each vertex owns a run of descriptors sorted ascending by dense
//! [`LabelId`]. A bucket describes its sub-span in the shared slab
//! (`edge_start`, absolute slot), liveness (`degree`/`stored`), a
//! leaf-log head, and an optional Tree-CSR state id. The bulk build and
//! the labeled rebalance pack buckets of a leaf end-to-end with slack
//! proportional to each bucket's live degree.
//!
//! Labels are abstracted behind the [`Label`] trait (default [`LabelId`]);
//! descriptors store the dense `u16` wire id and the conversion happens at
//! the API boundary.
//!
//! Rebalance preserves descriptor run positions and writes only its leaf.
//! Creating a bucket shifts or relocates only its vertex's run. Runs reserve
//! power-of-two capacity and reuse retired runs of the same size; their
//! physical order need not match vertex order.
//! The leaf overflow log is 16 bytes/entry, so the unlabeled core's log
//! entries carry two unused padding bytes.

use crate::block_store::BLOCK_PAYLOAD_BYTES;
use crate::bucket::{BucketDescriptor, BucketStore};
use crate::graph::{AdjacencyOrder, Edge, LaraGraph, LogEntry, OrderChangeError};
use crate::memory::{GrowFailed, Memory};
use crate::traits::{CsrEdge, CsrEdgeTombstone, CsrVertex, DynamicCsr, FixedWidth, RemovableCsr};
use crate::tree_csr::{B, assert_depth_one, root_len};
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

#[cfg(all(test, feature = "std"))]
mod order_tests;

// Re-export the derive macro alongside the trait so that `lara_graph::labeled::Label`
// (and, through it, `lara_graph::Label`) brings in both the trait and the macro.
#[cfg(feature = "derive")]
pub use lara_graph_macros::Label;

/// Dense label identifier stored in the slab (wire format: little-endian
/// `u16`). Max 65,536 distinct labels per graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabelId(u16);

impl LabelId {
    /// Maximum number of distinct labels per graph (`2^16`).
    pub const MAX_COUNT: usize = 65_536;

    /// Wraps a raw `u16` id.
    pub fn new(id: u16) -> Self {
        Self(id)
    }

    /// Returns the raw `u16` id.
    pub fn as_u16(self) -> u16 {
        self.0
    }
}

/// User-facing label type: converts between a domain type and the dense
/// wire id. Implement for `Copy` enums or newtypes; the graph never sees
/// anything but the conversion.
///
/// With the `derive` feature, fieldless enums can derive this trait
/// (declaration-order numbering). The derive macro shares the name `Label`
/// and is re-exported alongside the trait, so a single import brings in
/// both:
///
/// ```
/// use lara_graph::labeled::Label; // trait + derive macro
///
/// #[derive(Copy, Clone, Debug, PartialEq, Eq, Label)]
/// enum EdgeKind { Follows, Mentions, Blocks }
/// ```
///
/// The same works from the crate root (`use lara_graph::Label;`).
pub trait Label: Copy {
    /// Converts this label to its dense wire id.
    fn into_id(self) -> LabelId;
    /// Reconstructs a label from a dense wire id.
    fn from_id(id: LabelId) -> Self;
}

impl Label for LabelId {
    fn into_id(self) -> LabelId {
        self
    }
    fn from_id(id: LabelId) -> LabelId {
        id
    }
}

/// Labeled vertex row: the index and length of this vertex's bucket
/// descriptor run in the [`BucketStore`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct LabeledVertex {
    /// Global index of the first descriptor of this vertex's run.
    pub table_start: u32,
    /// Number of buckets (descriptors) in the run.
    pub bucket_count: u32,
}

impl FixedWidth for LabeledVertex {
    const BYTES: usize = 8;

    fn read_from(bytes: &[u8]) -> Self {
        Self {
            table_start: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            bucket_count: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        }
    }

    fn write_to(&self, bytes: &mut [u8]) {
        bytes[0..4].copy_from_slice(&self.table_start.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.bucket_count.to_le_bytes());
    }
}

impl CsrVertex for LabeledVertex {
    // The labeled layer manages its own bucket-aware slab layout; these
    // row primitives are inert (the unlabeled core's per-row operations are
    // never invoked on a labeled graph).
    fn base_slot_start(&self) -> u64 {
        0
    }
    fn degree(&self) -> u32 {
        0
    }
    fn stored_degree(&self) -> u32 {
        0
    }
    fn log_head(&self) -> i32 {
        -1
    }
    fn tree_bucket(&self) -> u32 {
        u32::MAX
    }
    fn with_base_slot_start(self, _s: u64) -> Self {
        self
    }
    fn with_degree(self, _n: u32) -> Self {
        self
    }
    fn with_live_degree(self, _n: u32) -> Self {
        self
    }
    fn with_stored_degree(self, _n: u32) -> Self {
        self
    }
    fn with_log_head(self, _h: i32) -> Self {
        self
    }
    fn with_tree_bucket(self, _b: u32) -> Self {
        self
    }
}

/// One label bucket (per-vertex, per-label view) — a label id and its
/// live count for one vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bucket {
    /// Label id.
    pub label: u16,
    /// Live rows in this bucket for one vertex.
    pub count: u32,
}

/// Multi-label LARA graph: `insert_edge(src, dst, label)` with per-label
/// scans. Slab records are plain 4-byte neighbors; each vertex's label
/// buckets live as a descriptor run in a separate [`BucketStore`].
///
/// `L` is the user-facing label type (default [`LabelId`]); the graph
/// converts it to a dense wire id at the API boundary and never stores the
/// domain type.
pub struct LabeledLaraGraph<M: Memory, L: Label = LabelId> {
    inner: LaraGraph<LabeledVertex, Edge, M>,
    buckets: BucketStore,
    // The sole policy owner: absent labels are Unordered, including future buckets.
    insertion_labels: Vec<LabelId>,
    _label: PhantomData<L>,
}

// -- descriptor / row plumbing -------------------------------------------------

impl<M: Memory, L: Label> LabeledLaraGraph<M, L> {
    /// Ordering within each `(source, label)` bucket; unset labels are Unordered.
    pub fn label_order(&self, label: L) -> AdjacencyOrder {
        if self.is_ordered(label.into_id().as_u16()) {
            AdjacencyOrder::Insertion
        } else {
            AdjacencyOrder::Unordered
        }
    }

    fn is_ordered(&self, label: u16) -> bool {
        self.insertion_labels
            .binary_search(&LabelId::new(label))
            .is_ok()
    }

    /// Changes one label's policy across all vertices, including future buckets.
    /// An actual change requires every bucket of this label to have degree zero;
    /// other labels may be populated. Same-mode calls are no-ops even when live.
    /// Empty extents are retired before changing interpretation; unused backing
    /// and shared log allocations need not be erased or returned to their pools.
    /// This does not recover old insertion history or provide failure rollback.
    pub fn set_label_order(
        &mut self,
        label: L,
        order: AdjacencyOrder,
    ) -> Result<(), OrderChangeError> {
        let id = label.into_id();
        let position = self.insertion_labels.binary_search(&id);
        if position.is_ok() == (order == AdjacencyOrder::Insertion) {
            return Ok(());
        }
        // LabeledVertex's core degree is inert: only real bucket degrees count.
        for v in 0..self.num_nodes() as u32 {
            if self
                .find_bucket(v, id.as_u16())
                .is_some_and(|(_, b)| b.degree != 0)
            {
                return Err(OrderChangeError);
            }
        }
        for v in 0..self.num_nodes() as u32 {
            if let Some((pos, mut b)) = self.find_bucket(v, id.as_u16()) {
                if b.stored != 0 || b.log_head != -1 {
                    b.stored = 0;
                    b.log_head = -1;
                    self.set_bucket(v, pos, &b);
                }
            }
        }
        // Do not reset a leaf cursor/free its shared chunk for one empty label.
        match position {
            Ok(i) => {
                self.insertion_labels.remove(i);
            }
            Err(i) => self.insertion_labels.insert(i, id),
        }
        Ok(())
    }

    #[inline]
    fn row(&self, v: u32) -> LabeledVertex {
        self.inner.get_row_pub(v)
    }

    #[inline]
    fn set_row(&mut self, v: u32, r: LabeledVertex) {
        self.inner.set_row_pub(v, r);
    }

    #[inline]
    fn bucket_of(&self, v: u32, i: u32) -> BucketDescriptor {
        let row = self.row(v);
        self.buckets
            .read(self.inner.buckets_mem(), row.table_start as u64 + i as u64)
    }

    fn set_bucket(&mut self, v: u32, i: u32, d: &BucketDescriptor) {
        let row = self.row(v);
        let mem = self.inner.buckets_mem_mut();
        self.buckets
            .write(mem, row.table_start as u64 + i as u64, d);
    }

    /// Binary-searches `v`'s run, returning its relative position and descriptor.
    fn find_bucket(&self, v: u32, label: u16) -> Option<(u32, BucketDescriptor)> {
        let row = self.row(v);
        let mut lo = 0u32;
        let mut hi = row.bucket_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let b = self.bucket_of(v, mid);
            match b.label.cmp(&label) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => return Some((mid, b)),
            }
        }
        None
    }

    /// Inserts a brand-new (empty) bucket into this vertex's sorted run.
    /// Other vertices keep their run locations, including empty vertices.
    fn ensure_bucket(&mut self, v: u32, label: u16) -> (u32, BucketDescriptor) {
        let row = self.row(v);
        let mut ins = 0u32;
        while ins < row.bucket_count && self.bucket_of(v, ins).label < label {
            ins += 1;
        }
        let d = BucketDescriptor {
            label,
            degree: 0,
            stored: 0,
            log_head: -1,
            edge_start: u32::MAX, // unpositioned: rebalance before first insert
            tree_bucket: u32::MAX,
        };
        let start = self
            .buckets
            .insert_run(
                self.inner.buckets_mem_mut(),
                row.table_start,
                row.bucket_count,
                ins,
                &d,
            )
            .expect("bucket grow");
        self.set_row(
            v,
            LabeledVertex {
                table_start: start,
                bucket_count: row.bucket_count + 1,
            },
        );
        if start != row.table_start {
            self.buckets.release_run(
                self.inner.buckets_mem_mut(),
                row.table_start,
                row.bucket_count,
            );
        }
        (ins, d)
    }

    /// The exclusive upper slot bound of bucket `(v, pos)`: the next
    /// **slab-mode** bucket's `edge_start` (skipping Tree-CSR buckets, which
    /// occupy no slab space), else the next vertex-with-buckets in the
    /// leaf, else the leaf span end.
    fn bucket_boundary(&self, v: u32, pos: u32, b: &BucketDescriptor) -> u64 {
        let row = self.row(v);
        debug_assert!(
            b.edge_start != u32::MAX,
            "unpositioned bucket reached boundary path"
        );
        // Skip tree-mode buckets: they hold their records in the tree, so
        // the previous slab bucket may extend up to the next slab bucket.
        let mut j = pos + 1;
        while j < row.bucket_count {
            let d = self.bucket_of(v, j);
            if d.tree_bucket == u32::MAX {
                return d.edge_start as u64;
            }
            j += 1;
        }
        let leaf = v as u64 / self.inner.segment_size() as u64;
        let (start, total, _) = self.inner.leaf_geometry(leaf);
        let leaf_end = if start == u64::MAX { 0 } else { start + total };
        let v0 = (leaf * self.inner.segment_size() as u64) as u32;
        let v1 = (v0 + self.inner.segment_size()).min(self.inner.num_nodes() as u32);
        for w in (v + 1)..v1 {
            let wr = self.row(w);
            for k in 0..wr.bucket_count {
                let d = self.bucket_of(w, k);
                if d.tree_bucket == u32::MAX {
                    return d.edge_start as u64;
                }
            }
        }
        leaf_end
    }

    /// Collects the live neighbors of one bucket (tree, then slab, then
    /// log chain).
    fn bucket_live(&self, v: u32, d: &BucketDescriptor) -> Vec<u32> {
        let mut out = Vec::new();
        if d.tree_bucket != u32::MAX {
            let t = d.tree_bucket as usize;
            let stored = self.inner.tree_state_stored(t);
            for slot in 0..stored {
                out.push(self.inner.tree_state_read(t, slot));
            }
            return out;
        }
        // A freshly ensured (unpositioned) bucket surfaces here during the
        // leaf rebalance that will position it; it holds no live edges yet.
        if d.edge_start == u32::MAX {
            return out;
        }
        let base = d.edge_start as u64;
        let ordered = self.is_ordered(d.label);
        for k in 0..d.stored {
            let e = self.inner.read_edge_pub(base + k as u64);
            if !ordered || !e.is_tombstone() {
                out.push(e.neighbor());
            }
        }
        let leaf = v as u64 / self.inner.segment_size() as u64;
        let log_start = out.len();
        let mut cur = d.log_head;
        while cur != -1 {
            let entry = self.inner.read_log_pub(leaf, cur as u32);
            if entry.label == d.label && entry.src == v {
                out.push(entry.dst);
            }
            cur = entry.prev;
        }
        if ordered {
            out[log_start..].reverse();
        }
        out
    }

    /// Lays out one leaf's buckets (given per-vertex `(label, neighbors)`
    /// runs) into a freshly allocated span, writing edge records. Returns
    /// `(start, span, per-vertex descriptors with edge_starts)`.
    ///
    /// Each bucket's allocation is `min(max(2*x, 1), T)` (see
    /// [`crate::graph::T_PROMOTE`]). Sweep result (paper §5): at equal
    /// totals the gap formula is irrelevant, so the single formula keeps
    /// `2x` headroom capped at `T`. Leaves with any bucket keep the 1024-slot
    /// floor, including retained empty buckets and tree-only leaves. A leaf
    /// with neither slab nor tree buckets reserves no slots.
    fn layout_leaf(
        &mut self,
        leaf: u64,
        grouped_local: &[Vec<(u16, Vec<u32>)>],
        has_tree_buckets: bool,
    ) -> Result<(u64, u64, Vec<Vec<BucketDescriptor>>), GrowFailed> {
        let ss = self.inner.segment_size() as u64;
        let v0 = leaf * ss;
        let nverts = grouped_local.len() as u64;
        let mut leaf_total = 0u64;
        let mut total_buckets = 0u64;
        for run in grouped_local {
            total_buckets += run.len() as u64;
            for (_, n) in run {
                leaf_total += n.len() as u64;
            }
        }
        let _ = (ss, nverts, leaf_total, total_buckets);
        // First pass: per-bucket allocations under the single formula
        // `bslots = min(max(2*n, 1), T)`. The `1` keeps one headroom slot
        // for empty buckets (for `n >= 1`, `2n >= n+1` already guarantees
        // a free slot). The cap limits headroom, never live data: when `n`
        // already exceeds the cap the bucket keeps exactly `n` slots
        // (reachable only with promotion disabled).
        let cap = self.inner.alloc_cap();
        let mut total_alloc = 0u64;
        for run in grouped_local {
            let mut vsum = 0u64;
            for (_, neigh) in run {
                let n = neigh.len() as u64;
                let bslots = (2u64.saturating_mul(n).max(1)).min(cap).max(n);
                vsum += bslots;
            }
            total_alloc += vsum;
        }
        let span = if total_buckets == 0 && !has_tree_buckets {
            0
        } else {
            total_alloc.max(1024)
        };
        let new_start = self.inner.allocate_span(span);
        // Second pass: pack buckets contiguously with their capped slots.
        let mut base = new_start;
        let mut out: Vec<Vec<BucketDescriptor>> = Vec::with_capacity(nverts as usize);
        for run in grouped_local {
            let mut cbase = base;
            let mut vdescs = Vec::with_capacity(run.len());
            for (label, neigh) in run {
                let n = neigh.len() as u64;
                let bslots = (2u64.saturating_mul(n).max(1)).min(cap).max(n);
                let start = cbase;
                for (k, &n) in neigh.iter().enumerate() {
                    self.inner
                        .write_edge_pub(start + k as u64, &Edge::with_neighbor(Edge::default(), n));
                }
                vdescs.push(BucketDescriptor {
                    label: *label,
                    degree: neigh.len() as u32,
                    stored: neigh.len() as u32,
                    log_head: -1,
                    edge_start: start as u32,
                    tree_bucket: u32::MAX,
                });
                cbase += bslots;
            }
            out.push(vdescs);
            // Advance by the vertex's capped allocation (never spills into
            // the next vertex).
            base = cbase;
            let _ = v0;
        }
        Ok((new_start, span, out))
    }

    /// Allocates and writes one leaf's independent descriptor runs during
    /// bulk replacement. The arena starts empty; backing may be retained.
    fn commit_append_leaf(
        &mut self,
        leaf: u64,
        descs: &[Vec<BucketDescriptor>],
    ) -> Result<(), GrowFailed> {
        let ss = self.inner.segment_size() as u64;
        let v0 = leaf * ss;
        for (lv, vdescs) in descs.iter().enumerate() {
            let v = (v0 as usize + lv) as u32;
            let count = vdescs.len() as u32;
            let start = if count == 0 {
                0
            } else {
                self.buckets
                    .alloc_run(self.inner.buckets_mem_mut(), count)?
            };
            for (pos, d) in vdescs.iter().enumerate() {
                self.buckets.write(
                    self.inner.buckets_mem_mut(),
                    u64::from(start) + pos as u64,
                    d,
                );
            }
            self.set_row(
                v,
                LabeledVertex {
                    table_start: start,
                    bucket_count: count,
                },
            );
        }
        self.inner.set_log_idx_pub(leaf, 0);
        Ok(())
    }

    /// Commits a rebalance without relocating descriptor runs. Slab/tree
    /// partitioning and merging preserve every label, including empty buckets;
    /// creation has already updated the canonical run positions and counts.
    fn commit_leaf_descriptors(&mut self, leaf: u64, descs: &[Vec<BucketDescriptor>]) {
        let ss = self.inner.segment_size() as u64;
        let v0 = leaf * ss;
        let v1 = (v0 + ss).min(self.inner.num_nodes());
        assert_eq!(
            descs.len() as u64,
            v1 - v0,
            "rebalance must preserve vertex runs"
        );
        // Check all run lengths before writing any descriptor. This is not
        // rollback of the preceding slab relocation or arbitrary Memory panics.
        for (lv, run) in descs.iter().enumerate() {
            assert_eq!(
                self.row((v0 + lv as u64) as u32).bucket_count as usize,
                run.len(),
                "rebalance must preserve bucket counts"
            );
        }
        for (lv, run) in descs.iter().enumerate() {
            for (pos, d) in run.iter().enumerate() {
                self.set_bucket((v0 + lv as u64) as u32, pos as u32, d);
            }
        }
    }
}

// -- construction & mutation --------------------------------------------------

impl<M: Memory, L: Label> LabeledLaraGraph<M, L> {
    /// Creates an empty labeled graph with `n` vertices.
    pub fn new(mem: crate::graph::LaraMemories<M>, n: u32) -> Result<Self, GrowFailed> {
        let inner = LaraGraph::new(mem, n)?;
        // The descriptor region is still empty; grow on first insertion.
        Ok(Self {
            inner,
            buckets: BucketStore::new(0),
            insertion_labels: Vec::new(),
            _label: PhantomData,
        })
    }

    /// Creates an empty labeled graph with an explicit per-leaf log cap.
    /// Like the core constructor, the log region is sized here — growing
    /// later panics (see [`LaraGraph::set_max_log_entries`]).
    pub fn new_with_max_log_entries(
        mem: crate::graph::LaraMemories<M>,
        n: u32,
        max_log_entries: usize,
    ) -> Result<Self, GrowFailed> {
        let inner = LaraGraph::new_with_max_log_entries(mem, n, max_log_entries)?;
        Ok(Self {
            inner,
            buckets: BucketStore::new(0),
            insertion_labels: Vec::new(),
            _label: PhantomData,
        })
    }

    /// Effective per-leaf overflow-log cap (delegates to the core).
    pub fn max_log_entries(&self) -> usize {
        self.inner.max_log_entries()
    }

    /// Number of leaf segments.
    pub fn segment_count(&self) -> u64 {
        self.inner.segment_count()
    }

    /// Legacy partial storage estimate delegated to the core. Excludes
    /// bucket descriptors and other allocations; NOT total heap or RSS.
    pub fn memory_bytes_total(&self) -> u64 {
        self.inner.memory_bytes_total()
    }

    /// Completed core `rebalance_leaf` calls (perf instrument passthrough).
    pub fn rebalance_count(&self) -> u64 {
        self.inner.rebalance_count()
    }

    /// Completed core `promote` calls (perf instrument passthrough).
    pub fn promote_count(&self) -> u64 {
        self.inner.promote_count()
    }

    /// Zeroes the core perf counters.
    pub fn reset_perf_counters(&mut self) {
        self.inner.reset_perf_counters()
    }

    /// Folds every leaf's residual overflow log into the slab (steady-state
    /// scan precondition: measures post-fold scan, not mid-log scan).
    pub fn rebalance_all(&mut self) {
        for leaf in 0..self.inner.segment_count() {
            self.rebalance_leaf(leaf);
        }
    }

    /// Sets the promote threshold (which doubles as the slab
    /// allocated-space cap), delegating to the inner core graph (builder).
    /// Defaults to `Some`([`crate::graph::T_PROMOTE`]); `None` disables both
    /// promotion and the cap.
    pub fn with_promote_threshold(mut self, threshold: Option<u32>) -> Self {
        self.inner.set_promote_threshold(threshold);
        self
    }

    /// Bulk build from labeled COO (grouped by ascending source).
    pub fn with_edge_list(
        mem: crate::graph::LaraMemories<M>,
        edge_list: &[(u32, u32, L)],
        n: u32,
    ) -> Result<Self, GrowFailed> {
        let mut g = Self::new(mem, n)?;
        g.build_edges(edge_list)?;
        Ok(g)
    }

    /// Replaces adjacency from labeled triples, retaining label policies and
    /// graph configuration. Each `(source, label)` keeps input-relative order;
    /// replacement starts new sequences, not recovered historical chronology.
    /// Source IDs and ordered markers are rejected before destructive writes.
    /// Later allocation/backend failures are not transactional.
    pub fn build_edges(&mut self, edge_list: &[(u32, u32, L)]) -> Result<(), GrowFailed> {
        let g = self;
        let nv = g.inner.num_nodes() as usize;
        let mut grouped: Vec<Vec<(u16, Vec<u32>)>> = vec![Vec::new(); nv];
        for &(src, dst, label) in edge_list {
            let l = L::into_id(label).as_u16();
            assert!((src as usize) < nv, "src out of range");
            assert!(
                !g.is_ordered(l) || !Edge::with_neighbor(Edge::default(), dst).is_tombstone(),
                "ordered insertion rejects tombstone records"
            );
            let run = &mut grouped[src as usize];
            match run.binary_search_by_key(&l, |&(x, _)| x) {
                Ok(idx) => run[idx].1.push(dst),
                Err(idx) => run.insert(idx, (l, vec![dst])),
            }
        }
        // All input has been checked/grouped before retiring the old owners.
        for t in (0..g.inner.tree_state_len()).rev() {
            g.inner.tree_state_release(t);
        }
        g.buckets.clear_runs();
        let segment_count = g.inner.segment_count();
        for leaf in 0..segment_count {
            g.inner.free_log_pub(leaf);
            let v0 = (leaf * g.inner.segment_size() as u64) as usize;
            let v1 = (v0 as u64 + g.inner.segment_size() as u64) as usize;
            let local = &grouped[v0..v1.min(nv)];
            let (start, span, descs) = g.layout_leaf(leaf, local, false)?;
            let leaf_total = local
                .iter()
                .map(|r| r.iter().map(|(_, n)| n.len() as u64).sum::<u64>())
                .sum();
            let (old_start, old_total, _) = g.inner.leaf_geometry(leaf);
            g.commit_append_leaf(leaf, &descs)?;
            if old_start != u64::MAX && old_total > 0 {
                g.inner.release_span(old_start, old_total);
            }
            g.inner.set_leaf_geometry(leaf, start, span, leaf_total);
        }
        g.inner.reset_edge_count();
        g.inner.add_edges(edge_list.len() as i64);
        Ok(())
    }

    /// Rebalances one leaf: collects every slab-mode bucket's live edges
    /// (slab + log), re-packs them into a fresh span with
    /// degree-proportional slack, preserves Tree-CSR buckets (their records
    /// live in the tree, not the slab), and writes only this leaf's descriptors
    /// at their existing positions. Descriptor counts and other leaves stay intact.
    pub fn rebalance_leaf(&mut self, leaf: u64) {
        let ss = self.inner.segment_size() as u64;
        let v0 = leaf * ss;
        let v1 = (v0 + ss).min(self.inner.num_nodes());
        let mut local: Vec<Vec<(u16, Vec<u32>)>> =
            Vec::with_capacity(v1.saturating_sub(v0) as usize);
        let mut tree_buckets: Vec<Vec<(u16, u32)>> =
            Vec::with_capacity(v1.saturating_sub(v0) as usize);
        for v in v0..v1 {
            let row = (self as &Self).row(v as u32);
            let bcnt = row.bucket_count;
            let mut run: Vec<(u16, Vec<u32>)> = Vec::with_capacity(bcnt as usize);
            let mut tb: Vec<(u16, u32)> = Vec::new();
            for i in 0..bcnt {
                let d = self.bucket_of(v as u32, i);
                if d.tree_bucket != u32::MAX {
                    tb.push((d.label, d.tree_bucket));
                } else {
                    let live = self.bucket_live(v as u32, &d);
                    // Preserve the bucket identity (label) even when empty so
                    // the run stays label-sorted and re-inserts reuse it.
                    run.push((d.label, live));
                }
            }
            local.push(run);
            tree_buckets.push(tb);
        }
        let has_tree_buckets = tree_buckets.iter().any(|run| !run.is_empty());
        let (start, span, descs) = self
            .layout_leaf(leaf, &local, has_tree_buckets)
            .expect("layout leaf");
        let leaf_end = start + span;
        // Merge tree-mode buckets back into the descriptor runs, preserving
        // label-sorted order. Tree buckets are parked at the leaf end
        // (`edge_start = leaf_end`, `stored = 0`); `bucket_boundary` skips
        // them so preceding slab buckets still bound correctly.
        let merge_rows = if has_tree_buckets { descs.len() } else { 0 };
        let mut merged: Vec<Vec<BucketDescriptor>> = Vec::with_capacity(merge_rows);
        for lv in 0..merge_rows {
            let mut m: Vec<BucketDescriptor> =
                Vec::with_capacity(descs[lv].len() + tree_buckets[lv].len());
            let mut di = 0;
            let mut ti = 0;
            while di < descs[lv].len() || ti < tree_buckets[lv].len() {
                let take_tree = if di >= descs[lv].len() {
                    true
                } else if ti >= tree_buckets[lv].len() {
                    false
                } else {
                    tree_buckets[lv][ti].0 < descs[lv][di].label
                };
                if take_tree {
                    let (label, tid) = tree_buckets[lv][ti];
                    let stored = self.inner.tree_state_stored(tid as usize);
                    m.push(BucketDescriptor {
                        label,
                        degree: stored,
                        stored: 0,
                        log_head: -1,
                        edge_start: leaf_end as u32,
                        tree_bucket: tid,
                    });
                    ti += 1;
                } else {
                    m.push(descs[lv][di]);
                    di += 1;
                }
            }
            merged.push(m);
        }
        // A tree-free leaf already holds these exact label-sorted runs, so
        // commit layout_leaf's descriptors instead of rebuilding equal vectors.
        let rows: &[Vec<BucketDescriptor>] = if has_tree_buckets { &merged } else { &descs };
        let leaf_total: u64 = local
            .iter()
            .map(|r| r.iter().map(|(_, n)| n.len() as u64).sum::<u64>())
            .sum();
        // Release the old span and publish the new geometry.
        let (old_start, old_total, _) = self.inner.leaf_geometry(leaf);
        if old_start != u64::MAX && old_total > 0 {
            self.inner.release_span(old_start, old_total);
        }
        self.inner.set_leaf_geometry(leaf, start, span, leaf_total);
        self.inner.set_log_idx_pub(leaf, 0);
        self.inner.free_log_pub(leaf);
        self.commit_leaf_descriptors(leaf, rows);
    }

    /// Inserts one labeled edge. With promotion enabled, a bucket is
    /// limited to `B * R_MAX` live neighbors; oversized inserts panic
    /// before changing the existing bucket. Unordered slab prefixes are dense;
    /// Insertion skips retired markers and appends only after older logs.
    /// A full prefix needs a rebalance or overflow log, not a hole search.
    pub fn insert_edge(&mut self, src: u32, dst: u32, label: L) {
        let label = L::into_id(label).as_u16();
        assert!((src as u64) < self.num_nodes(), "src out of range");
        assert!(
            !self.is_ordered(label) || !Edge::with_neighbor(Edge::default(), dst).is_tombstone(),
            "ordered insertion rejects tombstone records"
        );
        let (mut pos, mut b) = match self.find_bucket(src, label) {
            Some((i, b)) => {
                if self.inner.promote_threshold().is_some() {
                    assert_depth_one(u64::from(b.degree) + 1);
                }
                (i, b)
            }
            None => self.ensure_bucket(src, label),
        };
        // Tree-CSR bucket: O(1) append into the tree, no slab/log touch.
        if b.tree_bucket != u32::MAX {
            let t = b.tree_bucket as usize;
            self.inner.tree_state_insert(t, dst);
            let mut b2 = b;
            b2.degree += 1;
            self.set_bucket(src, pos, &b2);
            self.inner.add_edges(1);
            return;
        }
        // A freshly created (or previously emptied) bucket carries the
        // `u32::MAX` unpositioned marker; lay out the leaf first so the
        // bucket gets a real `edge_start` within its span before writing.
        if b.edge_start == u32::MAX {
            let leaf = src as u64 / self.inner.segment_size() as u64;
            self.rebalance_leaf(leaf);
            (pos, b) = match self.find_bucket(src, label) {
                Some(found) => found,
                None => self.ensure_bucket(src, label),
            };
        }
        for _ in 0..3 {
            let boundary = self.bucket_boundary(src, pos, &b);
            let base = b.edge_start as u64;
            if (base + b.stored as u64) < boundary && (!self.is_ordered(label) || b.log_head == -1)
            {
                // Physical room is usable only after any older ordered logs.
                self.inner.write_edge_pub(
                    base + b.stored as u64,
                    &Edge::with_neighbor(Edge::default(), dst),
                );
                let mut b2 = b;
                b2.stored += 1;
                b2.degree += 1;
                self.set_bucket(src, pos, &b2);
                self.inner.add_edges(1);
                self.maybe_promote(src, pos, b2);
                return;
            }
            // The occupied prefix is dense: grow the leaf and retry.
            let leaf = src as u64 / self.inner.segment_size() as u64;
            self.rebalance_leaf(leaf);
            // Layout rewrote the descriptors; discard the previous value.
            (pos, b) = match self.find_bucket(src, label) {
                Some(found) => found,
                None => self.ensure_bucket(src, label),
            };
        }
        // Last resort: the leaf overflow log. Guard the effective per-leaf
        // capacity — overflowing it would silently corrupt the neighboring
        // leaf's log region. Fold the log via a rebalance (which resets
        // `log_idx` to 0) first, then retry.
        let leaf = src as u64 / self.inner.segment_size() as u64;
        let mut idx = self.inner.log_idx_pub(leaf);
        if idx as usize >= self.inner.max_log_entries() {
            self.rebalance_leaf(leaf);
            (pos, b) = match self.find_bucket(src, label) {
                Some(found) => found,
                None => self.ensure_bucket(src, label),
            };
            idx = self.inner.log_idx_pub(leaf);
            debug_assert!(
                idx as usize <= self.inner.max_log_entries(),
                "rebalance must reset the leaf log"
            );
        }
        if self.inner.max_log_entries() == 0 {
            // No-log mode: the fold above guarantees slab room below the
            // cap (`min(max(2n,1),T) > stored` for `n < T`); at the capped
            // boundary (`n >= T`) there is no room by design and the
            // stored width already trips promotion. Never writes the log.
            let boundary = self.bucket_boundary(src, pos, &b);
            if (b.edge_start as u64 + b.stored as u64) < boundary
                && (!self.is_ordered(label) || b.log_head == -1)
            {
                self.inner.write_edge_pub(
                    b.edge_start as u64 + b.stored as u64,
                    &Edge::with_neighbor(Edge::default(), dst),
                );
                let mut b2 = b;
                b2.stored += 1;
                b2.degree += 1;
                self.set_bucket(src, pos, &b2);
                self.inner.add_edges(1);
                self.maybe_promote(src, pos, b2);
                return;
            }
            self.maybe_promote(src, pos, b);
            // Promotion may rewrite the layout; reload at the preserved
            // label-sorted relative position using the current row start.
            let b2 = self.bucket_of(src, pos);
            assert!(
                b2.tree_bucket != u32::MAX,
                "log cap 0 with no slab room and promotion ineligible (no-log mode needs promotion at the capped boundary)"
            );
            let t = b2.tree_bucket as usize;
            self.inner.tree_state_insert(t, dst);
            let mut b3 = b2;
            b3.degree += 1;
            self.set_bucket(src, pos, &b3);
            self.inner.add_edges(1);
            return;
        }
        let entry = LogEntry {
            src,
            dst,
            label,
            _pad: 0,
            prev: b.log_head,
        };
        self.inner.write_log_pub(leaf, idx, &entry);
        self.inner.set_log_idx_pub(leaf, idx + 1);
        b.log_head = idx as i32;
        b.degree += 1;
        self.set_bucket(src, pos, &b);
        self.inner.add_edges(1);
        self.maybe_promote(src, pos, b);
    }

    /// Promotes a slab-mode bucket to per-bucket Tree-CSR when its stored
    /// width reaches the threshold (not the degree including logs). The threshold
    /// doubles as the slab allocated-space cap (see
    /// [`crate::graph::T_PROMOTE`]); `None` disables promotion.
    fn maybe_promote(&mut self, src: u32, pos: u32, b: BucketDescriptor) {
        if b.tree_bucket != u32::MAX {
            return;
        }
        if let Some(t) = self.inner.promote_threshold() {
            if b.stored >= t {
                self.promote_bucket(src, pos, b);
            }
        }
    }

    /// Promotes one bucket to Tree-CSR mode: folds its slab + log records
    /// into a fresh tree bucket, parks the descriptor (stored = 0), and
    /// rebalances the leaf to reclaim the freed slab space.
    fn promote_bucket(&mut self, v: u32, pos: u32, b: BucketDescriptor) {
        assert_depth_one(u64::from(b.degree));
        let live = self.bucket_live(v, &b); // slab + log (bucket is slab-mode)
        assert_depth_one(live.len() as u64);
        let live_count = live.len() as u32;
        // Mint blocks and block-batch the transfer (4-byte neighbor wire).
        let mut root = Vec::new();
        let new_root_len = root_len(live_count);
        for _ in 0..new_root_len {
            let id = self.inner.mint_block().expect("promote mint");
            root.push(id);
        }
        for (root_index, &block_id) in root.iter().enumerate() {
            let first = root_index * B;
            let last = (first + B).min(live.len());
            let mut payload = [0u8; BLOCK_PAYLOAD_BYTES];
            for (s, t) in live[first..last].iter().enumerate() {
                let off = s * 4;
                payload[off..off + 4].copy_from_slice(&t.to_le_bytes());
            }
            self.inner
                .write_block_payload(block_id, &payload)
                .expect("promote write");
        }
        let tree_idx = self.inner.tree_state_push(root, live_count);
        // Convert the descriptor to tree mode: log folded, stored = 0,
        // edge_start parked at the leaf end (boundary marker; skipped by
        // `bucket_boundary`). The rebalance below re-parks it at the new
        // leaf end.
        let leaf = v as u64 / self.inner.segment_size() as u64;
        let (start, span, _) = self.inner.leaf_geometry(leaf);
        let leaf_end = if start == u64::MAX { 0 } else { start + span };
        let mut b2 = b;
        b2.tree_bucket = tree_idx;
        b2.stored = 0;
        b2.log_head = -1;
        b2.degree = live_count;
        b2.edge_start = leaf_end as u32;
        self.set_bucket(v, pos, &b2);
        // Rebalance to shrink the leaf span (preserves tree-mode buckets).
        self.rebalance_leaf(leaf);
    }

    /// Demotes a Tree-CSR bucket back to slab mode when its live degree
    /// falls to half the configured promotion threshold.
    /// Collects the tree records, stages them at the slab tail, converts
    /// the descriptor to slab mode, releases the tree bucket (fixing up
    /// later bucket ids), and lets [`LabeledLaraGraph::rebalance_leaf`]
    /// re-lay the bucket into the leaf span.
    ///
    /// The fresh slab bucket has `stored = degree <= T/2` (default T/2
    /// if promotion was disabled after trees had already been created).
    fn demote_bucket(&mut self, v: u32, pos: u32, b: BucketDescriptor) {
        let t = b.tree_bucket as usize;
        let leaf = v as u64 / self.inner.segment_size() as u64;
        // 1. Collect live records from the tree bucket.
        let stored = self.inner.tree_state_stored(t);
        let mut live: Vec<u32> = Vec::with_capacity(stored as usize);
        for slot in 0..stored {
            live.push(self.inner.tree_state_read(t, slot));
        }
        let live_count = live.len() as u32;
        // 2. Stage at the slab tail, outside every live leaf span.
        let staging = self.inner.edge_capacity();
        self.inner
            .ensure_slab(staging + u64::from(live_count))
            .expect("demote slab");
        for (k, &n) in live.iter().enumerate() {
            self.inner
                .write_edge_pub(staging + k as u64, &Edge::with_neighbor(Edge::default(), n));
        }
        // 3. Convert the descriptor to slab mode pointing at the staged
        //    records; rebalance_leaf re-reads them here.
        let mut b2 = b;
        b2.tree_bucket = u32::MAX;
        b2.stored = live_count;
        b2.log_head = -1;
        b2.degree = live_count;
        b2.edge_start = staging as u32;
        self.set_bucket(v, pos, &b2);
        // 4. Release the tree bucket and fix up later bucket ids.
        self.inner.tree_state_release(t);
        for w in 0..self.num_nodes() as u32 {
            let row = self.row(w);
            for i in 0..row.bucket_count {
                let d = self.bucket_of(w, i);
                if d.tree_bucket != u32::MAX && d.tree_bucket as usize > t {
                    let mut d2 = d;
                    d2.tree_bucket -= 1;
                    self.set_bucket(w, i, &d2);
                }
            }
        }
        // 5. Re-lay the leaf (positions the bucket and gives it a sub-span).
        self.rebalance_leaf(leaf);
        self.inner.release_span(staging, u64::from(live_count));
    }

    /// Removes one matching labeled edge. Unordered uses slab/tree tail
    /// replacement or newest-first log unlinking. Insertion preserves survivors:
    /// typed slab markers, dense tree suffix shifts and oldest-first log matching.
    /// Ordered tree removal is linear; edge ordinals are not stable.
    pub fn remove_edge(&mut self, src: u32, dst: u32, label: L) -> bool {
        let label = L::into_id(label).as_u16();
        let (pos, mut b) = match self.find_bucket(src, label) {
            Some(found) => found,
            None => return false,
        };
        // Tree-CSR bucket: linear value search + tail replacement, then demote
        // when the live degree falls to T_DEMOTE (hysteresis band).
        if b.tree_bucket != u32::MAX {
            let t = b.tree_bucket as usize;
            let stored = self.inner.tree_state_stored(t);
            for slot in 0..stored {
                if self.inner.tree_state_read(t, slot) == dst {
                    if self.is_ordered(label) {
                        self.inner.tree_state_remove_shift(t, slot);
                    } else {
                        self.inner.tree_remove_swap(t, slot);
                    }
                    b.degree -= 1;
                    self.set_bucket(src, pos, &b);
                    self.inner.add_edges(-1);
                    if b.degree <= self.inner.demote_threshold() {
                        self.demote_bucket(src, pos, b);
                    }
                    return true;
                }
            }
            return false;
        }
        debug_assert!(
            b.edge_start != u32::MAX,
            "unpositioned bucket reached remove path"
        );
        // Slab scan. Only this label decides whether MAX is a marker.
        let base = b.edge_start as u64;
        let ordered = self.is_ordered(label);
        for k in 0..b.stored {
            let e = self.inner.read_edge_pub(base + k as u64);
            if (!ordered || !e.is_tombstone()) && e.neighbor() == dst {
                if ordered {
                    self.inner
                        .write_edge_pub(base + u64::from(k), &Edge::tombstone_edge());
                } else {
                    let last = b.stored - 1;
                    if k != last {
                        let tail = self.inner.read_edge_pub(base + u64::from(last));
                        self.inner.write_edge_pub(base + u64::from(k), &tail);
                    }
                    b.stored = last;
                }
                b.degree -= 1;
                self.set_bucket(src, pos, &b);
                self.inner.add_edges(-1);
                return true;
            }
        }
        // Log chain unlink.
        let leaf = src as u64 / self.inner.segment_size() as u64;
        if b.log_head != -1 {
            let mut prev: i32 = -1;
            let mut cur = b.log_head;
            let mut found = None;
            while cur != -1 {
                let entry = self.inner.read_log_pub(leaf, cur as u32);
                if entry.label == b.label && entry.src == src && entry.dst == dst {
                    found = Some((prev, cur, entry));
                    // Last match in the newest-linked chain is the oldest.
                    if !ordered {
                        break;
                    }
                }
                prev = cur;
                cur = entry.prev;
            }
            if let Some((prev, cur, mut entry)) = found {
                if prev == -1 {
                    b.log_head = entry.prev;
                } else {
                    let mut p = self.inner.read_log_pub(leaf, prev as u32);
                    p.prev = entry.prev;
                    self.inner.write_log_pub(leaf, prev as u32, &p);
                }
                entry.prev = -1;
                self.inner.write_log_pub(leaf, cur as u32, &entry);
                b.degree -= 1;
                self.set_bucket(src, pos, &b);
                self.inner.add_edges(-1);
                return true;
            }
        }
        false
    }

    /// Visits label-ID groups in ascending order, using each label's local
    /// adjacency policy. This is not application chronology across labels.
    pub fn for_each_labeled<F: FnMut(u32, L)>(&self, v: u32, mut f: F) {
        let row = (self as &Self).row(v);
        for i in 0..row.bucket_count {
            let b = self.bucket_of(v, i);
            self.for_each_bucket(v, &b, |n| f(n, L::from_id(LabelId(b.label))));
        }
    }

    /// Visits the neighbors of exactly one label (per-label locality scan).
    pub fn for_each_edge_of_label<F: FnMut(u32)>(&self, v: u32, label: L, mut f: F) {
        let label = L::into_id(label).as_u16();
        let Some((_, b)) = self.find_bucket(v, label) else {
            return;
        };
        self.for_each_bucket(v, &b, &mut f);
    }

    fn for_each_bucket<F: FnMut(u32)>(&self, v: u32, b: &BucketDescriptor, mut f: F) {
        if b.tree_bucket != u32::MAX {
            let t = b.tree_bucket as usize;
            self.inner.for_each_in_tree(t, f);
            return;
        }
        let base = b.edge_start as u64;
        let ordered = self.is_ordered(b.label);
        self.inner.for_each_slab_record(base, b.stored, |e| {
            if !ordered || !e.is_tombstone() {
                f(e.neighbor());
            }
        });
        let leaf = v as u64 / self.inner.segment_size() as u64;
        let mut cur = b.log_head;
        if ordered && cur != -1 {
            let mut chain = Vec::new();
            while cur != -1 {
                let entry = self.inner.read_log_pub(leaf, cur as u32);
                if entry.label == b.label && entry.src == v {
                    chain.push(entry.dst);
                }
                cur = entry.prev;
            }
            for dst in chain.into_iter().rev() {
                f(dst);
            }
        } else {
            while cur != -1 {
                let entry = self.inner.read_log_pub(leaf, cur as u32);
                if entry.label == b.label && entry.src == v {
                    f(entry.dst);
                }
                cur = entry.prev;
            }
        }
    }

    /// Collects `(neighbor, label)` pairs (tests/benches).
    pub fn out_labeled_vec(&self, v: u32) -> Vec<(u32, L)> {
        let mut out = Vec::new();
        self.for_each_labeled(v, |n, l| out.push((n, l)));
        out
    }

    /// Number of vertices (including padding rows).
    pub fn num_nodes(&self) -> u64 {
        self.inner.num_nodes()
    }

    /// Segment size (vertices per leaf).
    pub fn segment_size(&self) -> u32 {
        self.inner.segment_size()
    }

    /// Leaf overflow-log insertion cursor (test introspection).
    pub fn leaf_log_idx(&self, leaf: u64) -> u32 {
        self.inner.log_idx_pub(leaf)
    }

    /// `(physical_start, segment_total, segment_actual)` for one leaf.
    pub fn leaf_geometry(&self, leaf: u64) -> (u64, u64, u64) {
        self.inner.leaf_geometry(leaf)
    }

    /// Number of live edges.
    pub fn num_edges(&self) -> u64 {
        self.inner.num_edges()
    }

    /// Number of Tree-CSR buckets (promoted buckets).
    pub fn tree_bucket_count(&self) -> usize {
        self.inner.tree_state_len()
    }

    /// Whether `v`'s bucket for `label` is in Tree-CSR mode.
    pub fn bucket_is_tree(&self, v: u32, label: L) -> bool {
        let label = L::into_id(label).as_u16();
        match self.find_bucket(v, label) {
            Some((_, b)) => b.tree_bucket != u32::MAX,
            None => false,
        }
    }

    /// Logical out-degree of `v` (sum of bucket live degrees incl. logs).
    pub fn out_degree(&self, v: u32) -> u64 {
        let row = (self as &Self).row(v);
        let mut total = 0u64;
        for i in 0..row.bucket_count {
            total += self.bucket_of(v, i).degree as u64;
        }
        total
    }

    /// Free writable slots before the boundary of `v`'s bucket for `label`
    /// (used by tests to assert per-bucket headroom).
    pub fn bucket_room(&self, v: u32, label: L) -> u64 {
        let label = L::into_id(label).as_u16();
        let Some((pos, b)) = self.find_bucket(v, label) else {
            return 0;
        };
        if b.edge_start == u32::MAX {
            return 0;
        }
        let boundary = self.bucket_boundary(v, pos, &b);
        boundary.saturating_sub(b.edge_start as u64 + b.stored as u64)
    }

    /// Allocated slab width of `v`'s bucket for `label` (boundary −
    /// edge_start), capped at the promote threshold by the leaf layout
    /// (see [`crate::graph::T_PROMOTE`]). Tree-mode buckets hold no slab
    /// data, so they report 0.
    pub fn bucket_allocated(&self, v: u32, label: L) -> u64 {
        let label = L::into_id(label).as_u16();
        let Some((pos, b)) = self.find_bucket(v, label) else {
            return 0;
        };
        if b.edge_start == u32::MAX || b.tree_bucket != u32::MAX {
            return 0;
        }
        let boundary = self.bucket_boundary(v, pos, &b);
        boundary.saturating_sub(b.edge_start as u64)
    }

    /// Checks allocation ownership, per-bucket live counts and boundaries,
    /// and unique acyclic log chains. Leaf counts are layout snapshots in
    /// the labeled layer, not continuously maintained density counters.
    pub fn verify_invariants(&self) {
        self.inner.verify_storage_invariants();
        self.buckets.verify_runs(
            self.inner.buckets_mem(),
            (0..self.num_nodes() as u32).map(|v| {
                let row = self.row(v);
                (row.table_start, row.bucket_count)
            }),
        );
        let mut total = 0u64;
        let mut tree_owners = vec![false; self.inner.tree_state_len()];
        let mut seen: Vec<Vec<bool>> = (0..self.inner.segment_count())
            .map(|leaf| vec![false; self.inner.log_idx_pub(leaf) as usize])
            .collect();
        for v in 0..self.num_nodes() as u32 {
            let row = self.row(v);
            let leaf = v as u64 / self.inner.segment_size() as u64;
            let (start, span, _) = self.inner.leaf_geometry(leaf);
            let mut previous_label = None;
            for i in 0..row.bucket_count {
                let b = self.bucket_of(v, i);
                assert!(
                    previous_label.is_none_or(|label| label < b.label),
                    "bucket labels not sorted/unique"
                );
                previous_label = Some(b.label);
                total += u64::from(b.degree);
                if b.tree_bucket != u32::MAX {
                    let t = b.tree_bucket as usize;
                    assert!(!tree_owners[t], "shared tree bucket");
                    tree_owners[t] = true;
                    assert_eq!(b.degree, self.inner.tree_state_stored(t));
                    assert_eq!(b.stored, 0);
                    assert_eq!(b.log_head, -1);
                    continue;
                }
                if b.edge_start == u32::MAX {
                    assert_eq!(b.degree, 0);
                    assert_eq!(b.stored, 0);
                    assert_eq!(b.log_head, -1);
                    continue;
                }
                let base = u64::from(b.edge_start);
                let boundary = self.bucket_boundary(v, i, &b);
                assert!(
                    start != u64::MAX
                        && base >= start
                        && boundary <= start + span
                        && base + u64::from(b.stored) <= boundary,
                    "bucket crosses slab boundary"
                );
                let mut live = b.stored;
                if self.is_ordered(b.label) {
                    live = 0;
                    for slot in 0..b.stored {
                        if !self
                            .inner
                            .read_edge_pub(base + u64::from(slot))
                            .is_tombstone()
                        {
                            live += 1;
                        }
                    }
                }
                let seen = &mut seen[leaf as usize];
                let mut cur = b.log_head;
                while cur != -1 {
                    assert!(
                        cur >= 0 && (cur as usize) < seen.len(),
                        "log index outside allocated prefix"
                    );
                    assert!(!seen[cur as usize], "shared or cyclic log chain");
                    seen[cur as usize] = true;
                    let entry = self.inner.read_log_pub(leaf, cur as u32);
                    assert_eq!((entry.src, entry.label), (v, b.label), "wrong log owner");
                    live += 1;
                    cur = entry.prev;
                }
                assert_eq!(live, b.degree, "bucket live count mismatch");
            }
        }
        assert!(
            tree_owners.iter().all(|&owned| owned),
            "unowned tree bucket"
        );
        assert_eq!(total, self.num_edges(), "bucket degree sum != num_edges");
    }
}

impl<M: Memory, L: Label> DynamicCsr for LabeledLaraGraph<M, L> {
    fn num_nodes(&self) -> u64 {
        LabeledLaraGraph::num_nodes(self)
    }
    fn num_edges(&self) -> u64 {
        LabeledLaraGraph::num_edges(self)
    }
    fn out_degree(&self, v: u32) -> u64 {
        LabeledLaraGraph::out_degree(self, v)
    }
    fn insert_edge(&mut self, src: u32, dst: u32) {
        self.insert_edge(src, dst, L::from_id(LabelId::new(0)))
    }
    fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, mut f: F) {
        self.for_each_labeled(v, |n, _| f(n))
    }
    fn scan_all<F: FnMut(u32)>(&self, mut f: F) {
        for v in 0..self.num_nodes() as u32 {
            self.for_each_neighbor(v, &mut f);
        }
    }
}

impl<M: Memory, L: Label> RemovableCsr for LabeledLaraGraph<M, L> {
    fn remove_edge(&mut self, src: u32, dst: u32) -> bool {
        self.remove_edge(src, dst, L::from_id(LabelId::new(0)))
    }
}
