//! Graph adapters exposing one uniform interface for the benchmark arms.

use dgap::DgapGraph;
use lara_graph::graph::{Edge, LaraGraph, Vertex};
use lara_graph::labeled::{LabelId, LabeledLaraGraph};
use lara_graph::memory::{GrowFailed, Memory, VecMemory};
use lara_graph::traits::{DynamicCsr, RemovableCsr};
use std::cell::RefCell;
use std::rc::Rc;

/// Bridge: lets the shared [`lara_graph::VecMemory`] drive the dgap crate's
/// [`dgap::Memory`] trait (one heap backend, two trait views).
struct DgapBridge(lara_graph::memory::VecMemory);

impl dgap::Memory for DgapBridge {
    fn read(&self, offset: u64, buf: &mut [u8]) {
        lara_graph::Memory::read(&self.0, offset, buf)
    }
    fn write(&mut self, offset: u64, buf: &[u8]) {
        lara_graph::Memory::write(&self.0, offset, buf)
    }
    fn size(&self) -> u64 {
        lara_graph::Memory::size(&self.0)
    }
    fn grow(&mut self, pages: u64) -> Result<u64, dgap::GrowFailed> {
        lara_graph::Memory::grow(&self.0, pages)
            .map(|_| lara_graph::Memory::size(&self.0))
            .map_err(|_| dgap::GrowFailed {
                current_size: lara_graph::Memory::size(&self.0),
                delta_pages: pages,
            })
    }
}

fn dgap_memory() -> DgapBridge {
    DgapBridge(VecMemory::new())
}

/// Static (read-only) CSR baseline: optimal memory layout, no update path.
pub struct StaticCsr {
    offsets: Vec<u64>,
    targets: Vec<u32>,
}

impl StaticCsr {
    /// Builds from grouped COO.
    pub fn new(num_vertices: u32, edges: &[(u32, u32)]) -> Self {
        let mut offsets = vec![0u64; num_vertices as usize + 1];
        for &(u, _) in edges {
            offsets[u as usize + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut targets = vec![0u32; edges.len()];
        let mut cursor = offsets.clone();
        for &(u, v) in edges {
            targets[cursor[u as usize] as usize] = v;
            cursor[u as usize] += 1;
        }
        Self { offsets, targets }
    }

    pub fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, mut f: F) {
        let (s, e) = (self.offsets[v as usize], self.offsets[v as usize + 1]);
        for t in &self.targets[s as usize..e as usize] {
            f(*t);
        }
    }

    pub fn num_nodes(&self) -> u64 {
        self.offsets.len() as u64 - 1
    }

    pub fn num_edges(&self) -> u64 {
        self.targets.len() as u64
    }
}

impl DynamicCsr for StaticCsr {
    fn num_nodes(&self) -> u64 {
        StaticCsr::num_nodes(self)
    }
    fn num_edges(&self) -> u64 {
        StaticCsr::num_edges(self)
    }
    fn out_degree(&self, v: u32) -> u64 {
        self.offsets[v as usize + 1] - self.offsets[v as usize]
    }
    fn insert_edge(&mut self, _src: u32, _dst: u32) {
        panic!("StaticCsr is read-only");
    }
    fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, f: F) {
        StaticCsr::for_each_neighbor(self, v, f)
    }
    fn scan_all<F: FnMut(u32)>(&self, mut f: F) {
        for v in 0..self.num_nodes() as u32 {
            self.for_each_neighbor(v, &mut f);
        }
    }
}

/// DGAP port adapter (`DgapGraph<u32, DgapBridge>`).
///
/// Fixed configuration — DGAP exposes no knobs comparable to LARA's:
/// segment size follows DGAP's `compute_capacity` rule
/// (`ceil_log2(n)` ideal size, hyperfloored count), PMA thresholds are the
/// hardcoded `up_0=1.0 / up_h=0.75 / low_0=0.25 / low_h=0.50`, and there is
/// no tree-CSR mode or free-span store. `LaraConfig::default()` is the
/// matching LARA arm:
///
/// | DGAP behavior                        | Matching `LaraConfig` field          |
/// |--------------------------------------|--------------------------------------|
/// | fixed segment geometry               | `segment_size: None` (same rule)     |
/// | full-window PMA rebalance on density | `promote_threshold: None` (slab arm) |
/// | resize on root overflow              | `use_free_spans: true` (full arm)    |
pub struct DgapAdapter(DgapGraph<u32, DgapBridge>);

impl DgapAdapter {
    pub fn new(num_vertices: u32) -> Result<Self, GrowFailed> {
        Ok(Self(
            DgapGraph::new(dgap_memory(), num_vertices, true).map_err(map_grow)?,
        ))
    }
    pub fn with_edges(num_vertices: u32, edges: &[(u32, u32)]) -> Result<Self, GrowFailed> {
        Ok(Self(
            DgapGraph::with_edge_list(dgap_memory(), edges, num_vertices, true)
                .map_err(map_grow)?,
        ))
    }
    pub fn verify(&self) {
        self.0.verify_invariants();
    }
}

fn map_grow(_: dgap::GrowFailed) -> GrowFailed {
    GrowFailed {
        current_pages: 0,
        delta_pages: 0,
    }
}

impl DynamicCsr for DgapAdapter {
    fn num_nodes(&self) -> u64 {
        self.0.num_nodes()
    }
    fn num_edges(&self) -> u64 {
        self.0.num_edges() as u64
    }
    fn out_degree(&self, v: u32) -> u64 {
        self.0.out_degree(v) as u64
    }
    fn insert_edge(&mut self, src: u32, dst: u32) {
        self.0.insert_edge(src, dst)
    }
    fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, f: F) {
        self.0.for_each_neighbor(v, f);
    }
    fn scan_all<F: FnMut(u32)>(&self, f: F) {
        self.0.scan_all(f)
    }
}

impl RemovableCsr for DgapAdapter {
    fn remove_edge(&mut self, src: u32, dst: u32) -> bool {
        self.0.remove_edge(src, dst)
    }
}

/// LARA knobs (ablation arms).
///
/// Correspondence to the benchmark arms:
///
/// | `LaraConfig` field         | value                | arm                  |
/// |----------------------------|----------------------|----------------------|
/// | `promote_threshold`        | `Some(T_PROMOTE)`    | full / `lara_tree`   |
/// | `promote_threshold`        | `None`               | `lara_slab`, no-tree |
/// | `use_free_spans`           | `true`               | full arm             |
/// | `use_free_spans`           | `false`              | no-free-span arm (slab bump-allocates; retired spans are never booked) |
/// | `segment_size`             | `None`               | DGAP-parity rule `ceil_log2(n)` ideal, hyperfloored count |
/// | `segment_size`             | `Some(ss)`           | segment sweep; the count is hyperfloored and the size recomputed so the padded vertex count covers `n` |
#[derive(Clone, Copy, Debug)]
pub struct LaraConfig {
    /// Tree-CSR promotion threshold (`None` = slab-only arm).
    pub promote_threshold: Option<u32>,
    /// Free-span reuse (`false` = bump-allocate only).
    pub use_free_spans: bool,
    /// Explicit segment size override (`None` = `ceil_log2(n)` rule).
    pub segment_size: Option<u32>,
    /// Per-leaf overflow-log cap (default [`MAX_LOG_ENTRIES`](lara_graph::graph::MAX_LOG_ENTRIES)).
    pub max_log_entries: usize,
}

impl Default for LaraConfig {
    fn default() -> Self {
        Self {
            promote_threshold: Some(lara_graph::graph::T_PROMOTE),
            use_free_spans: true,
            segment_size: None,
            max_log_entries: lara_graph::graph::MAX_LOG_ENTRIES,
        }
    }
}

/// Fresh memory bundle (8 regions).
fn fresh_mem() -> lara_graph::graph::LaraMemories<VecMemory> {
    lara_graph::graph::LaraMemories {
        vertices: VecMemory::new(),
        edges: VecMemory::new(),
        logs: VecMemory::new(),
        counts: VecMemory::new(),
        spans: VecMemory::new(),
        free_spans: VecMemory::new(),
        blocks: VecMemory::new(),
        buckets: VecMemory::new(),
    }
}

/// LARA adapter.
///
/// Constructed from a [`LaraConfig`]; see the struct's table for how each
/// knob maps to the paper's ablation arms. Memory is one
/// [`lara_graph::graph::LaraMemories`] bundle (8 regions of [`VecMemory`]).
pub struct LaraAdapter(LaraGraph<Vertex, Edge, VecMemory>);

impl LaraAdapter {
    pub fn new(num_vertices: u32, cfg: LaraConfig) -> Result<Self, GrowFailed> {
        let mem = fresh_mem();
        let mut g = match cfg.segment_size {
            Some(ss) => LaraGraph::new_with_segments_and_max_log(
                mem,
                num_vertices,
                ss,
                cfg.max_log_entries,
            )?,
            None => LaraGraph::new_with_max_log_entries(mem, num_vertices, cfg.max_log_entries)?,
        };
        g.set_promote_threshold(cfg.promote_threshold);
        g.set_use_free_spans(cfg.use_free_spans);
        Ok(Self(g))
    }
    pub fn with_edges(
        num_vertices: u32,
        edges: &[(u32, u32)],
        cfg: LaraConfig,
    ) -> Result<Self, GrowFailed> {
        let mem = fresh_mem();
        let mut g = match cfg.segment_size {
            Some(ss) => LaraGraph::new_with_segments_and_max_log(
                mem,
                num_vertices,
                ss,
                cfg.max_log_entries,
            )?,
            None => LaraGraph::new_with_max_log_entries(mem, num_vertices, cfg.max_log_entries)?,
        };
        g.set_promote_threshold(cfg.promote_threshold);
        g.set_use_free_spans(cfg.use_free_spans);
        g.build_pairs(edges)?;
        Ok(Self(g))
    }
    pub fn verify(&self) {
        self.0.verify_invariants();
    }
    pub fn free_span_stats(&self) -> (u32, u64, u64) {
        self.0.free_span_stats()
    }
    pub fn slab_capacity(&self) -> u64 {
        self.0.slab_capacity()
    }
    /// Total edge-slot footprint (slab + tree blocks); see
    /// [`lara_graph::graph::LaraGraph::memory_slots`].
    pub fn memory_slots(&self) -> u64 {
        self.0.memory_slots()
    }
    /// Legacy partial storage estimate, NOT total heap or resident bytes.
    /// Use the standalone `memory_probe` for allocated-heap comparisons.
    pub fn memory_bytes_total(&self) -> u64 {
        self.0.memory_bytes_total()
    }
    /// Perf-instrument passthroughs (measurement only, behavior unchanged).
    pub fn rebalance_count(&self) -> u64 {
        self.0.rebalance_count()
    }
    pub fn promote_count(&self) -> u64 {
        self.0.promote_count()
    }
    pub fn segment_count(&self) -> u64 {
        self.0.segment_count()
    }
    pub fn log_capacity_bytes(&self) -> u64 {
        self.0.segment_count() * self.0.max_log_entries() as u64 * 16
    }
    /// Components of the legacy partial estimate (bytes), not total memory.
    /// `slab = edge_capacity × 4`, `logs` = backed log memory minus the
    /// tiny term, `tiny` = bumped pooled-tiny chunks (32 bytes each at K=8),
    /// `vertex = seg × seg_size × 24` (`Vertex::BYTES`),
    /// `tree = remainder`.
    pub fn mem_breakdown(&self) -> (u64, u64, u64, u64, u64) {
        let slab = self.0.slab_capacity() * 4;
        let tiny = self.0.tiny_memory_bytes();
        let logs = self.0.log_memory_bytes() - tiny;
        let vertex = self.0.segment_count() * self.0.segment_size() as u64 * 24;
        let total = self.0.memory_bytes_total();
        let tree = total.saturating_sub(slab + logs + tiny + vertex);
        (slab, tree, logs, tiny, vertex)
    }
    /// Scan-breakdown stages (measurement only, behavior unchanged).
    /// `row_stage`: row fetch + tree-bucket branch, no edge loop.
    pub fn scan_row_stage(&self, v: u32) {
        use lara_graph::traits::CsrVertex;
        let row = self.0.get_row_pub(v);
        std::hint::black_box(row.tree_bucket());
    }
    /// Raw tree loop with discarded reads (no callback).
    pub fn scan_tree_raw_empty(&self, v: u32) {
        use lara_graph::traits::CsrVertex;
        let row = self.0.get_row_pub(v);
        let t = row.tree_bucket() as usize;
        let stored = self.0.tree_state_stored(t);
        for slot in 0..stored {
            std::hint::black_box(self.0.tree_state_read(t, slot));
        }
    }
    /// Raw tree loop with accumulation (no callback).
    pub fn scan_tree_raw_sum(&self, v: u32) -> u64 {
        use lara_graph::traits::CsrVertex;
        let row = self.0.get_row_pub(v);
        let t = row.tree_bucket() as usize;
        let stored = self.0.tree_state_stored(t);
        let mut sum = 0u64;
        for slot in 0..stored {
            sum += self.0.tree_state_read(t, slot) as u64;
        }
        sum
    }
    /// Row-state probes for residency histograms (measurement only).
    pub fn row_is_tiny(&self, v: u32) -> bool {
        self.0.row_is_tiny(v)
    }
    pub fn row_is_tree(&self, v: u32) -> bool {
        use lara_graph::traits::CsrVertex;
        self.0.get_row_pub(v).tree_bucket() != lara_graph::traits::NO_BUCKET
    }
    pub fn row_has_log(&self, v: u32) -> bool {
        use lara_graph::traits::CsrVertex;
        self.0.get_row_pub(v).log_head() != -1
    }
    pub fn row_stored(&self, v: u32) -> u64 {
        use lara_graph::traits::CsrVertex;
        let r = self.0.get_row_pub(v);
        (r.stored_degree() & !(1 << 31)) as u64
    }
    /// Folds every leaf (settled-scan precondition).
    pub fn settle(&mut self) {
        for leaf in 0..self.0.segment_count() {
            self.0.rebalance_leaf(leaf);
        }
    }
}

impl DynamicCsr for LaraAdapter {
    fn num_nodes(&self) -> u64 {
        self.0.num_nodes()
    }
    fn num_edges(&self) -> u64 {
        self.0.num_edges()
    }
    fn out_degree(&self, v: u32) -> u64 {
        self.0.out_degree(v)
    }
    fn insert_edge(&mut self, src: u32, dst: u32) {
        self.0.insert_edge(src, dst)
    }
    fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, f: F) {
        self.0.for_each_neighbor(v, f)
    }
    fn scan_all<F: FnMut(u32)>(&self, f: F) {
        self.0.scan_all(f)
    }
}

impl RemovableCsr for LaraAdapter {
    fn remove_edge(&mut self, src: u32, dst: u32) -> bool {
        self.0.remove_edge(src, dst)
    }
}

/// Labeled LARA adapter.
pub struct LabeledAdapter(LabeledLaraGraph<VecMemory>);

impl LabeledAdapter {
    pub fn new(num_vertices: u32) -> Result<Self, GrowFailed> {
        Ok(Self(LabeledLaraGraph::new(fresh_mem(), num_vertices)?))
    }
    /// Sweep-config constructor with an explicit promote threshold
    /// (= slab cap). Fixtures must be built with
    /// [`LabeledAdapter::insert_labeled`] (sequential inserts drive
    /// promotion; bulk build lays slab directly and would skip it).
    pub fn with_sweep_config(
        num_vertices: u32,
        threshold: Option<u32>,
    ) -> Result<Self, GrowFailed> {
        Ok(Self(
            LabeledLaraGraph::new(fresh_mem(), num_vertices)?.with_promote_threshold(threshold),
        ))
    }
    /// Sweep-config constructor with an explicit log cap. The log region
    /// is sized here, so this (not a post-construction setter) is the
    /// way to grow beyond 170.
    pub fn with_sweep_config_and_max_log(
        num_vertices: u32,
        threshold: Option<u32>,
        max_log_entries: usize,
    ) -> Result<Self, GrowFailed> {
        Ok(Self(
            LabeledLaraGraph::new_with_max_log_entries(fresh_mem(), num_vertices, max_log_entries)?
                .with_promote_threshold(threshold),
        ))
    }
    /// Folds residual overflow logs (steady-state scan precondition).
    pub fn settle(&mut self) {
        self.0.rebalance_all();
    }
    /// Sequential labeled insert (drives promotion; see
    /// [`LabeledAdapter::with_sweep_config`]).
    pub fn insert_labeled(&mut self, src: u32, dst: u32, label: u16) {
        self.0.insert_edge(src, dst, LabelId::new(label))
    }
    /// Whether `v`'s bucket for `label` is in Tree-CSR mode (fixture guard).
    pub fn is_tree(&self, v: u32, label: u16) -> bool {
        self.0.bucket_is_tree(v, LabelId::new(label))
    }
    /// Scan one `(vertex, label)` bucket.
    pub fn scan_label<F: FnMut(u32)>(&self, v: u32, label: u16, f: F) {
        self.0.for_each_edge_of_label(v, LabelId::new(label), f)
    }
    /// Scan all labels of one vertex.
    pub fn scan_vertex<F: FnMut(u32, LabelId)>(&self, v: u32, f: F) {
        self.0.for_each_labeled(v, f)
    }
    pub fn with_labeled_edges(
        num_vertices: u32,
        edges: &[(u32, u32, u16)],
    ) -> Result<Self, GrowFailed> {
        let labeled: Vec<(u32, u32, LabelId)> = edges
            .iter()
            .map(|&(u, v, l)| (u, v, LabelId::new(l)))
            .collect();
        Ok(Self(LabeledLaraGraph::with_edge_list(
            fresh_mem(),
            &labeled,
            num_vertices,
        )?))
    }
}

impl DynamicCsr for LabeledAdapter {
    fn num_nodes(&self) -> u64 {
        self.0.num_nodes()
    }
    fn num_edges(&self) -> u64 {
        self.0.num_edges()
    }
    fn out_degree(&self, v: u32) -> u64 {
        self.0.out_degree(v)
    }
    fn insert_edge(&mut self, src: u32, dst: u32) {
        self.0.insert_edge(src, dst, LabelId::new(0))
    }
    fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, mut f: F) {
        self.0.for_each_labeled(v, |n, _| f(n))
    }
    fn scan_all<F: FnMut(u32)>(&self, mut f: F) {
        for v in 0..self.num_nodes() as u32 {
            self.for_each_neighbor(v, &mut f);
        }
    }
}

impl RemovableCsr for LabeledAdapter {
    fn remove_edge(&mut self, src: u32, dst: u32) -> bool {
        self.0.remove_edge(src, dst, LabelId::new(0))
    }
}

/// Persist-counting LARA adapter: identical construction to [`LaraAdapter`]
/// but backed by [`PersistCounterMemory`](lara_graph::memory::PersistCounterMemory)
/// with one `Rc`-shared counter set across all 8 regions, so
/// [`CountingLaraAdapter::persisted_bytes`] is the graph-wide total.
/// Counts start at zero after construction. They describe backend writes,
/// not complete durable graph updates. Time the uninstrumented adapter.
pub struct CountingLaraAdapter {
    inner: LaraGraph<Vertex, Edge, lara_graph::memory::PersistCounterMemory<VecMemory>>,
    counters: Rc<lara_graph::memory::PersistCounters>,
}

impl CountingLaraAdapter {
    pub fn new(num_vertices: u32, cfg: LaraConfig) -> Result<Self, GrowFailed> {
        use lara_graph::memory::PersistCounterMemory;
        let counters = Rc::new(lara_graph::memory::PersistCounters::default());
        let mem = lara_graph::graph::LaraMemories {
            vertices: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            edges: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            logs: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            counts: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            spans: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            free_spans: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            blocks: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            buckets: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        };
        let mut g = match cfg.segment_size {
            Some(ss) => LaraGraph::new_with_segments_and_max_log(
                mem,
                num_vertices,
                ss,
                cfg.max_log_entries,
            )?,
            None => LaraGraph::new_with_max_log_entries(mem, num_vertices, cfg.max_log_entries)?,
        };
        g.set_promote_threshold(cfg.promote_threshold);
        g.set_use_free_spans(cfg.use_free_spans);
        counters.reset();
        Ok(Self { inner: g, counters })
    }
    /// Graph-wide persisted bytes (all regions aggregated).
    pub fn persisted_bytes(&self) -> u64 {
        self.counters.bytes()
    }
    /// Graph-wide `Memory` write-op count.
    pub fn write_ops(&self) -> u64 {
        self.counters.ops()
    }
    /// Graph-wide persisted bytes per inserted edge.
    pub fn persisted_bytes_per_edge(&self) -> f64 {
        self.persisted_bytes() as f64 / self.inner.num_edges() as f64
    }
    /// Graph-wide write ops per inserted edge.
    pub fn write_ops_per_edge(&self) -> f64 {
        self.write_ops() as f64 / self.inner.num_edges() as f64
    }
    pub fn verify(&self) {
        self.inner.verify_invariants();
    }
}

impl DynamicCsr for CountingLaraAdapter {
    fn num_nodes(&self) -> u64 {
        self.inner.num_nodes()
    }
    fn num_edges(&self) -> u64 {
        self.inner.num_edges()
    }
    fn out_degree(&self, v: u32) -> u64 {
        self.inner.out_degree(v)
    }
    fn insert_edge(&mut self, src: u32, dst: u32) {
        self.inner.insert_edge(src, dst)
    }
    fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, f: F) {
        self.inner.for_each_neighbor(v, f)
    }
    fn scan_all<F: FnMut(u32)>(&self, f: F) {
        self.inner.scan_all(f)
    }
}

/// File-backed (`mmap`) LARA adapter: one backing file per region in a
/// temp dir (removed on drop). Minimum useful arm is hub wall-clock vs
/// the `VecMemory` adapter; pair with [`CountingLaraAdapter`] (wrap
/// `MmapMemory` in the counter) for write-volume accounting on mmap.
pub struct MmapLaraAdapter {
    inner: LaraGraph<Vertex, Edge, lara_graph::MmapMemory>,
    dir: std::path::PathBuf,
}

impl MmapLaraAdapter {
    pub fn new(num_vertices: u32, cfg: LaraConfig) -> Result<Self, GrowFailed> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "lara_mmap_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).map_err(|_| GrowFailed {
            current_pages: 0,
            delta_pages: 0,
        })?;
        let region = |name: &str| {
            lara_graph::MmapMemory::create(dir.join(format!("{name}.bin")), 1 << 20).map_err(|_| {
                GrowFailed {
                    current_pages: 0,
                    delta_pages: 0,
                }
            })
        };
        let mem = lara_graph::graph::LaraMemories {
            vertices: region("vertices")?,
            edges: region("edges")?,
            logs: region("logs")?,
            counts: region("counts")?,
            spans: region("spans")?,
            free_spans: region("free_spans")?,
            blocks: region("blocks")?,
            buckets: region("buckets")?,
        };
        let mut g = match cfg.segment_size {
            Some(ss) => LaraGraph::new_with_segments_and_max_log(
                mem,
                num_vertices,
                ss,
                cfg.max_log_entries,
            )?,
            None => LaraGraph::new_with_max_log_entries(mem, num_vertices, cfg.max_log_entries)?,
        };
        g.set_promote_threshold(cfg.promote_threshold);
        g.set_use_free_spans(cfg.use_free_spans);
        Ok(Self { inner: g, dir })
    }
    pub fn verify(&self) {
        self.inner.verify_invariants();
    }
}

impl Drop for MmapLaraAdapter {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl DynamicCsr for MmapLaraAdapter {
    fn num_nodes(&self) -> u64 {
        self.inner.num_nodes()
    }
    fn num_edges(&self) -> u64 {
        self.inner.num_edges()
    }
    fn out_degree(&self, v: u32) -> u64 {
        self.inner.out_degree(v)
    }
    fn insert_edge(&mut self, src: u32, dst: u32) {
        self.inner.insert_edge(src, dst)
    }
    fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, f: F) {
        self.inner.for_each_neighbor(v, f)
    }
    fn scan_all<F: FnMut(u32)>(&self, f: F) {
        self.inner.scan_all(f)
    }
}

/// Shared handle type for benchmarking graphs behind `&mut`.
pub type SharedGraph<G> = Rc<RefCell<G>>;

#[allow(unused_imports)]
use Memory as _MemoryImport;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counted_adapter_excludes_constructor_writes() {
        let mut g = CountingLaraAdapter::new(64, LaraConfig::default()).unwrap();
        assert_eq!((g.persisted_bytes(), g.write_ops()), (0, 0));
        g.insert_edge(0, 7);
        assert_eq!((g.persisted_bytes(), g.write_ops()), (5, 2));
        g.verify();
        assert_eq!((g.persisted_bytes(), g.write_ops()), (5, 2));
    }
}
