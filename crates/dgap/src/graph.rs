//! Faithful Rust port of DGAP's `graph.h` (`CSRGraph`).
//!
//! DGAP (arXiv:2403.02665) stores adjacency lists in a PMA (Packed Memory
//! Array) where each vertex's list is prefixed by a `-vid` sentinel. Dynamic
//! inserts append in place while space remains inside the vertex's region;
//! overflow edges go to a per-leaf-segment edge log (`MAX_LOG_ENTRIES`);
//! density violations trigger weighted rebalances that walk up a conceptual
//! binary tree over leaf segments. `resize_v1` doubles the edge array.
//!
//! Deviations from the original, all forced by the single-threaded DRAM
//! setting:
//! - PMDK persistence (`flush_clwb_nolog`, PMEMoids, pool open/recover) is
//!   replaced by the [`crate::memory::Memory`] trait. Per-segment logs and the
//!   edge slab both live in `Memory`; vertex rows and PMA tree counters stay
//!   in DRAM vectors exactly like the C++ DRAM caches (`vertices_`,
//!   `segment_edges_actual/total`, `log_segment_idx_`).
//! - `PMALeafSegment` mutex/condvar and the undo log (`ulog`/`oplog`) are
//!   dropped: they exist for concurrency and crash consistency. The shift
//!   shift algorithm in `DgapGraph::rebalance_data_v1` is safe without them
//!   because positions from `calculate_positions` make every in-window shift
//!   a right-shift processed in descending vertex order.
//! - `remove_edge` is an extension: `graph.h` is insert-only. On-segment
//!   deletion shifts the victim's list tail left by one; when the vertex
//!   keeps a log chain, a leaf rebalance folds everything back so the
//!   `has_log ⟹ region full` invariant is restored. Underflow rebalancing
//!   (`low_0`/`low_h`) is out of scope.

use crate::memory::{GrowFailed, Memory, PAGE_SIZE, safe_write};
use crate::util::{ceil_div, ceil_log2, floor_log2, hyperfloor};

/// ~1.99 KB per leaf-segment edge log (matches `MAX_LOG_ENTRIES`).
pub const MAX_LOG_ENTRIES: usize = 170;

/// Chunk width in bytes for scan-path on-seg reads. DGAP's on-seg run is
/// contiguous per vertex, so the Rust port bulk-reads it in fixed chunks
/// (one `Memory::read` per chunk) and decodes in memory — the same
/// `SCAN_CHUNK_BYTES` pattern LARA uses. This is **not** an algorithmic
/// advantage over the C++ original: `graph.h` walks `seg_base_ptr + index`
/// directly with no per-edge call cost. The chunking only cancels the
/// per-edge `Memory::read` overhead the port artifact introduced, keeping
/// the DGAP-vs-LARA comparison about algorithms, not abstraction cost.
pub const SCAN_CHUNK_BYTES: usize = 512;

/// PMA density thresholds (height-based, from `graph.h`): leaves are denser
/// than the root. `low_*` are defined by the original for deletion paths.
pub const UP_0: f64 = 1.00; // leaf upper threshold
pub const UP_H: f64 = 0.75; // root upper threshold
pub const LOW_0: f64 = 0.25; // leaf lower threshold
pub const LOW_H: f64 = 0.50; // root lower threshold

/// One on-slab edge cell. The sign bit is the sentinel marker: `raw < 0`
/// encodes `-vid` (vertex id `(-raw) as u32`), `raw >= 0` is a real edge to
/// `raw as u32`.
pub type RawEdge = i32;

/// One vertex row (`struct vertex_element`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexElement {
    /// Absolute slab index where this vertex's sentinel sits.
    pub index: i64,
    /// 1 + logical out-degree (the sentinel counts).
    pub degree: i32,
    /// Head offset into the owning segment's edge log, or `-1`.
    pub offset: i32,
}

/// One per-segment log entry (`struct LogEntry`): a linked-list node in the
/// owning segment's insertion log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogEntry {
    /// Source vertex (parity with the C++; the chain head identifies it).
    pub u: i32,
    /// Destination vertex.
    pub v: i32,
    /// Previous (older) entry of the same vertex in this log, or `-1`.
    pub prev_offset: i32,
}

impl LogEntry {
    pub const BYTES: usize = 12;
}

/// Sum of segment-tree node counters: `(actual, total)` edge cells.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegMeta {
    pub actual: i64,
    pub total: i64,
}

/// DGAP mutable CSR graph, generic over the backing [`Memory`].
///
/// The logical edge payload `V` is decoded from the on-slab [`RawEdge`];
/// for plain unweighted graphs `V = u32` (the neighbor id).
///
/// Memory layout (all little-endian):
/// ```text
/// 0          : per-segment logs (segment_count * MAX_LOG_ENTRIES * 12)
/// edges_base : edge slab (elem_capacity * 4 bytes, grows on resize)
/// ```
pub struct DgapGraph<V = u32, M: Memory = crate::memory::VecMemory> {
    memory: M,
    edges_base: u64,
    log_base: u64,

    // DRAM caches (mirrors the C++ `vertices_`, tree counters, log index).
    vertices: Vec<VertexElement>,
    /// PMA tree counters; leaves live at `[segment_count, 2 * segment_count)`.
    segment_edges_actual: Vec<i64>,
    segment_edges_total: Vec<i64>,
    /// Next free slot per leaf-segment log (`log_segment_idx_`).
    log_segment_idx: Vec<i32>,
    /// Shared scratch for bulk cell moves (kept to avoid per-shift allocs).
    scratch: Vec<u8>,

    directed: bool,
    num_vertices: i32,
    num_edges: i64,
    elem_capacity: i64,
    segment_size: i32,
    segment_count: i64,
    tree_height: i32,
    delta_up: f64,
    delta_low: f64,

    _marker: std::marker::PhantomData<V>,
}

impl<V: From<u32> + Copy, M: Memory> DgapGraph<V, M> {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Creates an empty graph over `memory` and bulk-inserts `edge_list`
    /// (COO, grouped by ascending source vertex — mirrors the Builder's
    /// sort). `num_vertices` is rounded up to `segment_count * segment_size`
    /// exactly like `compute_capacity`.
    pub fn with_edge_list(
        memory: M,
        edge_list: &[(u32, u32)],
        num_vertices: u32,
        directed: bool,
    ) -> Result<Self, GrowFailed> {
        let mut g = Self::empty(memory, edge_list.len() as i64, num_vertices, directed)?;
        g.insert_edge_list(edge_list)?;
        Ok(g)
    }

    /// Creates an empty graph (all vertices sentinel-only).
    pub fn new(memory: M, num_vertices: u32, directed: bool) -> Result<Self, GrowFailed> {
        Self::with_edge_list(memory, &[], num_vertices, directed)
    }

    fn empty(
        mut memory: M,
        n_edges: i64,
        n_vertices: u32,
        directed: bool,
    ) -> Result<Self, GrowFailed> {
        // ---- compute_capacity() ----
        let mut num_vertices = n_vertices.max(1) as i64;
        let ideal_segment_size = ceil_log2(num_vertices as u64) as i64; // ideal seg size
        let mut segment_count = ceil_div(num_vertices as u64, ideal_segment_size as u64).max(1);
        segment_count = hyperfloor(segment_count); // power of two
        let segment_size = ceil_div(num_vertices as u64, segment_count) as i64;
        num_vertices = segment_count as i64 * segment_size;

        let max_sparseness = (1.0 / LOW_0) as i64; // 4
        let elem_capacity = (n_edges.max(0) + num_vertices) * max_sparseness;

        let tree_height = floor_log2(segment_count) as i32;
        let delta_up = (UP_0 - UP_H) / tree_height as f64;
        let delta_low = (LOW_H - LOW_0) / tree_height as f64;

        // Memory layout: logs first (fixed size), edges above.
        let log_base = 0u64;
        let log_bytes = segment_count * MAX_LOG_ENTRIES as u64 * LogEntry::BYTES as u64;
        let edges_base = log_bytes.div_ceil(PAGE_SIZE) * PAGE_SIZE;

        // Allocate both regions through the Memory abstraction.
        safe_write(&mut memory, log_base + log_bytes - 1, &[0u8])?;
        safe_write(
            &mut memory,
            edges_base + elem_capacity as u64 * 4 - 1,
            &[0u8],
        )?;

        let vertices = vec![VertexElement::default(); num_vertices as usize];
        Ok(Self {
            memory,
            edges_base,
            log_base,
            vertices,
            segment_edges_actual: vec![0i64; (segment_count * 2) as usize],
            segment_edges_total: vec![0i64; (segment_count * 2) as usize],
            log_segment_idx: vec![0i32; segment_count as usize],
            scratch: Vec::new(),
            directed,
            num_vertices: num_vertices as i32,
            num_edges: n_edges.max(0),
            elem_capacity,
            segment_size: segment_size as i32,
            segment_count: segment_count as i64,
            tree_height,
            delta_up,
            delta_low,
            _marker: std::marker::PhantomData,
        })
    }

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    pub fn directed(&self) -> bool {
        self.directed
    }

    pub fn num_nodes(&self) -> u64 {
        self.num_vertices as u64
    }

    /// Number of edges including per-vertex sentinels.
    pub fn num_edges_with_sentinels(&self) -> i64 {
        self.num_edges
    }

    pub fn num_edges(&self) -> i64 {
        self.num_edges - self.num_vertices as i64
    }

    pub fn out_degree(&self, v: u32) -> i64 {
        self.vertices[v as usize].degree as i64 - 1
    }

    pub fn elem_capacity(&self) -> i64 {
        self.elem_capacity
    }

    pub fn segment_size(&self) -> i32 {
        self.segment_size
    }

    pub fn segment_count(&self) -> i64 {
        self.segment_count
    }

    pub fn tree_height(&self) -> i32 {
        self.tree_height
    }

    /// Height-interpolated upper density threshold at `height`.
    pub fn up_threshold(&self, height: i32) -> f64 {
        UP_0 - (height as f64 * self.delta_up)
    }

    /// Height-interpolated lower density threshold at `height`.
    pub fn low_threshold(&self, height: i32) -> f64 {
        LOW_0 + (height as f64 * self.delta_low)
    }

    /// Leaf-segment tree node id for `vertex_id` (in
    /// `[segment_count, 2 * segment_count)`).
    #[inline]
    pub fn get_segment_id(&self, vertex_id: u32) -> i64 {
        (vertex_id as i64 / self.segment_size as i64) + self.segment_count
    }

    #[inline]
    fn log_seg_index(&self, tree_leaf: i64) -> usize {
        (tree_leaf - self.segment_count) as usize
    }

    // ------------------------------------------------------------------
    // Edge slab / log I/O through Memory
    // ------------------------------------------------------------------

    #[inline]
    fn read_edge(&self, slot: i64) -> RawEdge {
        let mut buf = [0u8; 4];
        self.memory
            .read(self.edges_base + slot as u64 * 4, &mut buf);
        RawEdge::from_le_bytes(buf)
    }

    #[inline]
    fn write_edge(&mut self, slot: i64, e: RawEdge) {
        self.memory
            .write(self.edges_base + slot as u64 * 4, &e.to_le_bytes());
    }

    #[inline]
    fn read_log_entry(&self, seg: usize, idx: i32) -> LogEntry {
        let off = self.log_base
            + (seg as u64) * MAX_LOG_ENTRIES as u64 * LogEntry::BYTES as u64
            + idx as u64 * LogEntry::BYTES as u64;
        let mut buf = [0u8; LogEntry::BYTES];
        self.memory.read(off, &mut buf);
        LogEntry {
            u: i32::from_le_bytes(buf[0..4].try_into().unwrap()),
            v: i32::from_le_bytes(buf[4..8].try_into().unwrap()),
            prev_offset: i32::from_le_bytes(buf[8..12].try_into().unwrap()),
        }
    }

    #[inline]
    fn write_log_entry(&mut self, seg: usize, idx: i32, e: &LogEntry) {
        let off = self.log_base
            + (seg as u64) * MAX_LOG_ENTRIES as u64 * LogEntry::BYTES as u64
            + idx as u64 * LogEntry::BYTES as u64;
        let mut bytes = [0u8; LogEntry::BYTES];
        bytes[0..4].copy_from_slice(&e.u.to_le_bytes());
        bytes[4..8].copy_from_slice(&e.v.to_le_bytes());
        bytes[8..12].copy_from_slice(&e.prev_offset.to_le_bytes());
        self.memory.write(off, &bytes);
    }

    /// `release_log`: zero the used entries and reset the insert index.
    fn release_log(&mut self, seg: usize) {
        let used = self.log_segment_idx[seg];
        if used == 0 {
            return;
        }
        let zero = LogEntry::default();
        for i in 0..used {
            self.write_log_entry(seg, i, &zero);
        }
        self.log_segment_idx[seg] = 0;
    }

    // ------------------------------------------------------------------
    // Bulk build (insert(edge_list))
    // ------------------------------------------------------------------

    /// `insert(const EdgeList&)`: COO bulk build. Requires `edge_list`
    /// grouped by ascending source vertex.
    pub fn insert_edge_list(&mut self, edge_list: &[(u32, u32)]) -> Result<(), GrowFailed> {
        let mut last_vid: i64 = -1;
        let mut ii: i64 = 0;

        for &(t_src, t_dst) in edge_list {
            debug_assert!((t_src as i64) < self.num_vertices as i64);
            let t_segment_id = self.get_segment_id(t_src);
            if self.vertices[t_src as usize].degree == 0 {
                // Fill skipped vertices with sentinels.
                if t_src as i64 != last_vid + 1 {
                    for vid in (last_vid + 1)..t_src as i64 {
                        self.write_edge(ii, -(vid as RawEdge));
                        self.vertices[vid as usize].degree = 1;
                        self.vertices[vid as usize].index = ii;
                        ii += 1;
                        let sid = self.get_segment_id(vid as u32) as usize;
                        self.segment_edges_actual[sid] += 1;
                    }
                }
                self.write_edge(ii, -(t_src as RawEdge));
                self.vertices[t_src as usize].degree = 1;
                self.vertices[t_src as usize].index = ii;
                ii += 1;
                self.segment_edges_actual[t_segment_id as usize] += 1;
                last_vid = t_src as i64;
            }

            self.write_edge(ii, t_dst as RawEdge);
            ii += 1;
            self.vertices[t_src as usize].degree += 1;
            self.segment_edges_actual[t_segment_id as usize] += 1;
        }

        // Trailing zero-degree vertices get sentinels; init offsets.
        for i in 0..self.num_vertices as usize {
            if self.vertices[i].degree == 0 {
                debug_assert!((i as i64) > last_vid, "zero-degree vertex before last_vid");
                self.write_edge(ii, -(i as RawEdge));
                self.vertices[i].degree = 1;
                self.vertices[i].index = ii;
                ii += 1;
                let sid = self.get_segment_id(i as u32) as usize;
                self.segment_edges_actual[sid] += 1;
            }
            self.vertices[i].offset = -1;
        }

        self.num_edges += self.num_vertices as i64;
        self.spread_weighted(0, self.num_vertices)?;
        Ok(())
    }

    /// `spread_weighted(0, num_vertices)`: redistribute the freshly built
    /// array by degree weights. Positions only move right; descending
    /// vertex order with buffered copies.
    fn spread_weighted(&mut self, start_vertex: i32, end_vertex: i32) -> Result<(), GrowFailed> {
        debug_assert_eq!(start_vertex, 0);
        let gaps = self.elem_capacity - self.num_edges;
        let new_positions =
            self.calculate_positions(start_vertex, end_vertex, gaps, self.num_edges);

        for curr_vertex in (start_vertex..end_vertex).rev() {
            let curr_degree = self.vertices[curr_vertex as usize].degree as i64;
            let read_index = self.vertices[curr_vertex as usize].index + curr_degree - 1;
            let write_index = new_positions[curr_vertex as usize] + curr_degree - 1;
            debug_assert!(
                write_index >= read_index,
                "index anomaly occurred while spreading elements"
            );
            self.move_cells(
                read_index - curr_degree + 1,
                curr_degree,
                write_index - curr_degree + 1,
            );
            self.vertices[curr_vertex as usize].index =
                new_positions[(curr_vertex - start_vertex) as usize];
        }
        self.recount_segment_total_all();
        Ok(())
    }

    /// `calculate_positions`: even-gap target positions for
    /// `[start_vertex, end_vertex)` given `gaps` free cells and
    /// `total_degree` used cells (both inclusive of sentinels). Returns a
    /// vector indexed by absolute vertex id (only the window filled, which
    /// matches the C++ calloc layout).
    fn calculate_positions(
        &self,
        start_vertex: i32,
        end_vertex: i32,
        gaps: i64,
        total_degree: i64,
    ) -> Vec<i64> {
        // Window-sized allocation (matches the C++ `calloc(size)`; callers
        // index window-relative via `(v - start_vertex)`).
        let size = (end_vertex - start_vertex) as usize;
        let mut new_index = vec![0i64; size];
        let total_degree = total_degree + size as i64;

        let index_boundary = if end_vertex >= self.num_vertices {
            self.elem_capacity
        } else {
            self.vertices[end_vertex as usize].index
        };

        let mut index_d = self.vertices[start_vertex as usize].index as f64;
        let mut step = gaps as f64 / total_degree as f64; // per-edge step
        let precision_pos = 100_000_000.0;
        debug_assert!(
            (step * precision_pos) as i64 > 0,
            "fixed-precision is going to cause problem!"
        );
        step = ((step * precision_pos) as i64) as f64 / precision_pos;

        for i in start_vertex..end_vertex {
            let deg = self.vertices[i as usize].degree as i64;
            new_index[(i - start_vertex) as usize] = index_d as i64;
            debug_assert!(
                new_index[(i - start_vertex) as usize] + deg <= index_boundary,
                "index calculation wrong at v={i}: pos={} deg={} boundary={} num_edges={} gaps={} cap={}",
                new_index[(i - start_vertex) as usize],
                deg,
                index_boundary,
                self.num_edges,
                gaps,
                self.elem_capacity
            );
            index_d = new_index[(i - start_vertex) as usize] as f64;
            index_d += (deg as f64) + (step * (deg + 1) as f64);
        }
        new_index
    }

    // ------------------------------------------------------------------
    // PMA tree counters
    // ------------------------------------------------------------------

    /// `recount_segment_actual`: recompute leaf `actual` counts (and
    /// `num_edges`) from vertex degrees.
    pub fn recount_segment_actual(&mut self) {
        self.segment_edges_actual = vec![0; (self.segment_count * 2) as usize];
        self.num_edges = 0;
        for seg_id in 0..self.segment_count as usize {
            let start_vertex = seg_id * self.segment_size as usize;
            let end_vertex =
                ((seg_id + 1) * self.segment_size as usize).min(self.num_vertices as usize);
            let mut actual = 0i64;
            for vid in start_vertex..end_vertex {
                actual += self.vertices[vid].degree as i64;
                self.num_edges += self.vertices[vid].degree as i64;
            }
            self.segment_edges_actual[seg_id + self.segment_count as usize] = actual;
        }
    }

    /// `recount_segment_total()` over all leaves.
    pub fn recount_segment_total_all(&mut self) {
        for i in 0..self.segment_count as usize {
            let next_starter = if i == self.segment_count as usize - 1 {
                self.elem_capacity
            } else {
                self.vertices[(i + 1) * self.segment_size as usize].index
            };
            let total = next_starter - self.vertices[i * self.segment_size as usize].index;
            self.segment_edges_total[i + self.segment_count as usize] = total;
        }
    }

    /// `recount_segment_total(start_vertex, end_vertex)` over the touched
    /// leaves only. Requires segment-aligned windows (all call sites are).
    fn recount_segment_total_range(&mut self, start_vertex: i32, end_vertex: i32) {
        let start_seg = self.get_segment_id(start_vertex as u32) - self.segment_count;
        let end_seg = self.get_segment_id(end_vertex as u32) - self.segment_count;
        for i in start_seg..end_seg {
            let next_starter = if i == self.segment_count - 1 {
                self.elem_capacity
            } else {
                self.vertices[((i + 1) * self.segment_size as i64) as usize].index
            };
            let total = next_starter - self.vertices[(i * self.segment_size as i64) as usize].index;
            self.segment_edges_total[(i + self.segment_count) as usize] = total;
        }
    }

    // ------------------------------------------------------------------
    // Dynamic insert path
    // ------------------------------------------------------------------

    /// `insert(int32_t src, int32_t dst)`: single dynamic edge insert with
    /// PMA rebalancing.
    pub fn insert_edge(&mut self, src: u32, dst: u32) {
        let current_segment = self.get_segment_id(src);

        // (single-threaded: no leaf lock / condvar)
        self.do_insertion(src, dst, current_segment);

        let leaf = self.segment_edges_actual[current_segment as usize] as f64
            / self.segment_edges_total[current_segment as usize] as f64;
        if leaf < UP_0 {
            return;
        }

        // Walk up the conceptual PMA tree for a window under the height's
        // upper threshold.
        let mut window = current_segment;
        let mut height = 0i32;
        let found = loop {
            window /= 2;
            height += 1;
            let window_size = self.segment_size as i64 * (1i64 << height);
            let left_index = (src as i64 / window_size) * window_size;
            let right_index = (left_index + window_size).min(self.num_vertices as i64);
            let left_segment = self.get_segment_id(left_index as u32) - self.segment_count;
            let right_segment = self.get_segment_id(right_index as u32) - self.segment_count;
            let seg_meta = self.sum_segments(left_segment, right_segment);
            let up_height = UP_0 - (height as f64 * self.delta_up);
            let d = seg_meta.actual as f64 / seg_meta.total as f64;
            let hit = (left_index, right_index, seg_meta);
            if d < up_height || window == 0 {
                break hit;
            }
        };
        let (left_index, right_index, seg_meta) = found;

        if window != 0 {
            // Found a window within threshold: weighted rebalance.
            let leaf_actual = self.segment_edges_actual[current_segment as usize];
            let leaf_total = self.segment_edges_total[current_segment as usize];
            if leaf_actual as f64 / leaf_total as f64 >= UP_0 {
                self.rebalance_weighted(
                    left_index as i32,
                    right_index as i32,
                    seg_meta.actual,
                    seg_meta.total,
                );
            }
        } else {
            // Whole tree too dense: resize the edge array.
            self.resize_v1();
        }
    }

    /// Sum `(actual, total)` over leaf ids `[left_segment, right_segment)`.
    fn sum_segments(&self, left_segment: i64, right_segment: i64) -> SegMeta {
        let mut m = SegMeta::default();
        for seg in left_segment..right_segment {
            let idx = (seg + self.segment_count) as usize;
            m.actual += self.segment_edges_actual[idx];
            m.total += self.segment_edges_total[idx];
        }
        m
    }

    /// `do_insertion`: append on-segment if space remains in the vertex's
    /// region; else (leaf log full → leaf rebalance) write to the log.
    fn do_insertion(&mut self, src: u32, dst: u32, src_segment: i64) {
        let mut loc = self.vertices[src as usize].index + self.vertices[src as usize].degree as i64;

        if self.have_space_onseg(src, loc) {
            self.write_edge(loc, dst as RawEdge);
        } else {
            let seg_log = self.log_seg_index(src_segment);
            if self.log_segment_idx[seg_log] as usize >= MAX_LOG_ENTRIES {
                // Fold the log via a leaf-level rebalance.
                let left_index = (src as i64 / self.segment_size as i64) * self.segment_size as i64;
                let right_index =
                    (left_index + self.segment_size as i64).min(self.num_vertices as i64);
                let actual = self.segment_edges_actual[src_segment as usize];
                let total = self.segment_edges_total[src_segment as usize];
                self.rebalance_weighted(left_index as i32, right_index as i32, actual, total);
            }

            loc = self.vertices[src as usize].index + self.vertices[src as usize].degree as i64;
            if self.have_space_onseg(src, loc) {
                self.write_edge(loc, dst as RawEdge);
            } else {
                self.insert_into_log(seg_log, src, dst);
            }
        }

        self.vertices[src as usize].degree += 1;
        self.segment_edges_actual[src_segment as usize] += 1;
        self.num_edges += 1;
    }

    /// `have_space_onseg`: can cell `loc` be claimed for `src`'s list?
    #[inline]
    fn have_space_onseg(&self, src: u32, loc: i64) -> bool {
        if src as i64 == self.num_vertices as i64 - 1 {
            self.elem_capacity > loc
        } else {
            self.vertices[src as usize + 1].index > loc
        }
    }

    /// `insert_into_log`: append `(src, dst)` to the segment log, linking
    /// the vertex's chain.
    fn insert_into_log(&mut self, seg_log: usize, src: u32, dst: u32) {
        let idx = self.log_segment_idx[seg_log];
        debug_assert!((idx as usize) < MAX_LOG_ENTRIES, "logs are full");
        debug_assert!(
            self.vertices[src as usize].offset < MAX_LOG_ENTRIES as i32,
            "vertex offset beyond log range"
        );
        let entry = LogEntry {
            u: src as i32,
            v: dst as i32,
            prev_offset: self.vertices[src as usize].offset,
        };
        self.write_log_entry(seg_log, idx, &entry);
        self.vertices[src as usize].offset = idx;
        self.log_segment_idx[seg_log] = idx + 1;
    }

    // ------------------------------------------------------------------
    // Rebalance
    // ------------------------------------------------------------------

    /// `rebalance_weighted`: redistribute `[start_vertex, end_vertex)`
    /// within its fixed region `[from, to)` by degree weights.
    /// `used_space` = sum of degrees in the window, `total_space` = region
    /// size. Windows must be segment-aligned (all call sites are).
    pub fn rebalance_weighted(
        &mut self,
        start_vertex: i32,
        end_vertex: i32,
        used_space: i64,
        total_space: i64,
    ) {
        let from = self.vertices[start_vertex as usize].index;
        let to = if end_vertex >= self.num_vertices {
            self.elem_capacity
        } else {
            self.vertices[end_vertex as usize].index
        };
        debug_assert!(
            to > from,
            "Invalid range found while doing weighted rebalance"
        );
        debug_assert_eq!(total_space, to - from, "Segment capacity mismatch");

        let gaps = total_space - used_space;
        let new_index = self.calculate_positions(start_vertex, end_vertex, gaps, used_space);
        self.rebalance_data_v1(start_vertex, end_vertex, &new_index);
        self.recount_segment_total_range(start_vertex, end_vertex);
    }

    /// `rebalance_data_V1` without the undo log. Shifts each vertex's
    /// on-segment cells to `new_index`, folds log chains onto the slab, and
    /// releases the touched segment logs.
    ///
    /// Safety without locks/undo-log: old region boundaries are snapshotted
    /// before any mutation; positions from `calculate_positions` give each
    /// vertex at least its old footprint plus a positive gap, so in-window
    /// moves are right-shifts processed in descending vertex order, each
    /// landing on stale cells only.
    fn rebalance_data_v1(&mut self, start_vertex: i32, end_vertex: i32, new_index: &[i64]) {
        // Snapshot only the rebalance window (C++ reads `vertices_` fresh,
        // but its update order makes stale reads impossible; our descending
        // processing needs the boundary of jj+1 BEFORE jj is rewritten, so
        // a window snapshot is the faithful equivalent without an O(V)
        // copy per rebalance).
        let win = (end_vertex - start_vertex) as usize;
        let mut old_index = vec![0i64; win + 1]; // +1: boundary of end_vertex
        for (k, slot) in old_index.iter_mut().enumerate().take(win) {
            *slot = self.vertices[start_vertex as usize + k].index;
        }
        // The boundary for the last window vertex: old index of end_vertex
        // (or elem_capacity) — captured BEFORE any mutation in this call.
        old_index[win] = if end_vertex >= self.num_vertices {
            self.elem_capacity
        } else {
            self.vertices[end_vertex as usize].index
        };
        let mut old_offsets = vec![0i32; win];
        for (k, slot) in old_offsets.iter_mut().enumerate().take(win) {
            *slot = self.vertices[start_vertex as usize + k].offset;
        }
        let old_degrees_win: Vec<i32> = {
            let mut d = vec![0i32; win];
            for (k, slot) in d.iter_mut().enumerate().take(win) {
                *slot = self.vertices[start_vertex as usize + k].degree;
            }
            d
        };

        let mut curr_vertex = start_vertex as i64;
        while curr_vertex < end_vertex as i64 {
            // First vertex whose new extent fits before the next vertex's
            // (old) region start. The window boundary stands in for
            // `vertices_[end_vertex]` (which the C++ reads out of bounds
            // when `end_vertex == num_vertices`).
            let mut ii = end_vertex as i64 - 1;
            for k in curr_vertex..end_vertex as i64 {
                let ki = (k - start_vertex as i64) as usize;
                // `old_index[win]` stands in for `vertices_[end_vertex].index`
                // (the C++ reads that slot out of bounds when the window
                // ends at `num_vertices`).
                let next_boundary = old_index[ki + 1];
                if new_index[ki] + old_degrees_win[ki] as i64 <= next_boundary {
                    ii = k;
                    break;
                }
            }

            // Shift vertices [curr_vertex, ii] in descending order.
            for jj in (curr_vertex..=ii).rev() {
                let jj_us = (jj - start_vertex as i64) as usize;
                let has_log = old_offsets[jj_us] != -1;
                // on-seg cell count: when a log exists the region is full,
                // so the count is the next vertex's old index minus ours;
                // otherwise it equals the (sentinel-inclusive) degree.
                let onseg_num_edges = if has_log {
                    let next_boundary = if jj >= self.num_vertices as i64 - 1 {
                        self.elem_capacity - 1
                    } else {
                        old_index[jj_us + 1] - 1
                    };
                    next_boundary - old_index[jj_us] + 1
                } else {
                    old_degrees_win[jj_us] as i64
                };

                let new_pos = new_index[jj_us];
                let old_pos = old_index[jj_us];

                if new_pos < old_pos {
                    // on-segment: left-shift (forward copy).
                    self.move_cells(old_pos, onseg_num_edges, new_pos);
                } else if new_pos > old_pos {
                    // on-segment: right-shift (backward copy).
                    self.move_cells(old_pos, onseg_num_edges, new_pos);
                }

                // Fold the log chain onto the slab tail (oldest lowest).
                if has_log {
                    let curr_seg = self.log_seg_index(self.get_segment_id(jj as u32));
                    let mut curr_off = old_offsets[jj_us];
                    let mut write_index = new_pos + old_degrees_win[jj_us] as i64 - 1;
                    while curr_off != -1 {
                        let entry = self.read_log_entry(curr_seg, curr_off);
                        self.write_edge(write_index, entry.v);
                        curr_off = entry.prev_offset;
                        write_index -= 1;
                    }
                }

                self.vertices[jj as usize].index = new_pos;
                self.vertices[jj as usize].offset = -1;
            }
            curr_vertex = ii + 1;
        }

        // Release logs of the touched segments.
        let st_seg = self.get_segment_id(start_vertex as u32);
        let nd_seg = self.get_segment_id(end_vertex as u32);
        for i in st_seg..nd_seg {
            let seg = self.log_seg_index(i);
            self.release_log(seg);
        }
    }

    /// Copy `count` edge cells from `src` to `dst` through `Memory`,
    /// overlap-safe (buffered like `memmove`).
    fn move_cells(&mut self, src: i64, count: i64, dst: i64) {
        if count <= 0 || src == dst {
            return;
        }
        let bytes = (count as usize) * 4;
        if self.scratch.len() < bytes {
            self.scratch.resize(bytes, 0);
        }
        self.memory
            .read(self.edges_base + src as u64 * 4, &mut self.scratch[..bytes]);
        self.memory
            .write(self.edges_base + dst as u64 * 4, &self.scratch[..bytes]);
    }

    // ------------------------------------------------------------------
    // Resize
    // ------------------------------------------------------------------

    /// `resize_V1`: double the edge array, re-spread all vertices, fold all
    /// logs, release all segment logs.
    pub fn resize_v1(&mut self) {
        self.elem_capacity *= 2;

        // Grow backing memory before touching the new tail.
        let need = self.edges_base + self.elem_capacity as u64 * 4;
        let have_pages = self.memory.size().div_ceil(PAGE_SIZE);
        let need_pages = need.div_ceil(PAGE_SIZE);
        if need_pages > have_pages {
            self.memory
                .grow(need_pages - have_pages)
                .expect("resize_v1: memory growth failed");
        }

        let gaps = self.elem_capacity - self.num_edges;
        let new_indices = self.calculate_positions(0, self.num_vertices, gaps, self.num_edges);

        // Snapshot old geometry (ascending update would read new indices of
        // not-yet-copied vertices through `vertices_[vi+1]`... descending
        // with snapshots matches the C++ fresh-array copy semantics).
        let old_index: Vec<i64> = self.vertices.iter().map(|r| r.index).collect();
        let old_offsets: Vec<i32> = self.vertices.iter().map(|r| r.offset).collect();
        let old_degrees: Vec<i32> = self.vertices.iter().map(|r| r.degree).collect();

        for vi in (0..self.num_vertices as usize).rev() {
            let has_log = old_offsets[vi] != -1;
            let onseg_num_edges = if has_log {
                let next_boundary = if vi as i64 >= self.num_vertices as i64 - 1 {
                    self.elem_capacity - 1
                } else {
                    old_index[vi + 1] - 1
                };
                next_boundary - old_index[vi] + 1
            } else {
                old_degrees[vi] as i64
            };

            let write_start = new_indices[vi];
            self.move_cells(old_index[vi], onseg_num_edges, write_start);

            if has_log {
                let curr_seg = self.log_seg_index(self.get_segment_id(vi as u32));
                let mut curr_off = old_offsets[vi];
                let mut write_index = write_start + old_degrees[vi] as i64 - 1;
                while curr_off != -1 {
                    let entry = self.read_log_entry(curr_seg, curr_off);
                    self.write_edge(write_index, entry.v);
                    curr_off = entry.prev_offset;
                    write_index -= 1;
                }
            }

            self.vertices[vi].index = write_start;
            self.vertices[vi].offset = -1;
        }

        self.recount_segment_total_all();
        for seg in 0..self.segment_count as usize {
            self.release_log(seg);
        }
    }

    // ------------------------------------------------------------------
    // Scan
    // ------------------------------------------------------------------

    /// `out_neigh`: iterate `v`'s neighbors in DGAP scan order — on-segment
    /// cells ascending, then log entries newest-to-oldest. Skips the
    /// sentinel.
    pub fn out_neighbors(&self, v: u32) -> DgapNeighborhood<'_, V, M> {
        let row = self.vertices[v as usize];
        let onseg_edges = if row.offset != -1 {
            let next_boundary = if v as i64 >= self.num_vertices as i64 - 1 {
                self.elem_capacity - 1
            } else {
                self.vertices[v as usize + 1].index - 1
            };
            next_boundary - row.index + 1
        } else {
            row.degree as i64
        };
        DgapNeighborhood {
            graph: self,
            vertex: v,
            row,
            onseg_edges,
            pos: 1, // skip sentinel
            onlog: false,
            log_index: row.offset,
            chunk: std::mem::MaybeUninit::uninit(),
            chunk_off: 0,
            chunk_len: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Collect neighbors into a Vec (convenience for tests).
    pub fn out_neighbors_vec(&self, v: u32) -> Vec<V> {
        self.out_neighbors(v).collect()
    }

    /// Visits `v`'s neighbors in DGAP scan order (on-seg ascending, then
    /// log newest-first) via a callback, bulk-reading the contiguous on-seg
    /// run in [`SCAN_CHUNK_BYTES`] chunks.
    ///
    /// This is the scan path the benchmark harness uses. It avoids the
    /// per-vertex iterator construction of [`DgapGraph::out_neighbors`]
    /// (which is kept for API parity with the C++ `Neighborhood`), so the
    /// DGAP-vs-LARA comparison measures the scan algorithm, not iterator
    /// setup. The chunked read is the same `SCAN_CHUNK_BYTES` pattern LARA
    /// uses and only cancels the port's per-edge `Memory::read` overhead
    /// (see [`SCAN_CHUNK_BYTES`] docs).
    pub fn for_each_neighbor<F: FnMut(V)>(&self, v: u32, mut f: F) {
        let row = self.vertices[v as usize];
        let onseg_edges = if row.offset != -1 {
            let next_boundary = if v as i64 >= self.num_vertices as i64 - 1 {
                self.elem_capacity - 1
            } else {
                self.vertices[v as usize + 1].index - 1
            };
            next_boundary - row.index + 1
        } else {
            row.degree as i64
        };

        // Chunked on-seg scan (skip the sentinel at `row.index`).
        let mut pos = 1i64;
        let mut chunk = [0u8; SCAN_CHUNK_BYTES];
        while pos < onseg_edges {
            let start = row.index + pos;
            let end = (start + (SCAN_CHUNK_BYTES / 4) as i64).min(row.index + onseg_edges);
            let n = (end - start) as usize;
            self.memory
                .read(self.edges_base + start as u64 * 4, &mut chunk[..n * 4]);
            for k in 0..n {
                let e = i32::from_le_bytes(chunk[k * 4..k * 4 + 4].try_into().unwrap());
                f(V::from(e as u32));
            }
            pos += n as i64;
        }

        // Log chain (newest → oldest).
        if row.offset != -1 && row.degree as i64 > onseg_edges {
            let seg_log = self.log_seg_index(self.get_segment_id(v));
            let mut cur = row.offset;
            while cur != -1 {
                let entry = self.read_log_entry(seg_log, cur);
                f(V::from(entry.v as u32));
                cur = entry.prev_offset;
            }
        }
    }

    /// Full-graph scan: visit every edge exactly once (bench helper).
    pub fn scan_all<F: FnMut(V)>(&self, mut f: F) {
        for v in 0..self.num_vertices as u32 {
            for n in self.out_neighbors(v) {
                f(n);
            }
        }
    }

    // ------------------------------------------------------------------
    // remove_edge (extension; graph.h is insert-only)
    // ------------------------------------------------------------------

    /// Removes edge `(src, dst)`. Log hits unlink the chain node; on-seg
    /// hits shift the list tail left by one. When the vertex keeps a log
    /// chain, a leaf rebalance afterwards restores the `has_log ⟹ region
    /// full` invariant. Returns `true` if removed.
    pub fn remove_edge(&mut self, src: u32, dst: u32) -> bool {
        let row = self.vertices[src as usize];
        let seg = self.get_segment_id(src);
        let seg_log = self.log_seg_index(seg);

        // Search the log chain first (newest → oldest).
        if row.offset != -1 {
            let mut prev_idx: i32 = -1;
            let mut cur = row.offset;
            while cur != -1 {
                let entry = self.read_log_entry(seg_log, cur);
                if entry.v == dst as i32 {
                    if prev_idx == -1 {
                        self.vertices[src as usize].offset = entry.prev_offset;
                    } else {
                        let mut p = self.read_log_entry(seg_log, prev_idx);
                        p.prev_offset = entry.prev_offset;
                        self.write_log_entry(seg_log, prev_idx, &p);
                    }
                    self.write_log_entry(seg_log, cur, &LogEntry::default());
                    self.vertices[src as usize].degree -= 1;
                    self.segment_edges_actual[seg as usize] -= 1;
                    self.num_edges -= 1;
                    return true;
                }
                prev_idx = cur;
                cur = entry.prev_offset;
            }
        }

        // Search the on-segment run.
        let has_log = row.offset != -1;
        let onseg = if has_log {
            let next_boundary = if src as i64 >= self.num_vertices as i64 - 1 {
                self.elem_capacity - 1
            } else {
                self.vertices[src as usize + 1].index - 1
            };
            next_boundary - row.index + 1
        } else {
            row.degree as i64
        };
        let run_start = row.index + 1; // skip sentinel
        let run_end = row.index + onseg; // exclusive
        for i in run_start..run_end {
            if self.read_edge(i) == dst as RawEdge {
                if i + 1 < run_end {
                    self.move_cells(i + 1, run_end - (i + 1), i);
                }
                self.vertices[src as usize].degree -= 1;
                self.segment_edges_actual[seg as usize] -= 1;
                self.num_edges -= 1;
                if has_log {
                    // Restore `has_log ⟹ region full` by folding the chain.
                    let left = (seg - self.segment_count) * self.segment_size as i64;
                    let right = (left + self.segment_size as i64).min(self.num_vertices as i64);
                    let actual = self.segment_edges_actual[seg as usize];
                    let total = self.segment_edges_total[seg as usize];
                    self.rebalance_weighted(left as i32, right as i32, actual, total);
                }
                return true;
            }
        }
        false
    }

    // ------------------------------------------------------------------
    // Debug integrity checks
    // ------------------------------------------------------------------

    /// Checks region boundaries, sentinels, and PMA counters.
    pub fn verify_invariants(&self) {
        for v in 0..self.num_vertices as usize {
            let row = self.vertices[v];
            assert!(row.index >= 0 && row.index < self.elem_capacity);
            assert!(row.degree >= 1, "every vertex holds a sentinel");
            let end = if v as i64 == self.num_vertices as i64 - 1 {
                self.elem_capacity
            } else {
                self.vertices[v + 1].index
            };
            // `degree` counts log-backed edges; the on-slab region only
            // holds min(degree, region width) cells. A log-backed vertex's
            // region is packed full (that is why it spilled to the log).
            let onseg = if row.offset != -1 {
                end - row.index
            } else {
                row.degree as i64
            };
            assert!(
                row.index + onseg <= end,
                "vertex {v} region overflow: idx={} deg={} next={}",
                row.index,
                row.degree,
                end
            );
            let sent = self.read_edge(row.index);
            assert_eq!(sent, -(v as i64) as RawEdge, "sentinel missing at {v}");
            if row.offset == -1 {
                for i in (row.index + 1)..(row.index + row.degree as i64) {
                    let e = self.read_edge(i);
                    assert!(e >= 0, "negative cell {e} in packed list of {v}");
                }
            }
        }
        let mut total_actual = 0i64;
        for seg in 0..self.segment_count {
            let s = (seg * self.segment_size as i64) as usize;
            let e =
                (((seg + 1) * self.segment_size as i64) as usize).min(self.num_vertices as usize);
            let actual: i64 = self.vertices[s..e].iter().map(|r| r.degree as i64).sum();
            assert_eq!(
                actual,
                self.segment_edges_actual[(seg + self.segment_count) as usize],
                "leaf {seg} actual mismatch"
            );
            total_actual += actual;
        }
        assert_eq!(total_actual, self.num_edges, "num_edges mismatch");
    }
}

/// Iterator over one vertex's neighborhood (`Neighborhood` in `graph.h`):
/// on-segment neighbors ascending, then log neighbors newest-first.
pub struct DgapNeighborhood<'a, V, M: Memory> {
    graph: &'a DgapGraph<V, M>,
    vertex: u32,
    row: VertexElement,
    onseg_edges: i64,
    pos: i64,
    onlog: bool,
    log_index: i32,
    // Chunk buffer for on-seg bulk reads (see [`SCAN_CHUNK_BYTES`]).
    // `MaybeUninit` avoids a per-vertex 512-byte zero-fill: the buffer is
    // written on first refill and only ever read over the initialized
    // prefix (`chunk_len * 4` bytes).
    chunk: std::mem::MaybeUninit<[u8; SCAN_CHUNK_BYTES]>,
    /// Next slot index within `chunk` to decode.
    chunk_off: usize,
    /// Number of valid slots currently buffered in `chunk`.
    chunk_len: usize,
    _marker: std::marker::PhantomData<V>,
}

impl<V: From<u32> + Copy, M: Memory> Iterator for DgapNeighborhood<'_, V, M> {
    type Item = V;

    fn next(&mut self) -> Option<V> {
        if !self.onlog {
            if self.pos < self.onseg_edges {
                // Refill the chunk when exhausted: bulk-read up to
                // `SCAN_CHUNK_BYTES / 4` contiguous on-seg cells in one
                // `Memory::read`, then decode in memory. The on-seg run is
                // contiguous per vertex, so this is a faithful equivalent of
                // the C++ pointer walk with the per-edge `Memory::read`
                // overhead removed (see [`SCAN_CHUNK_BYTES`] docs).
                if self.chunk_off >= self.chunk_len {
                    let start = self.row.index + self.pos;
                    let end = (start + (SCAN_CHUNK_BYTES / 4) as i64)
                        .min(self.row.index + self.onseg_edges);
                    let n = (end - start) as usize;
                    // SAFETY: `n * 4` bytes are written by `Memory::read`
                    // before any read of the buffer.
                    let buf: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(self.chunk.as_mut_ptr() as *mut u8, n * 4)
                    };
                    self.graph
                        .memory
                        .read(self.graph.edges_base + start as u64 * 4, buf);
                    self.chunk_off = 0;
                    self.chunk_len = n;
                }
                // SAFETY: the reference covers only the initialized prefix
                // (`chunk_len * 4` bytes written by the refill above).
                let buf: &[u8] = unsafe {
                    std::slice::from_raw_parts(self.chunk.as_ptr() as *const u8, self.chunk_len * 4)
                };
                let e = i32::from_le_bytes(
                    buf[self.chunk_off * 4..self.chunk_off * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                self.chunk_off += 1;
                self.pos += 1;
                return Some(V::from(e as u32));
            }
            if self.row.offset != -1 && self.row.degree as i64 > self.onseg_edges {
                self.onlog = true; // log_index already == row.offset (head)
            } else {
                return None;
            }
        }
        if self.log_index == -1 {
            return None;
        }
        let seg_log = self
            .graph
            .log_seg_index(self.graph.get_segment_id(self.vertex));
        let entry = self.graph.read_log_entry(seg_log, self.log_index);
        self.log_index = entry.prev_offset;
        Some(V::from(entry.v as u32))
    }
}
