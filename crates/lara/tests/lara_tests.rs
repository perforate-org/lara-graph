//! Behavioral tests for the generalized LARA graph.

use lara_graph::graph::{
    CAP_TREE, EDGE_TOMBSTONE, Edge, LaraGraph, LaraMemories, MAX_LOG_ENTRIES, T_DEMOTE, T_PROMOTE,
    Vertex,
};
use lara_graph::memory::VecMemory;
use lara_graph::traits::{CsrEdge, CsrVertex, NO_BUCKET};
use lara_graph::tree_csr::{B, R_MAX, root_len};

fn fresh_mem() -> LaraMemories<VecMemory> {
    LaraMemories {
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

fn fresh_labeled_mem() -> LaraMemories<VecMemory> {
    fresh_mem()
}

fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v
}

#[test]
fn bulk_build_and_scan() {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for i in 0..9u32 {
        edges.push((i, i + 1));
        edges.push((i + 1, i));
    }
    edges.sort_by_key(|&(u, _)| u);

    let g: LaraGraph<Vertex, Edge, VecMemory> =
        LaraGraph::with_edge_list(fresh_mem(), &edges, 10).unwrap();
    assert_eq!(g.num_nodes(), 10);
    assert_eq!(g.num_edges(), 18);
    for v in 0..10u32 {
        let expect: Vec<u32> = match v {
            0 => vec![1],
            9 => vec![8],
            x => vec![x - 1, x + 1],
        };
        assert_eq!(sorted(g.out_neighbors_vec(v)), expect, "vertex {v}");
    }
    g.verify_invariants();
}

#[test]
fn dynamic_insert_scan_remove() {
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 64).unwrap();
    for d in 0..100u32 {
        g.insert_edge(7, d);
    }
    assert_eq!(g.out_degree(7), 100);
    assert_eq!(
        sorted(g.out_neighbors_vec(7)),
        (0..100).collect::<Vec<u32>>()
    );
    g.verify_invariants();

    assert!(g.remove_edge(7, 50));
    assert!(!g.remove_edge(7, 50));
    assert_eq!(g.out_degree(7), 99);
    assert_eq!(g.num_edges(), 99);
    let neighbors = g.out_neighbors_vec(7);
    assert!(!neighbors.contains(&50));
    g.verify_invariants();
}

#[test]
fn leaf_rebalance_and_free_span_reuse() {
    // Many inserts into one leaf drive density ≥ 1.0 → rebalance → old span
    // released to the free-span store; later leaf growth reuses it.
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 16).unwrap();
    let (span_count_before, _, _) = g.free_span_stats();
    for d in 0..300u32 {
        g.insert_edge(3, d);
    }
    assert_eq!(g.out_degree(3), 300);
    assert_eq!(
        sorted(g.out_neighbors_vec(3)),
        (0..300).collect::<Vec<u32>>()
    );
    g.verify_invariants();
    // Rebalances happened: the free-span store saw released spans.
    let (_, free_slots, _) = g.free_span_stats();
    assert!(
        free_slots > 0 || g.slab_capacity() > 1024,
        "expected freed spans after rebalance (before={span_count_before})"
    );

    // Deletes compact the leaf back down.
    for d in 0..250u32 {
        assert!(g.remove_edge(3, d));
    }
    assert_eq!(g.out_degree(3), 50);
    g.verify_invariants();
}

#[test]
fn tree_csr_promotion_at_threshold() {
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 8).unwrap();
    for d in 0u32..(T_PROMOTE + 16) {
        g.insert_edge(2, d);
    }
    assert_eq!(g.out_degree(2), (T_PROMOTE + 16) as u64);
    assert!(
        g.tree_bucket_count() >= 1,
        "expected promotion at T_PROMOTE"
    );
    assert_eq!(
        sorted(g.out_neighbors_vec(2)),
        (0..T_PROMOTE + 16).collect::<Vec<u32>>()
    );
    g.verify_invariants();
    // Tree delete path.
    assert!(g.remove_edge(2, 3));
    assert_eq!(g.out_degree(2), (T_PROMOTE + 15) as u64);
    g.verify_invariants();
}

#[test]
fn scan_all_counts_every_edge() {
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 32).unwrap();
    for v in 0..30u32 {
        for d in 0..5u32 {
            g.insert_edge(v, (v * 7 + d) % 32);
        }
    }
    let mut count = 0u64;
    g.scan_all(|_| count += 1);
    assert_eq!(count, g.num_edges());
}

#[test]
fn labeled_graph_roundtrip() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};

    let mut edges: Vec<(u32, u32, LabelId)> = Vec::new();
    for i in 0..9u32 {
        edges.push((i, i + 1, LabelId::new(1)));
        edges.push((i + 1, i, LabelId::new(2)));
    }
    edges.sort_by_key(|&(u, _, _)| u);
    let mut g: LabeledLaraGraph<VecMemory> =
        LabeledLaraGraph::with_edge_list(fresh_labeled_mem(), &edges, 10).unwrap();

    assert_eq!(g.num_edges(), 18);
    let got = sorted(g.out_labeled_vec(4).iter().map(|&(n, _)| n).collect());
    assert_eq!(got, vec![3, 5]);

    // Label-filtered scan.
    let mut label1 = Vec::new();
    g.for_each_labeled(4, |n, l| {
        if l == LabelId::new(1) {
            label1.push(n)
        }
    });
    assert_eq!(label1, vec![5]);

    // Labeled remove.
    assert!(g.remove_edge(4, 5, LabelId::new(1)));
    assert!(!g.remove_edge(4, 5, LabelId::new(1)));
    assert_eq!(g.out_degree(4), 1);
}

#[test]
fn labeled_dynamic_inserts() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    let mut g: LabeledLaraGraph<VecMemory> =
        LabeledLaraGraph::new(fresh_labeled_mem(), 16).unwrap();
    for d in 0..40u32 {
        g.insert_edge(5, d, LabelId::new((d % 3) as u16));
    }
    assert_eq!(g.out_degree(5), 40);
    let mut per_label = [0usize; 3];
    g.for_each_labeled(5, |_, l| per_label[l.as_u16() as usize] += 1);
    assert_eq!(per_label, [14, 13, 13]);
    assert!(g.remove_edge(5, 10, LabelId::new(1)));
    assert_eq!(g.out_degree(5), 39);
}

#[test]
fn label_id_roundtrip() {
    use lara_graph::labeled::{Label, LabelId};
    for raw in [0u16, 1, 7, 65_535] {
        let id = LabelId::new(raw);
        assert_eq!(id.as_u16(), raw);
        assert_eq!(LabelId::from_id(id), id);
    }
    assert_eq!(LabelId::MAX_COUNT, 65_536);
}

/// A user-defined `Copy` enum label: the graph must hand back the same
/// enum it was given (zero-cost conversion at the API boundary).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rel {
    Follows,
    Blocks,
}

impl lara_graph::labeled::Label for Rel {
    fn into_id(self) -> lara_graph::labeled::LabelId {
        lara_graph::labeled::LabelId::new(match self {
            Rel::Follows => 0,
            Rel::Blocks => 1,
        })
    }
    fn from_id(id: lara_graph::labeled::LabelId) -> Self {
        match id.as_u16() {
            0 => Rel::Follows,
            _ => Rel::Blocks,
        }
    }
}

#[test]
fn user_defined_enum_label_roundtrip() {
    use lara_graph::labeled::LabeledLaraGraph;
    // Bulk build lays out per-vertex base slots; dynamic insert appends at
    // `base + stored` and needs that geometry.
    let edges = vec![
        (0u32, 1u32, Rel::Follows),
        (0, 2, Rel::Blocks),
        (1, 3, Rel::Follows),
    ];
    let mut g: LabeledLaraGraph<VecMemory, Rel> =
        LabeledLaraGraph::with_edge_list(fresh_labeled_mem(), &edges, 4).unwrap();

    let mut follows = Vec::new();
    let mut blocks = Vec::new();
    g.for_each_labeled(0, |n, l| match l {
        Rel::Follows => follows.push(n),
        Rel::Blocks => blocks.push(n),
    });
    assert_eq!(follows, vec![1]);
    assert_eq!(blocks, vec![2]);

    // Remove by enum label.
    assert!(g.remove_edge(0, 2, Rel::Blocks));
    assert!(!g.remove_edge(0, 2, Rel::Blocks));
    assert_eq!(g.out_degree(0), 1);
}

#[test]
fn label_catalog_register_lookup() {
    use lara_graph::catalog::LabelCatalog;
    use lara_graph::labeled::LabelId;

    let mut cat = LabelCatalog::new();
    let a = cat.register("FOLLOWS").unwrap();
    let b = cat.register("BLOCKS").unwrap();
    assert_eq!(a, LabelId::new(0));
    assert_eq!(b, LabelId::new(1));
    assert_eq!(cat.len(), 2);

    // Same name → same id.
    assert_eq!(cat.register("FOLLOWS").unwrap(), a);
    assert_eq!(cat.len(), 2);

    // name_of / lookup roundtrip.
    assert_eq!(cat.name_of(a), Some("FOLLOWS"));
    assert_eq!(cat.name_of(LabelId::new(9)), None);
    assert_eq!(cat.lookup("BLOCKS"), Some(b));
    assert_eq!(cat.lookup("MISSING"), None);
}

#[test]
fn label_catalog_limit() {
    use lara_graph::catalog::{LabelCatalog, LabelLimit};
    // Small capacity exercises the limit without 65,536 registrations.
    let mut cat = LabelCatalog::with_capacity(3);
    assert_eq!(
        cat.register("a").unwrap(),
        lara_graph::labeled::LabelId::new(0)
    );
    assert_eq!(
        cat.register("b").unwrap(),
        lara_graph::labeled::LabelId::new(1)
    );
    assert_eq!(
        cat.register("c").unwrap(),
        lara_graph::labeled::LabelId::new(2)
    );
    assert_eq!(cat.register("d"), Err(LabelLimit));
    assert_eq!(cat.len(), 3);
}

// -- derive(Label) ---------------------------------------------------------

/// A fieldless enum deriving `Label` (declaration-order numbering).
#[derive(Copy, Clone, Debug, PartialEq, Eq, lara_graph::Label)]
enum EdgeKind {
    Follows,
    Mentions,
    Blocks,
}

#[test]
fn derive_roundtrip() {
    use lara_graph::Label;
    use lara_graph::labeled::LabelId;

    // Declaration-order ids: Follows=0, Mentions=1, Blocks=2.
    assert_eq!(EdgeKind::Follows.into_id(), LabelId::new(0));
    assert_eq!(EdgeKind::Mentions.into_id(), LabelId::new(1));
    assert_eq!(EdgeKind::Blocks.into_id(), LabelId::new(2));

    // Roundtrip.
    assert_eq!(EdgeKind::from_id(LabelId::new(0)), EdgeKind::Follows);
    assert_eq!(EdgeKind::from_id(LabelId::new(1)), EdgeKind::Mentions);
    assert_eq!(EdgeKind::from_id(LabelId::new(2)), EdgeKind::Blocks);
}

#[test]
fn derive_with_graph() {
    use lara_graph::labeled::LabeledLaraGraph;
    let edges = vec![
        (0u32, 1u32, EdgeKind::Follows),
        (0, 2, EdgeKind::Blocks),
        (1, 3, EdgeKind::Mentions),
    ];
    let mut g: LabeledLaraGraph<VecMemory, EdgeKind> =
        LabeledLaraGraph::with_edge_list(fresh_labeled_mem(), &edges, 4).unwrap();

    let mut follows = Vec::new();
    let mut blocks = Vec::new();
    g.for_each_labeled(0, |n, l| match l {
        EdgeKind::Follows => follows.push(n),
        EdgeKind::Mentions => {}
        EdgeKind::Blocks => blocks.push(n),
    });
    assert_eq!(follows, vec![1]);
    assert_eq!(blocks, vec![2]);

    // Remove by derived enum label.
    assert!(g.remove_edge(0, 2, EdgeKind::Blocks));
    assert_eq!(g.out_degree(0), 1);
}

#[test]
fn tree_csr_bucket_unit() {
    use lara_graph::block_store::LtbRawBlockStore;
    use lara_graph::tree_csr::TreeCsrBucket;

    let mut b = TreeCsrBucket::new(LtbRawBlockStore::new(VecMemory::new()).unwrap());
    for s in 0..2500u32 {
        b.insert(s + 1);
    }
    assert_eq!(b.stored_slots(), 2500);
    assert_eq!(b.root_len(), 3);
    assert_eq!(b.read_slot(1024), 1025);
    assert_eq!(b.range_target(2500), None);
    assert_eq!(b.remove_at(0), Some(1));
    assert_eq!(b.read_slot(0), 2);
    assert_eq!(b.stored_slots(), 2499);
    assert_eq!(b.root_len(), root_len_for(2499));
}

fn root_len_for(n: u32) -> usize {
    (n as usize).div_ceil(lara_graph::tree_csr::B)
}

#[test]
fn free_span_best_fit_and_coalesce() {
    use lara_graph::free_span::FreeSpanStore;
    let mut fs = FreeSpanStore::new(VecMemory::new()).unwrap();
    fs.release(100, 10).unwrap();
    fs.release(200, 20).unwrap();
    fs.release(300, 30).unwrap();
    // Best-fit for 15 → the 20-slot span at 200.
    assert_eq!(fs.alloc(15), Some(200));
    // Best-fit for 5 → the exact-fit remainder of the split 20-slot span.
    assert_eq!(fs.alloc(5), Some(215));
    // Adjacent releases coalesce: 300+30 with 320+40.
    fs.release(330, 40).unwrap();
    let spans = fs.spans();
    assert!(spans.contains(&lara_graph::free_span::FreeSpan {
        start: 300,
        len: 70
    }));
    assert_eq!(fs.stats().2, 70); // largest
}

// -- insert-only (core) path ----------------------------------------------

/// A `CsrEdge`-only edge type (no [`CsrEdgeTombstone`]): exercises the core
/// `E: CsrEdge` impl block, which must scan every record as live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
struct PlainEdge(u32);

impl lara_graph::traits::FixedWidth for PlainEdge {
    const BYTES: usize = 4;
    fn read_from(bytes: &[u8]) -> Self {
        Self(u32::from_le_bytes(bytes[0..4].try_into().unwrap()))
    }
    fn write_to(&self, bytes: &mut [u8]) {
        bytes[0..4].copy_from_slice(&self.0.to_le_bytes());
    }
}

impl lara_graph::traits::CsrEdge for PlainEdge {
    fn neighbor(&self) -> u32 {
        self.0
    }
    fn with_neighbor(mut self, neighbor: u32) -> Self {
        self.0 = neighbor;
        self
    }
}

#[test]
fn insert_only_raw_scan_returns_all_records() {
    // Build an insert-only graph whose edge type is `CsrEdge` only (no
    // tombstone capability). The core block's `for_each_record_in_row`
    // must yield every record as live.
    let edges: Vec<(u32, u32)> = vec![(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (2, 6)];
    let g: LaraGraph<Vertex, PlainEdge, VecMemory> =
        LaraGraph::with_edge_list(fresh_mem(), &edges, 8).unwrap();

    let mut seen: Vec<u32> = Vec::new();
    for v in 0..8u32 {
        let row = g.get_row_pub(v);
        g.for_each_record_in_row(row.base_slot_start(), row.stored_degree(), |e| {
            seen.push(e.neighbor());
        });
    }
    assert_eq!(
        sorted(seen),
        sorted(edges.iter().map(|&(_, d)| d).collect())
    );
    assert_eq!(g.num_edges(), edges.len() as u64);
    assert_eq!(g.out_degree(0), 3);
    assert_eq!(g.out_degree(2), 2);
}

// -- record-preserving rebalance ------------------------------------------

/// 8-byte edge with an arbitrary payload (not a label): exercises the
/// record-preserving rebalance for wide edge types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
struct WideEdge {
    neighbor: u32,
    payload: u32,
}

impl lara_graph::traits::FixedWidth for WideEdge {
    const BYTES: usize = 8;
    fn read_from(bytes: &[u8]) -> Self {
        Self {
            neighbor: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            payload: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        }
    }
    fn write_to(&self, bytes: &mut [u8]) {
        bytes[0..4].copy_from_slice(&self.neighbor.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.payload.to_le_bytes());
    }
}

impl lara_graph::traits::CsrEdge for WideEdge {
    fn neighbor(&self) -> u32 {
        self.neighbor
    }
    fn with_neighbor(mut self, neighbor: u32) -> Self {
        self.neighbor = neighbor;
        self
    }
}

impl lara_graph::traits::CsrEdgeTombstone for WideEdge {
    fn tombstone_edge() -> Self {
        Self {
            neighbor: EDGE_TOMBSTONE,
            payload: 0,
        }
    }
    fn is_tombstone(&self) -> bool {
        self.neighbor == EDGE_TOMBSTONE
    }
}

#[test]
fn rebalance_preserves_labeled_records() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};

    // Bulk build with labels (lays out base slots), then dynamic inserts
    // with distinct labels. Vertex 4 lands in leaf 0 (segment_size = 5).
    let mut edges: Vec<(u32, u32, LabelId)> = Vec::new();
    for i in 0..9u32 {
        edges.push((i, i + 1, LabelId::new(1)));
        edges.push((i + 1, i, LabelId::new(2)));
    }
    edges.sort_by_key(|&(u, _, _)| u);
    let mut g: LabeledLaraGraph<VecMemory> =
        LabeledLaraGraph::with_edge_list(fresh_labeled_mem(), &edges, 10).unwrap();

    for d in 0..8u32 {
        g.insert_edge(4, 100 + d, LabelId::new((d % 3) as u16));
    }

    let mut before = g.out_labeled_vec(4);
    before.sort();

    // Explicit rebalance of leaf 0 (vertices 0..5, incl. vertex 4).
    g.rebalance_leaf(0);

    let mut after = g.out_labeled_vec(4);
    after.sort();
    assert_eq!(before, after, "labels must survive rebalance verbatim");
}

#[test]
fn rebalance_preserves_wide_records() {
    // 8-byte edge with an arbitrary payload: rebalance must copy records
    // verbatim so the payload survives (not just the neighbor).
    let edges: Vec<(u32, u32)> = vec![(4, 1), (4, 2), (4, 3), (4, 4), (4, 5)];
    let mut g: LaraGraph<Vertex, WideEdge, VecMemory> =
        LaraGraph::with_edge_list(fresh_mem(), &edges, 10).unwrap();

    // Overwrite vertex 4's records with payload-carrying wide edges.
    let row = g.get_row_pub(4);
    for k in 0..row.stored_degree() {
        let e = WideEdge {
            neighbor: 100 + k,
            payload: k * 7 + 1,
        };
        g.write_edge_pub(row.base_slot_start() + k as u64, &e);
    }

    let row = g.get_row_pub(4);
    let mut before: Vec<(u32, u32)> = Vec::new();
    g.for_each_record_in_row(row.base_slot_start(), row.stored_degree(), |e| {
        before.push((e.neighbor, e.payload));
    });
    before.sort();

    g.rebalance_leaf(0);

    let row = g.get_row_pub(4);
    let mut after: Vec<(u32, u32)> = Vec::new();
    g.for_each_record_in_row(row.base_slot_start(), row.stored_degree(), |e| {
        after.push((e.neighbor, e.payload));
    });
    after.sort();
    assert_eq!(before, after, "payloads must survive rebalance verbatim");
    g.verify_invariants();
}

// -- Phase 2: labeled veneer over core typed operations --------------------

#[test]
fn labeled_dynamic_insert_respects_boundary() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};

    // Dense leaf: vertex 0 gets a large degree so its region fills fast;
    // vertex 1's base sits right after vertex 0's region. Without the
    // core boundary check, dynamic inserts into vertex 0 would overwrite
    // vertex 1's first edge.
    let mut edges: Vec<(u32, u32, LabelId)> = Vec::new();
    for d in 0..12u32 {
        edges.push((0, 100 + d, LabelId::new(1)));
    }
    edges.push((1, 200, LabelId::new(2)));
    edges.push((1, 201, LabelId::new(2)));
    edges.sort_by_key(|&(u, _, _)| u);
    let mut g: LabeledLaraGraph<VecMemory> =
        LabeledLaraGraph::with_edge_list(fresh_labeled_mem(), &edges, 8).unwrap();

    let v1_before = g.out_labeled_vec(1);

    // Dynamically insert many edges into vertex 0 (fills its region and
    // would overflow into vertex 1 without the boundary check).
    for d in 0..200u32 {
        g.insert_edge(0, 1000 + d, LabelId::new((d % 3) as u16));
    }

    // Vertex 1's adjacency must be intact.
    assert_eq!(g.out_labeled_vec(1), v1_before, "neighbor vertex corrupted");
    assert_eq!(g.out_degree(0), 12 + 200);
    g.verify_invariants();
}

#[test]
fn labeled_dynamic_remove_updates_accounting() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    let mut g: LabeledLaraGraph<VecMemory> =
        LabeledLaraGraph::new(fresh_labeled_mem(), 16).unwrap();
    for d in 0..40u32 {
        g.insert_edge(5, d, LabelId::new((d % 3) as u16));
    }
    assert_eq!(g.num_edges(), 40);
    assert_eq!(g.out_degree(5), 40);

    // Remove every even edge.
    for d in (0..40u32).step_by(2) {
        assert!(g.remove_edge(5, d, LabelId::new((d % 3) as u16)));
    }
    assert_eq!(g.num_edges(), 20);
    assert_eq!(g.out_degree(5), 20);
    g.verify_invariants();

    // Remaining edges are the odd ones.
    let mut remaining: Vec<u32> = g.out_labeled_vec(5).iter().map(|&(n, _)| n).collect();
    remaining.sort();
    assert_eq!(remaining, (1..40).step_by(2).collect::<Vec<u32>>());
}

#[test]
fn labeled_log_overflow_folds_via_rebalance() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // Dense multi-leaf churn: enough inserts that, were the leaf overflow
    // log an unbounded fallback, it would blow past `MAX_LOG_ENTRIES` and
    // silently corrupt the neighboring leaf's log region. The guard must
    // fold the log (via a leaf rebalance) instead, keeping the cursor
    // bounded and every label intact.
    let mut g: LabeledLaraGraph<VecMemory> =
        LabeledLaraGraph::new(fresh_labeled_mem(), 16).unwrap();
    for d in 0..400u32 {
        g.insert_edge(5, 1000 + d, LabelId::new((d % 3) as u16));
        g.insert_edge(9, 2000 + d, LabelId::new((d % 2) as u16));
    }
    let ss = g.segment_size();
    let leaf5 = 5 / ss;
    let leaf9 = 9 / ss;
    assert!(
        (g.leaf_log_idx(leaf5 as u64) as usize) < MAX_LOG_ENTRIES,
        "leaf {leaf5} log overflowed"
    );
    assert!(
        (g.leaf_log_idx(leaf9 as u64) as usize) < MAX_LOG_ENTRIES,
        "leaf {leaf9} log overflowed"
    );
    g.verify_invariants();
    assert_eq!(g.out_degree(5), 400);
}

#[test]
fn dense_leaf_rebalance_gives_headroom() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // Pack one bucket to force a rebalance, then verify the rebalance left
    // at least one writable slot so the next insert lands in the slab (no
    // same-leaf rebalance retry loop).
    let mut g: LabeledLaraGraph<VecMemory> = LabeledLaraGraph::new(fresh_labeled_mem(), 8).unwrap();
    for d in 0..60u32 {
        g.insert_edge(2, d, LabelId::new(0));
    }
    let leaf = 2 / g.segment_size();
    g.rebalance_leaf(leaf as u64);
    assert!(
        g.bucket_room(2, LabelId::new(0)) >= 1,
        "rebalance must leave >= 1 headroom slot"
    );
    // Follow-up inserts must succeed without corruption.
    for d in 0..40u32 {
        g.insert_edge(2, 1000 + d, LabelId::new(0));
    }
    g.verify_invariants();
    assert_eq!(g.out_degree(2), 100);
}

#[test]
fn bucket_promotes_and_scans() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // Grow one bucket past T_PROMOTE: it must promote to per-bucket
    // Tree-CSR, and scans must read from the tree.
    let mut g: LabeledLaraGraph<VecMemory> = LabeledLaraGraph::new(fresh_labeled_mem(), 8).unwrap();
    for d in 0..(T_PROMOTE + 100) {
        g.insert_edge(2, d, LabelId::new(0));
    }
    assert!(
        g.bucket_is_tree(2, LabelId::new(0)),
        "bucket should promote"
    );
    assert!(g.tree_bucket_count() >= 1, "tree bucket registered");
    assert_eq!(g.out_degree(2), (T_PROMOTE + 100) as u64);
    // for_each_edge_of_label reads the tree.
    let mut n = 0u32;
    g.for_each_edge_of_label(2, LabelId::new(0), |_| n += 1);
    assert_eq!(n, T_PROMOTE + 100);
    // for_each_labeled reads the tree too.
    let mut m = 0u32;
    g.for_each_labeled(2, |_, l| {
        assert_eq!(l, LabelId::new(0));
        m += 1;
    });
    assert_eq!(m, T_PROMOTE + 100);
    // The tree keeps growing on further inserts.
    for d in 0..50u32 {
        g.insert_edge(2, 100_000 + d, LabelId::new(0));
    }
    assert_eq!(g.out_degree(2), (T_PROMOTE + 150) as u64);
    g.verify_invariants();
}

#[test]
fn bucket_demotes_below_t_demote() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    let mut g: LabeledLaraGraph<VecMemory> = LabeledLaraGraph::new(fresh_labeled_mem(), 8).unwrap();
    for d in 0..(T_PROMOTE + 50) {
        g.insert_edge(2, d, LabelId::new(0));
    }
    assert!(g.bucket_is_tree(2, LabelId::new(0)));
    // Remove down to T_DEMOTE: the bucket must demote back to slab.
    let mut removed = 0u32;
    for d in 0..(T_PROMOTE + 50) {
        if g.remove_edge(2, d, LabelId::new(0)) {
            removed += 1;
        }
        if removed >= (T_PROMOTE + 50) - T_DEMOTE {
            break;
        }
    }
    assert!(
        !g.bucket_is_tree(2, LabelId::new(0)),
        "bucket should demote at T_DEMOTE"
    );
    assert_eq!(g.out_degree(2), T_DEMOTE as u64);
    // Remaining edges are still returned by the slab scan.
    let mut n = 0u32;
    g.for_each_edge_of_label(2, LabelId::new(0), |_| n += 1);
    assert_eq!(n, T_DEMOTE);
    g.verify_invariants();
}

#[test]
fn bucket_tree_no_oscillation() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    let mut g: LabeledLaraGraph<VecMemory> = LabeledLaraGraph::new(fresh_labeled_mem(), 8).unwrap();
    for d in 0..(T_PROMOTE + 50) {
        g.insert_edge(2, d, LabelId::new(0));
    }
    assert!(g.bucket_is_tree(2, LabelId::new(0)));
    // Demote to T_DEMOTE.
    let mut removed = 0u32;
    for d in 0..(T_PROMOTE + 50) {
        if g.remove_edge(2, d, LabelId::new(0)) {
            removed += 1;
        }
        if removed >= (T_PROMOTE + 50) - T_DEMOTE {
            break;
        }
    }
    assert!(!g.bucket_is_tree(2, LabelId::new(0)));
    // Churn around T_DEMOTE must not re-promote (needs T_PROMOTE stored).
    for d in 0..100u32 {
        g.insert_edge(2, 200_000 + d, LabelId::new(0));
        g.remove_edge(2, 200_000 + d, LabelId::new(0));
    }
    assert!(
        !g.bucket_is_tree(2, LabelId::new(0)),
        "no promote/demote oscillation"
    );
    g.verify_invariants();
}

#[test]
fn mixed_buckets_scan_isolation() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // Same vertex: label 0 promotes to tree, label 1 stays in the slab.
    let mut g: LabeledLaraGraph<VecMemory> = LabeledLaraGraph::new(fresh_labeled_mem(), 8).unwrap();
    for d in 0..(T_PROMOTE + 20) {
        g.insert_edge(2, d, LabelId::new(0));
    }
    for d in 0..10u32 {
        g.insert_edge(2, 50_000 + d, LabelId::new(1));
    }
    assert!(g.bucket_is_tree(2, LabelId::new(0)));
    assert!(!g.bucket_is_tree(2, LabelId::new(1)));
    // for_each_edge_of_label isolates the tree bucket.
    let mut n = 0u32;
    g.for_each_edge_of_label(2, LabelId::new(0), |_| n += 1);
    assert_eq!(n, T_PROMOTE + 20);
    // for_each_labeled returns both buckets.
    let mut total = 0u32;
    g.for_each_labeled(2, |_, _| total += 1);
    assert_eq!(total, (T_PROMOTE + 20) + 10);
    g.verify_invariants();
}

#[test]
fn slab_bucket_alloc_capped_at_cap_normal() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // Grow one bucket to just under an explicit T=4096 (so it stays in
    // slab mode), rebalance, and verify its allocated sub-span never
    // exceeds the cap even though the gap_candidate would allocate more.
    // Explicit threshold: independent of the crate default (now 1024).
    let mut g: LabeledLaraGraph<VecMemory> = LabeledLaraGraph::new(fresh_labeled_mem(), 8)
        .unwrap()
        .with_promote_threshold(Some(4096));
    for d in 0..4000u32 {
        g.insert_edge(2, d, LabelId::new(0));
    }
    assert!(
        !g.bucket_is_tree(2, LabelId::new(0)),
        "still slab mode below T"
    );
    let leaf = 2 / g.segment_size();
    g.rebalance_leaf(leaf as u64);
    let alloc = g.bucket_allocated(2, LabelId::new(0));
    assert!(alloc <= 4096, "allocated {alloc} exceeds the cap");
    assert!(
        alloc >= 4000,
        "allocated {alloc} must fit the 4000 live edges"
    );
    g.verify_invariants();
}

#[test]
fn core_row_alloc_capped_at_cap_normal() {
    // Same cap on the unlabeled core: a single high-degree row's allocated
    // width must not exceed the cap after rebalance. Explicit T=4096 keeps
    // the row in slab mode regardless of the crate default.
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 8).unwrap();
    g.set_promote_threshold(Some(4096));
    for d in 0..4000u32 {
        g.insert_edge(2, d);
    }
    let leaf = 2 / g.segment_size();
    g.rebalance_leaf(leaf as u64);
    let alloc = g.row_allocated(2);
    assert!(alloc <= 4096, "allocated {alloc} exceeds the cap");
    assert!(
        alloc >= 4000,
        "allocated {alloc} must fit the 4000 live edges"
    );
    g.verify_invariants();
}

#[test]
fn leaf_span_equals_sum_of_capped_allocs() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // After a rebalance, the leaf span must equal the sum of the capped
    // per-bucket allocations (no hidden 4x overhead). Explicit T=4096
    // keeps both 3000-edge buckets in slab mode.
    let mut g: LabeledLaraGraph<VecMemory> = LabeledLaraGraph::new(fresh_labeled_mem(), 8)
        .unwrap()
        .with_promote_threshold(Some(4096));
    for d in 0..3000u32 {
        g.insert_edge(2, d, LabelId::new(0));
        g.insert_edge(3, 10_000 + d, LabelId::new(1));
    }
    let leaf = 2 / g.segment_size();
    g.rebalance_leaf(leaf as u64);
    let (start, span, _) = g.leaf_geometry(leaf as u64);
    let _ = start;
    // Sum the capped allocations of every bucket in the leaf.
    let mut sum = 0u64;
    for v in 0..g.num_nodes() as u32 {
        if v / g.segment_size() != leaf {
            continue;
        }
        for l in 0..2u16 {
            sum += g.bucket_allocated(v, LabelId::new(l));
        }
    }
    assert_eq!(span, sum, "leaf span must equal sum of capped allocations");
    g.verify_invariants();
}

#[test]
fn tree_root_array_capped_at_cap_tree() {
    // The tree root array length is capped at R_MAX = CAP_TREE blocks
    // (depth-1 coverage of B * R_MAX slots).
    assert_eq!(CAP_TREE as usize, R_MAX);
    assert!(root_len(B as u32 * R_MAX as u32) <= R_MAX);
    // A modest tree bucket's root length stays well under the cap.
    assert!(root_len(10_000) <= R_MAX);
}

#[test]
fn alloc_is_min_2x_cap() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // Single formula: allocated == min(max(2*n, 1), T). Two adjacent
    // 100-edge buckets: the first bucket's boundary is the second
    // bucket's edge_start, so its allocated width is exactly its own
    // bslots. An empty bucket keeps 1 headroom slot.
    let mut g: LabeledLaraGraph<VecMemory> = LabeledLaraGraph::new(fresh_labeled_mem(), 8).unwrap();
    for d in 0..100u32 {
        g.insert_edge(2, d, LabelId::new(0));
        g.insert_edge(3, 10_000 + d, LabelId::new(0));
    }
    let leaf = 2 / g.segment_size();
    g.rebalance_leaf(leaf as u64);
    assert_eq!(g.bucket_allocated(2, LabelId::new(0)), 200);
    // Trailing bucket absorbs the 1024-minimum span padding by design
    // (boundary = leaf end), so only the span total is asserted: 400
    // capped allocations padded to 1024.
    let (_, span, _) = g.leaf_geometry(leaf as u64);
    assert_eq!(span, 1024);
    // Empty row (core): 1 headroom slot so the first post-rebalance
    // insert lands in the slab.
    let mut e: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 8).unwrap();
    let leaf_e = 5 / e.segment_size();
    e.rebalance_leaf(leaf_e as u64);
    assert_eq!(e.row_allocated(5), 1);
    g.verify_invariants();
    e.verify_invariants();
}

#[test]
fn threshold_caps_allocation() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // Threshold doubles as the cap (CAP == T): with T=1024 a bucket grown
    // to 2000 promotes at stored >= 1024, so its slab allocation is 0
    // (tree buckets hold no slab data) — trivially <= 1024 — while all
    // 2000 edges stay retrievable through the tree.
    let mut g: LabeledLaraGraph<VecMemory> = LabeledLaraGraph::new(fresh_labeled_mem(), 8)
        .unwrap()
        .with_promote_threshold(Some(1024));
    for d in 0..2000u32 {
        g.insert_edge(2, d, LabelId::new(0));
    }
    assert!(g.bucket_is_tree(2, LabelId::new(0)));
    assert!(g.bucket_allocated(2, LabelId::new(0)) <= 1024);
    let mut seen = 0u32;
    g.for_each_edge_of_label(2, LabelId::new(0), |_| seen += 1);
    assert_eq!(seen, 2000);
    g.verify_invariants();
}

#[test]
fn scan_sweep_fixture_guards() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // hub-3k straddles the promote boundary (tree at T<=2048, slab
    // above); hub-10k is tree at every swept T. Fixtures are built with
    // sequential inserts so promotion fires per-T.
    for &t in &[1024u32, 2048, 4096, 8192] {
        let mut g3: LabeledLaraGraph<VecMemory> =
            LabeledLaraGraph::new(fresh_labeled_mem(), 16_384)
                .unwrap()
                .with_promote_threshold(Some(t));
        for d in 0..3000u32 {
            g3.insert_edge(0, 100 + d, LabelId::new(0));
        }
        assert_eq!(
            g3.bucket_is_tree(0, LabelId::new(0)),
            t <= 2048,
            "hub-3k T={t}"
        );
        let mut g10: LabeledLaraGraph<VecMemory> =
            LabeledLaraGraph::new(fresh_labeled_mem(), 16_384)
                .unwrap()
                .with_promote_threshold(Some(t));
        for d in 0..10_000u32 {
            g10.insert_edge(1, 5000 + d, LabelId::new(0));
        }
        assert!(g10.bucket_is_tree(1, LabelId::new(0)), "hub-10k T={t}");
    }
}

#[test]
fn threshold_default_is_t_promote() {
    let g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 8).unwrap();
    assert_eq!(g.promote_threshold(), Some(T_PROMOTE));
    assert_eq!(T_PROMOTE, 1024);
    assert_eq!(T_DEMOTE, 512);
    assert_eq!(g.alloc_cap(), T_PROMOTE as u64);
}

#[test]
fn max_log_entries_default_is_170() {
    let g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 8).unwrap();
    assert_eq!(g.max_log_entries(), MAX_LOG_ENTRIES);
    assert_eq!(MAX_LOG_ENTRIES, 170);
    // Shrinking post-construction is allowed; growing past the
    // construction capacity panics.
    let mut s = g;
    s.set_max_log_entries(32);
    assert_eq!(s.max_log_entries(), 32);
}

#[test]
#[should_panic(expected = "exceeds log region capacity")]
fn max_log_entries_cannot_grow_past_capacity() {
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 8).unwrap();
    g.set_max_log_entries(MAX_LOG_ENTRIES + 1);
}

#[test]
fn small_log_triggers_early_rebalance() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // Tiny log cap (32): 200 inserts into one slab bucket must fold
    // early and often, yet every edge stays retrievable.
    let mut g: LabeledLaraGraph<VecMemory> =
        LabeledLaraGraph::new_with_max_log_entries(fresh_labeled_mem(), 8, 32).unwrap();
    assert_eq!(g.max_log_entries(), 32);
    for d in 0..200u32 {
        g.insert_edge(2, d, LabelId::new(0));
    }
    let mut seen = 0u32;
    g.for_each_edge_of_label(2, LabelId::new(0), |_| seen += 1);
    assert_eq!(seen, 200);
    assert!((g.leaf_log_idx(2 / g.segment_size() as u64) as usize) <= 32);
    g.verify_invariants();
}

#[test]
fn memory_includes_log_region() {
    // Lazy backing: empty graphs at caps 32 vs 512 back nothing, so
    // totals are identical (header + directory only). After forcing one
    // overflow chunk each, the total differs by exactly one chunk delta.
    let mut a: LaraGraph<Vertex, Edge, VecMemory> =
        LaraGraph::new_with_max_log_entries(fresh_mem(), 256, 32).unwrap();
    let mut b: LaraGraph<Vertex, Edge, VecMemory> =
        LaraGraph::new_with_max_log_entries(fresh_mem(), 256, 512).unwrap();
    assert_eq!(a.segment_count(), b.segment_count());
    assert_eq!(b.memory_bytes_total(), a.memory_bytes_total());
    // First inserts go tiny-inline; K+1 on one row spills into the log.
    for d in 0..9u32 {
        a.insert_edge(1, d);
        b.insert_edge(1, d);
    }
    assert!(!a.row_is_tiny(1));
    assert_eq!(a.log_allocated_chunks(), 1);
    assert_eq!(b.log_allocated_chunks(), 1);
    let expected = (512 - 32) as u64 * 16;
    assert_eq!(b.memory_bytes_total() - a.memory_bytes_total(), expected);
}

#[test]
fn zero_log_inserts_correctly() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // No-log endpoint (cap=0): 1500 inserts into one bucket cross the
    // T=1024 capped boundary, exercising both the rebalance-every-time
    // path and the promote-instead-of-log path (no infinite loop).
    let mut g: LabeledLaraGraph<VecMemory> =
        LabeledLaraGraph::new_with_max_log_entries(fresh_labeled_mem(), 2048, 0).unwrap();
    assert_eq!(g.max_log_entries(), 0);
    for d in 0..1500u32 {
        g.insert_edge(2, d, LabelId::new(0));
    }
    assert!(
        g.bucket_is_tree(2, LabelId::new(0)),
        "1500-edge bucket must promote past T=1024"
    );
    let mut seen = 0u32;
    g.for_each_edge_of_label(2, LabelId::new(0), |_| seen += 1);
    assert_eq!(seen, 1500);
    for leaf in 0..g.segment_count() {
        assert_eq!(
            g.leaf_log_idx(leaf),
            0,
            "no-log mode must never write the log"
        );
    }
    g.verify_invariants();
}

#[test]
fn tiny_log_correctness() {
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    // cap=4 with dispersed inserts (unique dsts, spread sources).
    let mut g: LabeledLaraGraph<VecMemory> =
        LabeledLaraGraph::new_with_max_log_entries(fresh_labeled_mem(), 256, 4).unwrap();
    for i in 0..2000u32 {
        g.insert_edge((i * 7919) % 256, i, LabelId::new((i % 3) as u16));
    }
    assert_eq!(g.num_edges(), 2000);
    for leaf in 0..g.segment_count() {
        assert!(
            (g.leaf_log_idx(leaf) as usize) <= 4,
            "leaf {leaf} log overflow"
        );
    }
    g.verify_invariants();
}

#[test]
fn persist_counter_counts_writes() {
    use lara_graph::memory::{PersistCounterMemory, PersistCounters};
    use std::rc::Rc;
    let counters = Rc::new(PersistCounters::default());
    let mem = LaraMemories {
        vertices: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        edges: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        logs: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        counts: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        spans: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        free_spans: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        blocks: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        buckets: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
    };
    let mut g: LaraGraph<Vertex, Edge, PersistCounterMemory<VecMemory>> =
        LaraGraph::new(mem, 64).unwrap();
    for d in 0..100u32 {
        g.insert_edge(1, d);
    }
    assert!(counters.bytes() > 0, "inserts must persist bytes");
    assert!(counters.ops() > 0, "inserts must issue write ops");
    g.verify_invariants();
}

#[test]
fn persist_counter_reset_zeroes() {
    use lara_graph::memory::{PersistCounterMemory, PersistCounters};
    use std::rc::Rc;
    let counters = Rc::new(PersistCounters::default());
    let mem = LaraMemories {
        vertices: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        edges: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        logs: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        counts: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        spans: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        free_spans: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        blocks: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        buckets: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
    };
    let mut g: LaraGraph<Vertex, Edge, PersistCounterMemory<VecMemory>> =
        LaraGraph::new(mem, 64).unwrap();
    for d in 0..50u32 {
        g.insert_edge(1, d);
    }
    assert!(counters.bytes() > 0);
    counters.reset();
    assert_eq!(counters.bytes(), 0);
    assert_eq!(counters.ops(), 0);
    assert_eq!(counters.fences(), 0);
}

#[test]
fn rebalance_counter_counts_and_resets() {
    // Perf instrument: hub inserts must trip rebalances; reset zeroes.
    // Behavior unchanged (scan + invariants hold).
    let mut g: LaraGraph<Vertex, Edge, VecMemory> =
        LaraGraph::new_with_max_log_entries(fresh_mem(), 1024, 0).unwrap();
    for d in 0..500u32 {
        g.insert_edge(1, d);
    }
    assert!(g.rebalance_count() > 0, "500 hub inserts must rebalance");
    g.reset_perf_counters();
    assert_eq!(g.rebalance_count(), 0);
    assert_eq!(g.promote_count(), 0);
    let mut seen = 0u32;
    g.for_each_neighbor(1, |_| seen += 1);
    assert_eq!(seen, 500);
    g.verify_invariants();
}

#[test]
fn tree_block_scan_matches_edge_scan() {
    // Block-granular tree scan must return exactly the inserted set:
    // block-exact sizes (1024/2048/10240) and ragged tails (1025/10000).
    for &size in &[1024u32, 1025, 2048, 10_000, 10_240] {
        let mut g: LaraGraph<Vertex, Edge, VecMemory> =
            LaraGraph::new(fresh_mem(), 16_384).unwrap();
        for d in 0..size {
            g.insert_edge(1, d);
        }
        let got = sorted(g.out_neighbors_vec(1));
        let want: Vec<u32> = (0..size).collect();
        assert_eq!(got, want, "tree scan mismatch at size {size}");
        g.verify_invariants();
    }
    // Labeled tree path shares the block reader: one ragged case.
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    let mut l: LabeledLaraGraph<VecMemory> =
        LabeledLaraGraph::new(fresh_labeled_mem(), 4096).unwrap();
    for d in 0..1025u32 {
        l.insert_edge(2, d, LabelId::new(0));
    }
    assert!(l.bucket_is_tree(2, LabelId::new(0)));
    let mut got = Vec::new();
    l.for_each_edge_of_label(2, LabelId::new(0), |n| got.push(n));
    assert_eq!(sorted(got), (0..1025u32).collect::<Vec<_>>());
    l.verify_invariants();
}

#[test]
fn tree_scan_empty_and_single() {
    // Endpoints: empty row, single-edge (slab) row, and a promoted-then-
    // churned tree row must all scan exactly.
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 4096).unwrap();
    assert!(g.out_neighbors_vec(5).is_empty());
    g.insert_edge(5, 42);
    assert_eq!(g.out_neighbors_vec(5), vec![42]);
    for d in 0..1500u32 {
        g.insert_edge(6, d);
    }
    for d in 0..1000u32 {
        assert!(g.remove_edge(6, d));
    }
    let got = sorted(g.out_neighbors_vec(6));
    assert_eq!(got, (1000..1500u32).collect::<Vec<_>>());
    g.verify_invariants();
}

#[test]
fn insert_checks_equivalent() {
    // Slimmed insert check chain (row reuse + integer density) must
    // produce identical adjacency, invariants, and rebalance/promote
    // counts on both hub-growth and dispersed workloads.
    let mut h: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 20_480).unwrap();
    for d in 0..10_000u32 {
        h.insert_edge(1, d);
    }
    assert_eq!(
        sorted(h.out_neighbors_vec(1)),
        (0..10_000u32).collect::<Vec<_>>()
    );
    h.verify_invariants();
    assert_eq!((h.rebalance_count(), h.promote_count()), (3, 1));
    // Deterministic dispersed pattern (strided, no RNG dependency).
    let mut u: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 1024).unwrap();
    let mut want: Vec<Vec<u32>> = vec![Vec::new(); 1024];
    for i in 0..4096u32 {
        let (a, b) = ((i * 7 + 3) % 1024, (i * 13 + 5) % 1024);
        u.insert_edge(a, b);
        want[a as usize].push(b);
    }
    for v in 0..1024u32 {
        assert_eq!(
            sorted(u.out_neighbors_vec(v)),
            sorted(want[v as usize].clone()),
            "v={v}"
        );
    }
    u.verify_invariants();
    assert_eq!(u.rebalance_count(), 0);
    assert_eq!(u.promote_count(), 0);
}

#[test]
fn density_thinning_never_misses() {
    // Concentrated inserts below the promote threshold must still fire
    // the density rebalance (integer `actual >= total` never misses).
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 20_480).unwrap();
    for d in 0..900u32 {
        g.insert_edge(1, d);
    }
    assert!(g.rebalance_count() >= 1, "density rebalance must fire");
    assert_eq!(g.promote_count(), 0);
    assert_eq!(
        sorted(g.out_neighbors_vec(1)),
        (0..900u32).collect::<Vec<_>>()
    );
    g.verify_invariants();
}

#[test]
fn ingest_run_matches_per_edge() {
    use lara_graph::ingest::IngestBuffer;
    // Deterministic dispersed pattern.
    let edges: Vec<(u32, u32)> = (0..4096u32)
        .map(|i| ((i * 7 + 3) % 1024, (i * 13 + 5) % 1024))
        .collect();
    let mut ref_g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 1024).unwrap();
    for &(a, b) in &edges {
        ref_g.insert_edge(a, b);
    }
    for &cap in &[4096usize, 64] {
        let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 1024).unwrap();
        let mut ing = IngestBuffer::new(1024, cap);
        for &(a, b) in &edges {
            ing.push(a, b);
            if ing.len() >= cap {
                ing.flush(&mut g);
            }
        }
        ing.flush(&mut g);
        for v in 0..1024u32 {
            assert_eq!(
                sorted(g.out_neighbors_vec(v)),
                sorted(ref_g.out_neighbors_vec(v)),
                "v={v} cap={cap}"
            );
        }
        g.verify_invariants();
        let (rr, rp) = (ref_g.rebalance_count(), ref_g.promote_count());
        let (gr, gp) = (g.rebalance_count(), g.promote_count());
        let close =
            |a: u64, b: u64| (a as i64 - b as i64).abs() <= (a.max(1) as f64 * 0.1).ceil() as i64;
        assert!(close(rr, gr), "rb dispersed ref={rr} got={gr}");
        assert!(close(rp, gp), "pr dispersed ref={rp} got={gp}");
    }
    // Hub-10k growth via ingest (B=4096 and B=64).
    for &cap in &[4096usize, 64] {
        let mut ref_h: LaraGraph<Vertex, Edge, VecMemory> =
            LaraGraph::new(fresh_mem(), 20_480).unwrap();
        for d in 0..10_000u32 {
            ref_h.insert_edge(1, d);
        }
        let mut h: LaraGraph<Vertex, Edge, VecMemory> =
            LaraGraph::new(fresh_mem(), 20_480).unwrap();
        let mut ing = IngestBuffer::new(20_480, cap);
        for d in 0..10_000u32 {
            ing.push(1, d);
            if ing.len() >= cap {
                ing.flush(&mut h);
            }
        }
        ing.flush(&mut h);
        assert_eq!(
            sorted(h.out_neighbors_vec(1)),
            sorted(ref_h.out_neighbors_vec(1))
        );
        h.verify_invariants();
        let (rr, rp) = (ref_h.rebalance_count(), ref_h.promote_count());
        let (gr, gp) = (h.rebalance_count(), h.promote_count());
        let close =
            |a: u64, b: u64| (a as i64 - b as i64).abs() <= (a.max(1) as f64 * 0.1).ceil() as i64;
        assert!(close(rr, gr), "rb hub ref={rr} got={gr} cap={cap}");
        assert!(close(rp, gp), "pr hub ref={rp} got={gp} cap={cap}");
    }
}

#[test]
fn ingest_empty_and_single_run() {
    use lara_graph::ingest::IngestBuffer;
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 64).unwrap();
    let mut ing = IngestBuffer::new(64, 4096);
    ing.flush(&mut g); // empty no-op
    assert_eq!(g.out_neighbors_vec(1).len(), 0);
    ing.push(1, 7);
    ing.flush(&mut g);
    assert_eq!(g.out_neighbors_vec(1), vec![7]);
    // Same-source run exceeding the buffer cap (multiple flushes).
    let mut h: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 4096).unwrap();
    let mut ing2 = IngestBuffer::new(4096, 64);
    for d in 0..500u32 {
        ing2.push(2, d);
        if ing2.len() >= 64 {
            ing2.flush(&mut h);
        }
    }
    ing2.flush(&mut h);
    assert_eq!(
        sorted(h.out_neighbors_vec(2)),
        (0..500u32).collect::<Vec<_>>()
    );
    h.verify_invariants();
}

#[test]
fn ingest_interleaved_with_per_edge() {
    use lara_graph::ingest::IngestBuffer;
    let edges: Vec<(u32, u32)> = (0..2000u32)
        .map(|i| ((i * 11 + 1) % 256, (i * 17 + 3) % 256))
        .collect();
    let mut ref_g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 256).unwrap();
    for &(a, b) in &edges {
        ref_g.insert_edge(a, b);
    }
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 256).unwrap();
    let mut ing = IngestBuffer::new(256, 128);
    for (k, &(a, b)) in edges.iter().enumerate() {
        if k % 2 == 0 {
            g.insert_edge(a, b);
        } else {
            ing.push(a, b);
            if ing.len() >= 128 {
                ing.flush(&mut g);
            }
        }
    }
    ing.flush(&mut g);
    for v in 0..256u32 {
        assert_eq!(
            sorted(g.out_neighbors_vec(v)),
            sorted(ref_g.out_neighbors_vec(v)),
            "v={v}"
        );
    }
    g.verify_invariants();
}

#[test]
fn lazy_log_equivalent() {
    // Lazy log backing must not change observable behavior: identical
    // adjacency, invariants, and firing counts on dispersed + hub loads.
    let edges: Vec<(u32, u32)> = (0..4096u32)
        .map(|i| ((i * 7 + 3) % 1024, (i * 13 + 5) % 1024))
        .collect();
    let mut want: Vec<Vec<u32>> = vec![Vec::new(); 1024];
    for &(a, b) in &edges {
        want[a as usize].push(b);
    }
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 1024).unwrap();
    for &(a, b) in &edges {
        g.insert_edge(a, b);
    }
    for v in 0..1024u32 {
        assert_eq!(
            sorted(g.out_neighbors_vec(v)),
            sorted(want[v as usize].clone()),
            "v={v}"
        );
    }
    g.verify_invariants();
    assert_eq!((g.rebalance_count(), g.promote_count()), (0, 0));
    // Ingest path agrees on the same stream.
    let mut h: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 1024).unwrap();
    let mut ing = lara_graph::ingest::IngestBuffer::new(1024, 4096);
    for &(a, b) in &edges {
        ing.push(a, b);
    }
    ing.flush(&mut h);
    for v in 0..1024u32 {
        assert_eq!(
            sorted(h.out_neighbors_vec(v)),
            sorted(want[v as usize].clone()),
            "ingest v={v}"
        );
    }
    h.verify_invariants();
    // Hub growth: folds complete, adjacency exact, firing unchanged.
    let mut hub: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 20_480).unwrap();
    for d in 0..10_000u32 {
        hub.insert_edge(1, d);
    }
    assert_eq!(
        sorted(hub.out_neighbors_vec(1)),
        (0..10_000u32).collect::<Vec<_>>()
    );
    hub.verify_invariants();
    assert_eq!((hub.rebalance_count(), hub.promote_count()), (3, 1));
    assert_eq!(
        hub.log_allocated_chunks(),
        0,
        "settled hub must hold no log chunks"
    );
}

#[test]
fn lazy_log_alloc_free() {
    // Empty graphs back nothing but the header: directory only.
    for cap in [170usize, 0] {
        let g: LaraGraph<Vertex, Edge, VecMemory> =
            LaraGraph::new_with_max_log_entries(fresh_mem(), 1024, cap).unwrap();
        assert_eq!(g.log_allocated_chunks(), 0);
        assert_eq!(g.log_memory_bytes(), 4096 + g.segment_count() * 8);
    }
    // Tiny cap forces overflow → allocate → fold → retire → recycle.
    let mut g: LaraGraph<Vertex, Edge, VecMemory> =
        LaraGraph::new_with_max_log_entries(fresh_mem(), 64, 1).unwrap();
    let mut max_seen = 0u64;
    for d in 0..40u32 {
        g.insert_edge(1, d);
        max_seen = max_seen.max(g.log_allocated_chunks());
    }
    assert!(max_seen >= 1, "overflow must back at least one chunk");
    assert_eq!(
        sorted(g.out_neighbors_vec(1)),
        (0..40u32).collect::<Vec<_>>()
    );
    g.verify_invariants();
    assert_eq!(
        g.log_allocated_chunks(),
        0,
        "fold must retire the chunk (no leak)"
    );
    assert!(
        g.log_pool_depth() >= 1,
        "retired chunk must recycle through the pool"
    );
    // Reuse path: a new overflow reuses the pooled chunk (no re-bump).
    let bumped_before = g.log_memory_bytes();
    for d in 40..80u32 {
        g.insert_edge(1, d);
    }
    assert_eq!(
        g.log_memory_bytes(),
        bumped_before,
        "pool reuse must not bump"
    );
    // cap=0: never backs a chunk (rebalance-direct preserved).
    let mut z: LaraGraph<Vertex, Edge, VecMemory> =
        LaraGraph::new_with_max_log_entries(fresh_mem(), 64, 0).unwrap();
    for d in 0..40u32 {
        z.insert_edge(1, d);
        assert_eq!(z.log_allocated_chunks(), 0);
    }
    assert_eq!(
        sorted(z.out_neighbors_vec(1)),
        (0..40u32).collect::<Vec<_>>()
    );
    z.verify_invariants();
}

#[test]
fn lazy_log_memory() {
    // Uniform n=10k/m=40k strided stream: leaves never fold (per-leaf
    // load stays under the cap), so every touched leaf holds one chunk —
    // the log-as-initial-store design keeps ~27 B/e here by construction.
    // Lazy pays the same chunks plus the 8 B/leaf directory.
    let stream: Vec<(u32, u32)> = (0..40_000u32)
        .map(|i| ((i * 7 + 3) % 10_000, (i * 13 + 5) % 10_000))
        .collect();
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 10_000).unwrap();
    let t0 = std::time::Instant::now();
    for &(a, b) in &stream {
        g.insert_edge(a, b);
    }
    let ins_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let total = g.memory_bytes_total();
    let logs = g.log_memory_bytes();
    let slab = g.slab_capacity() * 4;
    let vertex = g.num_nodes() * 24;
    let tree = total.saturating_sub(slab + logs + vertex);
    let per_e = total as f64 / 40_000.0;
    let tiny = g.tiny_memory_bytes();
    let logs_pure = logs - tiny;
    println!(
        "lazy_log_memory/uniform10k: total={total} ({per_e:.1} B/e) slab={slab} tree={tree} logs={logs_pure} tiny={tiny} vtx={vertex} log_chunks={} tiny_rows={} insert={ins_ms:.1}ms",
        g.log_allocated_chunks(),
        g.tiny_row_count()
    );
    // Tiny era (K=8): uniform degree-4 rows stay tiny (see `tiny_memory`);
    // the lazy-log band assertion moved to the 14-16 B/e regime.
    assert!(
        total / 40_000 >= 12 && total / 40_000 <= 18,
        "uniform total must be in the 14-16 B/e band, got {per_e:.1}"
    );
    // Hub-10k settles chunk-free: lazy pays header + directory only
    // where eager paid seg × cap × 16 (MBs).
    let mut h: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 20_480).unwrap();
    for d in 0..10_000u32 {
        h.insert_edge(1, d);
    }
    let h_logs = h.log_memory_bytes();
    let h_total = h.memory_bytes_total();
    println!(
        "lazy_log_memory/hub10k: total={h_total} ({:.1} B/e) logs={h_logs} chunks={}",
        h_total as f64 / 10_000.0,
        h.log_allocated_chunks()
    );
    assert!(
        h_logs < 32_768,
        "settled hub logs must be header+directory only, got {h_logs}"
    );
}

#[test]
fn tiny_inline_equivalent() {
    // Tiny-inline must not change observable behavior: identical
    // adjacency, invariants, and firing counts on dispersed + hub loads.
    let edges: Vec<(u32, u32)> = (0..4096u32)
        .map(|i| ((i * 7 + 3) % 1024, (i * 13 + 5) % 1024))
        .collect();
    let mut want: Vec<Vec<u32>> = vec![Vec::new(); 1024];
    for &(a, b) in &edges {
        want[a as usize].push(b);
    }
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 1024).unwrap();
    for &(a, b) in &edges {
        g.insert_edge(a, b);
    }
    for v in 0..1024u32 {
        assert_eq!(
            sorted(g.out_neighbors_vec(v)),
            sorted(want[v as usize].clone()),
            "v={v}"
        );
    }
    g.verify_invariants();
    assert_eq!((g.rebalance_count(), g.promote_count()), (0, 0));
    // Tiny-active delete compacts inside the chunk.
    assert!(g.row_is_tiny(3));
    let before: Vec<u32> = sorted(g.out_neighbors_vec(3));
    assert!(g.remove_edge(3, before[0]));
    let after: Vec<u32> = sorted(g.out_neighbors_vec(3));
    assert_eq!(after.len(), before.len() - 1);
    // Multiset semantics: one occurrence removed (pattern has dup dsts).
    let cnt = |xs: &[u32], x: u32| xs.iter().filter(|&&y| y == x).count();
    assert_eq!(cnt(&after, before[0]), cnt(&before, before[0]) - 1);
    assert!(g.row_is_tiny(3), "delete must not spill a tiny row");
    g.verify_invariants();
    // Hub growth: spill replays into the log, firing unchanged.
    let mut hub: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 20_480).unwrap();
    for d in 0..10_000u32 {
        hub.insert_edge(1, d);
    }
    assert_eq!(
        sorted(hub.out_neighbors_vec(1)),
        (0..10_000u32).collect::<Vec<_>>()
    );
    hub.verify_invariants();
    assert_eq!((hub.rebalance_count(), hub.promote_count()), (3, 1));
}

#[test]
fn tiny_spill_boundary() {
    // K=8: eight inserts stay tiny (flag set, chunk held); the ninth
    // spills (flag down, chunk returned, pool +1).
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 64).unwrap();
    for d in 0..8u32 {
        g.insert_edge(1, d);
        assert!(g.row_is_tiny(1), "d={d}");
    }
    assert_eq!(g.out_degree(1), 8);
    assert_eq!(g.tiny_row_count(), 1);
    assert_eq!(g.tiny_pool_depth(), 0);
    g.insert_edge(1, 8);
    assert!(!g.row_is_tiny(1), "spill must clear the flag");
    assert_eq!(g.tiny_pool_depth(), 1, "spill must return the chunk");
    assert_eq!(
        sorted(g.out_neighbors_vec(1)),
        (0..9u32).collect::<Vec<_>>()
    );
    g.verify_invariants();
    // Post-spill appends use the ordinary path (no re-tiny).
    g.insert_edge(1, 9);
    assert!(!g.row_is_tiny(1));
    assert_eq!(
        sorted(g.out_neighbors_vec(1)),
        (0..10u32).collect::<Vec<_>>()
    );
    g.verify_invariants();
}

#[test]
fn tiny_pool_reuse() {
    // Empty graph backs nothing for tiny; a spilled chunk is re-popped
    // without moving the bump cursor.
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 64).unwrap();
    assert_eq!(g.tiny_memory_bytes(), 0);
    for d in 0..9u32 {
        g.insert_edge(1, d);
    }
    assert_eq!(g.tiny_pool_depth(), 1);
    let bumped = g.tiny_memory_bytes();
    assert_eq!(bumped, 32);
    g.insert_edge(2, 0);
    assert!(g.row_is_tiny(2));
    assert_eq!(
        g.tiny_pool_depth(),
        0,
        "fresh tiny row must reuse the pooled chunk"
    );
    assert_eq!(g.tiny_memory_bytes(), bumped, "pool reuse must not bump");
    assert_eq!(sorted(g.out_neighbors_vec(2)), vec![0]);
    g.verify_invariants();
}

#[test]
fn tiny_memory() {
    // Uniform n=10k/m=40k strided (degree exactly 4 = K): every row stays
    // tiny — no slab spans, no log chunks. Expect ~10-12 B/e total.
    let stream: Vec<(u32, u32)> = (0..40_000u32)
        .map(|i| ((i * 7 + 3) % 10_000, (i * 13 + 5) % 10_000))
        .collect();
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 10_000).unwrap();
    let t0 = std::time::Instant::now();
    for &(a, b) in &stream {
        g.insert_edge(a, b);
    }
    let ins_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let total = g.memory_bytes_total();
    let tiny = g.tiny_memory_bytes();
    let logs_pure = g.log_memory_bytes() - tiny;
    let slab = g.slab_capacity() * 4;
    let vertex = g.num_nodes() * 24;
    let tree = total.saturating_sub(slab + logs_pure + tiny + vertex);
    let per_e = total as f64 / 40_000.0;
    println!(
        "tiny_memory/uniform10k: total={total} ({per_e:.1} B/e) slab={slab} tree={tree} logs={logs_pure} tiny={tiny} vtx={vertex} tiny_rows={} log_chunks={} insert={ins_ms:.1}ms",
        g.tiny_row_count(),
        g.log_allocated_chunks()
    );
    assert_eq!(g.tiny_row_count(), 10_000);
    assert_eq!(tiny, 320_000, "10k rows × 32 B");
    assert_eq!(g.log_allocated_chunks(), 0);
    assert!(
        total / 40_000 >= 12 && total / 40_000 <= 18,
        "uniform total must be in the 14-16 B/e band, got {per_e:.1}"
    );
    // Hub-10k: the hub row spills once and settles slab/tree; no row
    // stays tiny.
    let mut h: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 20_480).unwrap();
    for d in 0..10_000u32 {
        h.insert_edge(1, d);
    }
    let h_total = h.memory_bytes_total();
    let h_tiny = h.tiny_memory_bytes();
    println!(
        "tiny_memory/hub10k: total={h_total} ({:.1} B/e) tiny={h_tiny} tiny_rows={} log_chunks={} rb={} pr={}",
        h_total as f64 / 10_000.0,
        h.tiny_row_count(),
        h.log_allocated_chunks(),
        h.rebalance_count(),
        h.promote_count()
    );
    assert_eq!(h.tiny_row_count(), 0);
    assert_eq!((h.rebalance_count(), h.promote_count()), (3, 1));
    h.verify_invariants();
}

#[test]
fn zero_log_persists_more_than_large_log() {
    use lara_graph::memory::{PersistCounterMemory, PersistCounters};
    use std::rc::Rc;
    fn hub_persisted(cap: usize) -> u64 {
        let counters = Rc::new(PersistCounters::default());
        let mem = LaraMemories {
            vertices: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            edges: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            logs: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            counts: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            spans: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            free_spans: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            blocks: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
            buckets: PersistCounterMemory::with_counters(VecMemory::new(), &counters),
        };
        let mut g: LaraGraph<Vertex, Edge, PersistCounterMemory<VecMemory>> =
            LaraGraph::new_with_max_log_entries(mem, 1024, cap).unwrap();
        // Slab-only: promotion would evacuate the slab at T=1024 and mute
        // the rebalance-volume difference this test measures. Cold-fill
        // every vertex first so the victim leaf holds cold data: each
        // no-log rebalance then rewrites the whole leaf span to absorb a
        // few overflow edges, while the large log batches them (16 B per
        // entry + one fold).
        g.set_promote_threshold(None);
        for v in 0..1024u32 {
            for d in 0..64u32 {
                g.insert_edge(v, 10_000_000 + v * 1000 + d);
            }
        }
        counters.reset();
        for d in 0..2000u32 {
            g.insert_edge(1, d);
        }
        g.verify_invariants();
        counters.bytes()
    }
    let b0 = hub_persisted(0);
    let b512 = hub_persisted(512);
    assert!(
        b0 > b512,
        "no-log must persist more than cap=512 (0={b0}, 512={b512})"
    );
}

#[cfg(feature = "mmap")]
#[test]
fn mmap_memory_roundtrip() {
    use lara_graph::MmapMemory;
    use lara_graph::memory::Memory;
    let dir = std::env::temp_dir().join(format!("lara_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let m = MmapMemory::create(dir.join("r.bin"), 1 << 16).unwrap();
    let v = VecMemory::new();
    let payload: Vec<u8> = (0..256u32).map(|i| (i % 251) as u8).collect();
    m.write(100, &payload);
    v.write(100, &payload);
    let mut a = vec![0u8; 256];
    let mut b = vec![0u8; 256];
    m.read(100, &mut a);
    v.read(100, &mut b);
    assert_eq!(a, b);
    // Relative-grow semantics: `grow(2)` adds two pages to the current
    // size (mmap starts with one preallocated page since an empty file
    // cannot be mapped; `VecMemory::grow` is allocation-hint-only and
    // `write` auto-extends, so only the mmap side is asserted exactly).
    let m0 = m.size();
    m.grow(2).unwrap();
    assert_eq!(m.size(), m0 + 2);
    // Functional parity at a far offset (past the initial page).
    let far: Vec<u8> = (0..64u32).map(|i| (i * 7 % 253) as u8).collect();
    m.write(100_000, &far);
    v.write(100_000, &far);
    let mut e = vec![0u8; 64];
    let mut f = vec![0u8; 64];
    m.read(100_000, &mut e);
    v.read(100_000, &mut f);
    assert_eq!(e, f);
    // Zero-fill past end matches.
    let mut c = vec![9u8; 16];
    let mut d = vec![9u8; 16];
    m.read(1 << 30, &mut c);
    v.read(1 << 30, &mut d);
    assert_eq!(c, d);
    m.flush().unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn unlabeled_insert_record_api_parity() {
    // insert_edge(src, dst) and insert_edge_record(src, Edge::with_neighbor(...))
    // must produce identical graphs.
    let mut a: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 8).unwrap();
    let mut b: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 8).unwrap();
    for d in 0..30u32 {
        a.insert_edge(3, d);
        b.insert_edge_record(3, Edge::with_neighbor(Edge::default(), d));
    }
    assert_eq!(a.num_edges(), b.num_edges());
    assert_eq!(a.out_degree(3), b.out_degree(3));
    assert_eq!(a.out_neighbors_vec(3), b.out_neighbors_vec(3));
    a.verify_invariants();
    b.verify_invariants();
}

// -- Phase A: cap alignment (T_PROMOTE/T_DEMOTE hysteresis, demotion) ------

#[test]
fn tree_demotes_below_t_demote() {
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 8).unwrap();
    // Grow past T_PROMOTE → promote.
    for d in 0u32..(T_PROMOTE + 16) {
        g.insert_edge(2, d);
    }
    assert!(
        g.tree_bucket_count() >= 1,
        "expected promotion at T_PROMOTE"
    );

    // Remove down to T_DEMOTE → demote back to slab.
    let target = T_DEMOTE;
    let mut removed = 0u32;
    for d in 0..(T_PROMOTE + 16) {
        if g.out_degree(2) <= target as u64 {
            break;
        }
        assert!(g.remove_edge(2, d));
        removed += 1;
    }
    assert_eq!(g.out_degree(2), target as u64);
    assert_eq!(g.tree_bucket_count(), 0, "expected demotion below T_DEMOTE");

    // Remaining edges are the ones not removed.
    let remaining: Vec<u32> = (0..(T_PROMOTE + 16)).filter(|&d| d >= removed).collect();
    assert_eq!(sorted(g.out_neighbors_vec(2)), remaining);
    g.verify_invariants();
}

#[test]
fn tree_demote_slab_geometry_correct() {
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 8).unwrap();
    for d in 0u32..(T_PROMOTE + 16) {
        g.insert_edge(2, d);
    }
    assert!(g.tree_bucket_count() >= 1);

    // Demote below T_DEMOTE.
    let target = T_DEMOTE;
    for d in 0..(T_PROMOTE + 16) {
        if g.out_degree(2) <= target as u64 {
            break;
        }
        g.remove_edge(2, d);
    }
    assert_eq!(g.tree_bucket_count(), 0, "expected demotion");

    // Slab geometry: the demoted vertex is a slab row with the live edges
    // readable from the slab.
    let row = g.get_row_pub(2);
    assert_eq!(row.tree_bucket(), NO_BUCKET);
    assert_eq!(row.stored_degree(), target);
    assert_eq!(g.out_degree(2), target as u64);
    let mut nbrs = g.out_neighbors_vec(2);
    nbrs.sort();
    assert_eq!(
        nbrs,
        ((T_PROMOTE + 16 - target)..(T_PROMOTE + 16)).collect::<Vec<u32>>()
    );
    g.verify_invariants();
}

#[test]
fn tree_no_promote_demote_oscillation() {
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 8).unwrap();
    for d in 0u32..(T_PROMOTE + 16) {
        g.insert_edge(2, d);
    }
    assert!(g.tree_bucket_count() >= 1);

    // Demote below T_DEMOTE.
    let target = T_DEMOTE;
    for d in 0..(T_PROMOTE + 16) {
        if g.out_degree(2) <= target as u64 {
            break;
        }
        g.remove_edge(2, d);
    }
    assert_eq!(g.tree_bucket_count(), 0);

    // Insert/remove churn just above T_DEMOTE must not re-promote (needs
    // T_PROMOTE - T_DEMOTE = 512 inserts).
    for i in 0..100u32 {
        g.insert_edge(2, 100_000 + i);
        assert!(g.remove_edge(2, 100_000 + i));
        assert_eq!(g.tree_bucket_count(), 0, "oscillation at iteration {i}");
    }
    g.verify_invariants();
}

#[test]
fn dense_churn_does_not_accumulate_stored_slots() {
    // Use public mutations rather than manufacturing a sparse prefix
    // inconsistent with the new default's live/stored contract.
    let edges: Vec<_> = (0..30).map(|d| (2, d)).collect();
    let mut g: LaraGraph<Vertex, Edge, VecMemory> =
        LaraGraph::with_edge_list(fresh_mem(), &edges, 8).unwrap();
    g.set_promote_threshold(Some(64));
    for d in 100..196 {
        g.insert_edge(2, d);
        assert!(g.remove_edge(2, d));
        assert_eq!(g.get_row_pub(2).stored_degree(), 30);
        assert_eq!(g.tree_bucket_count(), 0);
        assert_eq!(sorted(g.out_neighbors_vec(2)), (0..30).collect::<Vec<_>>());
        g.verify_invariants();
    }
    for d in 100..134 {
        g.insert_edge(2, d);
    }
    assert_eq!(g.tree_bucket_count(), 1);
    assert_eq!(
        sorted(g.out_neighbors_vec(2)),
        (0..30).chain(100..134).collect::<Vec<_>>()
    );
    g.verify_invariants();
}

#[test]
fn slab_block_scan_matches_edge_scan() {
    // Block-granular slab scan must return exactly the inserted set.
    // hub-3k at T=4096 stays slab (same fixture as the sweep slab arm);
    // boundaries: empty row, 1 edge, chunk-exact 128 (512 B, old loop),
    // 129 (fast-path entry), 8191 (32 KiB buffer edge — 8192 itself
    // promotes at T=8192), 9000 slab-only (chunked fallback past the
    // buffer).
    for &size in &[0u32, 1, 127, 128, 129, 3000, 8191] {
        let mut g: LaraGraph<Vertex, Edge, VecMemory> =
            LaraGraph::new(fresh_mem(), 16_384).unwrap();
        g.set_promote_threshold(Some(8192));
        for d in 0..size {
            g.insert_edge(1, d);
        }
        assert_eq!(g.tree_bucket_count(), 0, "must stay slab at size {size}");
        let got = sorted(g.out_neighbors_vec(1));
        let want: Vec<u32> = (0..size).collect();
        assert_eq!(got, want, "slab scan mismatch at size {size}");
        g.verify_invariants();
    }
    // Past the 32 KiB single-read buffer: slab-only arm (no promote,
    // no cap), 9000 edges = 36 KiB span → chunked fallback.
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 16_384).unwrap();
    g.set_promote_threshold(None);
    for d in 0..9000u32 {
        g.insert_edge(1, d);
    }
    assert_eq!(g.tree_bucket_count(), 0);
    assert_eq!(
        sorted(g.out_neighbors_vec(1)),
        (0..9000u32).collect::<Vec<_>>()
    );
    g.verify_invariants();
    // Dispersed uniform n=1k/m=4k: every row's scan matches its stream
    // slice (multiset order irrelevant — sorted compare).
    let stream: Vec<(u32, u32)> = (0..4000u32)
        .map(|i| ((i * 7 + 3) % 1000, (i * 13 + 5) % 1000))
        .collect();
    let mut g: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(fresh_mem(), 1024).unwrap();
    for &(a, b) in &stream {
        g.insert_edge(a, b);
    }
    for v in 0..1000u32 {
        let mut want: Vec<u32> = stream
            .iter()
            .filter(|&&(a, _)| a == v)
            .map(|&(_, b)| b)
            .collect();
        want.sort_unstable();
        assert_eq!(sorted(g.out_neighbors_vec(v)), want, "row {v}");
    }
    g.verify_invariants();
    // Labeled slab path shares the span reader: hub-3k at T=4096 stays
    // slab; both entry points must return the inserted set.
    use lara_graph::labeled::{LabelId, LabeledLaraGraph};
    let mut l: LabeledLaraGraph<VecMemory> = LabeledLaraGraph::new(fresh_labeled_mem(), 16_384)
        .unwrap()
        .with_promote_threshold(Some(4096));
    for d in 0..3000u32 {
        l.insert_edge(0, 100 + d, LabelId::new(0));
    }
    assert!(!l.bucket_is_tree(0, LabelId::new(0)));
    let mut got = Vec::new();
    l.for_each_edge_of_label(0, LabelId::new(0), |n| got.push(n));
    assert_eq!(sorted(got), (100..3100u32).collect::<Vec<_>>());
    let mut got2 = Vec::new();
    l.for_each_labeled(0, |n, _| got2.push(n));
    assert_eq!(sorted(got2), (100..3100u32).collect::<Vec<_>>());
    l.verify_invariants();
}
