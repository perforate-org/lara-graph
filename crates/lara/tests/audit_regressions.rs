//! Regression coverage for docs/audit-2026-09-11/report.md.
//! These tests use public mutation paths; multisets retain parallel edges.
#![cfg(feature = "std")]

use lara_graph::block_store::LtbRawBlockStore;
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::labeled::{LabelId, LabeledLaraGraph};
use lara_graph::memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
use lara_graph::traits::{CsrEdge, CsrEdgeTombstone, CsrVertex, FixedWidth};
use lara_graph::tree_csr::{B, R_MAX, TreeCsrBucket};
use std::cell::{Cell, RefCell};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

type Graph = LaraGraph<Vertex, Edge, VecMemory>;

fn memories<M: Memory + Default>() -> LaraMemories<M> {
    LaraMemories {
        vertices: M::default(),
        edges: M::default(),
        logs: M::default(),
        counts: M::default(),
        spans: M::default(),
        free_spans: M::default(),
        blocks: M::default(),
        buckets: M::default(),
    }
}

fn sorted(mut edges: Vec<u32>) -> Vec<u32> {
    edges.sort_unstable();
    edges
}

#[test]
fn middle_tree_preserves_next_slab_boundary() {
    let mut g = Graph::new_with_segments(memories(), 4, 4).unwrap();
    let mut pairs = vec![(0, 10); 2];
    pairs.extend(vec![(1, 100); 1024]);
    pairs.extend([(2, 200), (2, 201)]);
    g.build_pairs(&pairs).unwrap();
    g.insert_edge(1, 101);
    assert_ne!(g.get_row_pub(1).tree_bucket(), u32::MAX);
    g.rebalance_leaf(0);
    for dst in [500, 501, 502] {
        g.insert_edge(0, dst);
    }
    assert_eq!(g.out_neighbors_vec(2), vec![200, 201]);
    assert_eq!(sorted(g.out_neighbors_vec(0)), vec![10, 10, 500, 501, 502]);
    g.verify_invariants();
}

#[test]
fn promotion_preserves_sibling_log_entries() {
    let mut g = Graph::new_with_segments(memories(), 4, 4).unwrap();
    g.build_pairs(&vec![(0, 40); 512]).unwrap();
    for i in 0..10 {
        g.insert_edge(2, 200 + i);
    }
    let before = sorted(g.out_neighbors_vec(2));
    assert!(g.log_idx_pub(0) > 0);
    for i in 0..512 {
        g.insert_edge(0, 100 + i);
    }
    assert_ne!(g.get_row_pub(0).tree_bucket(), u32::MAX);
    for i in 0..5 {
        g.insert_edge(3, 300 + i);
    }
    assert_eq!(sorted(g.out_neighbors_vec(2)), before);
    g.verify_invariants();
}

#[test]
fn oversized_core_insert_is_rejected_before_mutation() {
    let limit = B * R_MAX;
    let mut g = Graph::with_edge_list(memories(), &vec![(0, 7); limit], 4).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| g.insert_edge(0, 8))).is_err());
    assert_eq!(g.num_edges(), limit as u64);
    assert_eq!(g.out_neighbors_vec(0), vec![7; limit]);
    assert_eq!(g.tree_bucket_count(), 0);
    g.verify_invariants();
}

#[test]
fn oversized_sorted_run_is_rejected_before_mutation() {
    let n = B * R_MAX - 2;
    let mut g = Graph::with_edge_list(memories(), &vec![(0, 7); n], 4).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| g.insert_sorted_run(0, &[1, 2, 3]))).is_err());
    assert_eq!(g.num_edges(), n as u64);
    assert_eq!(g.out_neighbors_vec(0), vec![7; n]);
    g.verify_invariants();
}

#[test]
fn oversized_labeled_insert_is_rejected_before_mutation() {
    let n = B * R_MAX;
    let label = LabelId::new(0);
    let mut g: LabeledLaraGraph<VecMemory> =
        LabeledLaraGraph::with_edge_list(memories(), &vec![(0, 7, label); n], 4).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| g.insert_edge(0, 8, label))).is_err());
    let mut got = Vec::new();
    g.for_each_edge_of_label(0, label, |dst| got.push(dst));
    assert_eq!(got, vec![7; n]);
    assert!(!g.bucket_is_tree(0, label));
    g.verify_invariants();
}

#[test]
fn standalone_bucket_checks_promotion_and_append_capacity() {
    let mut bucket = TreeCsrBucket::new(LtbRawBlockStore::new(VecMemory::new()).unwrap());
    let limit = B * R_MAX;
    assert!(
        catch_unwind(AssertUnwindSafe(|| bucket.promote_from_slice(&vec![
            7;
            limit
                + 1
        ])))
        .is_err()
    );
    assert_eq!(bucket.stored_slots(), 0);
    assert_eq!(bucket.root_len(), 0);
    bucket.promote_from_slice(&vec![7; limit]);
    assert!(catch_unwind(AssertUnwindSafe(|| bucket.insert(8))).is_err());
    assert_eq!(bucket.stored_slots(), limit as u32);
    assert_eq!(bucket.read_slot(limit as u32 - 1), 7);
}

#[test]
fn reused_tombstones_restore_leaf_counts() {
    let mut g = Graph::with_edge_list(memories(), &[(0, 1), (0, 2)], 4).unwrap();
    let room = g.row_allocated(0) as u32;
    for i in 2..room {
        g.insert_edge(0, 100 + i);
    }
    for dst in [1, 900, 901, 902] {
        assert!(g.remove_edge(0, dst));
        g.insert_edge(0, if dst == 1 { 900 } else { dst + 1 });
        assert_eq!(g.out_degree(0), room as u64);
        g.verify_invariants();
    }
}

#[derive(Clone, Copy, Default)]
struct Encoded(u32);
impl FixedWidth for Encoded {
    const BYTES: usize = 4;
    fn read_from(bytes: &[u8]) -> Self {
        Self(u32::from_le_bytes(bytes.try_into().unwrap()) ^ 0x8000_0000)
    }
    fn write_to(&self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&(self.0 ^ 0x8000_0000).to_le_bytes());
    }
}
impl CsrEdge for Encoded {
    fn neighbor(&self) -> u32 {
        self.0
    }
    fn with_neighbor(self, dst: u32) -> Self {
        Self(dst)
    }
}
impl CsrEdgeTombstone for Encoded {
    fn tombstone_edge() -> Self {
        Self(u32::MAX)
    }
    fn is_tombstone(&self) -> bool {
        self.0 == u32::MAX
    }
}

#[test]
fn typed_codec_applies_to_small_single_read_and_large_slab_scans() {
    for n in [1, 129, 9000] {
        let pairs: Vec<_> = (0..n).map(|dst| (0, dst)).collect();
        let mut g =
            LaraGraph::<Vertex, Encoded, VecMemory>::with_edge_list(memories(), &pairs, 4).unwrap();
        g.set_promote_threshold(None);
        assert_eq!(g.out_neighbors_vec(0), (0..n).collect::<Vec<_>>());
        assert!(g.remove_edge(0, 0));
        let mut got = g.out_neighbors_vec(0);
        got.sort_unstable();
        assert_eq!(got, (1..n).collect::<Vec<_>>());
        g.verify_invariants();
    }
}

#[test]
fn tiny_fold_returns_chunk_for_another_leaf() {
    let mut g = Graph::new(memories(), 64).unwrap();
    for i in 0..4 {
        g.insert_edge(0, i);
    }
    g.verify_invariants();
    let bytes = g.tiny_memory_bytes();
    g.rebalance_leaf(0);
    assert_eq!(g.tiny_row_count(), 0);
    assert_eq!(g.tiny_pool_depth(), 1);
    g.insert_edge(63, 42);
    assert_eq!(g.tiny_memory_bytes(), bytes);
    assert_eq!(g.tiny_pool_depth(), 0);
    g.verify_invariants();
}

#[test]
fn emptied_log_row_can_reuse_tiny_without_leaking() {
    let mut g = Graph::new(memories(), 64).unwrap();
    for _ in 0..3 {
        for i in 0..9 {
            g.insert_edge(0, i);
        }
        assert!(!g.row_is_tiny(0));
        for i in 0..9 {
            assert!(g.remove_edge(0, i));
        }
        g.insert_edge(0, 42);
        assert_eq!(g.out_neighbors_vec(0), vec![42]);
        assert!(g.row_is_tiny(0));
        assert!(g.remove_edge(0, 42));
        assert_eq!(g.tiny_memory_bytes(), 32);
        g.verify_invariants();
    }
}

#[test]
fn log_capacity_does_not_disable_fresh_tiny_rows() {
    for cap in [0, 170] {
        let mut g = Graph::new_with_max_log_entries(memories(), 64, cap).unwrap();
        for dst in 0..8 {
            g.insert_edge(0, dst);
            assert!(g.row_is_tiny(0));
            g.verify_invariants();
        }
        g.insert_edge(0, 8);
        assert!(!g.row_is_tiny(0));
        assert_eq!(sorted(g.out_neighbors_vec(0)), (0..9).collect::<Vec<_>>());
        g.verify_invariants();
    }
}

#[test]
fn customized_threshold_uses_half_threshold_for_demotion() {
    for threshold in [64, 2048] {
        let mut g = Graph::new(memories(), 64).unwrap();
        g.set_promote_threshold(Some(threshold));
        for dst in 0..threshold + 16 {
            g.insert_edge(0, dst);
        }
        g.rebalance_leaf(0);
        g.insert_edge(0, threshold + 16);
        assert_eq!(g.tree_bucket_count(), 1);
        let mut dst = threshold + 16;
        while g.out_degree(0) > u64::from(threshold / 2 + 1) {
            assert!(g.remove_edge(0, dst));
            dst -= 1;
            assert_eq!(g.tree_bucket_count(), 1);
        }
        assert!(g.remove_edge(0, dst));
        assert_eq!(g.out_degree(0), u64::from(threshold / 2));
        assert_eq!(g.tree_bucket_count(), 0);
        g.verify_invariants();
    }
}

#[test]
fn allocation_advances_tail_once_and_reuses_released_range() {
    let mut g = Graph::new(memories(), 64).unwrap();
    let before = g.slab_capacity();
    let first = g.allocate_span(100);
    assert_eq!(first, before);
    assert_eq!(g.slab_capacity(), before + 100);
    g.release_span(first, 100);
    assert_eq!(g.allocate_span(100), first);
    assert_eq!(g.slab_capacity(), before + 100);
}

#[test]
fn repeated_repacking_reuses_slab_and_tiny_allocations() {
    let mut g = Graph::new_with_segments(memories(), 16, 8).unwrap();
    for src in 0..8 {
        for dst in 0..4 {
            g.insert_edge(src, dst);
        }
    }
    g.rebalance_leaf(0);
    g.rebalance_leaf(0);
    let capacity = g.slab_capacity();
    for _ in 0..32 {
        g.rebalance_leaf(0);
        assert_eq!(g.slab_capacity(), capacity);
        assert_eq!(g.tiny_pool_depth(), 8);
        g.verify_invariants();
    }
}

#[test]
fn vec_growth_adds_pages() {
    let mem = VecMemory::new();
    assert_eq!(mem.grow(1).unwrap(), 1);
    mem.write(PAGE_SIZE - 1, &[42]);
    assert_eq!(mem.grow(1).unwrap(), 2);
    let mut b = [0];
    mem.read(PAGE_SIZE - 1, &mut b);
    assert_eq!(b, [42]);
    assert_eq!(mem.grow(0).unwrap(), 2);
    assert!(mem.grow(u64::MAX).is_err());
    assert_eq!(mem.size(), 2);
}

/// Unlike VecMemory/MmapMemory, never auto-extends writes.
#[derive(Clone, Default)]
struct StrictMemory {
    bytes: Rc<RefCell<Vec<u8>>>,
    fail_growth: Rc<Cell<bool>>,
}
impl Memory for StrictMemory {
    fn read(&self, offset: u64, buf: &mut [u8]) {
        let bytes = self.bytes.borrow();
        let start = offset as usize;
        buf.copy_from_slice(&bytes[start..start + buf.len()]);
    }
    fn write(&self, offset: u64, buf: &[u8]) {
        let mut bytes = self.bytes.borrow_mut();
        let start = offset as usize;
        bytes[start..start + buf.len()].copy_from_slice(buf);
    }
    fn size(&self) -> u64 {
        self.bytes.borrow().len() as u64 / PAGE_SIZE
    }
    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        let current = self.size();
        if self.fail_growth.get() {
            return Err(GrowFailed {
                current_pages: current,
                delta_pages: pages,
            });
        }
        self.bytes
            .borrow_mut()
            .resize(((current + pages) * PAGE_SIZE) as usize, 0);
        Ok(current + pages)
    }
}

#[test]
fn metadata_regions_need_no_backing_for_core_lifecycle() {
    type G = LaraGraph<Vertex, Edge, StrictMemory>;
    for n in [0, 1, 64] {
        for constructor in 0..5 {
            let mem = memories::<StrictMemory>();
            let metadata = [mem.vertices.clone(), mem.counts.clone()];
            for region in &metadata {
                region.fail_growth.set(true);
            }
            let mut g = match constructor {
                0 => G::new(mem, n),
                1 => G::new_with_max_log_entries(mem, n, 0),
                2 => G::new_with_segments(mem, n, 8),
                3 => G::new_with_segments_and_max_log(mem, n, 8, 17),
                _ => G::with_edge_list(mem, &[], n),
            }
            .expect("heap metadata must not require backend growth");
            assert_eq!(g.region_offsets(), (8, 4096));
            g.insert_sorted_run(0, &[0, 0, 0, 0]);
            assert!(g.remove_edge(0, 0));
            g.rebalance_leaf(0);
            assert_eq!(g.out_neighbors_vec(0), vec![0; 3]);
            assert_eq!(g.num_edges(), 3);
            g.verify_invariants();
            g.build_pairs(&[(0, 0)]).unwrap();
            assert_eq!(g.out_neighbors_vec(0), vec![0]);
            assert_eq!(g.num_edges(), 1);
            g.verify_invariants();
            for region in &metadata {
                assert_eq!(region.size(), 0);
            }
        }
    }
}

#[test]
fn metadata_regions_need_no_backing_for_labeled_lifecycle() {
    for constructor in 0..3 {
        let mem = memories::<StrictMemory>();
        let metadata = [mem.vertices.clone(), mem.counts.clone()];
        for region in &metadata {
            region.fail_growth.set(true);
        }
        let mut g = match constructor {
            0 => LabeledLaraGraph::<StrictMemory>::new(mem, 64),
            1 => LabeledLaraGraph::new_with_max_log_entries(mem, 64, 17),
            _ => LabeledLaraGraph::with_edge_list(mem, &[], 64),
        }
        .expect("labeled heap metadata must not require backend growth")
        .with_promote_threshold(Some(16));
        let label = LabelId::new(0);
        let other = LabelId::new(1);
        for _ in 0..20 {
            g.insert_edge(0, 0, label);
        }
        g.insert_edge(0, 0, other);
        assert!(g.bucket_is_tree(0, label));
        for _ in 0..12 {
            assert!(g.remove_edge(0, 0, label));
        }
        assert!(!g.bucket_is_tree(0, label));
        g.rebalance_leaf(0);
        assert_eq!(g.num_edges(), 9);
        let mut got = Vec::new();
        g.for_each_edge_of_label(0, label, |dst| got.push(dst));
        assert_eq!(got, vec![0; 8]);
        got.clear();
        g.for_each_edge_of_label(0, other, |dst| got.push(dst));
        assert_eq!(got, vec![0]);
        g.verify_invariants();
        for region in &metadata {
            assert_eq!(region.size(), 0);
        }
    }
}

#[test]
fn metadata_regions_preserve_preallocated_bytes() {
    let expected = vec![0xa5; PAGE_SIZE as usize];
    for labeled in [false, true] {
        let mem = memories::<StrictMemory>();
        let metadata = [mem.vertices.clone(), mem.counts.clone()];
        for region in &metadata {
            region.grow(1).unwrap();
            region.write(0, &expected);
            region.fail_growth.set(true);
        }
        if labeled {
            let label = LabelId::new(0);
            let mut g = LabeledLaraGraph::<StrictMemory>::with_edge_list(mem, &[(0, 0, label)], 64)
                .unwrap();
            g.insert_edge(0, 0, label);
            assert!(g.remove_edge(0, 0, label));
            g.rebalance_leaf(0);
            assert_eq!(g.out_labeled_vec(0), vec![(0, label)]);
            assert_eq!(g.num_edges(), 1);
            g.verify_invariants();
        } else {
            let mut g = LaraGraph::<Vertex, Edge, StrictMemory>::with_edge_list(mem, &[(0, 0)], 64)
                .unwrap();
            g.insert_edge(0, 0);
            assert!(g.remove_edge(0, 0));
            g.rebalance_leaf(0);
            assert_eq!(g.out_neighbors_vec(0), vec![0]);
            assert_eq!(g.num_edges(), 1);
            g.verify_invariants();
        }
        for region in &metadata {
            assert_eq!(region.size(), 1);
            let mut got = vec![0; expected.len()];
            region.read(0, &mut got);
            assert_eq!(got, expected, "unused metadata bytes were modified");
        }
    }
}

#[test]
fn required_constructor_regions_still_report_growth_failure() {
    for rejected in 0..5 {
        let mem = memories::<StrictMemory>();
        let required = [
            &mem.edges,
            &mem.logs,
            &mem.spans,
            &mem.free_spans,
            &mem.blocks,
        ];
        required[rejected].fail_growth.set(true);
        assert!(matches!(
            LaraGraph::<Vertex, Edge, StrictMemory>::new(mem, 64),
            Err(GrowFailed {
                current_pages: 0,
                delta_pages: 1
            }),
        ));
    }
}

#[test]
fn block_payloads_are_reserved_across_page_boundaries() {
    let mut store = LtbRawBlockStore::new(StrictMemory::default()).unwrap();
    for _ in 0..32 {
        store.mint_zeroed().unwrap();
    }
    let mut g = LaraGraph::<Vertex, Edge, StrictMemory>::new(memories(), 64).unwrap();
    for dst in 0..20_000 {
        g.insert_edge(0, dst);
    }
    assert_eq!(g.out_neighbors_vec(0), (0..20_000).collect::<Vec<_>>());
    g.rebalance_leaf(0);
    g.verify_invariants();
}

#[test]
fn failed_slab_growth_does_not_publish_capacity() {
    let mem = memories::<StrictMemory>();
    let edges = mem.edges.clone();
    let mut g = LaraGraph::<Vertex, Edge, StrictMemory>::new(mem, 64).unwrap();
    let before = g.slab_capacity();
    edges.fail_growth.set(true);
    assert!(g.ensure_slab(100_000).is_err());
    assert_eq!(g.slab_capacity(), before);
}

#[test]
fn rebuild_retires_tiny_and_tree_owners() {
    let mut g = Graph::new(memories(), 64).unwrap();
    g.insert_edge(63, 1);
    for dst in 0..1500 {
        g.insert_edge(0, dst);
    }
    assert!(g.tree_bucket_count() > 0);
    g.build_pairs(&[(0, 7), (1, 8)]).unwrap();
    assert_eq!(g.tree_bucket_count(), 0);
    assert_eq!(g.tiny_row_count(), 0);
    assert_eq!(g.out_neighbors_vec(0), vec![7]);
    assert_eq!(g.out_neighbors_vec(1), vec![8]);
    assert!(g.out_neighbors_vec(63).is_empty());
    g.verify_invariants();
}

#[test]
fn labeled_demotion_preserves_siblings_and_configured_hysteresis() {
    let label = LabelId::new(0);
    let other = LabelId::new(1);
    let mut g = LabeledLaraGraph::<StrictMemory>::new(memories(), 16)
        .unwrap()
        .with_promote_threshold(Some(64));
    for dst in 0..101 {
        g.insert_edge(0, dst, label);
    }
    for dst in 200..203 {
        g.insert_edge(0, dst, other);
    }
    for dst in 300..370 {
        g.insert_edge(1, dst, label);
    }
    assert!(g.bucket_is_tree(0, label));
    assert!(g.bucket_is_tree(1, label));
    for dst in (33..101).rev() {
        assert!(g.remove_edge(0, dst, label));
        assert!(g.bucket_is_tree(0, label));
    }
    assert!(g.remove_edge(0, 32, label));
    assert!(!g.bucket_is_tree(0, label));
    assert!(g.bucket_is_tree(1, label));
    assert_eq!(
        g.out_labeled_vec(0)
            .iter()
            .filter(|(_, l)| *l == other)
            .count(),
        3
    );
    g.insert_edge(1, 370, label);
    let mut neighbors = Vec::new();
    g.for_each_edge_of_label(1, label, |dst| neighbors.push(dst));
    assert_eq!(neighbors, (300..371).collect::<Vec<_>>());
    g.verify_invariants();
}

#[test]
fn zero_and_small_thresholds_do_not_promote_tiny_descriptors() {
    for threshold in [0, 1, 4, 8] {
        let mut g = Graph::new(memories(), 16).unwrap();
        g.set_promote_threshold(Some(threshold));
        for src in 0..4 {
            for dst in 0..12 {
                g.insert_edge(src, dst);
            }
        }
        for src in 0..4 {
            assert_eq!(
                sorted(g.out_neighbors_vec(src)),
                (0..12).collect::<Vec<_>>()
            );
        }
        g.verify_invariants();
    }
}

#[test]
fn mixed_labeled_churn_preserves_per_label_multisets() {
    let mut g = LabeledLaraGraph::<StrictMemory>::new(memories(), 16)
        .unwrap()
        .with_promote_threshold(Some(64));
    let mut expected = vec![vec![Vec::new(); 3]; 4];
    let mut state = 4321u64;
    for step in 0..2500 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let src = state as usize % 4;
        let label = (state >> 8) as usize % 3;
        let dst = ((state >> 16) % 200) as u32;
        let row = &mut expected[src][label];
        if step % 4 == 0 && !row.is_empty() {
            let old = row.remove(dst as usize % row.len());
            assert!(g.remove_edge(src as u32, old, LabelId::new(label as u16)));
        } else {
            row.push(dst);
            g.insert_edge(src as u32, dst, LabelId::new(label as u16));
        }
        if step % 41 == 0 {
            g.rebalance_leaf(src as u64 / u64::from(g.segment_size()));
        }
        if step % 13 == 0 {
            for (v, labels) in expected.iter().enumerate() {
                for (l, want) in labels.iter().enumerate() {
                    let mut got = Vec::new();
                    g.for_each_edge_of_label(v as u32, LabelId::new(l as u16), |n| got.push(n));
                    assert_eq!(
                        sorted(got),
                        sorted(want.clone()),
                        "step={step}, vertex={v}, label={l}"
                    );
                }
            }
            g.verify_invariants();
        }
    }
}

#[test]
fn mixed_source_churn_preserves_multisets() {
    for cap in [0, 32, 170] {
        let mut g = Graph::new_with_segments_and_max_log(memories(), 16, 8, cap).unwrap();
        g.set_promote_threshold(Some(64));
        let mut expected: Vec<Vec<u32>> = vec![Vec::new(); 16];
        let mut state = 1234u64;
        for step in 0..3000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let src = (state as usize) % 16;
            let dst = ((state >> 8) % 200) as u32;
            if step % 4 == 0 && !expected[src].is_empty() {
                let index = dst as usize % expected[src].len();
                let old = expected[src].remove(index);
                assert!(g.remove_edge(src as u32, old));
            } else {
                expected[src].push(dst);
                g.insert_edge(src as u32, dst);
            }
            if step % 41 == 0 {
                g.rebalance_leaf(src as u64 / u64::from(g.segment_size()));
            }
            if step % 13 == 0 {
                for (v, want) in expected.iter().enumerate() {
                    assert_eq!(
                        sorted(g.out_neighbors_vec(v as u32)),
                        sorted(want.clone()),
                        "cap={cap}, step={step}, vertex={v}"
                    );
                }
                g.verify_invariants();
            }
        }
    }
}
