//! A declared valid prepared log representation, not a natural pending-log workload.
//! Setup uses existing owners; every measured operation is a public graph call.
use super::*;
use crate::graph::LaraMemories;
use crate::{PAGE_SIZE, VecMemory};
#[path = "../../tests/support/label_order.rs"]
mod support;
use support::*;
type Graph = LabeledLaraGraph<Probe>;
fn l(n: u16) -> LabelId {
    LabelId::new(n)
}
fn neighbors(g: &Graph, src: u32, label: u16) -> Vec<u32> {
    let mut out = Vec::new();
    g.for_each_edge_of_label(src, l(label), |d| out.push(d));
    out
}
fn stage(g: &mut Graph, src: u32, label: u16, values: &[u32]) {
    let (pos, mut b) = g.find_bucket(src, label).unwrap();
    let leaf = src as u64 / g.segment_size() as u64;
    for &dst in values {
        let idx = g.inner.log_idx_pub(leaf);
        assert!((idx as usize) < g.max_log_entries());
        g.inner.write_log_pub(
            leaf,
            idx,
            &LogEntry {
                src,
                dst,
                label,
                _pad: 0,
                prev: b.log_head,
            },
        );
        g.inner.set_log_idx_pub(leaf, idx + 1);
        b.log_head = idx as i32;
        b.degree += 1;
        g.inner.add_edges(1);
    }
    g.set_bucket(src, pos, &b);
    g.verify_invariants();
}

#[test]
fn prepared_shared_logs_visit_and_remove_oldest_without_overtaking() {
    let r = regions();
    let mut g = Graph::new(bundle(&r), 64)
        .unwrap()
        .with_promote_threshold(None);
    g.set_label_order(l(0), AdjacencyOrder::Insertion).unwrap();
    for d in [3, 1] {
        g.insert_edge(8, d, l(0));
    }
    for d in [u32::MAX, 9] {
        g.insert_edge(8, d, l(1));
    }
    g.insert_edge(9, 22, l(1));
    stage(&mut g, 8, 0, &[5, 2, 7, 5]);
    stage(&mut g, 8, 1, &[4, 6]);
    stage(&mut g, 9, 1, &[44]);
    assert!(g.bucket_room(8, l(0)) > 0);
    let (got, _) = observe("prepared_ordered_log_scan", &r, || neighbors(&g, 8, 0));
    assert_eq!(got, [3, 1, 5, 2, 7, 5]);
    assert_eq!(neighbors(&g, 8, 1), [u32::MAX, 9, 6, 4]);
    let (found, t) = observe("prepared_ordered_log_duplicate", &r, || {
        g.remove_edge(8, 5, l(0))
    });
    assert!(found);
    assert_eq!(neighbors(&g, 8, 0), [3, 1, 2, 7, 5]);
    assert_eq!(t[2].writes.len(), 2);
    assert_eq!(neighbors(&g, 8, 1), [u32::MAX, 9, 6, 4]);
    assert_eq!(neighbors(&g, 9, 1), [22, 44]);
    let ((), _) = observe("prepared_ordered_no_overtake", &r, || {
        g.insert_edge(8, 99, l(0))
    });
    assert_eq!(neighbors(&g, 8, 0), [3, 1, 2, 7, 5, 99]);
    let ((), _) = observe("prepared_mixed_log_fold", &r, || g.rebalance_all());
    assert_eq!(neighbors(&g, 8, 0), [3, 1, 2, 7, 5, 99]);
    assert_eq!(neighbors(&g, 8, 1), [u32::MAX, 9, 6, 4]);
    assert_eq!(neighbors(&g, 9, 1), [22, 44]);
    assert_eq!(g.num_edges(), 12);
    g.verify_invariants();
    stage(&mut g, 8, 0, &[55, 56]);
    stage(&mut g, 8, 1, &[77]);
    let (result, _) = observe("prepared_bulk_retires_shared_logs", &r, || {
        g.build_edges(&[(9, 88, l(0))])
    });
    result.unwrap();
    assert_eq!(neighbors(&g, 9, 0), [88]);
    assert_eq!(neighbors(&g, 8, 0), []);
    assert_eq!(g.label_order(l(0)), AdjacencyOrder::Insertion);
    assert_eq!(g.num_edges(), 1);
    for leaf in 0..g.segment_count() {
        assert_eq!(g.leaf_log_idx(leaf), 0);
    }
    g.verify_invariants();
}

#[test]
fn empty_label_switch_preserves_sibling_log_owner_and_uses_canonical_degree() {
    let r = regions();
    let mut g = Graph::new(bundle(&r), 64).unwrap();
    g.set_label_order(l(0), AdjacencyOrder::Insertion).unwrap();
    g.insert_edge(8, 3, l(0));
    g.insert_edge(8, u32::MAX, l(1));
    stage(&mut g, 8, 0, &[7]);
    stage(&mut g, 8, 1, &[8, 9]);
    let count = g.num_edges();
    g.inner.add_edges(-(count as i64)); // deliberately stale aggregate
    let (result, t) = observe("prepared_stale_count_guard", &r, || {
        g.set_label_order(l(0), AdjacencyOrder::Unordered)
    });
    assert!(result.is_err());
    assert!(t.iter().all(|t| t.writes.is_empty() && t.grows.is_empty()));
    g.inner.add_edges(count as i64);
    assert!(g.remove_edge(8, 3, l(0)));
    assert!(g.remove_edge(8, 7, l(0)));
    let leaf = 8 / g.segment_size() as u64;
    let cursor = g.leaf_log_idx(leaf);
    let before = r[2].bytes();
    let (result, t) = observe("prepared_empty_label_switch", &r, || {
        g.set_label_order(l(0), AdjacencyOrder::Unordered)
    });
    result.unwrap();
    assert!(t[2].writes.is_empty());
    assert_eq!(r[2].bytes(), before);
    assert_eq!(g.leaf_log_idx(leaf), cursor);
    assert_eq!(neighbors(&g, 8, 0), []);
    assert_eq!(neighbors(&g, 8, 1), [u32::MAX, 9, 8]);
    g.insert_edge(8, u32::MAX, l(0));
    g.rebalance_all();
    assert_eq!(neighbors(&g, 8, 0), [u32::MAX]);
    assert_eq!(neighbors(&g, 8, 1), [u32::MAX, 9, 8]);
    assert_eq!(g.num_edges(), 4);
    g.verify_invariants();
}
