//! Leaf-local descriptor commits over strict real backing and independent runs.
//! Physical capacity/placement is distinct from label order and live records.
#![cfg(feature = "std")]
use lara_graph::graph::LaraMemories;
use lara_graph::{
    AdjacencyOrder, GrowFailed, LabelId, LabeledLaraGraph, Memory, PAGE_SIZE, VecMemory,
};
use std::collections::BTreeMap;
#[path = "support/label_order.rs"]
mod support;
use support::*;
type Graph = LabeledLaraGraph<Probe>;
type Model = BTreeMap<(u32, u16), Vec<u32>>;
const LOCAL_CONTRACT: bool = true;
fn l(n: u16) -> LabelId {
    LabelId::new(n)
}
fn local_contract(t: &[Trace; 8], reads: usize, writes: usize, range: Option<(u64, u64)>) {
    if LOCAL_CONTRACT {
        assert_eq!(t[7].reads.len(), reads, "leaf-local descriptor reads");
        assert_eq!(t[7].writes.len(), writes, "leaf-local descriptor writes");
        if let Some((first, end)) = range {
            assert!(
                t[7].reads
                    .iter()
                    .all(|&(off, n, m)| !m && n == 24 && off >= first && off + 24 <= end)
            );
            assert!(
                t[7].writes
                    .iter()
                    .all(|(off, b)| b.len() == 24 && *off >= first && *off + 24 <= end)
            );
        }
    }
}
fn verify(g: &Graph, m: &Model) {
    assert_eq!(
        g.num_edges(),
        m.values().map(|v| v.len() as u64).sum::<u64>()
    );
    for src in 0..g.num_nodes() as u32 {
        let got = g.out_labeled_vec(src);
        let want: Vec<_> = m
            .range((src, 0)..=(src, u16::MAX))
            .flat_map(|(&(_, label), ds)| ds.iter().map(move |&d| (d, l(label))))
            .collect();
        assert_eq!(got, want, "source {src}");
        assert_eq!(g.out_degree(src), want.len() as u64);
    }
    g.verify_invariants();
}
fn build(n: u32, labels: &[u16]) -> (Graph, [Probe; 8], Model) {
    let r = regions();
    let mut g = Graph::new(bundle(&r), n)
        .unwrap()
        .with_promote_threshold(None);
    g.set_label_order(l(2), AdjacencyOrder::Insertion).unwrap();
    let pairs: Vec<_> = (0..n)
        .flat_map(|src| {
            labels.iter().map(move |&label| {
                (
                    src,
                    if label == 10 && src % 7 == 0 {
                        u32::MAX
                    } else {
                        src
                    },
                    l(label),
                )
            })
        })
        .collect();
    g.build_edges(&pairs).unwrap();
    let mut m = Model::new();
    for (s, d, label) in pairs {
        m.entry((s, label.as_u16())).or_default().push(d);
    }
    verify(&g, &m);
    (g, r, m)
}
fn geometry(g: &Graph) -> Vec<(u64, u64, u64)> {
    (0..g.segment_count()).map(|i| g.leaf_geometry(i)).collect()
}
fn unaffected_leaves(g: &Graph, before: &[(u64, u64, u64)], leaf: u64) {
    for (i, &geo) in before.iter().enumerate() {
        if i as u64 != leaf {
            assert_eq!(g.leaf_geometry(i as u64), geo);
        }
    }
}

#[test]
fn rebalance_requests_follow_leaf_size_not_global_descriptor_count() {
    for n in [64, 256, 1024, 4096] {
        for labels in [&[2][..], &[2, 6, 10][..]] {
            for position in ["first", "middle", "last"] {
                let (mut g, r, m) = build(n, labels);
                let leaf = match position {
                    "first" => 0,
                    "middle" => g.segment_count() / 2,
                    _ => g.segment_count() - 1,
                };
                let first = leaf * g.segment_size() as u64;
                let end = (first + g.segment_size() as u64).min(g.num_nodes());
                let capacity = labels.len().next_power_of_two() as u64;
                let first_slot = first * capacity;
                let end_slot = end * capacity;
                let live_count = ((end - first) * labels.len() as u64) as usize;
                let before = r[7].bytes();
                let geo = geometry(&g);
                let ((), t) = observe(
                    &format!("local_rebalance_n{n}_labels{}_{}", labels.len(), position),
                    &r,
                    || g.rebalance_leaf(leaf),
                );
                verify(&g, &m);
                unaffected_leaves(&g, &geo, leaf);
                let after = r[7].bytes();
                assert_eq!(
                    &after[..first_slot as usize * 24],
                    &before[..first_slot as usize * 24]
                );
                assert_eq!(
                    &after[end_slot as usize * 24..],
                    &before[end_slot as usize * 24..]
                );
                local_contract(
                    &t,
                    live_count,
                    live_count,
                    Some((first_slot * 24, end_slot * 24)),
                );
            }
        }
    }
}

#[test]
fn creation_keeps_independent_runs_and_leaf_only_repacking() {
    for n in [64, 256, 1024, 4096] {
        for src in [0, n / 2, n - 1] {
            for new_label in [5, 20] {
                let (mut g, r, mut m) = build(n, &[10]);
                let geo = geometry(&g);
                let leaf = src as u64 / g.segment_size() as u64;
                let local = g.segment_size() as usize + 1;
                let lookup = if new_label < 10 { 2 } else { 1 };
                let boundary = usize::from(new_label < 10 || (src + 1) % g.segment_size() != 0);
                let ((), t) = observe(
                    &format!("local_create_n{n}_src{src}_label{new_label}"),
                    &r,
                    || g.insert_edge(src, 17, l(new_label)),
                );
                m.entry((src, new_label)).or_default().push(17);
                verify(&g, &m);
                unaffected_leaves(&g, &geo, leaf);
                // The last vertex extends at the frontier: no retirement,
                // and an append also avoids copying its old descriptor.
                let frontier = src == n - 1;
                let prefix = usize::from(frontier && new_label > 10);
                local_contract(
                    &t,
                    3 + local + lookup + boundary - prefix,
                    local + 5 - prefix - usize::from(frontier),
                    None,
                );
                assert_eq!(
                    t[7].writes.iter().filter(|w| w.1.len() == 8).count(),
                    usize::from(!frontier)
                );
                assert_eq!(t[7].writes.iter().filter(|w| w.1.len() == 1).count(), 1);
                g.insert_edge(src, 19, l(new_label));
                m.get_mut(&(src, new_label)).unwrap().push(19);
                verify(&g, &m);
            }
        }
    }
}

#[test]
fn creation_keeps_empty_vertices_independent_and_records_canonical() {
    for cap in [0, 170] {
        let r = regions();
        let mut g = Graph::new_with_max_log_entries(bundle(&r), 64, cap).unwrap();
        g.set_label_order(l(2), AdjacencyOrder::Insertion).unwrap();
        let mut m = Model::new();
        for (src, label, dst) in [(63, 10, u32::MAX), (32, 6, 31), (0, 2, 17)] {
            let ((), _) = observe(&format!("local_empty_tie_cap{cap}_src{src}"), &r, || {
                g.insert_edge(src, dst, l(label));
            });
            m.insert((src, label), vec![dst]);
            verify(&g, &m);
            // Each source occupies a different leaf with one bucket. Discover
            // its active offset through a public scan, never by scanning slack.
            let bytes = r[7].bytes();
            for &(source, id) in m.keys() {
                let (_, t) = capture(&r, || g.for_each_labeled(source, |_, _| {}));
                assert_eq!(t[7].reads.len(), 1);
                let (off, size, mapped) = t[7].reads[0];
                assert_eq!((size, mapped), (24, false));
                let start = g.leaf_geometry(source as u64 / g.segment_size() as u64).0;
                let mut wanted = Vec::new();
                wanted.extend_from_slice(&id.to_le_bytes());
                wanted.extend_from_slice(&0u16.to_le_bytes());
                for word in [1, 1, u32::MAX, start as u32, u32::MAX] {
                    wanted.extend_from_slice(&word.to_le_bytes());
                }
                assert_eq!(
                    &bytes[off as usize..off as usize + 24],
                    wanted,
                    "canonical owned record"
                );
            }
        }
    }
}

#[test]
fn empty_and_sparse_leaves_do_not_rewrite_distant_descriptors() {
    for n in [64, 1024] {
        let r = regions();
        let mut g = Graph::new(bundle(&r), n).unwrap();
        g.set_label_order(l(2), AdjacencyOrder::Insertion).unwrap();
        let last = g.num_nodes() as u32 - 1;
        g.build_edges(&[(last, 7, l(2)), (last, u32::MAX, l(10))])
            .unwrap();
        let mut m = Model::from([((last, 2), vec![7]), ((last, 10), vec![u32::MAX])]);
        let before = r[7].bytes();
        let geo = geometry(&g);
        let ((), t) = observe(&format!("local_empty_leaf_n{n}"), &r, || {
            g.rebalance_leaf(0)
        });
        assert_eq!(r[7].bytes(), before);
        verify(&g, &m);
        unaffected_leaves(&g, &geo, 0);
        local_contract(&t, 0, 0, Some((0, 0)));
        assert!(g.remove_edge(last, 7, l(2)));
        m.get_mut(&(last, 2)).unwrap().clear();
        let ((), t) = observe(&format!("local_empty_bucket_n{n}"), &r, || {
            g.rebalance_leaf(g.segment_count() - 1)
        });
        verify(&g, &m);
        local_contract(&t, 2, 2, Some((0, 48)));
        g.insert_edge(last, 9, l(2));
        m.get_mut(&(last, 2)).unwrap().push(9);
        verify(&g, &m);
        g.build_edges(&[]).unwrap();
        m.clear();
        verify(&g, &m);
    }
}
