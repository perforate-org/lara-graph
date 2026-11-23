//! Mixed per-label ordering through public APIs, with strict real backing.
#![cfg(feature = "std")]
use lara_graph::graph::LaraMemories;
use lara_graph::{
    AdjacencyOrder::{Insertion, Unordered},
    GrowFailed, LabelId, LabeledLaraGraph, Memory, PAGE_SIZE, VecMemory,
};
#[path = "support/label_order.rs"]
mod support;
use support::*;

type Graph = LabeledLaraGraph<Probe>;
fn label(l: u16) -> LabelId {
    LabelId::new(l)
}

#[test]
fn mixed_scans_keep_chunk_extents_and_label_local_markers() {
    for n in [127, 128, 129, 8193] {
        let r = regions();
        let mut g = Graph::new(bundle(&r), 64)
            .unwrap()
            .with_promote_threshold(None);
        g.set_label_order(label(0), Insertion).unwrap();
        let pairs: Vec<_> = (0..n)
            .rev()
            .map(|d| (8, d, label(0)))
            .chain([(8, u32::MAX, label(1)), (9, 9, label(1))])
            .collect();
        g.build_edges(&pairs).unwrap();
        for d in [n - 1, n / 2, 0] {
            assert!(g.remove_edge(8, d, label(0)));
        }
        let want: Vec<_> = (0..n)
            .rev()
            .filter(|d| ![n - 1, n / 2, 0].contains(d))
            .collect();
        let (got, t) = observe(&format!("label_scan_stored{n}"), &r, || neighbors(&g, 8, 0));
        assert_eq!(got, want);
        assert_eq!(
            t[1].reads.iter().map(|r| r.1).sum::<usize>(),
            n as usize * 4
        );
        assert_eq!(t[1].reads.len(), 1);
        assert!(t[1].reads.iter().all(|r| !r.2));
        no_mutation(&t);
        let ((), _) = observe(&format!("label_scan_fold{n}"), &r, || {
            g.rebalance_leaf(8 / g.segment_size() as u64)
        });
        assert_eq!(neighbors(&g, 8, 0), want);
        assert_eq!(neighbors(&g, 8, 1), [u32::MAX]);
        assert_eq!(neighbors(&g, 9, 1), [9]);
        g.verify_invariants();
    }
}

#[test]
fn source_marker_and_depth_preflight_keep_real_backing_unchanged() {
    use lara_graph::{B, R_MAX};
    let r = regions();
    let mut g = Graph::new(bundle(&r), 64).unwrap();
    g.set_label_order(label(0), Insertion).unwrap();
    g.build_edges(&(0..15359).map(|d| (8, d, label(0))).collect::<Vec<_>>())
        .unwrap();
    g.insert_edge(8, 15359, label(0));
    assert_eq!(r[6].size(), 1);
    let ((), t) = observe("label_actual_sixteenth_block_grow", &r, || {
        g.insert_edge(8, 15360, label(0))
    });
    assert_eq!(t[6].grows, [1]);
    assert_eq!(r[6].size(), 2);
    assert_eq!(neighbors(&g, 8, 0), (0..15361).collect::<Vec<_>>());
    g.verify_invariants();
    let max = (B * R_MAX) as u32;
    g.build_edges(&(0..max - 1).map(|d| (8, d, label(0))).collect::<Vec<_>>())
        .unwrap();
    g.insert_edge(8, max - 1, label(0));
    assert!(g.bucket_is_tree(8, label(0)));
    assert_eq!(neighbors(&g, 8, 0), (0..max).collect::<Vec<_>>());
    for (name, src, dst) in [
        ("depth", 8, 2),
        ("marker", 8, u32::MAX),
        ("source", g.num_nodes() as u32, 2),
    ] {
        let before: Vec<_> = r.iter().map(Probe::bytes).collect();
        let (result, t) = observe(&format!("label_reject_{name}"), &r, || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                g.insert_edge(src, dst, label(0))
            }))
        });
        let err = result.unwrap_err();
        let message = err
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| err.downcast_ref::<&str>().copied())
            .unwrap();
        assert!(message.contains(match name {
            "depth" => "depth-1 capacity",
            "marker" => "tombstone",
            _ => "src out of range",
        }));
        no_mutation(&t);
        assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), before);
        assert_eq!(g.num_edges(), u64::from(max));
        g.verify_invariants();
    }
    g = g.with_promote_threshold(None);
    let before: Vec<_> = r.iter().map(Probe::bytes).collect();
    let (result, t) = observe("label_existing_tree_depth_disabled_promotion", &r, || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g.insert_edge(8, 2, label(0))
        }))
    });
    assert!(result.is_err());
    no_mutation(&t);
    assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), before);
    assert_eq!(neighbors(&g, 8, 0), (0..max).collect::<Vec<_>>());
    g.verify_invariants();
}

#[cfg(feature = "derive")]
#[test]
fn typed_labels_and_dynamic_traits_follow_the_selected_bucket() {
    use lara_graph::{DynamicCsr, Label, RemovableCsr};
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Label)]
    enum Rel {
        Feed = 7,
        Other = 9,
    }
    let r = regions();
    let mut g = LabeledLaraGraph::<_, Rel>::new(bundle(&r), 64).unwrap();
    g.set_label_order(Rel::Feed, Insertion).unwrap();
    for d in [30, 10, 20, 10] {
        DynamicCsr::insert_edge(&mut g, 8, d);
    }
    for d in [u32::MAX, 7] {
        g.insert_edge(8, d, Rel::Other);
    }
    let (found, _) = observe("typed_label_zero_trait_delete", &r, || {
        RemovableCsr::remove_edge(&mut g, 8, 10)
    });
    assert!(found);
    let (got, _) = observe("typed_label_grouped_trait_scan", &r, || {
        let mut v = Vec::new();
        DynamicCsr::for_each_neighbor(&g, 8, |d| v.push(d));
        v
    });
    assert_eq!(got, [30, 20, 10, u32::MAX, 7]);
    assert_eq!(
        g.out_labeled_vec(8),
        [
            (30, Rel::Feed),
            (20, Rel::Feed),
            (10, Rel::Feed),
            (u32::MAX, Rel::Other),
            (7, Rel::Other)
        ]
    );
    assert_eq!(DynamicCsr::out_degree(&g, 8), 5);
    assert_eq!(DynamicCsr::num_edges(&g), 5);
    g.verify_invariants();
}

#[test]
fn mixed_lifecycle_matches_independent_bucket_sequences() {
    use std::collections::BTreeMap;
    type Model = BTreeMap<(u32, u16), Vec<u32>>;
    fn verify(g: &Graph, m: &Model) {
        assert_eq!(
            g.num_edges(),
            m.values().map(|v| v.len() as u64).sum::<u64>()
        );
        for src in 0..g.num_nodes() as u32 {
            let all = g.out_labeled_vec(src);
            assert!(all.windows(2).all(|w| w[0].1.as_u16() <= w[1].1.as_u16()));
            assert_eq!(all.len() as u64, g.out_degree(src));
            let mut expected = Vec::new();
            for (&(v, l), want) in m.range((src, 0)..=(src, u16::MAX)) {
                let mut got = neighbors(g, v, l);
                let mut want = want.clone();
                if ![0, 60000].contains(&l) {
                    got.sort_unstable();
                    want.sort_unstable();
                }
                assert_eq!(got, want, "bucket ({v},{l})");
                expected.extend(want.into_iter().map(|d| (d, l)));
            }
            let mut got: Vec<_> = all.into_iter().map(|(d, l)| (d, l.as_u16())).collect();
            got.sort_unstable();
            expected.sort_unstable();
            assert_eq!(got, expected);
        }
        g.verify_invariants();
    }
    for cap in [0, 170] {
        let r = regions();
        let mut g = Graph::new_with_max_log_entries(bundle(&r), 128, cap)
            .unwrap()
            .with_promote_threshold(Some(64));
        for l in [60000, 0] {
            g.set_label_order(label(l), Insertion).unwrap();
        }
        let mut model = Model::new();
        for src in [8, 9, 16, 17] {
            for l in [60000, 0, 1] {
                for i in 0..70 {
                    let d = (i * 7 + src) % 43;
                    g.insert_edge(src, d, label(l));
                    model.entry((src, l)).or_default().push(d);
                }
            }
        }
        assert_eq!(g.tree_bucket_count(), 12);
        verify(&g, &model);
        for (&(src, l), values) in &mut model {
            while values.len() > 32 {
                let d = values.remove(0);
                assert!(g.remove_edge(src, d, label(l)));
            }
        }
        assert_eq!(g.tree_bucket_count(), 0);
        verify(&g, &model);
        let mut rng = 0xdecafbad12345678u64;
        for batch in 0..24 {
            let mut ops = Vec::new();
            for _ in 0..25 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let src = [8, 9, 16, 17][(rng as usize >> 8) % 4];
                let l = [0, 1, 2, 60000][(rng as usize >> 12) % 4];
                let d = if l == 1 && rng % 11 == 0 {
                    u32::MAX
                } else {
                    (rng % 53) as u32
                };
                ops.push((rng % 5, src, l, d));
            }
            let (results, _) = observe(&format!("mixed_model_cap{cap}_b{batch}"), &r, || {
                ops.iter()
                    .map(|&(op, src, l, d)| match op {
                        0 => g.remove_edge(src, d, label(l)),
                        1 => {
                            g.rebalance_leaf(src as u64 / g.segment_size() as u64);
                            true
                        }
                        _ => {
                            g.insert_edge(src, d, label(l));
                            true
                        }
                    })
                    .collect::<Vec<_>>()
            });
            for ((op, src, l, d), found) in ops.into_iter().zip(results) {
                let values = model.entry((src, l)).or_default();
                match op {
                    0 => {
                        let pos = values.iter().position(|&v| v == d);
                        assert_eq!(found, pos.is_some());
                        if let Some(i) = pos {
                            values.remove(i);
                        }
                    }
                    1 => {}
                    _ => values.push(d),
                }
            }
            verify(&g, &model);
        }
    }
}
#[test]
fn public_bulk_overflow_is_folded_before_tree_publication() {
    for cap in [0, 170] {
        let r = regions();
        let mut g = Graph::new_with_max_log_entries(bundle(&r), 64, cap)
            .unwrap()
            .with_promote_threshold(Some(64));
        g.set_label_order(label(0), Insertion).unwrap();
        let pairs: Vec<_> = (0..64)
            .rev()
            .map(|d| (8, d, label(0)))
            .chain([(8, u32::MAX, label(1))])
            .collect();
        g.build_edges(&pairs).unwrap();
        assert_eq!(g.bucket_room(8, label(0)), 0);
        let ((), t) = observe(&format!("public_transient_overflow_cap{cap}"), &r, || {
            g.insert_edge(8, 777, label(0))
        });
        assert_eq!(
            neighbors(&g, 8, 0),
            (0..64).rev().chain([777]).collect::<Vec<_>>()
        );
        assert_eq!(neighbors(&g, 8, 1), [u32::MAX]);
        assert!(g.bucket_is_tree(8, label(0)));
        let expected = if cap == 0 {
            Vec::new()
        } else {
            vec![
                (4096 + cap as u64 * 16 - 1, vec![0]), // first chunk reservation
                (
                    4096,
                    vec![8, 0, 0, 0, 9, 3, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255],
                ),
            ]
        };
        assert_eq!(t[2].writes, expected);
        assert_eq!(g.leaf_log_idx(8 / g.segment_size() as u64), 0);
        assert_eq!(g.num_edges(), 66);
        g.verify_invariants();
    }
}

#[test]
fn configured_bulk_replaces_history_and_preserves_all_settings() {
    for cap in [0, 170] {
        let r = regions();
        let mut g = Graph::new_with_max_log_entries(bundle(&r), 64, cap)
            .unwrap()
            .with_promote_threshold(Some(64));
        g.set_label_order(label(2), Insertion).unwrap();
        g.set_label_order(label(60000), Insertion).unwrap();
        let pairs = [
            (8, 40, label(2)),
            (9, 30, label(2)),
            (8, u32::MAX, label(1)),
            (8, 2, label(2)),
            (8, 40, label(2)),
            (9, 1, label(2)),
        ];
        let (result, _) = observe(&format!("bulk_initial_cap{cap}"), &r, || {
            g.build_edges(&pairs)
        });
        result.unwrap();
        assert_eq!(neighbors(&g, 8, 2), [40, 2, 40]);
        assert_eq!(neighbors(&g, 9, 2), [30, 1]);
        assert_eq!(neighbors(&g, 8, 1), [u32::MAX]);
        g.verify_invariants();
        for d in 100..1200 {
            g.insert_edge(8, d, label(2));
        }
        assert!(g.bucket_is_tree(8, label(2)));
        let replacement = [
            (8, 7, label(2)),
            (8, 3, label(2)),
            (8, 7, label(2)),
            (9, u32::MAX, label(1)),
        ];
        for round in 0..3 {
            let (result, _) = observe(&format!("bulk_replace_cap{cap}_r{round}"), &r, || {
                g.build_edges(&replacement)
            });
            result.unwrap();
            assert_eq!(g.tree_bucket_count(), 0);
            assert_eq!(g.num_edges(), 4);
            assert_eq!(neighbors(&g, 8, 2), [7, 3, 7]);
            assert_eq!(neighbors(&g, 9, 1), [u32::MAX]);
            assert_eq!(g.label_order(label(2)), Insertion);
            assert_eq!(g.label_order(label(60000)), Insertion);
            assert_eq!(g.max_log_entries(), cap);
            g.verify_invariants();
        }
        for (name, bad) in [
            ("marker", vec![(8, 1, label(2)), (8, u32::MAX, label(2))]),
            (
                "source",
                vec![(8, 1, label(2)), (g.num_nodes() as u32, 2, label(1))],
            ),
        ] {
            let before: Vec<_> = r.iter().map(Probe::bytes).collect();
            let (result, t) = observe(&format!("bulk_reject_{name}_cap{cap}"), &r, || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.build_edges(&bad)))
            });
            assert!(result.is_err());
            no_mutation(&t);
            assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), before);
            assert_eq!(g.num_edges(), 4);
            assert_eq!(neighbors(&g, 8, 2), [7, 3, 7]);
            g.verify_invariants();
        }
        let (result, _) = observe(&format!("bulk_empty_cap{cap}"), &r, || g.build_edges(&[]));
        result.unwrap();
        assert_eq!(g.num_edges(), 0);
        assert_eq!(g.label_order(label(2)), Insertion);
        g.verify_invariants();
        for d in 0..64 {
            g.insert_edge(8, d, label(2));
        }
        assert!(g.bucket_is_tree(8, label(2))); // threshold was retained too
    }
}

fn neighbors(g: &Graph, src: u32, l: u16) -> Vec<u32> {
    let mut out = Vec::new();
    g.for_each_edge_of_label(src, label(l), |d| out.push(d));
    out
}
fn no_mutation(t: &[Trace; 8]) {
    assert!(t.iter().all(|t| t.writes.is_empty() && t.grows.is_empty()));
}

#[test]
fn ordered_trees_shift_actual_roots_and_preserve_siblings() {
    use lara_graph::block_store::BLOCK_STRIDE;
    for n in [4096, 4097] {
        for pos in [0, 1023, n / 2, n - 1, n + 10] {
            let r = regions();
            let mut g = Graph::new(bundle(&r), 64).unwrap();
            g.set_label_order(label(0), Insertion).unwrap();
            for d in 0..n {
                g.insert_edge(8, d, label(0));
            }
            g.insert_edge(8, u32::MAX, label(1));
            let (found, t) = observe(&format!("ordered_tree_n{n}_p{pos}"), &r, || {
                g.remove_edge(8, pos, label(0))
            });
            assert_eq!(found, pos < n);
            assert_eq!(
                neighbors(&g, 8, 0),
                (0..n).filter(|&d| d != pos).collect::<Vec<_>>()
            );
            assert_eq!(neighbors(&g, 8, 1), [u32::MAX]);
            assert_eq!(t[6].reads.len(), n as usize);
            assert_eq!(t[6].writes.len(), n.saturating_sub(pos + 1) as usize);
            for (j, (off, b)) in t[6].writes.iter().enumerate() {
                let slot = pos as u64 + j as u64;
                assert_eq!(*off, 80 + (slot / 1024) * BLOCK_STRIDE + (slot % 1024) * 4);
                assert_eq!(*b, (pos + j as u32 + 1).to_le_bytes());
            }
            g.verify_invariants();
        }
    }
    let r = regions();
    let mut g = Graph::new(bundle(&r), 64)
        .unwrap()
        .with_promote_threshold(Some(64));
    g.set_label_order(label(0), Insertion).unwrap();
    for d in 0..1024 {
        g.insert_edge(8, d, label(0));
    }
    for d in 7000..7064 {
        g.insert_edge(8, d, label(1));
    }
    g.insert_edge(8, 4090, label(0));
    let before = r[6].bytes();
    let (found, t) = observe("ordered_actual_roots_0_2", &r, || {
        g.remove_edge(8, 0, label(0))
    });
    assert!(found);
    assert_eq!(
        t[6].writes.last().unwrap(),
        &(80 + 1023 * 4, 4090u32.to_le_bytes().to_vec())
    );
    assert!(t[6].reads.contains(&(80 + 2 * BLOCK_STRIDE, 4, false)));
    assert_eq!(&r[6].bytes()[4176..8288], &before[4176..8288]);
    let ((), t) = observe("ordered_retained_root_regrow", &r, || {
        g.insert_edge(8, 4091, label(0))
    });
    assert_eq!(
        t[6].writes,
        [(80 + 2 * BLOCK_STRIDE, 4091u32.to_le_bytes().to_vec())]
    );
    assert_eq!(
        neighbors(&g, 8, 0),
        (1..1024).chain([4090, 4091]).collect::<Vec<_>>()
    );
    for d in 1..994 {
        assert!(g.remove_edge(8, d, label(0)));
    }
    assert!(!g.bucket_is_tree(8, label(0))); // 32 remain; sibling tree ID is fixed up
    assert!(g.bucket_is_tree(8, label(1)));
    let mut want = (994..1024).chain([4090, 4091]).collect::<Vec<_>>();
    for d in (5000..5032).rev() {
        g.insert_edge(8, d, label(0));
        want.push(d);
    }
    assert!(g.bucket_is_tree(8, label(0)));
    assert_eq!(neighbors(&g, 8, 0), want);
    assert_eq!(neighbors(&g, 8, 1), (7000..7064).collect::<Vec<_>>());
    assert_eq!(g.num_edges(), 128);
    g.verify_invariants();
}

#[test]
fn label_settings_check_every_vertex_without_blocking_other_labels() {
    let r = regions();
    let mut g = Graph::new(bundle(&r), 64).unwrap();
    let last = g.num_nodes() as u32 - 1;
    g.insert_edge(last, 7, label(1));
    assert_eq!(g.label_order(label(0)), Unordered);
    g.set_label_order(label(0), Insertion).unwrap(); // another label is live
    g.insert_edge(8, 3, label(0));
    let before: Vec<_> = r.iter().map(Probe::bytes).collect();
    let (result, t) = observe("scope_live_last", &r, || {
        g.set_label_order(label(1), Insertion)
    });
    assert!(result.is_err());
    no_mutation(&t);
    assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), before);
    let (result, t) = observe("scope_same_live", &r, || {
        g.set_label_order(label(0), Insertion)
    });
    result.unwrap();
    no_mutation(&t);
    assert!(t.iter().all(|t| t.reads.is_empty()));
    let (result, t) = observe("scope_reverse_live", &r, || {
        g.set_label_order(label(0), Unordered)
    });
    assert!(result.is_err());
    no_mutation(&t);
    assert!(g.remove_edge(8, 3, label(0)));
    g.set_label_order(label(0), Unordered).unwrap();
    g.set_label_order(label(u16::MAX), Insertion).unwrap();
    g.insert_edge(9, 9, label(u16::MAX));
    assert_eq!(neighbors(&g, last, 1), [7]);
    assert_eq!(neighbors(&g, 9, u16::MAX), [9]);
    assert_eq!(g.num_edges(), 2);
    g.verify_invariants();
}

#[test]
fn mixed_slab_markers_preserve_survivors_and_unordered_max() {
    use lara_graph::{BucketDescriptor, FixedWidth};
    for cap in [0, 170] {
        let r = regions();
        let mut g = Graph::new_with_max_log_entries(bundle(&r), 64, cap)
            .unwrap()
            .with_promote_threshold(None);
        g.set_label_order(label(0), Insertion).unwrap();
        for d in [30, 10, 20, 10] {
            g.insert_edge(8, d, label(0));
        }
        for d in [u32::MAX, 7, 2, 7] {
            g.insert_edge(8, d, label(1));
        }
        let (_, locations) = capture(&r, || g.for_each_labeled(8, |_, _| {}));
        let off = locations[7].reads[0].0 as usize;
        let b = BucketDescriptor::read_from(&r[7].bytes()[off..off + 24]);
        assert_eq!(b.label, 0);
        let (found, t) = observe(&format!("mixed_slab_delete_cap{cap}"), &r, || {
            g.remove_edge(8, 10, label(0))
        });
        assert!(found);
        assert_eq!(neighbors(&g, 8, 0), [30, 20, 10]);
        assert_eq!(
            t[1].writes,
            [(
                u64::from(b.edge_start + 1) * 4,
                u32::MAX.to_le_bytes().to_vec()
            )]
        );
        assert_eq!(
            BucketDescriptor::read_from(&r[7].bytes()[off..off + 24]).stored,
            4
        );
        let ((), t) = observe(&format!("mixed_slab_append_cap{cap}"), &r, || {
            g.insert_edge(8, 17, label(0))
        });
        assert_eq!(
            t[1].writes,
            [(
                u64::from(b.edge_start + 4) * 4,
                17u32.to_le_bytes().to_vec()
            )]
        );
        assert!(g.remove_edge(8, 7, label(1)));
        assert_eq!(neighbors(&g, 8, 1), [u32::MAX, 7, 2]); // existing dense swap
        let ((), _) = observe(&format!("mixed_slab_fold_cap{cap}"), &r, || {
            g.rebalance_all()
        });
        assert_eq!(neighbors(&g, 8, 0), [30, 20, 10, 17]);
        assert_eq!(neighbors(&g, 8, 1), [u32::MAX, 7, 2]);
        let (nested, _) = observe(&format!("mixed_nested_cap{cap}"), &r, || {
            let mut out = Vec::new();
            g.for_each_labeled(8, |d, l| out.push((d, l, neighbors(&g, 8, 0))));
            out
        });
        assert_eq!(nested.len(), 7);
        assert!(nested.iter().all(|r| r.2 == [30, 20, 10, 17]));
        assert_eq!(g.num_edges(), 7);
        assert_eq!(g.out_degree(8), 7);
        g.verify_invariants();
        let before: Vec<_> = r.iter().map(Probe::bytes).collect();
        let (result, t) = observe(&format!("mixed_reject_marker_cap{cap}"), &r, || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                g.insert_edge(9, u32::MAX, label(0))
            }))
        });
        assert!(result.is_err());
        no_mutation(&t);
        assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), before);
        for d in [30, 20, 10, 17] {
            assert!(g.remove_edge(8, d, label(0)));
        }
        let (result, _) = observe(&format!("mixed_empty_switch_cap{cap}"), &r, || {
            g.set_label_order(label(0), Unordered)
        });
        result.unwrap();
        g.insert_edge(8, u32::MAX, label(0));
        assert_eq!(neighbors(&g, 8, 0), [u32::MAX]);
        assert_eq!(neighbors(&g, 8, 1), [u32::MAX, 7, 2]);
        g.verify_invariants();
    }
}
