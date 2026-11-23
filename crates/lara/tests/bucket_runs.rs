//! Independent descriptor-run ownership through public graph and Memory seams.
#![cfg(feature = "std")]
use lara_graph::graph::LaraMemories;
use lara_graph::{
    AdjacencyOrder, BucketDescriptor, FixedWidth, GrowFailed, LabelId, LabeledLaraGraph, Memory,
    PAGE_SIZE, VecMemory,
};
use std::collections::BTreeMap;
#[path = "support/label_order.rs"]
mod support;
use support::*;
type Graph<M = Probe> = LabeledLaraGraph<M>;
type Model = BTreeMap<(u32, u16), Vec<u32>>;
type Descriptors = BTreeMap<(u32, u16), (u64, Vec<u8>)>;
// Only new locality/reserve/reuse budgets are disabled in the old observation control.
const LOCAL: bool = true;
fn l(n: u16) -> LabelId {
    LabelId::new(n)
}
fn insert<M: Memory>(g: &mut Graph<M>, m: &mut Model, s: u32, d: u32, label: u16) {
    g.insert_edge(s, d, l(label));
    m.entry((s, label)).or_default().push(d);
}
fn verify<M: Memory>(g: &Graph<M>, m: &Model) {
    assert_eq!(
        g.num_edges(),
        m.values().map(|v| v.len() as u64).sum::<u64>()
    );
    for s in 0..g.num_nodes() as u32 {
        let got = g.out_labeled_vec(s);
        assert!(got.windows(2).all(|w| w[0].1 <= w[1].1));
        let want = m
            .range((s, 0)..=(s, u16::MAX))
            .map(|(_, ds)| ds.len())
            .sum::<usize>();
        assert_eq!(got.len(), want);
        assert_eq!(g.out_degree(s), want as u64);
        let mut cursor = 0;
        for (&(_, label), ds) in m.range((s, 0)..=(s, u16::MAX)) {
            let end = cursor + ds.len();
            let mut actual: Vec<_> = got[cursor..end]
                .iter()
                .map(|&(dst, id)| {
                    assert_eq!(id, l(label));
                    dst
                })
                .collect();
            cursor = end;
            let mut expected = ds.clone();
            if g.label_order(l(label)) == AdjacencyOrder::Unordered {
                actual.sort_unstable();
                expected.sort_unstable();
            }
            assert_eq!(actual, expected, "source {s}, label {label}");
        }
    }
    g.verify_invariants();
}
fn descriptors<M: Memory>(g: &Graph<M>, r: &[Probe; 8], m: &Model, tag: &str) -> Descriptors {
    let (_, t) = capture(r, || {
        for s in 0..g.num_nodes() as u32 {
            g.for_each_labeled(s, |_, _| {});
        }
    });
    let bytes = r[7].bytes();
    assert_eq!(t[7].reads.len(), m.len());
    let mut out = Descriptors::new();
    let mut previous = None;
    for ((&(src, label), ds), &(off, width, mapped)) in m.iter().zip(&t[7].reads) {
        assert_eq!((width, mapped), (24, false));
        let b = bytes[off as usize..off as usize + 24].to_vec();
        let d = BucketDescriptor::read_from(&b);
        assert_eq!(d.label, label);
        assert_eq!(d.degree, ds.len() as u32);
        assert_eq!(&b[2..4], &[0, 0]);
        if let Some((v, p)) = previous {
            if v == src {
                assert_eq!(off, p + 24);
            }
        }
        previous = Some((src, off));
        if !tag.is_empty() {
            println!(
                "DESC,{tag},{src},{label},{}",
                b.iter().map(|b| format!("{b:02x}")).collect::<String>()
            );
            println!("LOC,{tag},{src},{label},{off}");
        }
        out.insert((src, label), (off, b));
    }
    // Active records never overlap, independently of the run allocator.
    let mut slots: Vec<_> = out.values().map(|d| d.0).collect();
    slots.sort_unstable();
    assert!(slots.windows(2).all(|w| w[0] + 24 <= w[1]));
    out
}
fn others_stay(before: &Descriptors, after: &Descriptors, src: u32) {
    if LOCAL {
        for (&key, (off, _)) in before {
            if key.0 != src {
                assert_eq!(after[&key].0, *off, "unrelated vertex moved");
            }
        }
    }
}

#[test]
fn creation_moves_only_the_target_run() {
    for n in [64, 256, 1024, 4096] {
        for src in [0, n / 2, n - 1] {
            for label in [5, 20, 40] {
                let r = regions();
                let mut g = Graph::new(bundle(&r), n)
                    .unwrap()
                    .with_promote_threshold(None);
                g.set_label_order(l(10), AdjacencyOrder::Insertion).unwrap();
                let pairs: Vec<_> = (0..n)
                    .flat_map(|v| [(v, v, l(10)), (v, u32::MAX, l(30))])
                    .collect();
                g.build_edges(&pairs).unwrap();
                let mut model = Model::new();
                for (s, d, l) in pairs {
                    model.entry((s, l.as_u16())).or_default().push(d);
                }
                let before = descriptors(&g, &r, &model, "");
                let tag = format!("run_create_n{n}_src{src}_label{label}");
                let (_, t) = observe(&tag, &r, || g.insert_edge(src, 17, l(label)));
                model.insert((src, label), vec![17]);
                verify(&g, &model);
                let after = descriptors(&g, &r, &model, &tag);
                others_stay(&before, &after, src);
                let leaf = src / g.segment_size();
                for (&key, (_, b)) in &before {
                    if key.0 / g.segment_size() != leaf {
                        assert_eq!(&after[&key].1, b);
                    }
                }
                if LOCAL {
                    let local = 2 * g.segment_size() as usize + 1;
                    let tail = label == 40 && (src + 1) % g.segment_size() == 0;
                    let frontier = src == n - 1;
                    let prefix = if frontier {
                        usize::from(label > 10) + usize::from(label > 30)
                    } else {
                        0
                    };
                    assert_eq!(t[7].reads.len(), local + 8 - usize::from(tail) - prefix);
                    assert_eq!(
                        t[7].writes.len(),
                        local + 6 - prefix - usize::from(frontier)
                    );
                    assert_eq!(
                        t[7].writes.iter().filter(|w| w.1.len() == 8).count(),
                        usize::from(!frontier)
                    );
                }
            }
        }
    }
}

#[test]
fn reserved_capacity_and_free_runs_preserve_siblings() {
    let r = regions();
    let mut g = Graph::new(bundle(&r), 64)
        .unwrap()
        .with_promote_threshold(None);
    g.set_label_order(l(2), AdjacencyOrder::Insertion).unwrap();
    g.build_edges(&[(8, 30, l(2)), (32, u32::MAX, l(6))])
        .unwrap();
    let mut m = Model::from([((8, 2), vec![30]), ((32, 6), vec![u32::MAX])]);
    for (i, (src, label, dst)) in [
        (8, 4, 11),
        (8, 8, 12),
        (8, 10, 13),
        (8, 0, 14),
        (63, 2, 22),
        (63, 4, 23),
        (63, 8, 24),
    ]
    .into_iter()
    .enumerate()
    {
        let before = descriptors(&g, &r, &m, "");
        let tag = format!("run_reuse_{i}");
        let (_, t) = observe(&tag, &r, || g.insert_edge(src, dst, l(label)));
        m.entry((src, label)).or_default().push(dst);
        verify(&g, &m);
        let after = descriptors(&g, &r, &m, &tag);
        others_stay(&before, &after, src);
        if LOCAL {
            if i == 2 {
                assert_eq!(before[&(8, 2)].0, after[&(8, 2)].0);
            }
            if i == 3 {
                // Adding the new first label extends the same frontier run.
                assert_eq!(before[&(8, 2)].0, after[&(8, 0)].0);
            }
            if i >= 4 {
                // The free one-slot run is reused; larger old runs were not
                // retired by the earlier frontier extensions.
                assert_eq!(
                    t[7].reads.iter().filter(|r| r.1 == 8).count(),
                    usize::from(i == 4)
                );
                assert!(t[7].grows.is_empty());
                assert!(!t[7].writes.iter().any(|w| w.1.len() == 1));
            }
            if i == 5 {
                assert_ne!(before[&(63, 2)].0, after[&(63, 2)].0);
            }
            if i == 6 {
                assert_eq!(before[&(63, 2)].0, after[&(63, 2)].0);
            }
        }
    }
    for (src, dst) in [(8, 30), (63, 22)] {
        assert!(g.remove_edge(src, dst, l(2)));
        m.get_mut(&(src, 2)).unwrap().clear();
    }
    // Stale retired descriptor bytes are not an emptiness authority.
    let (_, _) = observe("run_empty_policy_change", &r, || {
        g.set_label_order(l(2), AdjacencyOrder::Unordered).unwrap()
    });
    verify(&g, &m);
    descriptors(&g, &r, &m, "run_empty_policy_change");
    insert(&mut g, &mut m, 8, u32::MAX, 2);
    g.rebalance_all();
    verify(&g, &m);
    let mut nested = Vec::new();
    g.for_each_labeled(8, |n, _| g.for_each_labeled(32, |s, _| nested.push((n, s))));
    assert_eq!(nested.len(), g.out_degree(8) as usize);
    assert!(nested.iter().all(|e| e.1 == u32::MAX));
}

fn model(pairs: &[(u32, u32, LabelId)]) -> Model {
    let mut m = Model::new();
    for &(s, d, id) in pairs {
        m.entry((s, id.as_u16())).or_default().push(d);
    }
    m
}
fn remove_first(g: &mut Graph, m: &mut Model, src: u32, label: u16) {
    let dst = m[&(src, label)][0];
    assert!(g.remove_edge(src, dst, l(label)));
    m.get_mut(&(src, label)).unwrap().remove(0);
}
#[test]
fn bulk_replacement_keeps_mixed_storage_and_retires_run_owners() {
    for cap in [0, 170] {
        let r = regions();
        let mut g = Graph::new_with_max_log_entries(bundle(&r), 64, cap)
            .unwrap()
            .with_promote_threshold(Some(64));
        for label in [0, 60000] {
            g.set_label_order(l(label), AdjacencyOrder::Insertion)
                .unwrap();
        }
        let mut pairs: Vec<_> = (0..1025).rev().map(|d| (8, d, l(0))).collect();
        pairs.extend((0..70).map(|d| (32, d, l(60000))));
        pairs.extend([
            (8, u32::MAX, l(1)),
            (8, 7, l(60000)),
            (9, 9, l(0)),
            (9, u32::MAX, l(1)),
            (32, u32::MAX, l(1)),
        ]);
        g.build_edges(&pairs).unwrap();
        let mut m = model(&pairs);
        insert(&mut g, &mut m, 8, 777, 0);
        insert(&mut g, &mut m, 32, 888, 60000);
        assert_eq!(g.tree_bucket_count(), 2);
        for label in [2, 3] {
            let tag = format!("run_mixed_create_cap{cap}_label{label}");
            let (_, _) = observe(&tag, &r, || g.insert_edge(8, 42, l(label)));
            m.insert((8, label), vec![42]);
            verify(&g, &m);
            descriptors(&g, &r, &m, &tag);
            assert!(g.bucket_is_tree(8, l(0)));
        }
        while m[&(8, 0)].len() > 33 {
            remove_first(&mut g, &mut m, 8, 0);
        }
        let tag = format!("run_mixed_demote_cap{cap}");
        let dst = m[&(8, 0)][0];
        let (ok, _) = observe(&tag, &r, || g.remove_edge(8, dst, l(0)));
        assert!(ok);
        m.get_mut(&(8, 0)).unwrap().remove(0);
        verify(&g, &m);
        descriptors(&g, &r, &m, &tag);
        assert!(!g.bucket_is_tree(8, l(0)));
        assert_eq!(g.tree_bucket_count(), 1);
        for rep in 0..3 {
            let replacement = [
                (9, 7, l(0)),
                (0, u32::MAX, l(1)),
                (9, 3, l(0)),
                (63, 8, l(60000)),
                (9, 7, l(0)),
            ];
            let tag = format!("run_bulk_cap{cap}_rep{rep}");
            let (_, _) = observe(&tag, &r, || g.build_edges(&replacement).unwrap());
            m = model(&replacement);
            verify(&g, &m);
            descriptors(&g, &r, &m, &tag);
            assert_eq!(g.tree_bucket_count(), 0);
            let bytes: Vec<_> = r.iter().map(Probe::bytes).collect();
            let bad = [(0, 2, l(1)), (9, u32::MAX, l(0))];
            let (result, t) = observe(&format!("run_bulk_reject_cap{cap}_rep{rep}"), &r, || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    g.build_edges(&bad).unwrap()
                }))
            });
            assert!(result.is_err());
            assert!(t.iter().all(|t| t.writes.is_empty() && t.grows.is_empty()));
            assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), bytes);
            verify(&g, &m);
        }
        let tag = format!("run_bulk_empty_cap{cap}");
        let (_, _) = observe(&tag, &r, || g.build_edges(&[]).unwrap());
        m.clear();
        verify(&g, &m);
        descriptors(&g, &r, &m, &tag);
        for label in [0, 60000] {
            assert_eq!(g.label_order(l(label)), AdjacencyOrder::Insertion);
        }
        insert(&mut g, &mut m, 63, 13, 60000);
        verify(&g, &m);
    }
}

#[test]
fn large_runs_cover_the_full_label_id_domain() {
    let r = regions();
    let mut g = Graph::new(bundle(&r), 64)
        .unwrap()
        .with_promote_threshold(None);
    g.set_label_order(l(u16::MAX), AdjacencyOrder::Insertion)
        .unwrap();
    for (count, next, tag) in [
        (32768, 32768, "run_class_65536"),
        (65535, 65535, "run_all_label_ids"),
    ] {
        let mut pairs: Vec<_> = (0..count).map(|id| (0, id, l(id as u16))).collect();
        pairs.push((32, u32::MAX, l(1)));
        g.build_edges(&pairs).unwrap();
        let mut m = model(&pairs);
        let before = descriptors(&g, &r, &m, "");
        let (_, t) = observe(tag, &r, || g.insert_edge(0, 7, l(next as u16)));
        m.insert((0, next as u16), vec![7]);
        verify(&g, &m);
        let after = descriptors(&g, &r, &m, tag);
        others_stay(&before, &after, 0);
        if LOCAL && count == 65535 {
            assert_eq!(before[&(0, 0)].0, after[&(0, 0)].0);
            assert!(t[7].grows.is_empty());
            assert!(t[7].writes.iter().all(|w| w.1.len() == 24));
        }
    }
    let bytes: Vec<_> = r.iter().map(Probe::bytes).collect();
    let (result, t) = observe("run_max_ordered_marker_reject", &r, || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g.insert_edge(0, u32::MAX, l(u16::MAX))
        }))
    });
    assert!(result.is_err());
    assert!(t.iter().all(|t| t.writes.is_empty() && t.grows.is_empty()));
    assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), bytes);
    assert_eq!(g.label_order(l(u16::MAX)), AdjacencyOrder::Insertion);
    assert_eq!(g.num_edges(), 65_537);
    assert_eq!(
        g.out_labeled_vec(0),
        (0..65_535)
            .map(|id| (id, l(id as u16)))
            .chain([(7, l(u16::MAX))])
            .collect::<Vec<_>>()
    );
    assert_eq!(g.out_labeled_vec(32), [(u32::MAX, l(1))]);
    g.verify_invariants();
}

#[derive(Clone)]
struct Limited {
    inner: Probe,
    deny: std::rc::Rc<std::cell::Cell<bool>>,
    attempts: std::rc::Rc<std::cell::Cell<u32>>,
}
impl Memory for Limited {
    fn read(&self, o: u64, b: &mut [u8]) {
        self.inner.read(o, b);
    }
    fn read_mapped(&self, o: u64, b: &mut [u8]) {
        self.inner.read_mapped(o, b);
    }
    fn write(&self, o: u64, b: &[u8]) {
        self.inner.write(o, b);
    }
    fn size(&self) -> u64 {
        self.inner.size()
    }
    fn fence(&self) {
        self.inner.fence();
    }
    fn grow(&self, n: u64) -> Result<u64, GrowFailed> {
        self.attempts.set(self.attempts.get() + 1);
        if self.deny.get() {
            Err(GrowFailed {
                current_pages: self.size(),
                delta_pages: n,
            })
        } else {
            self.inner.grow(n)
        }
    }
}
#[test]
fn failed_descriptor_growth_preserves_published_owners() {
    let r = regions();
    let limited: [Limited; 8] = std::array::from_fn(|i| Limited {
        inner: r[i].clone(),
        deny: Default::default(),
        attempts: Default::default(),
    });
    let mem = LaraMemories {
        vertices: limited[0].clone(),
        edges: limited[1].clone(),
        logs: limited[2].clone(),
        counts: limited[3].clone(),
        spans: limited[4].clone(),
        free_spans: limited[5].clone(),
        blocks: limited[6].clone(),
        buckets: limited[7].clone(),
    };
    let mut g = Graph::new(mem, 2048).unwrap().with_promote_threshold(None);
    let pairs: Vec<_> = (0..2048).map(|s| (s, s, l(10))).collect();
    g.build_edges(&pairs).unwrap();
    let mut m = model(&pairs);
    let before = descriptors(&g, &r, &m, "");
    let bytes: Vec<_> = r.iter().map(Probe::bytes).collect();
    assert_eq!(r[7].size(), 1);
    limited[7].attempts.set(0);
    limited[7].deny.set(true);
    let (result, t) = observe("run_denied_growth", &r, || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.insert_edge(0, 17, l(20))))
    });
    let error = result.unwrap_err();
    let message = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .unwrap();
    assert!(message.contains("bucket grow"));
    assert_eq!(limited[7].attempts.get(), 1);
    println!("DENIED,run_denied_growth,buckets,1");
    assert!(t.iter().all(|t| t.writes.is_empty() && t.grows.is_empty()));
    assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), bytes);
    verify(&g, &m);
    assert_eq!(descriptors(&g, &r, &m, "run_denied_growth"), before);
    limited[7].deny.set(false);
    let (_, t) = observe("run_growth_retry", &r, || g.insert_edge(0, 17, l(20)));
    m.insert((0, 20), vec![17]);
    assert_eq!(t[7].grows, [1]);
    assert_eq!(limited[7].attempts.get(), 2);
    assert_eq!(r[7].size(), 2);
    verify(&g, &m);
    let after = descriptors(&g, &r, &m, "run_growth_retry");
    others_stay(&before, &after, 0);
}
