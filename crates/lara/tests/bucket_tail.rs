//! Frontier extension, reuse priority and failure through public graph/Memory seams.
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
// Only new frontier placement/copy/retirement budgets are disabled in the old observation control.
const TAIL_CONTRACT: bool = true;
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
    for (&key, (off, _)) in before {
        if key.0 != src {
            assert_eq!(after[&key].0, *off, "unrelated vertex moved");
        }
    }
}
fn first(records: &Descriptors, src: u32) -> u64 {
    records
        .range((src, 0)..=(src, u16::MAX))
        .next()
        .unwrap()
        .1
        .0
}
fn model(pairs: &[(u32, u32, LabelId)]) -> Model {
    let mut m = Model::new();
    for &(s, d, id) in pairs {
        m.entry((s, id.as_u16())).or_default().push(d);
    }
    m
}
#[test]
fn frontier_growth_keeps_sorted_records_and_run_location() {
    for cap in [0, 170] {
        for k in [1, 2, 4, 8] {
            for label in [0, 3, 4 * k + 2] {
                let r = regions();
                let mut g = Graph::new_with_max_log_entries(bundle(&r), 64, cap)
                    .unwrap()
                    .with_promote_threshold(None);
                for id in [2, 3] {
                    g.set_label_order(l(id), AdjacencyOrder::Insertion).unwrap();
                }
                let mut pairs = vec![(0, u32::MAX, l(6))];
                for i in 0..k {
                    let id = 2 + 4 * i;
                    for d in if id == 2 {
                        vec![30, 10, 20, 10]
                    } else {
                        vec![u32::MAX, 7]
                    } {
                        pairs.push((8, d, l(id)));
                    }
                }
                g.build_edges(&pairs).unwrap();
                let mut m = model(&pairs);
                let before = descriptors(&g, &r, &m, "");
                assert!(first(&before, 8) > 0);
                let tag = format!("tail_cap{cap}_k{k}_label{label}");
                let (_, t) = observe(&tag, &r, || g.insert_edge(8, 17, l(label)));
                m.insert((8, label), vec![17]);
                verify(&g, &m);
                let after = descriptors(&g, &r, &m, &tag);
                others_stay(&before, &after, 8);
                assert_eq!(before[&(0, 6)], after[&(0, 6)]);
                if TAIL_CONTRACT {
                    assert_eq!(first(&before, 8), first(&after, 8), "frontier run moved");
                    assert!(
                        t[7].writes.iter().all(|w| w.1.len() != 8),
                        "frontier extension retired a run"
                    );
                    let initial = BucketDescriptor {
                        label,
                        edge_start: u32::MAX,
                        ..Default::default()
                    };
                    let mut encoded = vec![0; 24];
                    initial.write_to(&mut encoded);
                    let writes: Vec<_> = t[7]
                        .writes
                        .iter()
                        .filter(|w| w.1.len() == 24)
                        .cloned()
                        .collect();
                    let end = writes.iter().position(|w| w.1 == encoded).unwrap();
                    let expected: Vec<_> = before
                        .range((8, label)..=(8, u16::MAX))
                        .rev()
                        .map(|(_, d)| (d.0 + 24, d.1.clone()))
                        .collect();
                    assert_eq!(writes[..end], expected, "only the sorted suffix moves");
                }
            }
        }
    }
}

#[test]
fn reuse_precedes_tail_growth_and_non_tail_runs_still_move() {
    for cap in [0, 170] {
        let r = regions();
        let mut g = Graph::new_with_max_log_entries(bundle(&r), 128, cap)
            .unwrap()
            .with_promote_threshold(Some(64));
        g.set_label_order(l(2), AdjacencyOrder::Insertion).unwrap();
        let mut pairs: Vec<_> = (0..64).map(|d| (8, d, l(2))).collect();
        pairs.extend([
            (8, u32::MAX, l(6)),
            (32, 32, l(2)),
            (32, u32::MAX, l(6)),
            (63, 63, l(2)),
            (63, u32::MAX, l(6)),
        ]);
        g.build_edges(&pairs).unwrap();
        let mut m = model(&pairs);
        insert(&mut g, &mut m, 8, 777, 2);
        assert!(g.bucket_is_tree(8, l(2)));
        let mut free_four = 0;
        for (i, (src, label)) in [(8, 8), (32, 8), (8, 10), (8, 0)].into_iter().enumerate() {
            let before = descriptors(&g, &r, &m, "");
            let tag = format!("reuse_setup_cap{cap}_{i}");
            let (_, t) = observe(&tag, &r, || g.insert_edge(src, 17, l(label)));
            m.insert((src, label), vec![17]);
            verify(&g, &m);
            let after = descriptors(&g, &r, &m, &tag);
            others_stay(&before, &after, src);
            if i != 2 {
                assert_ne!(
                    first(&before, src),
                    first(&after, src),
                    "non-frontier run must relocate"
                );
                assert_eq!(t[7].writes.iter().filter(|w| w.1.len() == 8).count(), 1);
            }
            if i == 3 {
                free_four = first(&before, 8);
            }
        }
        // Consume both free two-slot runs. The next two-slot run will end
        // at the frontier, while the earlier free four-slot run still exists.
        for src in [0, 9, 16] {
            insert(&mut g, &mut m, src, src, 2);
            insert(&mut g, &mut m, src, u32::MAX, 6);
        }
        let before = descriptors(&g, &r, &m, "");
        let tag = format!("reuse_wins_at_frontier_cap{cap}");
        let (_, t) = observe(&tag, &r, || g.insert_edge(16, 19, l(8)));
        m.insert((16, 8), vec![19]);
        verify(&g, &m);
        let after = descriptors(&g, &r, &m, &tag);
        others_stay(&before, &after, 16);
        assert_eq!(first(&after, 16), free_four);
        assert_ne!(first(&before, 16), first(&after, 16));
        assert!(t[7].reads.contains(&(free_four, 8, false)));
        assert!(t[7].grows.is_empty());
        assert!(!t[7].writes.iter().any(|w| w.1.len() == 1));
        assert!(g.bucket_is_tree(8, l(2)));
        let replacement = [(16, 3, l(2)), (16, 1, l(2)), (8, u32::MAX, l(6))];
        let tag = format!("tail_reuse_bulk_cap{cap}");
        let (_, _) = observe(&tag, &r, || g.build_edges(&replacement).unwrap());
        m = model(&replacement);
        verify(&g, &m);
        descriptors(&g, &r, &m, &tag);
        assert_eq!(g.tree_bucket_count(), 0);
        g.build_edges(&[]).unwrap();
        m.clear();
        verify(&g, &m);
        assert_eq!(g.label_order(l(2)), AdjacencyOrder::Insertion);
        insert(&mut g, &mut m, 32, 7, 2);
        insert(&mut g, &mut m, 32, u32::MAX, 6);
        verify(&g, &m);
    }
}

#[derive(Clone)]
struct Limited {
    inner: Probe,
    deny: std::rc::Rc<std::cell::Cell<bool>>,
    attempts: std::rc::Rc<std::cell::RefCell<Vec<u64>>>,
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
        self.attempts.borrow_mut().push(n);
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
fn frontier_growth_failure_and_retry_preserve_owners() {
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
    let mut g = Graph::new(mem, 64).unwrap().with_promote_threshold(None);
    g.set_label_order(l(2), AdjacencyOrder::Insertion).unwrap();
    let pairs: Vec<_> = (0..2048)
        .map(|i| (8, if i == 1 { u32::MAX } else { i }, l((2 + 4 * i) as u16)))
        .collect();
    g.build_edges(&pairs).unwrap();
    let mut m = model(&pairs);
    let before = descriptors(&g, &r, &m, "");
    let bytes: Vec<_> = r.iter().map(Probe::bytes).collect();
    assert_eq!(r[7].size(), 1);
    limited[7].attempts.borrow_mut().clear();
    limited[7].deny.set(true);
    let (result, t) = observe("tail_denied_growth", &r, || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.insert_edge(8, 17, l(3))))
    });
    let error = result.unwrap_err();
    let message = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .unwrap();
    assert!(message.contains("bucket grow"));
    assert_eq!(limited[7].attempts.borrow().len(), 1);
    assert!(limited[7].attempts.borrow()[0] > 0);
    println!(
        "DENIED,tail_denied_growth,buckets,{}",
        limited[7].attempts.borrow()[0]
    );
    assert!(t.iter().all(|t| t.writes.is_empty() && t.grows.is_empty()));
    assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), bytes);
    verify(&g, &m);
    assert_eq!(descriptors(&g, &r, &m, "tail_denied_growth"), before);
    assert_eq!(g.label_order(l(2)), AdjacencyOrder::Insertion);
    limited[7].deny.set(false);
    let (_, t) = observe("tail_growth_retry", &r, || g.insert_edge(8, 17, l(3)));
    m.insert((8, 3), vec![17]);
    let attempts = limited[7].attempts.borrow();
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0], attempts[1]);
    assert_eq!(t[7].grows, [attempts[1]]);
    assert_eq!(r[7].size(), 1 + attempts[1]);
    verify(&g, &m);
    let after = descriptors(&g, &r, &m, "tail_growth_retry");
    if TAIL_CONTRACT {
        assert_eq!(first(&before, 8), first(&after, 8), "frontier run moved");
        assert_eq!(t[7].grows, [1]);
        assert!(!t[7].writes.iter().any(|w| w.1.len() == 8));
    }
}

#[test]
fn large_frontier_run_preserves_label_domain_and_bulk_history() {
    let r = regions();
    let mut g = Graph::new(bundle(&r), 64)
        .unwrap()
        .with_promote_threshold(None);
    for id in [0, u16::MAX] {
        g.set_label_order(l(id), AdjacencyOrder::Insertion).unwrap();
    }
    for (count, id, tag) in [
        (32768, 32768, "tail_class65536"),
        (65535, 65535, "tail_all_labels"),
    ] {
        let pairs: Vec<_> = (0..count)
            .map(|label| {
                (
                    8,
                    if label == 1 { u32::MAX } else { label },
                    l(label as u16),
                )
            })
            .collect();
        g.build_edges(&pairs).unwrap();
        let mut m = model(&pairs);
        let before = descriptors(&g, &r, &m, "");
        let (_, t) = observe(tag, &r, || g.insert_edge(8, 7, l(id as u16)));
        m.insert((8, id as u16), vec![7]);
        verify(&g, &m);
        let after = descriptors(&g, &r, &m, tag);
        if TAIL_CONTRACT {
            assert_eq!(first(&before, 8), first(&after, 8), "frontier run moved");
            assert!(!t[7].writes.iter().any(|w| w.1.len() == 8));
        }
        let bytes: Vec<_> = r.iter().map(Probe::bytes).collect();
        let (result, t) = observe(&format!("{tag}_marker_reject"), &r, || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                g.insert_edge(8, u32::MAX, l(0))
            }))
        });
        assert!(result.is_err());
        assert!(t.iter().all(|t| t.writes.is_empty() && t.grows.is_empty()));
        assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), bytes);
        verify(&g, &m);
        assert_eq!(g.label_order(l(0)), AdjacencyOrder::Insertion);
    }
    let (_, _) = observe("tail_large_empty", &r, || g.build_edges(&[]).unwrap());
    verify(&g, &Model::new());
    assert_eq!(g.label_order(l(u16::MAX)), AdjacencyOrder::Insertion);
}
