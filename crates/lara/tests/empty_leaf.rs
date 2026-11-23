//! Bucket-free leaf reservations and lifecycle through public graph/Memory seams.
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
// Only new bucket-free geometry/reservation budgets are disabled in the old observation control.
const EMPTY_CONTRACT: bool = true;
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
fn model(pairs: &[(u32, u32, LabelId)]) -> Model {
    let mut m = Model::new();
    for &(s, d, id) in pairs {
        m.entry((s, id.as_u16())).or_default().push(d);
    }
    m
}
fn snapshot<M: Memory>(g: &Graph<M>, r: &[Probe; 8], m: &Model, tag: &str) -> Descriptors {
    verify(g, m);
    let raw = descriptors(g, r, m, tag);
    for (&(src, label), (_, bytes)) in &raw {
        let mut d = BucketDescriptor::read_from(bytes);
        if d.edge_start != u32::MAX {
            let (start, span, _) = g.leaf_geometry(u64::from(src) / u64::from(g.segment_size()));
            assert!(u64::from(d.edge_start) >= start && u64::from(d.edge_start) <= start + span);
            d.edge_start = (u64::from(d.edge_start) - start) as u32;
        }
        let mut b = [0; 24];
        d.write_to(&mut b);
        println!(
            "NORMDESC,{tag},{src},{label},{}",
            b.iter().map(|b| format!("{b:02x}")).collect::<String>()
        );
    }
    for leaf in 0..g.segment_count() {
        let (s, n, a) = g.leaf_geometry(leaf);
        println!("GEOM,{tag},{leaf},{s},{n},{a}");
    }
    raw
}
#[test]
fn empty_layouts_leave_no_slab_reservation() {
    for n in [0, 1, 64, 1024] {
        for cap in [0, 170] {
            let r = regions();
            let mut g = Graph::new_with_max_log_entries(bundle(&r), n, cap).unwrap();
            for phase in ["bulk", "rebalance"] {
                let tag = format!("empty_n{n}_cap{cap}_{phase}");
                let (_, t) = observe(&tag, &r, || {
                    if phase == "bulk" {
                        g.build_edges(&[]).unwrap()
                    } else {
                        g.rebalance_all()
                    }
                });
                snapshot(&g, &r, &Model::new(), &tag);
                if EMPTY_CONTRACT {
                    for leaf in 0..g.segment_count() {
                        assert_eq!(g.leaf_geometry(leaf).1, 0, "empty leaf reserved a slab");
                    }
                    assert!(
                        t[1].writes.is_empty() && t[1].grows.is_empty(),
                        "empty layout requested edge backing"
                    );
                }
            }
        }
    }
}

#[test]
fn first_insert_and_empty_replacement_preserve_policies_and_siblings() {
    for cap in [0, 170] {
        let r = regions();
        let mut g = Graph::new_with_max_log_entries(bundle(&r), 256, cap)
            .unwrap()
            .with_promote_threshold(None);
        g.set_label_order(l(2), AdjacencyOrder::Insertion).unwrap();
        g.build_edges(&[]).unwrap();
        let mut m = Model::new();
        for (i, (src, dst, id)) in [
            (17, 30, 2),
            (200, u32::MAX, 6),
            (17, 10, 2),
            (17, 20, 2),
            (17, 10, 2),
            (200, 7, 2),
            (200, 7, 2),
            (17, u32::MAX, 6),
        ]
        .into_iter()
        .enumerate()
        {
            let before = descriptors(&g, &r, &m, "");
            let tag = format!("empty_first_cap{cap}_{i}");
            let (_, _) = observe(&tag, &r, || g.insert_edge(src, dst, l(id)));
            m.entry((src, id)).or_default().push(dst);
            let after = snapshot(&g, &r, &m, &tag);
            others_stay(&before, &after, src);
            for (key, d) in before {
                if key.0 / g.segment_size() != src / g.segment_size() {
                    assert_eq!(after[&key], d);
                }
            }
        }
        let tag = format!("empty_first_settled_cap{cap}");
        let (_, _) = observe(&tag, &r, || g.rebalance_all());
        snapshot(&g, &r, &m, &tag);
        if EMPTY_CONTRACT {
            for leaf in 0..g.segment_count() {
                assert_eq!(
                    g.leaf_geometry(leaf).1,
                    if [
                        17 / u64::from(g.segment_size()),
                        200 / u64::from(g.segment_size())
                    ]
                    .contains(&leaf)
                    {
                        1024
                    } else {
                        0
                    }
                );
            }
        }
        let bytes: Vec<_> = r.iter().map(Probe::bytes).collect();
        let tag = format!("empty_marker_reject_cap{cap}");
        let (result, t) = observe(&tag, &r, || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                g.insert_edge(17, u32::MAX, l(2))
            }))
        });
        assert!(result.is_err());
        assert!(t.iter().all(|t| t.writes.is_empty() && t.grows.is_empty()));
        assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), bytes);
        snapshot(&g, &r, &m, &tag);
        let replacement = [
            (63, 7, l(2)),
            (0, u32::MAX, l(6)),
            (63, 3, l(2)),
            (63, 7, l(2)),
        ];
        for epoch in 0..2 {
            let tag = format!("empty_replace_cap{cap}_{epoch}");
            let (_, _) = observe(&tag, &r, || g.build_edges(&replacement).unwrap());
            m = model(&replacement);
            snapshot(&g, &r, &m, &tag);
        }
        let tag = format!("empty_clear_cap{cap}");
        let (_, _) = observe(&tag, &r, || g.build_edges(&[]).unwrap());
        m.clear();
        snapshot(&g, &r, &m, &tag);
        assert_eq!(g.label_order(l(2)), AdjacencyOrder::Insertion);
        if EMPTY_CONTRACT {
            assert!((0..g.segment_count()).all(|leaf| g.leaf_geometry(leaf).1 == 0));
        }
        let tag = format!("empty_reinsert_cap{cap}");
        let (_, _) = observe(&tag, &r, || {
            g.insert_edge(255, 9, l(2));
            g.insert_edge(255, 9, l(2));
            g.insert_edge(0, u32::MAX, l(6));
        });
        m.insert((255, 2), vec![9, 9]);
        m.insert((0, 6), vec![u32::MAX]);
        snapshot(&g, &r, &m, &tag);
    }
}

#[test]
fn empty_buckets_and_tree_only_leaves_keep_the_existing_floor() {
    for cap in [0, 170] {
        for threshold in [0, 64] {
            let r = regions();
            let mut g = Graph::new_with_max_log_entries(bundle(&r), 128, cap)
                .unwrap()
                .with_promote_threshold(Some(threshold));
            g.set_label_order(l(2), AdjacencyOrder::Insertion).unwrap();
            g.build_edges(&[]).unwrap();
            let mut m = Model::new();
            insert(&mut g, &mut m, 8, 7, 2);
            assert!(g.remove_edge(8, 7, l(2)));
            m.get_mut(&(8, 2)).unwrap().clear();
            assert_eq!(g.num_edges(), 0);
            let tag = format!("retained_empty_cap{cap}_t{threshold}");
            let (_, _) = observe(&tag, &r, || g.rebalance_all());
            snapshot(&g, &r, &m, &tag);
            let leaf = 8 / u64::from(g.segment_size());
            assert_eq!(
                g.leaf_geometry(leaf).1,
                1024,
                "empty bucket keeps its floor"
            );
            g.set_label_order(l(2), AdjacencyOrder::Unordered).unwrap();
            insert(&mut g, &mut m, 8, u32::MAX, 2);
            verify(&g, &m);
        }
        let r = regions();
        let mut g = Graph::new_with_max_log_entries(bundle(&r), 128, cap)
            .unwrap()
            .with_promote_threshold(Some(64));
        g.set_label_order(l(2), AdjacencyOrder::Insertion).unwrap();
        g.build_edges(&[]).unwrap();
        let mut m = Model::new();
        for d in 0..65 {
            insert(&mut g, &mut m, 8, d, 2);
        }
        insert(&mut g, &mut m, 32, u32::MAX, 6);
        assert!(g.bucket_is_tree(8, l(2)));
        let leaf = 8 / u64::from(g.segment_size());
        let tag = format!("tree_only_empty_siblings_cap{cap}");
        let (_, _) = observe(&tag, &r, || g.rebalance_all());
        snapshot(&g, &r, &m, &tag);
        assert_eq!(
            g.leaf_geometry(leaf).1,
            1024,
            "tree-only leaf keeps its floor"
        );
        assert_eq!(g.tree_bucket_count(), 1);
        let tag = format!("tree_only_append_cap{cap}");
        let (_, _) = observe(&tag, &r, || g.insert_edge(8, 777, l(2)));
        m.get_mut(&(8, 2)).unwrap().push(777);
        snapshot(&g, &r, &m, &tag);
        for d in 0..34 {
            assert!(g.remove_edge(8, d, l(2)));
            m.get_mut(&(8, 2)).unwrap().remove(0);
        }
        assert!(!g.bucket_is_tree(8, l(2)));
        let tag = format!("tree_only_demoted_cap{cap}");
        let (_, _) = observe(&tag, &r, || g.rebalance_all());
        snapshot(&g, &r, &m, &tag);
        assert_eq!(g.tree_bucket_count(), 0);
        assert_eq!(g.leaf_geometry(leaf).1, 1024);
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
fn skipping_empty_leaves_does_not_hide_real_growth_failures() {
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
    g.build_edges(&[]).unwrap();
    let pairs: Vec<_> = (0..8192).map(|d| (0, d, l(2))).collect();
    g.build_edges(&pairs).unwrap();
    let mut m = model(&pairs);
    assert_eq!(g.bucket_allocated(0, l(2)), 16384);
    for d in 8192..16384 {
        insert(&mut g, &mut m, 0, d, 2);
    }
    assert_eq!(g.bucket_room(0, l(2)), 0);
    let before = descriptors(&g, &r, &m, "");
    let geom: Vec<_> = (0..g.segment_count()).map(|i| g.leaf_geometry(i)).collect();
    let bytes: Vec<_> = r.iter().map(Probe::bytes).collect();
    limited[1].attempts.borrow_mut().clear();
    limited[1].deny.set(true);
    let (result, t) = observe("empty_slab_denied", &r, || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| g.insert_edge(0, 17, l(2))))
    });
    let e = result.unwrap_err();
    let msg = e
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| e.downcast_ref::<&str>().copied())
        .unwrap();
    assert!(msg.contains("slab grow"));
    assert_eq!(limited[1].attempts.borrow().len(), 1);
    assert!(limited[1].attempts.borrow()[0] > 0);
    println!(
        "DENGROW,empty_slab_denied,edges,{}",
        limited[1].attempts.borrow()[0]
    );
    assert!(t.iter().all(|t| t.writes.is_empty() && t.grows.is_empty()));
    assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), bytes);
    assert_eq!(
        (0..g.segment_count())
            .map(|i| g.leaf_geometry(i))
            .collect::<Vec<_>>(),
        geom
    );
    assert_eq!(snapshot(&g, &r, &m, "empty_slab_denied"), before);
    limited[1].deny.set(false);
    let (_, t) = observe("empty_slab_retry", &r, || g.insert_edge(0, 17, l(2)));
    m.get_mut(&(0, 2)).unwrap().push(17);
    let attempts = limited[1].attempts.borrow();
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0], attempts[1]);
    assert_eq!(t[1].grows, [attempts[1]]);
    snapshot(&g, &r, &m, "empty_slab_retry");
}
