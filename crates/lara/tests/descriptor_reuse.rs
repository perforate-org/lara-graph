//! Descriptor request budgets at the public labeled/Memory seams.
//! Real strict VecMemory backing; freeze all regions before semantic oracles.
#![cfg(feature = "std")]

use lara_graph::graph::LaraMemories;
use lara_graph::labeled::{LabelId, LabeledLaraGraph};
use lara_graph::memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::rc::Rc;

const REGIONS: [&str; 8] = [
    "vertices",
    "edges",
    "logs",
    "counts",
    "spans",
    "free_spans",
    "blocks",
    "buckets",
];

#[derive(Default, Debug)]
struct Trace {
    reads: Vec<(u64, usize, bool)>,
    writes: Vec<(u64, Vec<u8>)>,
    grows: Vec<u64>,
}

#[derive(Clone, Default)]
struct Probe {
    memory: VecMemory,
    active: Rc<RefCell<Option<Trace>>>,
}

impl Probe {
    fn record_read(&self, off: u64, len: usize, mapped: bool) {
        assert!(off + len as u64 <= self.memory.size() * PAGE_SIZE);
        if let Some(c) = self.active.borrow_mut().as_mut() {
            c.reads.push((off, len, mapped));
        }
    }
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0; (self.memory.size() * PAGE_SIZE) as usize];
        self.memory.read(0, &mut bytes);
        bytes
    }
    fn backing(&self) -> (u64, usize) {
        (self.memory.size(), self.memory.capacity_bytes())
    }
}

impl Memory for Probe {
    fn read(&self, off: u64, out: &mut [u8]) {
        self.record_read(off, out.len(), false);
        self.memory.read(off, out);
    }
    fn read_mapped(&self, off: u64, out: &mut [u8]) {
        self.record_read(off, out.len(), true);
        self.memory.read_mapped(off, out);
    }
    fn write(&self, off: u64, bytes: &[u8]) {
        assert!(off + bytes.len() as u64 <= self.memory.size() * PAGE_SIZE);
        if let Some(c) = self.active.borrow_mut().as_mut() {
            c.writes.push((off, bytes.to_vec()));
        }
        self.memory.write(off, bytes);
    }
    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        if let Some(c) = self.active.borrow_mut().as_mut() {
            c.grows.push(pages);
        }
        self.memory.grow(pages)
    }
    fn size(&self) -> u64 {
        self.memory.size()
    }
    fn fence(&self) {
        self.memory.fence();
    }
}

type Graph = LabeledLaraGraph<Probe>;
type Model = BTreeMap<(u32, u16), Vec<u32>>;

fn label(n: u16) -> LabelId {
    LabelId::new(n)
}

fn bundle(r: &[Probe; 8]) -> LaraMemories<Probe> {
    LaraMemories {
        vertices: r[0].clone(),
        edges: r[1].clone(),
        logs: r[2].clone(),
        counts: r[3].clone(),
        spans: r[4].clone(),
        free_spans: r[5].clone(),
        blocks: r[6].clone(),
        buckets: r[7].clone(),
    }
}

fn bulk(n: u32, pairs: &[(u32, u32, u16)]) -> (Graph, [Probe; 8], Model) {
    let r = std::array::from_fn(|_| Probe::default());
    let edges: Vec<_> = pairs.iter().map(|&(s, d, l)| (s, d, label(l))).collect();
    let g = Graph::with_edge_list(bundle(&r), &edges, n).unwrap();
    let mut model = Model::new();
    for &(s, d, l) in pairs {
        model.entry((s, l)).or_default().push(d);
    }
    (g, r, model)
}

fn verify(g: &Graph, model: &Model) {
    let total: usize = model.values().map(Vec::len).sum();
    assert_eq!(g.num_edges(), total as u64);
    for src in 0..g.num_nodes() as u32 {
        let mut want: Vec<_> = model
            .range((src, 0)..=(src, u16::MAX))
            .flat_map(|(&(_, l), dsts)| dsts.iter().map(move |&d| (d, l)))
            .collect();
        let mut got: Vec<_> = g
            .out_labeled_vec(src)
            .into_iter()
            .map(|(d, l)| (d, l.as_u16()))
            .collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "source {src}");
        assert_eq!(g.out_degree(src), want.len() as u64);
    }
    g.verify_invariants();
}

fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}

fn observe<T>(tag: &str, r: &[Probe; 8], op: impl FnOnce() -> T) -> (T, [Trace; 8]) {
    for region in r {
        assert!(region.active.replace(Some(Trace::default())).is_none());
    }
    let result = op();
    let traces: [Trace; 8] = std::array::from_fn(|i| r[i].active.take().unwrap());
    for (i, c) in traces.iter().enumerate() {
        let region = REGIONS[i];
        let mapped = c.reads.iter().filter(|r| r.2).count();
        let rbytes: usize = c.reads.iter().map(|r| r.1).sum();
        let wbytes: usize = c.writes.iter().map(|w| w.1.len()).sum();
        let (pages, capacity) = r[i].backing();
        println!(
            "OBS,{tag},{region},{},{mapped},{rbytes},{},{wbytes},{},{pages},{capacity}",
            c.reads.len() - mapped,
            c.writes.len(),
            c.grows.len()
        );
        for (k, &(off, len, mapped)) in c.reads.iter().enumerate() {
            println!("READ,{tag},{region},{k},{off},{len},{mapped}");
        }
        for (k, (off, bytes)) in c.writes.iter().enumerate() {
            let mut hex = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                write!(hex, "{byte:02x}").unwrap();
            }
            println!("WRITE,{tag},{region},{k},{off},{},{hex}", bytes.len());
        }
        for (k, pages) in c.grows.iter().enumerate() {
            println!("GROW,{tag},{region},{k},{pages}");
        }
        println!("STATE,{tag},{region},{:016x}", fingerprint(&r[i].bytes()));
    }
    assert!(
        traces[7]
            .reads
            .iter()
            .all(|&(_, n, mapped)| matches!(n, 8 | 24) && !mapped)
    );
    (result, traces)
}

// This sole budget assertion is replaced by logging in the archived old-source
// observation control. Semantic, write, backing and invariant oracles stay on.
fn read_budget(c: &[Trace; 8], reads: usize) {
    assert_eq!(c[7].reads.len(), reads);
}

#[test]
fn existing_insert_reads_target_and_boundary_once() {
    for n in [64, 256, 1024] {
        for promotion in [true, false] {
            let pairs: Vec<_> = (0..n).map(|v| (v, v, 0)).collect();
            let (mut g, r, mut model) = bulk(n, &pairs);
            if !promotion {
                g = g.with_promote_threshold(None);
            }
            let before: Vec<_> = r.iter().map(Probe::bytes).collect();
            let backing: Vec<_> = r.iter().map(Probe::backing).collect();
            let geometry = g.leaf_geometry(0);
            let (_, c) = observe(&format!("existing_n{n}_promote{promotion}"), &r, || {
                g.insert_edge(0, 0, label(0));
            });
            model.get_mut(&(0, 0)).unwrap().push(0);
            verify(&g, &model);
            assert_eq!(g.leaf_geometry(0), geometry);
            assert_eq!(g.tree_bucket_count(), 0);
            assert_eq!(r.iter().map(Probe::backing).collect::<Vec<_>>(), backing);
            assert_eq!(c[7].writes.len(), 1);
            assert_eq!(c[7].writes[0].0, 0);
            let mut descriptor = before[7][..24].to_vec();
            descriptor[4..8].copy_from_slice(&2u32.to_le_bytes());
            descriptor[8..12].copy_from_slice(&2u32.to_le_bytes());
            assert_eq!(c[7].writes[0].1, descriptor);
            assert_eq!(r[7].bytes()[24..], before[7][24..]);
            assert_eq!(
                c[1].writes,
                [(4 * (geometry.0 + 1), 0u32.to_le_bytes().to_vec())]
            );
            for (i, trace) in c.iter().enumerate() {
                assert!(trace.grows.is_empty());
                if i != 1 && i != 7 {
                    assert!(trace.writes.is_empty());
                    assert_eq!(r[i].bytes(), before[i]);
                }
            }
            read_budget(&c, 2);
        }
    }
}

fn insert(g: &mut Graph, model: &mut Model, src: u32, dst: u32, l: u16) {
    g.insert_edge(src, dst, label(l));
    model.entry((src, l)).or_default().push(dst);
}

fn removed(model: &mut Model, src: u32, dst: u32, l: u16) {
    let values = model.get_mut(&(src, l)).unwrap();
    values.remove(values.iter().position(|&d| d == dst).unwrap());
}

fn read_only(c: &[Trace; 8]) {
    assert!(c.iter().all(|t| t.writes.is_empty() && t.grows.is_empty()));
}

#[test]
fn direct_lookups_preserve_missing_labels_deletion_and_nested_scans() {
    let pairs = [
        (8, 8, 2),
        (8, 8, 2),
        (8, 9, 6),
        (8, 8, 6),
        (8, 8, 6),
        (8, 10, 10),
        (9, 9, 0),
        (63, 63, 0),
    ];
    let (mut g, r, mut model) = bulk(64, &pairs);
    for (l, search, allocated) in [(6, 1, 6), (2, 2, 4), (10, 2, 2)] {
        let (got, c) = observe(&format!("lookup_scan{l}"), &r, || {
            let mut out = Vec::new();
            g.for_each_edge_of_label(8, label(l), |d| out.push(d));
            out
        });
        assert_eq!(got, model[&(8, l)]);
        verify(&g, &model);
        read_only(&c);
        read_budget(&c, search);
        let (tree, c) = observe(&format!("lookup_tree{l}"), &r, || {
            g.bucket_is_tree(8, label(l))
        });
        assert!(!tree);
        read_only(&c);
        read_budget(&c, search);
        let (room, c) = observe(&format!("lookup_room{l}"), &r, || {
            g.bucket_room(8, label(l))
        });
        assert_eq!(room, allocated / 2);
        read_only(&c);
        read_budget(&c, search + 1);
        let (width, c) = observe(&format!("lookup_allocated{l}"), &r, || {
            g.bucket_allocated(8, label(l))
        });
        assert_eq!(width, allocated);
        read_only(&c);
        read_budget(&c, search + 1);
    }
    for l in [0, 4, 8, 12] {
        let (got, c) = observe(&format!("lookup_absent{l}"), &r, || {
            let mut out = Vec::new();
            g.for_each_edge_of_label(8, label(l), |d| out.push(d));
            out
        });
        assert!(got.is_empty());
        verify(&g, &model);
        read_only(&c);
        read_budget(&c, 2);
    }
    let ((got, tree, room, allocated), c) = observe("lookup_empty_vertex", &r, || {
        let mut out = Vec::new();
        g.for_each_edge_of_label(0, label(6), |d| out.push(d));
        (
            out,
            g.bucket_is_tree(0, label(6)),
            g.bucket_room(0, label(6)),
            g.bucket_allocated(0, label(6)),
        )
    });
    assert!(got.is_empty());
    assert!(!tree);
    assert_eq!((room, allocated), (0, 0));
    read_only(&c);
    read_budget(&c, 0);
    let (ok, c) = observe("lookup_remove_duplicate", &r, || {
        g.remove_edge(8, 8, label(6))
    });
    assert!(ok);
    removed(&mut model, 8, 8, 6);
    verify(&g, &model);
    assert_eq!(c[7].writes.len(), 1);
    read_budget(&c, 1);
    let (ok, c) = observe("lookup_remove_missing_value", &r, || {
        g.remove_edge(8, 63, label(6))
    });
    assert!(!ok);
    verify(&g, &model);
    read_only(&c);
    read_budget(&c, 1);
    let (ok, c) = observe("lookup_remove_missing_label", &r, || {
        g.remove_edge(8, 8, label(4))
    });
    assert!(!ok);
    verify(&g, &model);
    read_only(&c);
    read_budget(&c, 2);
    for d in [9, 8] {
        assert!(g.remove_edge(8, d, label(6)));
        removed(&mut model, 8, d, 6);
    }
    let (_, c) = observe("lookup_empty_bucket_reinsert", &r, || {
        g.insert_edge(8, 8, label(6))
    });
    model.get_mut(&(8, 6)).unwrap().push(8);
    verify(&g, &model);
    assert_eq!(c[7].writes.len(), 1);
    read_budget(&c, 2);
    let (got, c) = observe("lookup_nested", &r, || {
        let mut out = Vec::new();
        g.for_each_edge_of_label(8, label(6), |a| {
            g.for_each_edge_of_label(8, label(2), |b| out.push((a, b)));
        });
        out
    });
    assert_eq!(got, [(8, 8), (8, 8)]);
    verify(&g, &model);
    read_only(&c);
    read_budget(&c, 3);
    let (got, c) = observe("lookup_all_labels", &r, || g.out_labeled_vec(8));
    assert_eq!(got.len(), 4);
    verify(&g, &model);
    read_only(&c);
    read_budget(&c, 3); // No lookup: unchanged one descriptor per bucket.
}

#[test]
fn new_labels_refresh_layout_and_preserve_global_neighbors() {
    for n in [64, 1024] {
        let pairs: Vec<_> = (0..n).map(|v| (v, v, 10)).collect();
        let (mut g, r, mut model) = bulk(n, &pairs);
        let (_, c) = observe(&format!("create_n{n}"), &r, || {
            g.insert_edge(0, 0, label(20))
        });
        model.entry((0, 20)).or_default().push(0);
        verify(&g, &model);
        // Grow only this vertex's run, retire its old slot, and commit
        // this leaf. The graph-wide descriptor count no longer appears.
        assert_eq!(c[7].writes.len(), g.segment_size() as usize + 6);
        assert_eq!(c[7].writes.iter().filter(|w| w.1.len() == 1).count(), 1);
        assert_eq!(c[7].writes.iter().filter(|w| w.1.len() == 8).count(), 1);
        read_budget(&c, g.segment_size() as usize + 6);
        // Creation laid out the empty bucket with one slot, now occupied.
        // Reserve ordinary headroom before observing the resident append.
        g.rebalance_leaf(0);
        assert_eq!(g.bucket_room(0, label(20)), 1);
        let (_, c) = observe(&format!("created_hot_n{n}"), &r, || {
            g.insert_edge(0, 0, label(20))
        });
        model.get_mut(&(0, 20)).unwrap().push(0);
        verify(&g, &model);
        assert_eq!(c[7].writes.len(), 1);
        read_budget(&c, 2);
        for l in [0, 5, u16::MAX] {
            let (_, _) = observe(&format!("create_n{n}_label{l}"), &r, || {
                g.insert_edge(0, 0, label(l))
            });
            model.entry((0, l)).or_default().push(0);
            verify(&g, &model);
        }
        let (_, _) = observe(&format!("create_last_n{n}"), &r, || {
            g.insert_edge(n - 1, n - 1, label(5))
        });
        model.entry((n - 1, 5)).or_default().push(n - 1);
        verify(&g, &model);
        // Last label at the last vertex uses the leaf end, not a neighbor.
        let (_, c) = observe(&format!("last_hot_n{n}"), &r, || {
            g.insert_edge(n - 1, n - 1, label(10))
        });
        model.get_mut(&(n - 1, 10)).unwrap().push(n - 1);
        verify(&g, &model);
        assert_eq!(c[7].writes.len(), 1);
        read_budget(&c, 1);
    }
}

#[test]
fn dense_reuse_retries_and_log_modes_keep_current_descriptors() {
    let pairs: Vec<_> = (0..160)
        .map(|d| (8, d, 0))
        .chain((0..128).map(|d| (8, d, 1)))
        .collect();
    let (g, r, mut model) = bulk(512, &pairs);
    let mut g = g.with_promote_threshold(None);
    for d in 160..320 {
        insert(&mut g, &mut model, 8, d, 0);
    }
    assert_eq!(g.bucket_room(8, label(0)), 0);
    assert!(g.remove_edge(8, 129, label(0)));
    removed(&mut model, 8, 129, 0);
    let (_, c) = observe("full_hole", &r, || g.insert_edge(8, 500, label(0)));
    model.get_mut(&(8, 0)).unwrap().push(500);
    verify(&g, &model);
    assert_eq!(c[1].reads.iter().map(|r| r.1).sum::<usize>(), 0);
    assert_eq!(c[1].writes.len(), 1);
    assert_eq!(c[7].writes.len(), 1);
    read_budget(&c, 3);
    let leaf = 8 / u64::from(g.segment_size());
    let geometry = g.leaf_geometry(leaf);
    let (_, c) = observe("full_rebalance", &r, || g.insert_edge(8, 501, label(0)));
    model.get_mut(&(8, 0)).unwrap().push(501);
    verify(&g, &model);
    assert_ne!(g.leaf_geometry(leaf).0, geometry.0);
    assert_eq!(g.bucket_allocated(8, label(0)), 640);
    assert_eq!(g.bucket_room(8, label(0)), 319);
    assert_eq!(g.tree_bucket_count(), 0);
    read_budget(&c, 8);

    let capped = |cap| {
        let r = std::array::from_fn(|_| Probe::default());
        let mut g = Graph::new_with_max_log_entries(bundle(&r), 64, cap)
            .unwrap()
            .with_promote_threshold(None);
        let mut model = Model::new();
        for d in 0..64 {
            insert(&mut g, &mut model, 8, d, 6);
        }
        for (s, d, l) in [(8, 9, 10), (8, 9, 10), (9, 9, 0), (0, 0, 0), (63, 63, 0)] {
            insert(&mut g, &mut model, s, d, l);
        }
        g = g.with_promote_threshold(Some(64));
        let leaf = 8 / u64::from(g.segment_size());
        g.rebalance_leaf(leaf);
        assert_eq!(g.bucket_allocated(8, label(6)), 64);
        (g, r, model, leaf)
    };
    for cap in [0, 170] {
        let (mut g, r, mut model, _) = capped(cap);
        assert!(g.remove_edge(8, 0, label(6)));
        removed(&mut model, 8, 0, 6);
        let (_, c) = observe(&format!("capped_hole_cap{cap}"), &r, || {
            g.insert_edge(8, 63, label(6))
        });
        model.get_mut(&(8, 6)).unwrap().push(63);
        verify(&g, &model);
        // Tail append restores stored=64 and now reaches normal promotion.
        assert!(g.bucket_is_tree(8, label(6)));
        assert!(c[2].writes.is_empty());
        read_budget(&c, 6); // lookup/boundary plus promotion's leaf-local descriptor commit
        // A separate full-prefix fixture retains the no-room retry/log
        // coverage; it has no preceding deletion or already-promoted tree.
        let (mut g, r, mut model, leaf) = capped(cap);
        let (_, c) = observe(&format!("capped_promote_cap{cap}"), &r, || {
            g.insert_edge(8, 8, label(6))
        });
        model.get_mut(&(8, 6)).unwrap().push(8);
        verify(&g, &model);
        assert!(g.bucket_is_tree(8, label(6)));
        assert_eq!(g.tree_bucket_count(), 1);
        assert_eq!(g.leaf_log_idx(leaf), 0);
        assert!(!c[6].writes.is_empty());
        if cap == 0 {
            assert!(c[2].writes.is_empty());
        } else {
            assert!(c[2].writes.iter().any(|w| w.1.len() == 16));
        }
    }
}

#[test]
fn tree_transitions_refresh_degree_and_sibling_ids() {
    let r = std::array::from_fn(|_| Probe::default());
    let mut g = Graph::new(bundle(&r), 64)
        .unwrap()
        .with_promote_threshold(Some(16));
    let mut model = Model::new();
    for l in [2, 6, 10] {
        for d in 0..16 {
            insert(&mut g, &mut model, 8, d, l);
        }
    }
    insert(&mut g, &mut model, 9, 9, 0);
    assert_eq!(g.tree_bucket_count(), 3);
    let (_, c) = observe("tree_append", &r, || g.insert_edge(8, 8, label(6)));
    model.get_mut(&(8, 6)).unwrap().push(8);
    verify(&g, &model);
    assert_eq!(c[6].writes.len(), 1);
    assert_eq!(c[7].writes.len(), 1);
    read_budget(&c, 1);
    let (got, c) = observe("tree_lookup_scan", &r, || {
        let mut out = Vec::new();
        g.for_each_edge_of_label(8, label(6), |d| out.push(d));
        out
    });
    assert_eq!(got, model[&(8, 6)]);
    read_only(&c);
    read_budget(&c, 1);
    g = g.with_promote_threshold(None);
    let (_, c) = observe("tree_append_promotion_disabled", &r, || {
        g.insert_edge(8, 8, label(6))
    });
    model.get_mut(&(8, 6)).unwrap().push(8);
    verify(&g, &model);
    read_budget(&c, 1);
    g = g.with_promote_threshold(Some(16));
    for d in 0..9 {
        assert!(g.remove_edge(8, d, label(6)));
        removed(&mut model, 8, d, 6);
    }
    let (ok, _) = observe("tree_demote_middle", &r, || g.remove_edge(8, 9, label(6)));
    assert!(ok);
    removed(&mut model, 8, 9, 6);
    verify(&g, &model);
    assert!(!g.bucket_is_tree(8, label(6)));
    assert_eq!(g.tree_bucket_count(), 2);
    let (_, c) = observe("tree_sibling_after_id_shift", &r, || {
        g.insert_edge(8, 8, label(10))
    });
    model.get_mut(&(8, 10)).unwrap().push(8);
    verify(&g, &model);
    read_budget(&c, 2);
    let (_, _) = observe("tree_insert_earlier_label", &r, || {
        g.insert_edge(8, 8, label(0))
    });
    model.entry((8, 0)).or_default().push(8);
    verify(&g, &model);
    for _ in 0..7 {
        insert(&mut g, &mut model, 8, 8, 6);
    }
    assert!(!g.bucket_is_tree(8, label(6)));
    let (_, _) = observe("tree_repromote_updated_width", &r, || {
        g.insert_edge(8, 8, label(6))
    });
    model.get_mut(&(8, 6)).unwrap().push(8);
    verify(&g, &model);
    assert!(g.bucket_is_tree(8, label(6)));
    assert!(g.bucket_is_tree(8, label(2)));
    assert!(g.bucket_is_tree(8, label(10)));
    assert!(!g.bucket_is_tree(8, label(0)));
    assert_eq!(g.tree_bucket_count(), 3);
}

#[test]
fn depth_rejection_keeps_slab_tree_and_all_backing_unchanged() {
    use lara_graph::tree_csr::{B, R_MAX};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let limit = (B * R_MAX) as u32;
    for tree in [false, true] {
        let count = limit - u32::from(tree);
        let pairs: Vec<_> = [(0, 0, 0)]
            .into_iter()
            .chain((0..count).map(|_| (8, 0, u16::MAX)))
            .chain([(9, 9, 0)])
            .collect();
        let (mut g, r, mut model) = bulk(64, &pairs);
        if tree {
            insert(&mut g, &mut model, 8, 0, u16::MAX);
            g = g.with_promote_threshold(None);
        }
        assert_eq!(g.bucket_is_tree(8, label(u16::MAX)), tree);
        let before: Vec<_> = r.iter().map(Probe::bytes).collect();
        let backing: Vec<_> = r.iter().map(Probe::backing).collect();
        let (result, c) = observe(&format!("depth_tree{tree}"), &r, || {
            catch_unwind(AssertUnwindSafe(|| g.insert_edge(8, 1, label(u16::MAX))))
        });
        let error = result.unwrap_err();
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap();
        assert!(message.contains("depth-1 capacity exceeded"));
        verify(&g, &model);
        assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), before);
        assert_eq!(r.iter().map(Probe::backing).collect::<Vec<_>>(), backing);
        read_only(&c);
        assert!(c[..7].iter().all(|t| t.reads.is_empty()));
        read_budget(&c, 1);
        let (result, c) = observe(&format!("source_over_tree{tree}"), &r, || {
            catch_unwind(AssertUnwindSafe(|| g.insert_edge(u32::MAX, 1, label(0))))
        });
        assert!(result.is_err());
        verify(&g, &model);
        assert_eq!(r.iter().map(Probe::bytes).collect::<Vec<_>>(), before);
        read_only(&c);
        assert!(c.iter().all(|t| t.reads.is_empty()));
    }
}
