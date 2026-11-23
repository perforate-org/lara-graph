//! Default deletion through public graph and strict, real Memory seams.
//! Observations are frozen before semantic, ownership and request oracles.
#![cfg(feature = "std")]

use lara_graph::block_store::BLOCK_STRIDE;
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::labeled::{LabelId, LabeledLaraGraph};
use lara_graph::memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
use lara_graph::traits::{CsrEdge, CsrEdgeTombstone, CsrVertex, FixedWidth, NO_BUCKET};
use lara_graph::tree_csr::B;
use std::cell::RefCell;
use std::fmt::Write as _;
use std::rc::Rc;

const SRC: u32 = 8;
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
    mem: VecMemory,
    trace: Rc<RefCell<Option<Trace>>>,
}
impl Probe {
    fn reading(&self, off: u64, len: usize, mapped: bool) {
        assert!(off + len as u64 <= self.mem.size() * PAGE_SIZE);
        if let Some(t) = self.trace.borrow_mut().as_mut() {
            t.reads.push((off, len, mapped));
        }
    }
    fn bytes(&self) -> Vec<u8> {
        let mut b = vec![0; (self.mem.size() * PAGE_SIZE) as usize];
        self.mem.read(0, &mut b);
        b
    }
}
impl Memory for Probe {
    fn read(&self, off: u64, b: &mut [u8]) {
        self.reading(off, b.len(), false);
        self.mem.read(off, b);
    }
    fn read_mapped(&self, off: u64, b: &mut [u8]) {
        self.reading(off, b.len(), true);
        self.mem.read_mapped(off, b);
    }
    fn write(&self, off: u64, b: &[u8]) {
        assert!(off + b.len() as u64 <= self.mem.size() * PAGE_SIZE);
        if let Some(t) = self.trace.borrow_mut().as_mut() {
            t.writes.push((off, b.to_vec()));
        }
        self.mem.write(off, b);
    }
    fn grow(&self, n: u64) -> Result<u64, GrowFailed> {
        if let Some(t) = self.trace.borrow_mut().as_mut() {
            t.grows.push(n);
        }
        self.mem.grow(n)
    }
    fn size(&self) -> u64 {
        self.mem.size()
    }
    fn fence(&self) {
        self.mem.fence();
    }
}
fn regions() -> [Probe; 8] {
    std::array::from_fn(|_| Probe::default())
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
fn observe<T>(tag: &str, r: &[Probe; 8], op: impl FnOnce() -> T) -> (T, [Trace; 8]) {
    for p in r {
        assert!(p.trace.replace(Some(Trace::default())).is_none());
    }
    let result = op();
    let ts: [Trace; 8] = std::array::from_fn(|i| r[i].trace.take().unwrap());
    for (i, t) in ts.iter().enumerate() {
        let region = REGIONS[i];
        let mapped = t.reads.iter().filter(|r| r.2).count();
        println!(
            "OBS,{tag},{region},{},{mapped},{},{},{},{},{},{}",
            t.reads.len() - mapped,
            t.reads.iter().map(|r| r.1).sum::<usize>(),
            t.writes.len(),
            t.writes.iter().map(|w| w.1.len()).sum::<usize>(),
            t.grows.len(),
            r[i].size(),
            r[i].mem.capacity_bytes()
        );
        for (k, (off, n, m)) in t.reads.iter().enumerate() {
            println!("READ,{tag},{region},{k},{off},{n},{m}");
        }
        for (k, (off, b)) in t.writes.iter().enumerate() {
            let mut hex = String::new();
            for x in b {
                write!(hex, "{x:02x}").unwrap();
            }
            println!("WRITE,{tag},{region},{k},{off},{},{hex}", b.len());
        }
        for (k, n) in t.grows.iter().enumerate() {
            println!("GROW,{tag},{region},{k},{n}");
        }
    }
    (result, ts)
}
type Core<E = Edge> = LaraGraph<Vertex, E, Probe>;
fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v
}
fn core(n: u32, threshold: u32) -> (Core, [Probe; 8]) {
    let r = regions();
    let mut g = Core::new_with_segments(bundle(&r), (n + 16).max(4096), 8).unwrap();
    g.set_promote_threshold(Some(threshold));
    let pairs: Vec<_> = [1, 2, 2]
        .into_iter()
        .map(|d| (0, d))
        .chain((1..n).map(|d| (SRC, d)))
        .chain([9, 9, 10].into_iter().map(|d| (9, d)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    g.insert_edge(SRC, n);
    assert_ne!(g.get_row_pub(SRC).tree_bucket(), NO_BUCKET);
    (g, r)
}
fn verify(g: &Core, want: &[u32]) {
    assert_eq!(sorted(g.out_neighbors_vec(SRC)), sorted(want.to_vec()));
    assert_eq!(g.out_neighbors_vec(0), [1, 2, 2]);
    assert_eq!(g.out_neighbors_vec(9), [9, 9, 10]);
    assert_eq!(g.out_degree(SRC), want.len() as u64);
    assert_eq!(g.num_edges(), want.len() as u64 + 6);
    g.verify_invariants();
}
fn no_writes_elsewhere(t: &[Trace; 8], allowed: &[usize]) {
    for (i, t) in t.iter().enumerate() {
        assert!(t.grows.is_empty());
        if !allowed.contains(&i) {
            assert!(t.writes.is_empty(), "{}", REGIONS[i]);
        }
    }
}
fn offset(s: u32) -> u64 {
    80 + u64::from(s / B as u32) * BLOCK_STRIDE + u64::from(s % B as u32) * 4
}
// Shared replacement-request assertions. The evidence's old-source control
// changes only these and the separately identified dense-policy assertions.
fn swap_contract(t: &Trace, reads: usize, write: Option<(u64, Vec<u8>)>) {
    assert_eq!(t.reads.len(), reads, "search + at most one tail read");
    assert_eq!(
        t.writes,
        write.into_iter().collect::<Vec<_>>(),
        "one replacement, no suffix shift"
    );
}
#[test]
fn core_tree_deletion_replaces_only_the_removed_slot() {
    for n in [4096, 4097] {
        for p in [0, 1023, n / 2, n - 1, n + 10] {
            let (mut g, r) = core(n, 1024);
            let geometry = g.leaf_geometry(1);
            let (found, t) = observe(&format!("core_n{n}_p{p}"), &r, || g.remove_edge(SRC, p + 1));
            let want: Vec<_> = (1..=n).filter(|&x| x != p + 1).collect();
            assert_eq!(found, p < n);
            verify(&g, &want);
            assert_eq!(
                g.tree_state_stored(g.get_row_pub(SRC).tree_bucket() as usize),
                want.len() as u32
            );
            assert_eq!(g.leaf_geometry(1), geometry);
            no_writes_elsewhere(&t, &[6]);
            assert!(t[6].reads.iter().all(|r| r.1 == 4 && !r.2));
            let moved = p < n - 1;
            swap_contract(
                &t[6],
                if found {
                    p as usize + 1 + usize::from(moved)
                } else {
                    n as usize
                },
                moved.then(|| (offset(p), n.to_le_bytes().to_vec())),
            );
        }
    }
    let (mut g, r) = core(1024, 1024);
    g.insert_edge(SRC, 7);
    let (found, t) = observe("core_tree_duplicate", &r, || g.remove_edge(SRC, 7));
    assert!(found);
    verify(&g, &(1..=1024).collect::<Vec<_>>());
    assert_eq!(
        g.out_neighbors_vec(SRC).iter().filter(|&&d| d == 7).count(),
        1
    );
    no_writes_elsewhere(&t, &[6]);
    swap_contract(&t[6], 8, Some((104, 7u32.to_le_bytes().to_vec())));
}

#[test]
fn labeled_deletion_keeps_other_labels_and_actual_root_ids() {
    let l = LabelId::new(0);
    let other = LabelId::new(1);
    for n in [4096, 4097] {
        for p in [0, 1023, n / 2, n - 1, n + 10] {
            let r = regions();
            let pairs: Vec<_> = (1..n)
                .map(|d| (SRC, d, l))
                .chain([8, 8, 10].into_iter().map(|d| (SRC, d, other)))
                .collect();
            let mut g = LabeledLaraGraph::with_edge_list(bundle(&r), &pairs, n + 32).unwrap();
            g.insert_edge(SRC, n, l);
            let (found, t) = observe(&format!("labeled_n{n}_p{p}"), &r, || {
                g.remove_edge(SRC, p + 1, l)
            });
            let want: Vec<_> = (1..=n).filter(|&d| d != p + 1).collect();
            let mut got = Vec::new();
            g.for_each_edge_of_label(SRC, l, |d| got.push(d));
            assert_eq!(sorted(got), want);
            assert_eq!(found, p < n);
            let mut got = Vec::new();
            g.for_each_edge_of_label(SRC, other, |d| got.push(d));
            assert_eq!(got, [8, 8, 10]);
            assert_eq!(g.out_degree(SRC), want.len() as u64 + 3);
            assert_eq!(g.num_edges(), want.len() as u64 + 3);
            g.verify_invariants();
            no_writes_elsewhere(&t, &[6, 7]);
            assert_eq!(t[7].writes.len(), usize::from(found));
            assert!(t[7].writes.iter().all(|w| w.1.len() == 24));
            let moved = p < n - 1;
            swap_contract(
                &t[6],
                if found {
                    p as usize + 1 + usize::from(moved)
                } else {
                    n as usize
                },
                moved.then(|| (offset(p), n.to_le_bytes().to_vec())),
            );
        }
    }
    let r = regions();
    let pairs: Vec<_> = (1..1024)
        .map(|d| (SRC, d, l))
        .chain([8, 8, 10].into_iter().map(|d| (SRC, d, other)))
        .collect();
    let mut g = LabeledLaraGraph::with_edge_list(bundle(&r), &pairs, 4096).unwrap();
    g.insert_edge(SRC, 1024, l);
    for d in 2000..3021 {
        g.insert_edge(SRC, d, other);
    }
    assert!(g.bucket_is_tree(SRC, other));
    g.insert_edge(SRC, 4090, l);
    let before = r[6].bytes();
    let (found, t) = observe("labeled_interleaved", &r, || g.remove_edge(SRC, 1, l));
    assert!(found);
    let mut got = Vec::new();
    g.for_each_edge_of_label(SRC, l, |d| got.push(d));
    assert_eq!(sorted(got), (2..=1024).chain([4090]).collect::<Vec<_>>());
    let mut got = Vec::new();
    g.for_each_edge_of_label(SRC, other, |d| got.push(d));
    assert_eq!(
        got,
        [8, 8, 10].into_iter().chain(2000..3021).collect::<Vec<_>>()
    );
    assert_eq!(g.num_edges(), 2048);
    g.verify_invariants();
    let after = r[6].bytes();
    assert_eq!(&before[4176..8304], &after[4176..8304]);
    no_writes_elsewhere(&t, &[6, 7]);
    swap_contract(&t[6], 2, Some((80, 4090u32.to_le_bytes().to_vec())));
    assert_eq!(t[6].reads[1], (8304, 4, false));
    let ((), t) = observe("labeled_retained_root_regrow", &r, || {
        g.insert_edge(SRC, 4091, l)
    });
    let mut got = Vec::new();
    g.for_each_edge_of_label(SRC, l, |d| got.push(d));
    assert_eq!(
        sorted(got),
        (2..=1024).chain([4090, 4091]).collect::<Vec<_>>()
    );
    assert_eq!(g.num_edges(), 2049);
    g.verify_invariants();
    assert_eq!(&before[4176..8304], &r[6].bytes()[4176..8304]);
    assert_eq!(t[6].writes, [(8304, 4091u32.to_le_bytes().to_vec())]);
    no_writes_elsewhere(&t, &[6, 7]);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Record<const N: usize> {
    dst: u32,
    tag: u32,
}
impl<const N: usize> FixedWidth for Record<N> {
    const BYTES: usize = N;
    fn read_from(b: &[u8]) -> Self {
        assert_eq!(b.len(), N);
        if N > 8 {
            assert!(b[8..].iter().all(|&x| x == 0x5a));
        }
        Self {
            dst: u32::from_le_bytes(b[..4].try_into().unwrap()) ^ 0x8000_0000,
            tag: if N >= 8 {
                u32::from_le_bytes(b[4..8].try_into().unwrap())
            } else {
                0
            },
        }
    }
    fn write_to(&self, b: &mut [u8]) {
        b.fill(0x5a);
        b[..4].copy_from_slice(&(self.dst ^ 0x8000_0000).to_le_bytes());
        if N >= 8 {
            b[4..8].copy_from_slice(&self.tag.to_le_bytes());
        }
    }
}
impl<const N: usize> CsrEdge for Record<N> {
    fn neighbor(&self) -> u32 {
        self.dst
    }
    fn with_neighbor(self, dst: u32) -> Self {
        Self { dst, ..self }
    }
}
impl<const N: usize> CsrEdgeTombstone for Record<N> {
    fn tombstone_edge() -> Self {
        Self {
            dst: u32::MAX,
            tag: 0,
        }
    }
    fn is_tombstone(&self) -> bool {
        self.dst == u32::MAX
    }
}
fn slab_case<const N: usize>() {
    for p in [0, 128, 324, 999] {
        let r = regions();
        let mut g = Core::<Record<N>>::new_with_segments(bundle(&r), 512, 8).unwrap();
        g.set_promote_threshold(None);
        let pairs: Vec<_> = (0..80)
            .map(|d| (SRC, d))
            .chain((0..128).map(|d| (9, d)))
            .collect();
        g.build_pairs(&pairs).unwrap();
        assert_eq!(g.row_allocated(SRC), 325);
        let mut want: Vec<_> = (0..80).map(|dst| Record { dst, tag: 0 }).collect();
        for dst in 80..325 {
            let e = Record {
                dst,
                tag: if N >= 8 { 1000 + dst } else { 0 },
            };
            g.insert_edge_record(SRC, e);
            want.push(e);
        }
        let base = g.get_row_pub(SRC).base_slot_start();
        let before = r[1].bytes();
        let (found, t) = observe(&format!("slab_w{N}_p{p}"), &r, || {
            g.remove_edge_matching(SRC, |e| e.dst == p)
        });
        assert_eq!(found, p < 325);
        if found {
            want.remove(p as usize);
        }
        let mut got = Vec::new();
        g.for_each_record_in_row(base, g.get_row_pub(SRC).stored_degree(), |e| got.push(*e));
        let mut live = got.clone();
        live.retain(|e| !e.is_tombstone());
        live.sort_by_key(|e| e.dst);
        assert_eq!(live, want); // positive record oracle also runs on the old source
        assert_eq!(
            sorted(g.out_neighbors_vec(SRC)),
            want.iter().map(|e| e.dst).collect::<Vec<_>>()
        );
        assert_eq!(g.out_neighbors_vec(9), (0..128).collect::<Vec<_>>());
        assert_eq!(g.num_edges(), want.len() as u64 + 128);
        g.verify_invariants();
        no_writes_elsewhere(&t, &[1]);
        let moved = p < 324;
        let mut wire = vec![0; N];
        Record::<N> {
            dst: 324,
            tag: if N >= 8 { 1324 } else { 0 },
        }
        .write_to(&mut wire);
        swap_contract(
            &t[1],
            if found {
                p as usize + 1 + usize::from(moved)
            } else {
                325
            },
            moved.then(|| ((base + u64::from(p)) * N as u64, wire)),
        );
        assert_eq!(got.len(), want.len());
        if p >= 324 {
            assert_eq!(r[1].bytes(), before);
        }
        let ((), t) = observe(&format!("slab_w{N}_reappend{p}"), &r, || {
            if found {
                g.insert_edge_record(
                    SRC,
                    Record {
                        dst: 400,
                        tag: if N >= 8 { 42 } else { 0 },
                    },
                );
            }
        });
        if found {
            want.push(Record {
                dst: 400,
                tag: if N >= 8 { 42 } else { 0 },
            });
        }
        let mut got = Vec::new();
        g.for_each_record_in_row(base, want.len() as u32, |e| got.push(*e));
        got.sort_by_key(|e| e.dst);
        assert_eq!(got, want);
        g.verify_invariants();
        assert!(t[1].reads.is_empty());
        no_writes_elsewhere(&t, &[1]);
    }
}
#[test]
fn slab_deletion_keeps_dense_full_records_and_boundaries() {
    slab_case::<4>();
    slab_case::<12>();
    slab_case::<516>();
    let r = regions();
    let mut g = Core::<Record<12>>::new_with_segments(bundle(&r), 512, 8).unwrap();
    g.set_promote_threshold(None);
    let pairs: Vec<_> = (0..96)
        .map(|d| (SRC, d))
        .chain((0..128).map(|d| (9, d)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    let mut want: Vec<_> = (0..96).map(|dst| Record { dst, tag: 0 }).collect();
    for tag in 42..54 {
        let e = Record { dst: 7, tag };
        g.insert_edge_record(SRC, e);
        want.push(e);
    }
    let base = g.get_row_pub(SRC).base_slot_start();
    let (found, t) = observe("slab_duplicate_payload_match", &r, || {
        g.remove_edge_matching(SRC, |e| e.dst == 7 && e.tag == 45)
    });
    assert!(found);
    want.retain(|e| e.tag != 45);
    let mut got = Vec::new();
    g.for_each_record_in_row(base, g.get_row_pub(SRC).stored_degree(), |e| {
        if !e.is_tombstone() {
            got.push(*e);
        }
    });
    want.sort_by_key(|e| (e.dst, e.tag));
    got.sort_by_key(|e| (e.dst, e.tag));
    assert_eq!(got, want);
    assert_eq!(got.iter().filter(|e| e.dst == 7).count(), 12);
    assert_eq!(g.num_edges(), 235);
    assert_eq!(g.out_neighbors_vec(9), (0..128).collect::<Vec<_>>());
    g.verify_invariants();
    no_writes_elsewhere(&t, &[1]);
    let mut wire = vec![0; 12];
    Record::<12> { dst: 7, tag: 53 }.write_to(&mut wire);
    swap_contract(&t[1], 101, Some(((base + 99) * 12, wire)));
}

#[test]
fn tiny_swap_and_log_unlink_keep_counts_and_reentry() {
    for cap in [0, 170] {
        for dst in [0, 5, 7, 99] {
            let r = regions();
            let mut g = Core::<Edge>::new_with_max_log_entries(bundle(&r), 64, cap).unwrap();
            let mut want = vec![0, 5, 5, 7];
            for &d in &want {
                g.insert_edge(SRC, d);
            }
            let off = g.get_row_pub(SRC).log_head() as u64;
            let pos = want.iter().position(|&x| x == dst);
            let (found, t) = observe(&format!("tiny_cap{cap}_dst{dst}"), &r, || {
                g.remove_edge(SRC, dst)
            });
            if let Some(p) = pos {
                want.remove(p);
            }
            assert_eq!(found, pos.is_some());
            assert_eq!(sorted(g.out_neighbors_vec(SRC)), sorted(want.clone()));
            assert_eq!(g.num_edges(), want.len() as u64);
            g.verify_invariants();
            no_writes_elsewhere(&t, &[2]);
            let replacement = pos
                .filter(|&p| p < 3)
                .map(|p| (off + p as u64 * 4, 7u32.to_le_bytes().to_vec()));
            swap_contract(&t[2], 1, replacement);
            assert_eq!(t[2].reads[0], (off, 16, false));
            for d in want {
                assert!(g.remove_edge(SRC, d));
            }
            let (got, t) = observe(&format!("tiny_empty_cap{cap}_dst{dst}"), &r, || {
                g.out_neighbors_vec(SRC)
            });
            assert!(got.is_empty());
            assert!(
                t.iter()
                    .all(|t| t.reads.is_empty() && t.writes.is_empty() && t.grows.is_empty())
            );
            let bytes = g.tiny_memory_bytes();
            g.insert_edge(SRC, 8);
            assert_eq!(g.tiny_memory_bytes(), bytes);
            g.verify_invariants();
        }
    }
    let r = regions();
    let mut g = Core::<Edge>::new_with_segments(bundle(&r), 512, 8).unwrap();
    g.set_promote_threshold(None);
    let pairs: Vec<_> = (0..16)
        .map(|d| (SRC, d))
        .chain((0..128).map(|d| (9, d)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    assert_eq!(g.row_allocated(SRC), 67);
    for d in 16..67 {
        g.insert_edge(SRC, d);
    }
    for _ in 0..64 {
        g.insert_edge(SRC, 500);
    }
    let (found, t) = observe("logged_slab_delete", &r, || g.remove_edge(SRC, 0));
    assert!(found);
    let want: Vec<_> = (1..67).chain(std::iter::repeat_n(500, 64)).collect();
    assert_eq!(sorted(g.out_neighbors_vec(SRC)), want);
    g.verify_invariants();
    no_writes_elsewhere(&t, &[1]);
    let base = g.get_row_pub(SRC).base_slot_start();
    swap_contract(&t[1], 2, Some((base * 4, 66u32.to_le_bytes().to_vec())));
    let (found, t) = observe("logged_unlink", &r, || g.remove_edge(SRC, 500));
    assert!(found);
    let want: Vec<_> = (1..67).chain(std::iter::repeat_n(500, 63)).collect();
    assert_eq!(sorted(g.out_neighbors_vec(SRC)), want);
    g.verify_invariants();
    no_writes_elsewhere(&t, &[2]);
    assert_eq!(t[2].writes.len(), 1);
    let ((), t) = observe("logged_reappend", &r, || g.insert_edge(SRC, 501));
    let want: Vec<_> = (1..67)
        .chain(std::iter::repeat_n(500, 63))
        .chain([501])
        .collect();
    assert_eq!(sorted(g.out_neighbors_vec(SRC)), want);
    g.verify_invariants();
    no_writes_elsewhere(&t, &[1, 2]);
    assert_eq!(
        t[1].writes,
        [((base + 66) * 4, 501u32.to_le_bytes().to_vec())]
    );
    assert!(t[1].reads.is_empty());
    assert!(t[2].writes.is_empty());
    g.rebalance_leaf(1);
    assert_eq!(sorted(g.out_neighbors_vec(SRC)), want);
    assert_eq!(g.out_neighbors_vec(9), (0..128).collect::<Vec<_>>());
    g.verify_invariants();
    let r = regions();
    let l = LabelId::new(0);
    let other = LabelId::new(1);
    let pairs: Vec<_> = (0..160)
        .map(|d| (SRC, d, l))
        .chain((0..128).map(|d| (SRC, d, other)))
        .collect();
    let mut g = LabeledLaraGraph::with_edge_list(bundle(&r), &pairs, 512)
        .unwrap()
        .with_promote_threshold(None);
    for d in 160..320 {
        g.insert_edge(SRC, d, l);
    }
    let room = g.bucket_room(SRC, l);
    let (found, t) = observe("labeled_slab_delete", &r, || g.remove_edge(SRC, 129, l));
    assert!(found);
    let mut got = Vec::new();
    g.for_each_edge_of_label(SRC, l, |d| got.push(d));
    assert_eq!(
        sorted(got),
        (0..320).filter(|&x| x != 129).collect::<Vec<_>>()
    );
    let mut got = Vec::new();
    g.for_each_edge_of_label(SRC, other, |d| got.push(d));
    assert_eq!(got, (0..128).collect::<Vec<_>>());
    g.verify_invariants();
    no_writes_elsewhere(&t, &[1, 7]);
    assert_eq!(g.bucket_room(SRC, l), room + 1);
    assert_eq!(t[1].writes.len(), 1);
    assert_eq!(t[1].writes[0].1, 319u32.to_le_bytes());
}

#[test]
fn mutation_and_ingest_cross_storage_transitions_without_losing_edges() {
    use lara_graph::IngestBuffer;
    let sources = [8, 9, 16, 17];
    for cap in [0, 170] {
        let r = regions();
        let mut g = Core::<Edge>::new_with_segments_and_max_log(bundle(&r), 512, 8, cap).unwrap();
        g.set_promote_threshold(Some(64));
        let mut model: Vec<Vec<u32>> = sources.iter().map(|&s| vec![0, 1, 1, s]).collect();
        for (i, &s) in sources.iter().enumerate() {
            for &d in &model[i] {
                g.insert_edge(s, d);
            }
        }
        assert_eq!(g.tiny_row_count(), 4);
        let ((), _) = observe(&format!("transitions_fill_cap{cap}"), &r, || {
            for d in 0..96 {
                for &s in &sources {
                    g.insert_edge(s, d % 64);
                }
            }
        });
        for m in &mut model {
            m.extend((0..96).map(|d| d % 64));
        }
        for (i, &s) in sources.iter().enumerate() {
            assert_eq!(sorted(g.out_neighbors_vec(s)), sorted(model[i].clone()));
        }
        assert_eq!(g.tree_bucket_count(), 4);
        assert_eq!(g.num_edges(), 400);
        g.verify_invariants();
        let erase: Vec<Vec<_>> = model.iter().map(|m| m[..68].to_vec()).collect();
        let (removed, _) = observe(&format!("transitions_demote_cap{cap}"), &r, || {
            let mut removed = Vec::new();
            for (i, &s) in sources.iter().enumerate() {
                for &d in &erase[i] {
                    removed.push(g.remove_edge(s, d));
                }
            }
            removed
        });
        for m in &mut model {
            m.drain(..68);
        }
        assert!(removed.iter().all(|&x| x));
        for (i, &s) in sources.iter().enumerate() {
            assert_eq!(sorted(g.out_neighbors_vec(s)), sorted(model[i].clone()));
        }
        assert_eq!(g.tree_bucket_count(), 0);
        assert_eq!(g.num_edges(), 128);
        g.verify_invariants();
        let mut buf = IngestBuffer::new(4, 4);
        for d in 8..48 {
            for &s in sources.iter().rev() {
                buf.push(s, d);
            }
        }
        assert_eq!(buf.len(), 160);
        assert_eq!(g.num_edges(), 128);
        let ((), _) = observe(&format!("transitions_flush_cap{cap}"), &r, || {
            buf.flush(&mut g)
        });
        for m in &mut model {
            m.extend(8..48);
        }
        assert!(buf.is_empty());
        for (i, &s) in sources.iter().enumerate() {
            assert_eq!(sorted(g.out_neighbors_vec(s)), sorted(model[i].clone()));
        }
        assert_eq!(g.tree_bucket_count(), 4);
        assert_eq!(g.num_edges(), 288);
        g.verify_invariants();
        let ((), t) = observe(&format!("transitions_empty_flush_cap{cap}"), &r, || {
            buf.flush(&mut g)
        });
        assert!(
            t.iter()
                .all(|t| t.reads.is_empty() && t.writes.is_empty() && t.grows.is_empty())
        );
    }
}

#[test]
fn rejected_mutations_and_explicit_raw_shifts_keep_their_contracts() {
    use lara_graph::block_store::LtbRawBlockStore;
    use lara_graph::tree_csr::{R_MAX, TreeCsrBucket};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let r = regions();
    let mut g = Core::<Edge>::new_with_segments(bundle(&r), 128, 8).unwrap();
    let limit = (B * R_MAX) as u32;
    let pairs: Vec<_> = (0..limit - 1).map(|i| (SRC, i % 64)).collect();
    g.build_pairs(&pairs).unwrap();
    g.insert_edge(SRC, 64);
    let before = r.each_ref().map(Probe::bytes);
    for (tag, kind) in [("depth_scalar", 0), ("depth_run", 1), ("invalid_source", 2)] {
        let (result, t) = observe(tag, &r, || {
            catch_unwind(AssertUnwindSafe(|| match kind {
                0 => g.insert_edge(SRC, 65),
                1 => g.insert_sorted_run(SRC, &[65, 66]),
                _ => g.insert_edge(u32::MAX, 0),
            }))
        });
        let error = result.unwrap_err();
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap();
        assert!(message.contains(if kind < 2 {
            "tree-csr: depth-1 capacity exceeded"
        } else {
            "src out of range"
        }));
        no_writes_elsewhere(&t, &[]);
        assert_eq!(r.each_ref().map(Probe::bytes), before);
        assert_eq!(g.out_degree(SRC), u64::from(limit));
        g.verify_invariants();
    }
    assert_eq!(
        g.out_neighbors_vec(SRC),
        (0..limit - 1)
            .map(|i| i % 64)
            .chain([64])
            .collect::<Vec<_>>()
    );
    let r = regions();
    let mut raw = Core::<Edge>::new(bundle(&r), 64).unwrap();
    let idx = raw.tree_state_push(Vec::new(), 0) as usize;
    for d in [0, u32::MAX, 7] {
        raw.tree_state_insert(idx, d);
    }
    let ((), t) = observe("explicit_raw_shift", &r, || {
        raw.tree_state_remove_shift(idx, 0)
    });
    assert_eq!(raw.tree_state_stored(idx), 2);
    assert_eq!(raw.tree_state_read(idx, 0), u32::MAX);
    assert_eq!(raw.tree_state_read(idx, 1), 7);
    assert_eq!(
        t[6].writes,
        [
            (80, u32::MAX.to_le_bytes().to_vec()),
            (84, 7u32.to_le_bytes().to_vec())
        ]
    );
    no_writes_elsewhere(&t, &[6]);
    raw.tree_state_release(idx);
    raw.verify_invariants();
    let mem = Probe::default();
    let mut bucket = TreeCsrBucket::new(LtbRawBlockStore::new(mem).unwrap());
    bucket.promote_from_slice(&[0, u32::MAX, 7, 7]);
    assert_eq!(bucket.remove_at(0), Some(0));
    let mut got = Vec::new();
    bucket.for_each_ascending(|i, n| got.push((i, n)));
    assert_eq!(got, [(0, u32::MAX), (1, 7), (2, 7)]);
    assert_eq!(bucket.remove_at(3), None);
}
