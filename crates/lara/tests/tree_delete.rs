//! Current dense deletion I/O and unchanged raw/ordinal contracts.
//! The original shift baseline is preserved in the B1a source archive.
//! All eight real-memory regions are frozen before semantic and budget oracles.
#![cfg(feature = "std")]

use lara_graph::block_store::{BLOCK_STRIDE, LtbRawBlockStore};
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::labeled::{LabelId, LabeledLaraGraph};
use lara_graph::memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
use lara_graph::traits::{CsrEdge, CsrEdgeTombstone, CsrVertex, FixedWidth, NO_BUCKET};
use lara_graph::tree_csr::{B, TreeCsrBucket};
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
    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        if let Some(t) = self.trace.borrow_mut().as_mut() {
            t.grows.push(pages);
        }
        self.mem.grow(pages)
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
    let traces: [Trace; 8] = std::array::from_fn(|i| r[i].trace.take().unwrap());
    for (i, t) in traces.iter().enumerate() {
        let region = REGIONS[i];
        let mapped = t.reads.iter().filter(|r| r.2).count();
        let rb: usize = t.reads.iter().map(|r| r.1).sum();
        let wb: usize = t.writes.iter().map(|w| w.1.len()).sum();
        println!(
            "OBS,{tag},{region},{},{mapped},{rb},{},{wb},{},{},{}",
            t.reads.len() - mapped,
            t.writes.len(),
            t.grows.len(),
            r[i].size(),
            r[i].mem.capacity_bytes()
        );
        for (k, (off, len, mapped)) in t.reads.iter().enumerate() {
            println!("READ,{tag},{region},{k},{off},{len},{mapped}");
        }
        for (k, (off, bytes)) in t.writes.iter().enumerate() {
            let mut hex = String::with_capacity(bytes.len() * 2);
            for b in bytes {
                write!(hex, "{b:02x}").unwrap();
            }
            println!("WRITE,{tag},{region},{k},{off},{},{hex}", bytes.len());
        }
        for (k, pages) in t.grows.iter().enumerate() {
            println!("GROW,{tag},{region},{k},{pages}");
        }
    }
    (result, traces)
}

type Core<E = Edge> = LaraGraph<Vertex, E, Probe>;

fn core<E: CsrEdgeTombstone>(n: u32, threshold: u32) -> (Core<E>, [Probe; 8]) {
    let r = regions();
    let mut g = Core::<E>::new_with_segments(bundle(&r), (n + 16).max(4096), 8).unwrap();
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

fn verify_core<E: CsrEdgeTombstone>(g: &Core<E>, want: &[u32]) {
    assert_eq!(g.out_neighbors_vec(SRC), want);
    assert_eq!(g.out_neighbors_vec(0), [1, 2, 2]);
    assert_eq!(g.out_neighbors_vec(9), [9, 9, 10]);
    assert_eq!(g.out_degree(SRC), want.len() as u64);
    assert_eq!(g.num_edges(), want.len() as u64 + 6);
    g.verify_invariants();
}

fn no_writes_elsewhere(t: &[Trace; 8], allowed: &[usize]) {
    for (i, p) in t.iter().enumerate() {
        assert!(p.grows.is_empty());
        if !allowed.contains(&i) {
            assert!(p.writes.is_empty(), "{} writes", REGIONS[i]);
        }
    }
}

fn payload_offset(slot: u32) -> u64 {
    80 + u64::from(slot / B as u32) * BLOCK_STRIDE + u64::from(slot % B as u32) * 4
}

fn swapped_payload(t: &Trace, n: u32, position: u32) {
    let moved = position < n - 1;
    assert_eq!(
        t.reads.len(),
        if position < n {
            position as usize + 1 + usize::from(moved)
        } else {
            n as usize
        }
    );
    assert!(t.reads.iter().all(|r| r.1 == 4 && !r.2));
    assert_eq!(
        t.writes,
        if moved {
            vec![(payload_offset(position), n.to_le_bytes().to_vec())]
        } else {
            vec![]
        }
    );
}

#[test]
fn core_deletion_replaces_from_the_tail() {
    for n in [4096, 4097] {
        for position in [0, 1023, n / 2, n - 1, n + 10] {
            let (mut g, r) = core::<Edge>(n, 1024);
            let geometry = g.leaf_geometry(1);
            let before = r[6].bytes();
            let (found, t) = observe(&format!("core_n{n}_pos{position}"), &r, || {
                g.remove_edge(SRC, position + 1)
            });
            let mut want: Vec<_> = (1..=n).collect();
            if position < n {
                want.swap_remove(position as usize);
            }
            assert_eq!(found, position < n);
            verify_core(&g, &want);
            assert_eq!(
                g.tree_state_stored(g.get_row_pub(SRC).tree_bucket() as usize),
                want.len() as u32
            );
            assert_eq!(g.leaf_geometry(1), geometry);
            assert_eq!(g.tree_bucket_count(), 1);
            if position >= n - 1 {
                assert_eq!(r[6].bytes(), before);
            }
            no_writes_elsewhere(&t, &[6]);
            swapped_payload(&t[6], n, position);
        }
    }
}

#[test]
fn labeled_deletion_has_the_same_tail_replacement() {
    for n in [4096, 4097] {
        for position in [0, 1023, n / 2, n - 1, n + 10] {
            let r = regions();
            let l = LabelId::new(0);
            let other = LabelId::new(1);
            let pairs: Vec<_> = [1, 2, 2]
                .into_iter()
                .map(|d| (0, d, l))
                .chain((1..n).map(|d| (SRC, d, l)))
                .chain([8, 8, 10].into_iter().map(|d| (SRC, d, other)))
                .chain([9, 9, 10].into_iter().map(|d| (9, d, l)))
                .collect();
            let mut g =
                LabeledLaraGraph::with_edge_list(bundle(&r), &pairs, (n + 16).max(4096)).unwrap();
            g.insert_edge(SRC, n, l);
            let (found, t) = observe(&format!("labeled_n{n}_pos{position}"), &r, || {
                g.remove_edge(SRC, position + 1, l)
            });
            let mut want: Vec<_> = (1..=n).collect();
            if position < n {
                want.swap_remove(position as usize);
            }
            assert_eq!(found, position < n);
            let mut got = Vec::new();
            g.for_each_edge_of_label(SRC, l, |d| got.push(d));
            assert_eq!(got, want);
            let mut got = Vec::new();
            g.for_each_edge_of_label(SRC, other, |d| got.push(d));
            assert_eq!(got, [8, 8, 10]);
            assert_eq!(g.out_labeled_vec(0), [(1, l), (2, l), (2, l)]);
            assert_eq!(g.out_labeled_vec(9), [(9, l), (9, l), (10, l)]);
            assert_eq!(g.num_edges(), want.len() as u64 + 9);
            assert_eq!(g.out_degree(SRC), want.len() as u64 + 3);
            assert!(g.bucket_is_tree(SRC, l));
            g.verify_invariants();
            no_writes_elsewhere(&t, &[6, 7]);
            swapped_payload(&t[6], n, position);
            assert_eq!(t[7].writes.len(), usize::from(found));
            assert!(t[7].writes.iter().all(|w| w.1.len() == 24));
        }
    }
}

#[test]
fn standalone_ordinals_shift_and_accept_all_u32_values() {
    let n = 4097;
    let original: Vec<_> = (0..n).map(|v| if v == 1 { u32::MAX } else { v }).collect();
    for position in [0, 1, 1023, n - 1, n + 10] {
        let r = regions();
        let mut tree = TreeCsrBucket::new(LtbRawBlockStore::new(r[6].clone()).unwrap());
        tree.promote_from_slice(&original);
        let (removed, t) = observe(&format!("ordinal_pos{position}"), &r, || {
            tree.remove_at(position)
        });
        let mut want = original.clone();
        let expected = if position < n {
            Some(want.remove(position as usize))
        } else {
            None
        };
        assert_eq!(removed, expected);
        assert_eq!(tree.stored_slots(), want.len() as u32);
        assert_eq!(tree.root_len(), if position < n { 4 } else { 5 });
        let mut got = Vec::new();
        tree.for_each_ascending(|slot, target| got.push((slot, target)));
        assert_eq!(
            got,
            want.iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (i as u32, v))
                .collect::<Vec<_>>()
        );
        for (i, &v) in want.iter().enumerate() {
            assert_eq!(tree.range_target(i as u32), Some(v));
        }
        assert_eq!(tree.range_target(want.len() as u32), None);
        no_writes_elsewhere(&t, &[6]);
        if position < n {
            assert_eq!(t[6].reads.len(), (n - position) as usize);
            let payload_writes = (n - position) as usize;
            assert_eq!(t[6].writes.len(), payload_writes + 4); // clear tail, release block + global header
            assert_eq!(
                t[6].writes[payload_writes - 1],
                (payload_offset(n - 1), vec![0; 4])
            );
            let mut store = tree.into_store();
            assert_eq!(store.mint_zeroed().unwrap(), 4); // popped block is reusable
        } else {
            assert!(t[6].reads.is_empty() && t[6].writes.is_empty());
        }
    }
}

#[test]
fn raw_staging_preserves_max_neighbor_through_shift_and_scan() {
    let r = regions();
    let mut g = Core::<Edge>::new_with_segments(bundle(&r), 64, 8).unwrap();
    let idx = g.tree_state_push(Vec::new(), 0) as usize;
    g.tree_state_insert(idx, 0);
    let (_, t) = observe("raw_append_max", &r, || g.tree_state_insert(idx, u32::MAX));
    assert_eq!(g.tree_state_read(idx, 1), u32::MAX);
    assert_eq!(g.tree_state_stored(idx), 2);
    assert_eq!(t[6].writes, [(84, u32::MAX.to_le_bytes().to_vec())]);
    g.tree_state_insert(idx, 7);
    let (_, t) = observe("raw_shift_before_max", &r, || {
        g.tree_state_remove_shift(idx, 0)
    });
    assert_eq!(g.tree_state_stored(idx), 2);
    assert_eq!(g.tree_state_read(idx, 0), u32::MAX);
    assert_eq!(g.tree_state_read(idx, 1), 7);
    assert_eq!(
        t[6].writes,
        [
            (80, u32::MAX.to_le_bytes().to_vec()),
            (84, 7u32.to_le_bytes().to_vec())
        ]
    );
    let (got, t) = observe("raw_scan_max", &r, || {
        let mut out = Vec::new();
        g.for_each_in_tree(idx, |d| out.push(d));
        out
    });
    assert_eq!(got, [u32::MAX, 7]);
    assert_eq!(t[6].reads, [(80, 8, false)]);
    assert!(t.iter().all(|r| r.writes.is_empty() && r.grows.is_empty()));
    // Detached staging must be released before whole-graph ownership checks.
    g.tree_state_release(idx);
    assert_eq!(g.num_edges(), 0);
    g.verify_invariants();
}

#[derive(Clone, Copy, Default)]
struct FlagEdge(u32);
impl FixedWidth for FlagEdge {
    const BYTES: usize = 4;
    fn read_from(b: &[u8]) -> Self {
        Self(u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn write_to(&self, b: &mut [u8]) {
        b.copy_from_slice(&self.0.to_le_bytes());
    }
}
impl CsrEdge for FlagEdge {
    fn neighbor(&self) -> u32 {
        self.0 & 0x7fff_ffff
    }
    fn with_neighbor(self, n: u32) -> Self {
        Self(n & 0x7fff_ffff)
    }
}
impl CsrEdgeTombstone for FlagEdge {
    fn tombstone_edge() -> Self {
        Self(0x8000_0007)
    }
    fn is_tombstone(&self) -> bool {
        self.0 & 0x8000_0000 != 0
    }
}

#[derive(Clone, Copy, Default)]
struct XorZeroEdge(u32);
impl FixedWidth for XorZeroEdge {
    const BYTES: usize = 4;
    fn read_from(b: &[u8]) -> Self {
        Self(u32::from_le_bytes(b.try_into().unwrap()) ^ 7)
    }
    fn write_to(&self, b: &mut [u8]) {
        b.copy_from_slice(&(self.0 ^ 7).to_le_bytes());
    }
}
impl CsrEdge for XorZeroEdge {
    fn neighbor(&self) -> u32 {
        self.0
    }
    fn with_neighbor(self, n: u32) -> Self {
        Self(n)
    }
}
impl CsrEdgeTombstone for XorZeroEdge {
    fn tombstone_edge() -> Self {
        Self(0)
    }
    fn is_tombstone(&self) -> bool {
        self.0 == 0
    }
}

fn marker_case<E: CsrEdgeTombstone>(tag: &str, marker_neighbor: u32, marker_wire: u32) {
    let (mut g, r) = core::<E>(1024, 1024);
    g.insert_edge(SRC, 7); // another live7; not a tombstone
    let mut want: Vec<_> = (1..=1024).chain([7]).collect();
    let (got, t) = observe(tag, &r, || g.out_neighbors_vec(SRC));
    assert_eq!(got, want);
    verify_core(&g, &want);
    assert!(!E::with_neighbor(E::default(), 7).is_tombstone());
    assert_eq!(E::tombstone_edge().neighbor(), marker_neighbor);
    let mut wire = [0; 4];
    E::tombstone_edge().write_to(&mut wire);
    assert_eq!(u32::from_le_bytes(wire), marker_wire);
    // Tree slot6 holds raw7, independently of either type's wire marker.
    assert_eq!(&r[6].bytes()[104..108], &7u32.to_le_bytes());
    assert!(t.iter().all(|r| r.writes.is_empty() && r.grows.is_empty()));
    let (ok, _) = observe(&format!("{tag}_delete1"), &r, || g.remove_edge(SRC, 1));
    assert!(ok);
    want.swap_remove(0);
    verify_core(&g, &want);
    assert_eq!(
        g.out_neighbors_vec(SRC).iter().filter(|&&v| v == 7).count(),
        2
    );
}

#[test]
fn a_tombstone_neighbor_or_wire_word_can_be_a_live_raw_id() {
    // A valid flagged record and its tombstone share neighbor7.
    marker_case::<FlagEdge>("marker_neighbor_collision", 7, 0x8000_0007);
    // The tombstone's XOR wire word is7, a different live raw neighbor.
    marker_case::<XorZeroEdge>("marker_wire_collision", 0, 7);
    assert!(!XorZeroEdge::with_neighbor(XorZeroEdge::default(), u32::MAX).is_tombstone());
}

#[test]
fn interleaved_roots_and_demotion_are_distinct_delete_costs() {
    let (mut g, r) = core::<Edge>(1024, 1024);
    for d in 2000..3021 {
        g.insert_edge(9, d);
    }
    assert_eq!(g.tree_bucket_count(), 2);
    g.insert_edge(SRC, 4090); // root0 for SRC, root1 for sibling, root2 for SRC
    let before = r[6].bytes();
    let (ok, t) = observe("interleaved_remove1024", &r, || g.remove_edge(SRC, 1024));
    assert!(ok);
    assert_eq!(
        g.out_neighbors_vec(SRC),
        (1..1024).chain([4090]).collect::<Vec<_>>()
    );
    assert_eq!(
        g.out_neighbors_vec(9),
        [9, 9, 10].into_iter().chain(2000..3021).collect::<Vec<_>>()
    );
    assert_eq!(g.out_neighbors_vec(0), [1, 2, 2]);
    assert_eq!(g.num_edges(), 2051);
    g.verify_invariants();
    assert_eq!(t[6].reads.len(), 1025);
    assert_eq!(t[6].reads.last(), Some(&(8304, 4, false)));
    assert_eq!(t[6].writes, [(4172, 4090u32.to_le_bytes().to_vec())]);
    assert_eq!(r[6].bytes()[4176..], before[4176..]);
    no_writes_elsewhere(&t, &[6]);

    let (mut g, r) = core::<Edge>(65, 64);
    for d in 1..33 {
        assert!(g.remove_edge(SRC, d));
    }
    assert_eq!(g.out_degree(SRC), 33);
    let (ok, t) = observe("core_demote33_to32", &r, || g.remove_edge(SRC, 33));
    assert!(ok);
    verify_core(&g, &(34..=65).rev().collect::<Vec<_>>());
    assert_eq!(g.tree_bucket_count(), 0);
    assert_eq!(g.get_row_pub(SRC).tree_bucket(), NO_BUCKET);
    assert_eq!(t[6].reads.len(), 65); // search33 (now the tail) + collection32
    assert_eq!(t[6].writes.len(), 4); // release/global headers only
    assert_eq!(t[6].writes.iter().map(|w| w.1.len()).sum::<usize>(), 28);
    assert!(!t[1].reads.is_empty() && !t[1].writes.is_empty());
}
