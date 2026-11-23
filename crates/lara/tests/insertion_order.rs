//! Explicit core ordering through public APIs and strict real backing.
//! Freeze all eight regions before value, count, record and ownership checks.
#![cfg(feature = "std")]

use lara_graph::block_store::BLOCK_STRIDE;
use lara_graph::graph::LaraMemories;
use lara_graph::memory::{GrowFailed, PAGE_SIZE};
use lara_graph::traits::NO_BUCKET;
use lara_graph::{
    AdjacencyOrder, CsrEdge, CsrEdgeTombstone, CsrVertex, Edge, FixedWidth, IngestBuffer,
    LaraGraph, Memory, VecMemory, Vertex,
};
use std::cell::RefCell;
use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};
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
#[derive(Debug, Default)]
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
        let hash = r[i].bytes().iter().fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        println!("STATE,{tag},{region},{hash:016x}");
    }
    (result, ts)
}
type Core<E = Edge> = LaraGraph<Vertex, E, Probe>;
fn verify<E: CsrEdge>(g: &Core<E>, rows: &[(u32, &[u32])]) {
    for src in 0..g.num_nodes() as u32 {
        let want = rows.iter().find(|r| r.0 == src).map(|r| r.1).unwrap_or(&[]);
        assert_eq!(g.out_neighbors_vec(src), want, "source {src}");
        let mut through_trait = Vec::new();
        lara_graph::DynamicCsr::for_each_neighbor(g, src, |d| through_trait.push(d));
        assert_eq!(through_trait, want);
        assert_eq!(g.out_degree(src), want.len() as u64, "degree {src}");
    }
    assert_eq!(
        g.num_edges(),
        rows.iter().map(|r| r.1.len() as u64).sum::<u64>()
    );
    g.verify_invariants();
}

#[test]
fn tiny_delete_preserves_successful_application_order() {
    for cap in [0, 170] {
        let r = regions();
        let mut g = Core::<Edge>::new_with_segments(bundle(&r), 64, 8).unwrap();
        g.set_max_log_entries(cap);
        assert_eq!(g.order(), AdjacencyOrder::Unordered);
        g.set_order(AdjacencyOrder::Insertion).unwrap();
        for d in [30, 10, 20, 10] {
            g.insert_edge(SRC, d);
        }
        assert!(g.row_is_tiny(SRC));
        let off = g.get_row_pub(SRC).log_head() as u64;
        let (removed, t) = observe(&format!("tiny_delete_cap{cap}"), &r, || {
            g.remove_edge(SRC, 10)
        });
        assert!(removed);
        verify(&g, &[(SRC, &[30, 20, 10])]);
        assert_eq!(
            t[2].writes,
            [
                (off + 4, 20u32.to_le_bytes().to_vec()),
                (off + 8, 10u32.to_le_bytes().to_vec())
            ]
        );
        let (nested, _) = observe(&format!("tiny_nested_cap{cap}"), &r, || {
            let mut nested = Vec::new();
            g.for_each_neighbor(SRC, |d| nested.push((d, g.out_neighbors_vec(SRC))));
            nested
        });
        assert_eq!(nested, [30, 20, 10].map(|d| (d, vec![30, 20, 10])));
        for d in [30, 20, 10] {
            assert!(g.remove_edge(SRC, d));
        }
        g.insert_sorted_run(SRC, &[7, 3, 7]);
        verify(&g, &[(SRC, &[7, 3, 7])]);
        assert_eq!(g.get_row_pub(SRC).log_head() as u64, off);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Record<const N: usize> {
    dst: u32,
    tag: u64,
}
impl<const N: usize> FixedWidth for Record<N> {
    const BYTES: usize = N;
    fn read_from(b: &[u8]) -> Self {
        Self {
            dst: u32::from_le_bytes(b[..4].try_into().unwrap()) ^ 0x8000_0000,
            tag: if N >= 12 {
                u64::from_le_bytes(b[4..12].try_into().unwrap())
            } else {
                0
            },
        }
    }
    fn write_to(&self, b: &mut [u8]) {
        b.fill(0x5a);
        b[..4].copy_from_slice(&(self.dst ^ 0x8000_0000).to_le_bytes());
        if N >= 12 {
            b[4..12].copy_from_slice(&self.tag.to_le_bytes());
        }
    }
}
impl<const N: usize> CsrEdge for Record<N> {
    fn neighbor(&self) -> u32 {
        self.dst
    }
    fn with_neighbor(mut self, n: u32) -> Self {
        self.dst = n;
        self
    }
}
impl<const N: usize> CsrEdgeTombstone for Record<N> {
    fn tombstone_edge() -> Self {
        Self {
            dst: u32::MAX,
            tag: 99,
        }
    }
    fn is_tombstone(&self) -> bool {
        self.dst == u32::MAX
    }
}
fn slab_case<const N: usize>(n: u32) {
    let r = regions();
    let mut g = Core::<Record<N>>::new_with_segments(bundle(&r), 16384, 8).unwrap();
    g.set_promote_threshold(None);
    g.set_order(AdjacencyOrder::Insertion).unwrap();
    let pairs: Vec<_> = (0..n)
        .rev()
        .map(|d| (SRC, d))
        .chain((0..256).map(|d| (9, d)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    g.rebalance_leaf(1); // Established density avoids unrelated low-density folding.
    let base = g.get_row_pub(SRC).base_slot_start();
    let mut want: Vec<_> = (0..n)
        .rev()
        .map(|dst| Record::<N> { dst, tag: 0 })
        .collect();
    let (removed, t) = observe(&format!("slab{N}_n{n}_delete"), &r, || {
        g.remove_edge(SRC, n - 1)
    });
    assert!(removed);
    want.remove(0);
    let dsts: Vec<_> = want.iter().map(|e| e.dst).collect();
    let sibling: Vec<_> = (0..256).collect();
    verify(&g, &[(SRC, &dsts), (9, &sibling)]);
    assert_eq!(g.get_row_pub(SRC).stored_degree(), n);
    let mut marker = vec![0; N];
    Record::<N>::tombstone_edge().write_to(&mut marker);
    assert_eq!(t[1].writes, [(base * N as u64, marker)]);
    let added = Record {
        dst: 7777,
        tag: if N >= 12 { 12345 } else { 0 },
    };
    observe(&format!("slab{N}_n{n}_append"), &r, || {
        g.insert_edge_record(SRC, added)
    });
    want.push(added);
    assert_eq!(g.get_row_pub(SRC).stored_degree(), n + 1);
    let mut raw = Vec::new();
    g.for_each_record_in_row(base, n + 1, |e| raw.push(*e));
    assert!(raw[0].is_tombstone());
    assert_eq!(&raw[1..], want);
    let (got, _) = observe(&format!("slab{N}_n{n}_scan"), &r, || {
        g.out_neighbors_vec(SRC)
    });
    assert_eq!(got, want.iter().map(|e| e.dst).collect::<Vec<_>>());
    observe(&format!("slab{N}_n{n}_fold"), &r, || g.rebalance_leaf(1));
    let row = g.get_row_pub(SRC);
    assert_eq!(row.stored_degree(), want.len() as u32);
    let mut packed = Vec::new();
    g.for_each_record_in_row(row.base_slot_start(), row.stored_degree(), |e| {
        packed.push(*e)
    });
    assert_eq!(packed, want);
    verify(&g, &[(SRC, &got), (9, &sibling)]);
}
#[test]
fn slab_markers_preserve_live_records_and_never_receive_new_edges() {
    for n in [127, 128, 129, 8193] {
        slab_case::<4>(n);
    }
    for n in [129, 2731] {
        slab_case::<12>(n);
    }
    for n in [65, 129] {
        slab_case::<516>(n);
    }
}

#[test]
fn logs_follow_application_order_and_cannot_be_overtaken() {
    for cap in [0, 170] {
        let r = regions();
        let mut g = Core::<Edge>::new_with_segments_and_max_log(bundle(&r), 64, 8, cap).unwrap();
        g.set_order(AdjacencyOrder::Insertion).unwrap();
        let mut want = vec![7, 3, 7, 4, 6, 1, 9, 2, 8, 5];
        let sibling = [16, 11, 19, 12, 18, 13, 17, 14, 15];
        for &d in &want {
            g.insert_edge(SRC, d);
        }
        for &d in &sibling {
            g.insert_edge(9, d);
        }
        let (got, _) = observe(&format!("spill_scan_cap{cap}"), &r, || {
            g.out_neighbors_vec(SRC)
        });
        assert_eq!(got, want);
        let (removed, _) = observe(&format!("spill_delete_cap{cap}"), &r, || {
            g.remove_edge(SRC, 7)
        });
        assert!(removed);
        want.remove(0);
        verify(&g, &[(SRC, &want), (9, &sibling)]);
        observe(&format!("spill_fold_cap{cap}"), &r, || g.rebalance_leaf(1));
        verify(&g, &[(SRC, &want), (9, &sibling)]);
        assert_eq!(g.log_allocated_chunks(), 0);
    }
    let r = regions();
    let mut g = Core::<Edge>::new_with_segments(bundle(&r), 64, 8).unwrap();
    g.set_promote_threshold(None);
    g.set_order(AdjacencyOrder::Insertion).unwrap();
    let sibling: Vec<_> = (0..128).collect();
    let pairs: Vec<_> = (0..16)
        .map(|d| (SRC, d))
        .chain(sibling.iter().map(|&d| (10, d)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    let width = (g.get_row_pub(9).base_slot_start() - g.get_row_pub(SRC).base_slot_start()) as u32;
    assert_eq!(width, 67);
    for d in 16..width {
        g.insert_edge(SRC, d);
    }
    for d in [500, 501, 500] {
        g.insert_edge(SRC, d);
    }
    assert!(!g.row_has_slab_room(SRC));
    // Public, valid geometry control: lend two slack cells from the next
    // EMPTY row, without changing live counts, spans or allocation owners.
    let next = g.get_row_pub(9);
    assert_eq!(next.degree(), 0);
    assert!(next.base_slot_start() + 2 <= g.get_row_pub(10).base_slot_start());
    g.set_row_pub(9, next.with_base_slot_start(next.base_slot_start() + 2));
    g.verify_invariants();
    assert!(g.row_has_slab_room(SRC));
    let (_, scalar) = observe("pending_log_scalar", &r, || g.insert_edge(SRC, 600));
    let mut want: Vec<_> = (0..width).chain([500, 501, 500, 600]).collect();
    verify(&g, &[(SRC, &want), (10, &sibling)]);
    assert!(scalar[1].writes.is_empty());
    assert_eq!(scalar[2].writes.len(), 1);
    let (_, run) = observe("pending_log_run", &r, || {
        g.insert_sorted_run(SRC, &[602, 601])
    });
    want.extend([602, 601]);
    verify(&g, &[(SRC, &want), (10, &sibling)]);
    assert!(run[1].writes.is_empty());
    assert_eq!(run[2].writes.len(), 2);
    let seen = RefCell::new(Vec::new());
    let (removed, _) = observe("oldest_log_duplicate", &r, || {
        g.remove_edge_matching(SRC, |e| {
            seen.borrow_mut().push(e.neighbor());
            e.neighbor() == 500
        })
    });
    assert!(removed);
    assert_eq!(
        seen.into_inner(),
        (0..width).chain([500]).collect::<Vec<_>>()
    );
    want.remove(width as usize);
    verify(&g, &[(SRC, &want), (10, &sibling)]);
    observe("pending_log_fold", &r, || g.rebalance_leaf(1));
    g.insert_sorted_run(SRC, &[700, 603, 700, 604]);
    want.extend([700, 603, 700, 604]);
    verify(&g, &[(SRC, &want), (10, &sibling)]);
}

#[test]
fn trees_shift_survivors_and_keep_order_through_transitions() {
    for n in [4096, 4097] {
        for p in [0, 1023, n / 2, n - 1, n + 10] {
            let r = regions();
            let mut g = Core::<Edge>::new_with_segments(bundle(&r), 8192, 8).unwrap();
            g.set_order(AdjacencyOrder::Insertion).unwrap();
            let pairs: Vec<_> = (1..n)
                .map(|d| (SRC, d))
                .chain([9, 9, 10].map(|d| (9, d)))
                .collect();
            g.build_pairs(&pairs).unwrap();
            g.insert_edge(SRC, n);
            assert_ne!(g.get_row_pub(SRC).tree_bucket(), NO_BUCKET);
            let (removed, t) =
                observe(&format!("tree_n{n}_p{p}"), &r, || g.remove_edge(SRC, p + 1));
            let want: Vec<_> = (1..=n).filter(|&d| d != p + 1).collect();
            assert_eq!(removed, p < n);
            verify(&g, &[(SRC, &want), (9, &[9, 9, 10])]);
            assert_eq!(t[6].reads.len(), n as usize);
            assert_eq!(
                t[6].writes.len(),
                if p < n { (n - p - 1) as usize } else { 0 }
            );
            for (k, (off, b)) in t[6].writes.iter().enumerate() {
                let slot = u64::from(p) + k as u64;
                assert_eq!(*off, 80 + (slot / 1024) * BLOCK_STRIDE + (slot % 1024) * 4);
                assert_eq!(b, &(p + k as u32 + 2).to_le_bytes());
            }
        }
    }
    let r = regions();
    let mut g = Core::<Edge>::new_with_segments(bundle(&r), 8192, 8).unwrap();
    g.set_order(AdjacencyOrder::Insertion).unwrap();
    let sibling: Vec<_> = (1001..=2024).collect();
    let pairs: Vec<_> = (1..1024)
        .map(|d| (SRC, d))
        .chain((1001..2024).map(|d| (16, d)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    g.insert_edge(SRC, 1024); // root 0
    g.insert_edge(16, 2024); // root 1
    g.insert_edge(SRC, 4090); // root 2, noncontiguous with 0
    let sibling_start = (64 + BLOCK_STRIDE) as usize;
    let sibling_end = sibling_start + BLOCK_STRIDE as usize;
    let before = r[6].bytes();
    let (removed, t) = observe("tree_noncontiguous_shift", &r, || g.remove_edge(SRC, 1));
    assert!(removed);
    assert_eq!(t[6].reads.last(), Some(&(80 + 2 * BLOCK_STRIDE, 4, false)));
    assert_eq!(
        t[6].writes.last(),
        Some(&(80 + 1023 * 4, 4090u32.to_le_bytes().to_vec()))
    );
    let mut want: Vec<_> = (2..=1024).chain([4090]).collect();
    verify(&g, &[(SRC, &want), (16, &sibling)]);
    while want.len() > 600 {
        assert!(g.remove_edge(SRC, want.pop().unwrap()));
    }
    let run: Vec<_> = (5000..5424).rev().collect();
    observe("tree_resident_run424", &r, || {
        g.insert_sorted_run(SRC, &run)
    });
    want.extend(run);
    let (_, t) = observe("tree_retained_root_run", &r, || {
        g.insert_sorted_run(SRC, &[6002, 6000, 6002])
    });
    want.extend([6002, 6000, 6002]);
    assert_eq!(
        t[6].writes,
        [(
            80 + 2 * BLOCK_STRIDE,
            [6002u32, 6000, 6002]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect()
        )]
    );
    verify(&g, &[(SRC, &want), (16, &sibling)]);
    assert_eq!(
        &r[6].bytes()[sibling_start..sibling_end],
        &before[sibling_start..sibling_end]
    );
    assert!(g.remove_edge(SRC, 6002));
    want.remove(want.iter().position(|&d| d == 6002).unwrap());
    verify(&g, &[(SRC, &want), (16, &sibling)]);
    g.set_promote_threshold(Some(64));
    while want.len() > 33 {
        assert!(g.remove_edge(SRC, want.remove(0)));
    }
    let (removed, _) = observe("tree_demote33_to32", &r, || g.remove_edge(SRC, want[0]));
    assert!(removed);
    want.remove(0);
    assert_eq!(g.get_row_pub(SRC).tree_bucket(), NO_BUCKET);
    verify(&g, &[(SRC, &want), (16, &sibling)]);
    let run: Vec<_> = (20..52).rev().collect();
    observe("tree_repromote32_to64", &r, || {
        g.insert_sorted_run(SRC, &run)
    });
    want.extend(run);
    assert_ne!(g.get_row_pub(SRC).tree_bucket(), NO_BUCKET);
    verify(&g, &[(SRC, &want), (16, &sibling)]);
}

#[test]
fn configured_bulk_and_explicit_ingest_use_source_relative_input_order() {
    let r = regions();
    let mut g = Core::<Edge>::new_with_segments(bundle(&r), 256, 8).unwrap();
    g.set_order(AdjacencyOrder::Insertion).unwrap();
    let pairs = [(8, 40), (9, 30), (8, 2), (8, 40), (9, 1), (8, 7)];
    let (built, _) = observe("configured_bulk", &r, || g.build_pairs(&pairs));
    built.unwrap();
    verify(&g, &[(8, &[40, 2, 40, 7]), (9, &[30, 1])]);
    assert_eq!(g.order(), AdjacencyOrder::Insertion);
    g.insert_sorted_run(8, &[22, 3, 22]);
    let mut buf = IngestBuffer::new(0, 2);
    let (_, t) = observe("buffered_push_is_invisible", &r, || {
        for (s, d) in [(13, 7), (8, 17), (13, 3), (9, 8), (8, 1), (13, 7)] {
            buf.push(s, d);
        }
    });
    assert!(
        t.iter()
            .all(|t| t.reads.is_empty() && t.writes.is_empty() && t.grows.is_empty())
    );
    assert_eq!(buf.len(), 6);
    verify(&g, &[(8, &[40, 2, 40, 7, 22, 3, 22]), (9, &[30, 1])]);
    g.insert_edge(13, 99); // Direct B is applied before buffered A=7.
    observe("ordered_ingest_flush", &r, || buf.flush(&mut g));
    assert!(buf.is_empty());
    verify(
        &g,
        &[
            (8, &[40, 2, 40, 7, 22, 3, 22, 17, 1]),
            (9, &[30, 1, 8]),
            (13, &[99, 7, 3, 7]),
        ],
    );
    let (_, t) = observe("ordered_ingest_empty", &r, || buf.flush(&mut g));
    assert!(
        t.iter()
            .all(|t| t.reads.is_empty() && t.writes.is_empty() && t.grows.is_empty())
    );
    let replacement = [(16, 77), (8, 9), (16, 7), (8, 2), (16, 77)];
    let (built, _) = observe("ordered_bulk_replacement", &r, || {
        g.build_pairs(&replacement)
    });
    built.unwrap();
    assert_eq!(g.order(), AdjacencyOrder::Insertion);
    verify(&g, &[(8, &[9, 2]), (16, &[77, 7, 77])]);
    g.build_pairs(&[]).unwrap();
    verify(&g, &[]);
    assert_eq!(g.order(), AdjacencyOrder::Insertion);
}

fn images(r: &[Probe; 8]) -> [Vec<u8>; 8] {
    std::array::from_fn(|i| r[i].bytes())
}
fn no_io(t: &[Trace; 8]) {
    assert!(
        t.iter()
            .all(|t| t.reads.is_empty() && t.writes.is_empty() && t.grows.is_empty())
    );
}

#[test]
fn order_changes_use_canonical_empty_rows_and_retire_empty_extents() {
    let r = regions();
    let mut g = Core::<Edge>::new_with_segments(bundle(&r), 64, 8).unwrap();
    g.insert_edge(SRC, 4);
    g.add_edges(-1); // Public raw count seam: a stale aggregate must not admit a change.
    let before = images(&r);
    let (changed, t) = observe("switch_reject_stale_global", &r, || {
        g.set_order(AdjacencyOrder::Insertion)
    });
    assert_eq!(changed, Err(lara_graph::OrderChangeError));
    no_io(&t);
    assert_eq!(images(&r), before);
    assert_eq!(g.order(), AdjacencyOrder::Unordered);
    assert_eq!(g.out_degree(SRC), 1);
    assert_eq!(g.num_edges(), 0);
    g.add_edges(1);
    let (same, t) = observe("switch_same_unordered_live", &r, || {
        g.set_order(AdjacencyOrder::Unordered)
    });
    same.unwrap();
    no_io(&t);
    assert!(g.remove_edge(SRC, 4));
    g.set_order(AdjacencyOrder::Insertion).unwrap();
    g.insert_edge(63, 3);
    let (changed, t) = observe("switch_reject_last_vertex", &r, || {
        g.set_order(AdjacencyOrder::Unordered)
    });
    assert!(changed.is_err());
    no_io(&t);
    let (same, t) = observe("switch_same_insertion_live", &r, || {
        g.set_order(AdjacencyOrder::Insertion)
    });
    same.unwrap();
    no_io(&t);
    verify(&g, &[(63, &[3])]);
    assert!(g.remove_edge(63, 3));
    g.set_order(AdjacencyOrder::Unordered).unwrap();
    verify(&g, &[]);

    let r = regions();
    let mut g = Core::<Edge>::new_with_segments(bundle(&r), 64, 8).unwrap();
    g.set_promote_threshold(None);
    g.set_order(AdjacencyOrder::Insertion).unwrap();
    let pairs: Vec<_> = (0..16)
        .map(|d| (SRC, d))
        .chain((0..128).map(|d| (10, d)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    for d in 16..67 {
        g.insert_edge(SRC, d);
    }
    for d in 1000..1169 {
        g.insert_edge(SRC, d);
    }
    // Logs alone keep the leaf above LOW_0 while all slab records are deleted.
    // Deleting the remaining logs unlinks without a low-density slab fold.
    for d in 0..67 {
        assert!(g.remove_edge(SRC, d));
    }
    for d in 0..128 {
        assert!(g.remove_edge(10, d));
    }
    for d in 1000..1169 {
        assert!(g.remove_edge(SRC, d));
    }
    verify(&g, &[]);
    assert_eq!(g.get_row_pub(SRC).stored_degree(), 67);
    assert_eq!(g.get_row_pub(10).stored_degree(), 128);
    assert_eq!(g.log_allocated_chunks(), 1);
    let pooled = g.log_pool_depth();
    let before = images(&r);
    let (changed, t) = observe("switch_retires_empty_markers_and_logs", &r, || {
        g.set_order(AdjacencyOrder::Unordered)
    });
    changed.unwrap();
    no_io(&t);
    assert_eq!(images(&r), before, "retirement is not erasure");
    assert_eq!(g.get_row_pub(SRC).stored_degree(), 0);
    assert_eq!(g.get_row_pub(10).stored_degree(), 0);
    assert_eq!(g.log_allocated_chunks(), 0);
    assert_eq!(g.log_pool_depth(), pooled + 1);
    verify(&g, &[]);
    g.insert_edge(SRC, 7);
    verify(&g, &[(SRC, &[7])]);
    assert!(g.remove_edge(SRC, 7));
    g.set_order(AdjacencyOrder::Insertion).unwrap();
    g.insert_sorted_run(SRC, &[9, 2, 9]);
    verify(&g, &[(SRC, &[9, 2, 9])]);
}

fn rejected<E: CsrEdge>(
    tag: &str,
    g: &mut Core<E>,
    r: &[Probe; 8],
    message: &str,
    op: impl FnOnce(&mut Core<E>),
) {
    let before = images(r);
    let rows: Vec<_> = (0..g.num_nodes() as u32)
        .map(|v| g.get_row_pub(v))
        .collect();
    let count = g.num_edges();
    let order = g.order();
    let (result, t) = observe(tag, r, || catch_unwind(AssertUnwindSafe(|| op(g))));
    let error = result.expect_err("input must be rejected before writes");
    let text = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic");
    assert!(text.contains(message), "{text}");
    assert!(t.iter().all(|t| t.writes.is_empty() && t.grows.is_empty()));
    assert_eq!(images(r), before);
    assert_eq!((g.num_edges(), g.order()), (count, order));
    assert_eq!(
        (0..g.num_nodes() as u32)
            .map(|v| g.get_row_pub(v))
            .collect::<Vec<_>>(),
        rows
    );
    g.verify_invariants();
}
#[derive(Clone, Copy, Default)]
struct Flag(u32);
impl FixedWidth for Flag {
    const BYTES: usize = 4;
    fn read_from(b: &[u8]) -> Self {
        Self(u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn write_to(&self, b: &mut [u8]) {
        b.copy_from_slice(&self.0.to_le_bytes());
    }
}
impl CsrEdge for Flag {
    fn neighbor(&self) -> u32 {
        self.0 & 0x7fff_ffff
    }
    fn with_neighbor(self, n: u32) -> Self {
        Self(n & 0x7fff_ffff)
    }
}
impl CsrEdgeTombstone for Flag {
    fn tombstone_edge() -> Self {
        Self(0x8000_0007)
    }
    fn is_tombstone(&self) -> bool {
        self.0 == 0x8000_0007
    }
}
#[derive(Clone, Copy, Default)]
struct ReconstructionMarker(Record<12>);
impl FixedWidth for ReconstructionMarker {
    const BYTES: usize = 12;
    fn read_from(b: &[u8]) -> Self {
        Self(Record::read_from(b))
    }
    fn write_to(&self, b: &mut [u8]) {
        self.0.write_to(b);
    }
}
impl CsrEdge for ReconstructionMarker {
    fn neighbor(&self) -> u32 {
        self.0.dst
    }
    fn with_neighbor(mut self, n: u32) -> Self {
        self.0.dst = n;
        self
    }
}
impl CsrEdgeTombstone for ReconstructionMarker {
    fn tombstone_edge() -> Self {
        Self(Record { dst: 7, tag: 0 })
    }
    fn is_tombstone(&self) -> bool {
        self.0.dst == 7 && self.0.tag == 0
    }
}

#[test]
fn typed_markers_are_rejected_without_reserving_raw_collisions() {
    for mode in ["tiny", "slab", "log", "tree"] {
        let r = regions();
        let mut g = Core::<Flag>::new_with_segments(bundle(&r), 256, 8).unwrap();
        g.set_promote_threshold(Some(64));
        g.set_order(AdjacencyOrder::Insertion).unwrap();
        let n = match mode {
            "tiny" => 3,
            "slab" => 32,
            "log" => 10,
            _ => 64,
        };
        if mode == "slab" || mode == "tree" {
            g.build_pairs(&(0..n - 1).map(|d| (SRC, d)).collect::<Vec<_>>())
                .unwrap();
            g.insert_edge(SRC, n - 1);
        } else {
            for d in 0..n {
                g.insert_edge(SRC, d);
            }
        }
        assert_eq!(g.row_is_tiny(SRC), mode == "tiny");
        assert_eq!(
            g.get_row_pub(SRC).tree_bucket() != NO_BUCKET,
            mode == "tree"
        );
        if mode == "log" {
            assert!(g.get_row_pub(SRC).log_head() >= 0);
        }
        rejected(
            &format!("reject_marker_{mode}"),
            &mut g,
            &r,
            "tombstone",
            |g| g.insert_edge_record(SRC, Flag::tombstone_edge()),
        );
        // A marker's neighbor and wire word are not universally reserved IDs.
        // The existing builder normalizes this raw input to the LIVE Flag(7).
        observe(&format!("live_marker_collision_{mode}"), &r, || {
            g.insert_sorted_run(SRC, &[7, 0x8000_0007])
        });
        let want: Vec<_> = (0..n).chain([7, 7]).collect();
        verify(&g, &[(SRC, &want)]);
        g.rebalance_leaf(1);
        verify(&g, &[(SRC, &want)]);
    }
    let r = regions();
    let mut g = Core::<Edge>::new_with_segments(bundle(&r), 64, 8).unwrap();
    g.set_order(AdjacencyOrder::Insertion).unwrap();
    g.insert_edge(SRC, 3);
    rejected("reject_scalar_marker", &mut g, &r, "tombstone", |g| {
        g.insert_edge(SRC, u32::MAX)
    });
    rejected("reject_run_marker", &mut g, &r, "tombstone", |g| {
        g.insert_sorted_run(SRC, &[2, 1, u32::MAX])
    });
    rejected("reject_bulk_marker", &mut g, &r, "tombstone", |g| {
        g.build_pairs(&[(9, 2), (SRC, u32::MAX)]).unwrap();
    });
    let mut buf = IngestBuffer::new(0, 2);
    buf.push(9, 3);
    buf.push(SRC, 2);
    buf.push(16, u32::MAX);
    rejected("reject_buffer_marker", &mut g, &r, "tombstone", |g| {
        buf.flush(g)
    });
    assert_eq!(buf.len(), 3);
    let mut buf = IngestBuffer::new(0, 1);
    buf.push(SRC, 2);
    buf.push(64, 3);
    rejected(
        "reject_buffer_source",
        &mut g,
        &r,
        "src out of range",
        |g| buf.flush(g),
    );
    assert_eq!(buf.len(), 2);
    verify(&g, &[(SRC, &[3])]);

    let r = regions();
    let mut g = Core::<ReconstructionMarker>::new_with_segments(bundle(&r), 64, 8).unwrap();
    g.set_order(AdjacencyOrder::Insertion).unwrap();
    g.build_pairs(&[(SRC, 1), (9, 2)]).unwrap();
    let record = ReconstructionMarker(Record { dst: 7, tag: 9 });
    assert!(!record.is_tombstone());
    assert!(
        ReconstructionMarker::default()
            .with_neighbor(record.neighbor())
            .is_tombstone()
    );
    assert!(g.row_has_slab_room(SRC));
    rejected(
        "reject_marker_reconstruction",
        &mut g,
        &r,
        "tombstone",
        |g| g.insert_edge_record(SRC, record),
    );
    verify(&g, &[(SRC, &[1]), (9, &[2])]);
}

#[test]
fn depth_rejection_and_explicit_raw_domains_remain_separate() {
    let r = regions();
    let mut g = Core::<Edge>::new_with_segments(bundle(&r), 64, 8).unwrap();
    g.set_order(AdjacencyOrder::Insertion).unwrap();
    let pairs: Vec<_> = (0..15359).map(|k| (SRC, k % 64)).collect();
    g.build_pairs(&pairs).unwrap();
    g.insert_edge(SRC, 63); // Fifteen blocks fit in the initial backed page.
    assert_eq!(r[6].size(), 1);
    let (_, t) = observe("ordered_tree_grows_sixteenth_block", &r, || {
        g.insert_edge(SRC, 31)
    });
    assert_eq!(t[6].grows, [1]);
    assert_eq!(r[6].size(), 2);
    let want: Vec<_> = pairs.iter().map(|p| p.1).chain([63, 31]).collect();
    verify(&g, &[(SRC, &want)]);
    let limit = lara_graph::B * lara_graph::R_MAX;
    g.insert_sorted_run(SRC, &vec![7; limit - want.len()]);
    assert_eq!(g.out_degree(SRC), limit as u64);
    rejected(
        "ordered_depth_scalar",
        &mut g,
        &r,
        "tree-csr: depth-1 capacity exceeded",
        |g| g.insert_edge(SRC, 4),
    );
    rejected(
        "ordered_depth_run",
        &mut g,
        &r,
        "tree-csr: depth-1 capacity exceeded",
        |g| g.insert_sorted_run(SRC, &[4, 3]),
    );
    g.set_promote_threshold(None);
    rejected(
        "ordered_depth_retained_tree",
        &mut g,
        &r,
        "tree-csr: depth-1 capacity exceeded",
        |g| g.insert_edge(SRC, 4),
    );
    rejected(
        "ordered_source_scalar",
        &mut g,
        &r,
        "src out of range",
        |g| g.insert_edge(64, 1),
    );
    rejected("ordered_source_run", &mut g, &r, "src out of range", |g| {
        g.insert_sorted_run(64, &[1])
    });
    rejected("ordered_source_bulk", &mut g, &r, "src out of range", |g| {
        g.build_pairs(&[(SRC, 1), (64, 2)]).unwrap();
    });
    // Replacement retires actual trees while retaining the selected policy.
    g.build_pairs(&[(9, 7), (SRC, 3), (9, 2)]).unwrap();
    assert_eq!(g.tree_state_len(), 0);
    verify(&g, &[(SRC, &[3]), (9, &[7, 2])]);
    assert_eq!(g.order(), AdjacencyOrder::Insertion);

    let r = regions();
    let mut g = Core::<Edge>::new_with_segments(bundle(&r), 64, 8).unwrap();
    g.set_order(AdjacencyOrder::Insertion).unwrap();
    let idx = g.tree_state_push(Vec::new(), 0) as usize;
    for d in [0, u32::MAX, 7, 7] {
        g.tree_state_insert(idx, d);
    }
    let (_, t) = observe("ordered_explicit_raw_shift", &r, || {
        g.tree_state_remove_shift(idx, 0)
    });
    assert_eq!(
        t[6].writes,
        [
            (80, u32::MAX.to_le_bytes().to_vec()),
            (84, 7u32.to_le_bytes().to_vec()),
            (88, 7u32.to_le_bytes().to_vec())
        ]
    );
    let mut raw = Vec::new();
    g.for_each_in_tree(idx, |d| raw.push(d));
    assert_eq!(raw, [u32::MAX, 7, 7]);
    g.tree_state_release(idx); // Detached raw owners are not graph adjacency.
    verify(&g, &[]);
}

#[derive(Clone, Copy)]
enum Op {
    Insert(u32, u32),
    Remove(u32, u32),
    Run(u32, [u32; 3]),
    Push(u32, u32),
    Flush,
    Fold(u64),
}
#[test]
fn mixed_lifecycle_matches_independent_ordered_sequences() {
    use std::collections::BTreeMap;
    for cap in [0, 170] {
        let r = regions();
        let mut g = Core::<Edge>::new_with_segments_and_max_log(bundle(&r), 128, 8, cap).unwrap();
        g.set_promote_threshold(Some(64));
        g.set_order(AdjacencyOrder::Insertion).unwrap();
        let sources = [8, 9, 16, 17];
        let mut model: BTreeMap<u32, Vec<u32>> =
            sources.into_iter().map(|s| (s, vec![7, 4, s, 7])).collect();
        observe(&format!("mixed_tiny_cap{cap}"), &r, || {
            for (&s, row) in &model {
                for &d in row {
                    lara_graph::DynamicCsr::insert_edge(&mut g, s, d);
                }
            }
        });
        assert_eq!(g.tiny_row_count(), 4);
        for &s in &sources {
            let run: Vec<_> = (0..96).map(|k| (k * 37 + s) % 64).collect();
            g.insert_sorted_run(s, &run);
            model.get_mut(&s).unwrap().extend(run);
        }
        for &s in &sources {
            g.rebalance_leaf(u64::from(s / 8));
            g.insert_edge(s, 31);
            model.get_mut(&s).unwrap().push(31);
        }
        assert_eq!(g.tree_state_len(), 4);
        for &s in &sources {
            while model[&s].len() > 32 {
                let d = model.get_mut(&s).unwrap().remove(0);
                assert!(lara_graph::RemovableCsr::remove_edge(&mut g, s, d));
            }
        }
        assert_eq!(g.tree_state_len(), 0);
        verify(
            &g,
            &model
                .iter()
                .map(|(&s, ds)| (s, ds.as_slice()))
                .collect::<Vec<_>>(),
        );
        let mut rng = 0xdecafbad12345678u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 32) as u32
        };
        let mut ops = Vec::new();
        for k in 0..600 {
            let s = sources[(next() % 4) as usize];
            let d = next() % 64;
            ops.push(match next() % 10 {
                0..=3 => Op::Insert(s, d),
                4..=6 => Op::Remove(s, d),
                7 => Op::Run(s, [d, next() % 64, d]),
                _ => Op::Push(s, d),
            });
            if k % 17 == 0 {
                ops.push(Op::Flush);
            }
            if k % 61 == 0 {
                ops.push(Op::Fold(u64::from(s / 8)));
            }
        }
        ops.push(Op::Flush);
        let mut buf = IngestBuffer::new(0, 3);
        let mut pending = Vec::new();
        for (batch, ops) in ops.chunks(32).enumerate() {
            let (results, _) = observe(&format!("mixed_cap{cap}_batch{batch}"), &r, || {
                let mut results = Vec::new();
                for &op in ops {
                    match op {
                        Op::Insert(s, d) => lara_graph::DynamicCsr::insert_edge(&mut g, s, d),
                        Op::Remove(s, d) => {
                            results.push(lara_graph::RemovableCsr::remove_edge(&mut g, s, d))
                        }
                        Op::Run(s, ds) => g.insert_sorted_run(s, &ds),
                        Op::Push(s, d) => buf.push(s, d),
                        Op::Flush => buf.flush(&mut g),
                        Op::Fold(leaf) => g.rebalance_leaf(leaf),
                    }
                }
                results
            });
            // Independent model and all assertions run only after freezing.
            let mut expected = Vec::new();
            for &op in ops {
                match op {
                    Op::Insert(s, d) => model.get_mut(&s).unwrap().push(d),
                    Op::Remove(s, d) => {
                        let row = model.get_mut(&s).unwrap();
                        let found = row.iter().position(|&x| x == d);
                        if let Some(k) = found {
                            row.remove(k);
                        }
                        expected.push(found.is_some());
                    }
                    Op::Run(s, ds) => model.get_mut(&s).unwrap().extend(ds),
                    Op::Push(s, d) => pending.push((s, d)),
                    Op::Flush => {
                        for (s, d) in pending.drain(..) {
                            model.get_mut(&s).unwrap().push(d);
                        }
                    }
                    Op::Fold(_) => {}
                }
            }
            assert_eq!(results, expected);
            assert_eq!(buf.len(), pending.len());
            verify(
                &g,
                &model
                    .iter()
                    .map(|(&s, ds)| (s, ds.as_slice()))
                    .collect::<Vec<_>>(),
            );
        }
        assert!(pending.is_empty());
        assert!(buf.is_empty());
    }
}
