//! Dense slab reuse/query contracts; the historical hole-search tests remain archived.
//! Counters are sampled before multiset/invariant oracles.
//! These are operation counts, not timings. Every region has independent backing.
#![cfg(feature = "std")]

use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::labeled::{LabelId, LabeledLaraGraph};
use lara_graph::memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
use lara_graph::traits::{CsrEdge, CsrEdgeTombstone, CsrVertex, FixedWidth};
use std::cell::Cell;
use std::rc::Rc;

const SRC: u32 = 8;
const SIBLING: u32 = 9;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Ops {
    reads: u64,
    read_bytes: u64,
    writes: u64,
    write_bytes: u64,
    first_byte: u64,
    end_byte: u64,
}

#[derive(Clone, Default)]
struct Counted {
    memory: VecMemory,
    ops: Rc<Cell<Ops>>,
}

impl Memory for Counted {
    fn read(&self, offset: u64, buf: &mut [u8]) {
        assert!(offset + buf.len() as u64 <= self.memory.size() * PAGE_SIZE);
        let mut c = self.ops.get();
        c.first_byte = if c.reads == 0 {
            offset
        } else {
            c.first_byte.min(offset)
        };
        c.end_byte = c.end_byte.max(offset + buf.len() as u64);
        c.reads += 1;
        c.read_bytes += buf.len() as u64;
        self.ops.set(c);
        self.memory.read(offset, buf);
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        assert!(offset + buf.len() as u64 <= self.memory.size() * PAGE_SIZE);
        let mut c = self.ops.get();
        c.writes += 1;
        c.write_bytes += buf.len() as u64;
        self.ops.set(c);
        self.memory.write(offset, buf);
    }

    fn size(&self) -> u64 {
        self.memory.size()
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        self.memory.grow(pages)
    }
}

type Graph<E = Edge> = LaraGraph<Vertex, E, Counted>;

fn bundle(r: &[Counted; 8]) -> LaraMemories<Counted> {
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

fn observe<T>(tag: &str, r: &[Counted; 8], f: impl FnOnce() -> T) -> (T, [Ops; 8]) {
    for region in r {
        region.ops.set(Ops::default());
    }
    let value = f();
    let snapshot = std::array::from_fn(|i| r[i].ops.get());
    for (region, i) in [("edges", 1), ("logs", 2)] {
        let c = snapshot[i];
        println!(
            "OBS,{tag},{region},{},{},{},{},{},{}",
            c.reads, c.read_bytes, c.writes, c.write_bytes, c.first_byte, c.end_byte,
        );
    }
    (value, snapshot)
}

fn full_row<E: CsrEdgeTombstone>(initial: u32) -> (Graph<E>, [Counted; 8], u32) {
    let regions = std::array::from_fn(|_| Counted::default());
    let mut g = Graph::<E>::new_with_segments(bundle(&regions), 512, 8).unwrap();
    let pairs: Vec<_> = (0..initial)
        .map(|n| (SRC, n))
        .chain((0..128).map(|n| (SIBLING, n)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    g.set_promote_threshold(None);
    let width = g.row_allocated(SRC) as u32;
    assert!(initial <= width && width < 500);
    for dst in initial..width {
        g.insert_edge(SRC, dst);
    }
    assert!(!g.row_has_slab_room(SRC));
    (g, regions, width)
}

fn verify<E: CsrEdgeTombstone>(g: &Graph<E>, want: &[u32]) {
    g.verify_invariants();
    let mut got = g.out_neighbors_vec(SRC);
    got.sort_unstable();
    let mut want = want.to_vec();
    want.sort_unstable();
    assert_eq!(got, want);
    assert_eq!(g.out_neighbors_vec(SIBLING), (0..128).collect::<Vec<_>>());
    assert_eq!(g.num_edges(), want.len() as u64 + 128);
    for v in 0..g.num_nodes() as u32 {
        if v != SRC && v != SIBLING {
            assert!(g.out_neighbors_vec(v).is_empty());
        }
    }
}

#[test]
fn clean_full_rows_skip_tombstone_search() {
    let (mut g, r, width) = full_row::<Edge>(16);
    assert_eq!(width, 67);
    let rebalances = g.rebalance_count();
    let (_, c) = observe("clean67_append64", &r, || {
        for _ in 0..64 {
            g.insert_edge(SRC, 500);
        }
    });
    assert_eq!((c[1].reads, c[1].read_bytes, c[1].writes), (0, 0, 0));
    assert_eq!(g.rebalance_count(), rebalances);
    let want: Vec<_> = (0..width).chain(std::iter::repeat_n(500, 64)).collect();
    verify(&g, &want);
}

#[derive(Clone, Copy, Default)]
struct Encoded<const N: usize> {
    dst: u32,
    tag: u32,
}

impl<const N: usize> FixedWidth for Encoded<N> {
    const BYTES: usize = N;
    fn read_from(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), N);
        let dst = u32::from_le_bytes(bytes[..4].try_into().unwrap()) ^ 0x8000_0000;
        let tag = if N >= 8 {
            assert!(bytes[8..].iter().all(|&b| b == 0x5a));
            u32::from_le_bytes(bytes[4..8].try_into().unwrap())
        } else {
            0
        };
        Self { dst, tag }
    }
    fn write_to(&self, bytes: &mut [u8]) {
        bytes.fill(0x5a);
        bytes[..4].copy_from_slice(&(self.dst ^ 0x8000_0000).to_le_bytes());
        if N >= 8 {
            bytes[4..8].copy_from_slice(&self.tag.to_le_bytes());
        }
    }
}

impl<const N: usize> CsrEdge for Encoded<N> {
    fn neighbor(&self) -> u32 {
        self.dst
    }
    fn with_neighbor(self, dst: u32) -> Self {
        Self { dst, ..self }
    }
}

impl<const N: usize> CsrEdgeTombstone for Encoded<N> {
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

fn check_codec<const N: usize>() {
    let chunk_slots = (512 / N).max(1) as u32;
    let mut holes = vec![
        0,
        1,
        chunk_slots - 1,
        chunk_slots,
        chunk_slots + 1,
        2 * chunk_slots - 1,
        2 * chunk_slots,
        324,
    ];
    holes.sort_unstable_by_key(|&hole| (hole != chunk_slots, hole));
    holes.dedup();
    for hole in holes {
        let (mut g, r, width) = full_row::<Encoded<N>>(80);
        assert_eq!(width, 325);
        let row = g.get_row_pub(SRC);
        let base = row.base_slot_start();
        assert!(base > 0);
        assert!(g.remove_edge(SRC, hole));
        let (_, c) = observe(&format!("codec{N}_hole{hole}"), &r, || {
            g.insert_edge_record(SRC, Encoded { dst: 501, tag: 42 });
        });
        let want: Vec<_> = (0..width).filter(|&n| n != hole).chain([501]).collect();
        verify(&g, &want);
        assert_eq!(g.get_row_pub(SRC).stored_degree(), width);
        assert_eq!(c[2].writes, 0);
        assert_eq!(
            (c[1].reads, c[1].read_bytes, c[1].first_byte, c[1].end_byte),
            (0, 0, 0, 0)
        );
        assert_eq!((c[1].writes, c[1].write_bytes), (1, N as u64));
        let mut records = Vec::new();
        g.for_each_record_in_row(base, width, |e| records.push((e.dst, e.tag)));
        let mut expected: Vec<_> = (0..width).map(|n| (n, 0)).collect();
        expected.swap_remove(hole as usize);
        expected.push((501, if N >= 8 { 42 } else { 0 }));
        assert_eq!(records, expected);
        let (room, c) = observe(&format!("codec{N}_full_after{hole}"), &r, || {
            g.row_has_slab_room(SRC)
        });
        assert!(!room);
        assert_eq!(
            (c[1].reads, c[1].read_bytes, c[1].first_byte, c[1].end_byte),
            (0, 0, 0, 0)
        );
    }
}

#[test]
fn dense_tail_reuse_preserves_codecs_payloads_and_row_bounds() {
    check_codec::<4>();
    check_codec::<12>();
    check_codec::<516>();
}

#[test]
fn log_backed_rows_preserve_dense_tail_room_and_multisets() {
    let (mut g, r, width) = full_row::<Edge>(16);
    for _ in 0..64 {
        g.insert_edge(SRC, 500);
    }
    assert!(g.remove_edge(SRC, 0));
    assert!(g.get_row_pub(SRC).degree() >= width);
    let (room, query) = observe("logged_hole_exact_query", &r, || g.row_has_slab_room(SRC));
    assert!(room);
    let (_, c) = observe("logged_hole_deferred_insert", &r, || {
        g.insert_edge(SRC, 501)
    });
    assert_eq!((c[1].reads, c[1].writes, c[2].writes), (0, 1, 0));
    assert_eq!((query[1].reads, query[1].read_bytes), (0, 0));
    let mut want: Vec<_> = (1..width)
        .chain(std::iter::repeat_n(500, 64))
        .chain([501])
        .collect();
    verify(&g, &want);
    for _ in 0..64 {
        assert!(g.remove_edge(SRC, 500));
    }
    assert!(g.remove_edge(SRC, 501));
    assert_eq!(g.get_row_pub(SRC).log_head(), -1);
    let (_, c) = observe("unlinked_logs_allow_reuse", &r, || g.insert_edge(SRC, 502));
    assert_eq!((c[1].reads, c[1].writes, c[2].writes), (0, 1, 0));
    want = (1..width).chain([502]).collect();
    verify(&g, &want);
    g.rebalance_leaf(1);
    verify(&g, &want);

    // Two deletions free two tail slots, independently of the live log.
    let (mut g, r, width) = full_row::<Edge>(16);
    g.insert_edge(SRC, 500);
    assert!(g.remove_edge(SRC, 0));
    assert!(g.remove_edge(SRC, 1));
    assert!(g.get_row_pub(SRC).degree() < width);
    assert!(g.get_row_pub(SRC).log_head() >= 0);
    let (_, c) = observe("more_holes_than_logs_reuse", &r, || g.insert_edge(SRC, 501));
    assert_eq!((c[1].reads, c[1].writes, c[2].writes), (0, 1, 0));
    assert_eq!(g.get_row_pub(SRC).stored_degree(), width - 1);
    assert_eq!(
        g.read_edge_pub(g.get_row_pub(SRC).base_slot_start() + 1)
            .neighbor(),
        width - 2
    );
    let (_, c) = observe("equal_holes_and_logs_defer", &r, || g.insert_edge(SRC, 502));
    assert_eq!((c[1].reads, c[1].writes, c[2].writes), (0, 1, 0));
    let want: Vec<_> = (2..width).chain([500, 501, 502]).collect();
    verify(&g, &want);
    g.rebalance_leaf(1);
    g.insert_edge(SRC, SRC);
    let mut want = want;
    want.push(SRC);
    verify(&g, &want);
}

fn full_label() -> (LabeledLaraGraph<Counted>, [Counted; 8]) {
    let r = std::array::from_fn(|_| Counted::default());
    let label = LabelId::new(0);
    let other = LabelId::new(1);
    let pairs: Vec<_> = (0..160)
        .map(|n| (SRC, n, label))
        .chain((0..128).map(|n| (SRC, n, other)))
        .collect();
    let mut g = LabeledLaraGraph::with_edge_list(bundle(&r), &pairs, 512)
        .unwrap()
        .with_promote_threshold(None);
    for dst in 160..320 {
        g.insert_edge(SRC, dst, label);
    }
    assert_eq!(g.bucket_room(SRC, label), 0);
    assert_eq!(g.num_edges(), 448);
    (g, r)
}

fn verify_labeled(g: &LabeledLaraGraph<Counted>, want: &[u32]) {
    g.verify_invariants();
    let mut got: Vec<_> = g
        .out_labeled_vec(SRC)
        .into_iter()
        .map(|(dst, label)| (dst, label.as_u16()))
        .collect();
    got.sort_unstable();
    let mut expected: Vec<_> = want
        .iter()
        .map(|&n| (n, 0))
        .chain((0..128).map(|n| (n, 1)))
        .collect();
    expected.sort_unstable();
    assert_eq!(got, expected);
    assert_eq!(g.num_edges(), expected.len() as u64);
    for v in 0..g.num_nodes() as u32 {
        if v != SRC {
            assert!(g.out_labeled_vec(v).is_empty());
        }
    }
}

#[test]
fn labeled_tail_reuse_is_read_free_and_label_local() {
    let (mut g, r) = full_label();
    let label = LabelId::new(0);
    assert!(g.remove_edge(SRC, 129, label));
    let (_, c) = observe("labeled_hole129", &r, || g.insert_edge(SRC, 500, label));
    assert_eq!((c[1].reads, c[1].read_bytes, c[1].writes), (0, 0, 1));
    assert_eq!(c[2].writes, 0);
    let want: Vec<_> = (0..320).filter(|&n| n != 129).chain([500]).collect();
    verify_labeled(&g, &want);
}

#[test]
fn labeled_dense_overflow_skips_search_before_rebalance() {
    let (mut g, r) = full_label();
    let (_, c) = observe("labeled_dense_overflow", &r, || {
        g.insert_edge(SRC, 500, LabelId::new(0))
    });
    // Rebalance still collects the 320 + 128 live records per-slot.
    // Only the preceding 320-slot speculative hole search is removed.
    assert_eq!((c[1].reads, c[1].read_bytes), (448, 1792));
    assert_eq!(c[2].writes, 0);
    let want: Vec<_> = (0..320).chain([500]).collect();
    verify_labeled(&g, &want);
}

#[test]
fn empty_and_tiny_room_queries_do_not_scan_edges() {
    let r = std::array::from_fn(|_| Counted::default());
    let mut g = Graph::<Edge>::new(bundle(&r), 64).unwrap();
    for tag in ["empty", "tiny_one", "tiny_empty"] {
        if tag == "tiny_one" {
            g.insert_edge(SRC, SRC);
            assert_eq!(g.tiny_row_count(), 1);
        } else if tag == "tiny_empty" {
            assert!(g.remove_edge(SRC, SRC));
        }
        let (room, c) = observe(tag, &r, || g.row_has_slab_room(SRC));
        assert!(!room);
        assert_eq!(c[1].reads, 0);
        assert_eq!(g.num_edges(), u64::from(tag == "tiny_one"));
        g.verify_invariants();
    }
}
