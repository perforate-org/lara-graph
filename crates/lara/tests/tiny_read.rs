//! Tiny-prefix I/O contracts. Counters are frozen before result/invariant checks.
#![cfg(feature = "std")]

use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
use lara_graph::traits::{CsrEdge, CsrEdgeTombstone, CsrVertex, FixedWidth};
use std::cell::Cell;
use std::rc::Rc;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Ops {
    reads: u64,
    bytes: u64,
    writes: u64,
    write_bytes: u64,
    grows: u64,
    first: u64,
    end: u64,
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
        c.first = if c.reads == 0 {
            offset
        } else {
            c.first.min(offset)
        };
        c.end = c.end.max(offset + buf.len() as u64);
        c.reads += 1;
        c.bytes += buf.len() as u64;
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
        let mut c = self.ops.get();
        c.grows += 1;
        self.ops.set(c);
        self.memory.grow(pages)
    }
}

type Graph<E = Edge> = LaraGraph<Vertex, E, Counted>;

fn fresh<E: CsrEdgeTombstone>(cap: usize) -> (Graph<E>, [Counted; 8]) {
    let r: [Counted; 8] = std::array::from_fn(|_| Counted::default());
    let memories = LaraMemories {
        vertices: r[0].clone(),
        edges: r[1].clone(),
        logs: r[2].clone(),
        counts: r[3].clone(),
        spans: r[4].clone(),
        free_spans: r[5].clone(),
        blocks: r[6].clone(),
        buckets: r[7].clone(),
    };
    let g = Graph::new_with_segments_and_max_log(memories, 64, 8, cap).unwrap();
    (g, r)
}

fn observe<T>(tag: &str, r: &[Counted; 8], f: impl FnOnce() -> T) -> (T, [Ops; 8]) {
    for m in r {
        m.ops.set(Ops::default());
    }
    let value = f();
    let snapshot = std::array::from_fn(|i| r[i].ops.get());
    for (name, c) in [
        "vertices", "edges", "logs", "counts", "spans", "free", "blocks", "buckets",
    ]
    .iter()
    .zip(&snapshot)
    {
        println!(
            "OBS,{tag},{name},{},{},{},{},{},{},{}",
            c.reads, c.bytes, c.writes, c.write_bytes, c.grows, c.first, c.end
        );
    }
    (value, snapshot)
}

fn assert_prefix(c: Ops, off: u64, n: usize) {
    let expected = if n == 0 {
        (0, 0, 0, 0)
    } else {
        (1, n as u64 * 4, off, off + n as u64 * 4)
    };
    assert_eq!((c.reads, c.bytes, c.first, c.end), expected);
}

fn check_scan<E: CsrEdgeTombstone>(
    g: &Graph<E>,
    r: &[Counted; 8],
    tag: &str,
    src: u32,
    want: &[u32],
) {
    assert!(g.row_is_tiny(src));
    let off = g.get_row_pub(src).log_head() as u64;
    assert!(off > 0);
    let (got, c) = observe(tag, r, || g.out_neighbors_vec(src));
    assert_eq!(got, want);
    assert_eq!(g.out_degree(src), want.len() as u64);
    g.verify_invariants();
    assert_prefix(c[2], off, want.len());
    assert_eq!((c[2].writes, c[2].grows), (0, 0));
    assert!(
        c.iter()
            .enumerate()
            .all(|(i, c)| i == 2 || *c == Ops::default())
    );
}

#[test]
fn tiny_scans_read_only_live_prefix() {
    let values = [0, 5, 5, 7, 42, 63, 1, 2];
    for cap in [170, 0] {
        let (mut g, r) = fresh::<Edge>(cap);
        for (i, &dst) in values.iter().enumerate() {
            g.insert_edge(0, dst);
            check_scan(
                &g,
                &r,
                &format!("scan_cap{cap}_n{}", i + 1),
                0,
                &values[..=i],
            );
            assert_eq!(g.num_edges(), (i + 1) as u64);
            assert_eq!(g.tiny_memory_bytes(), 32);
        }
        let (all, c) = observe(&format!("scan_all_cap{cap}"), &r, || {
            let mut all = Vec::new();
            g.scan_all(|dst| all.push(dst));
            all
        });
        assert_eq!(all, values);
        assert_prefix(c[2], g.get_row_pub(0).log_head() as u64, 8);
        for dst in values {
            assert!(g.remove_edge(0, dst));
        }
        check_scan(&g, &r, &format!("empty_cap{cap}"), 0, &[]);
        let (removed, c) = observe(&format!("empty_delete_cap{cap}"), &r, || {
            g.remove_edge(0, 0)
        });
        assert!(!removed);
        assert!(c.iter().all(|c| *c == Ops::default()));
        g.insert_edge(0, 7);
        check_scan(&g, &r, &format!("regrow_cap{cap}"), 0, &[7]);
        assert_eq!(g.tiny_memory_bytes(), 32);
        assert_eq!(g.tiny_pool_depth(), 0);
    }
}

#[test]
fn tiny_deletion_reads_prefix_and_replaces_from_tail() {
    let (mut g, r) = fresh::<Edge>(170);
    let mut want = vec![0, 5, 5, 7];
    g.insert_sorted_run(0, &want);
    let off = g.get_row_pub(0).log_head() as u64;
    let steps = [
        (5, true, 1),
        (63, false, 0),
        (5, true, 0),
        (7, true, 0),
        (0, true, 0),
        (0, false, 0),
    ];
    for (i, (dst, removed, writes)) in steps.into_iter().enumerate() {
        let n = want.len();
        let (got, c) = observe(&format!("delete_step{i}"), &r, || g.remove_edge(0, dst));
        assert_eq!(got, removed);
        if removed {
            want.swap_remove(want.iter().position(|&v| v == dst).unwrap());
        }
        assert_eq!(g.out_neighbors_vec(0), want);
        assert_eq!(g.num_edges(), want.len() as u64);
        assert!(g.row_is_tiny(0));
        g.verify_invariants();
        assert_prefix(c[2], off, n);
        assert_eq!(
            (c[2].writes, c[2].write_bytes, c[2].grows),
            (writes, writes * 4, 0)
        );
        assert!(
            c.iter()
                .enumerate()
                .all(|(i, c)| i == 2 || *c == Ops::default())
        );
    }
}

#[test]
fn tiny_fold_and_pool_reuse_preserve_partial_rows() {
    let (mut g, r) = fresh::<Edge>(170);
    g.insert_sorted_run(0, &[0, 5, 5]);
    g.insert_sorted_run(1, &[11, 12, 13, 14, 15, 16, 17, 18]);
    for dst in (11..=18).rev() {
        assert!(g.remove_edge(1, dst));
    }
    assert!(g.row_is_tiny(0) && g.row_is_tiny(1));
    let offsets = [
        g.get_row_pub(0).log_head() as u64,
        g.get_row_pub(1).log_head() as u64,
    ];
    let (_, c) = observe("fold_partial_and_empty", &r, || g.rebalance_leaf(0));
    assert_eq!(g.out_neighbors_vec(0), vec![0, 5, 5]);
    assert!(g.out_neighbors_vec(1).is_empty());
    assert_eq!(g.num_edges(), 3);
    assert_eq!(g.tiny_row_count(), 0);
    assert_eq!(g.tiny_pool_depth(), 2);
    assert_eq!(g.tiny_memory_bytes(), 64);
    g.verify_invariants();
    assert_prefix(c[2], offsets[0], 3);
    assert_eq!((c[2].writes, c[2].grows), (0, 0));

    g.insert_edge(63, 3); // an unplaced leaf can reuse a returned chunk
    assert!(offsets.contains(&(g.get_row_pub(63).log_head() as u64)));
    assert_eq!(g.tiny_memory_bytes(), 64);
    assert_eq!(g.tiny_pool_depth(), 1);
    check_scan(&g, &r, "pooled_singleton", 63, &[3]);
    assert_eq!(g.num_edges(), 4);
    let (_, c) = observe("bulk_replace", &r, || {
        g.build_pairs(&[(0, 11), (0, 11), (63, 3)]).unwrap()
    });
    assert_eq!(c[2].reads, 0); // rebuild retires chunks without reading them
    assert_eq!(g.out_neighbors_vec(0), vec![11, 11]);
    assert_eq!(g.out_neighbors_vec(63), vec![3]);
    assert_eq!(g.num_edges(), 3);
    assert_eq!(g.tiny_row_count(), 0);
    assert_eq!(g.tiny_pool_depth(), 2);
    assert_eq!(g.tiny_memory_bytes(), 64);
    g.verify_invariants();
}

#[derive(Clone, Copy, Default)]
struct XorEdge(u32);

impl FixedWidth for XorEdge {
    const BYTES: usize = 4;
    fn read_from(bytes: &[u8]) -> Self {
        Self(u32::from_le_bytes(bytes.try_into().unwrap()) ^ 0x8000_0000)
    }
    fn write_to(&self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&(self.0 ^ 0x8000_0000).to_le_bytes());
    }
}
impl CsrEdge for XorEdge {
    fn neighbor(&self) -> u32 {
        self.0
    }
    fn with_neighbor(self, dst: u32) -> Self {
        Self(dst)
    }
}
impl CsrEdgeTombstone for XorEdge {
    fn tombstone_edge() -> Self {
        Self(u32::MAX)
    }
    fn is_tombstone(&self) -> bool {
        self.0 == u32::MAX
    }
}

fn check_spill<E: CsrEdgeTombstone>(codec: &str, cap: usize) {
    let (mut g, r) = fresh::<E>(cap);
    g.insert_sorted_run(0, &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(g.row_is_tiny(0));
    let off = g.get_row_pub(0).log_head() as u64;
    let (_, c) = observe(&format!("{codec}_cap{cap}_spill"), &r, || {
        g.insert_edge(0, 8)
    });
    let mut got = g.out_neighbors_vec(0);
    got.sort_unstable();
    assert_eq!(got, (0..=8).collect::<Vec<_>>());
    assert_eq!(g.num_edges(), 9);
    assert!(!g.row_is_tiny(0));
    assert_eq!(g.tiny_pool_depth(), 1);
    assert_eq!(g.tiny_memory_bytes(), 32);
    g.verify_invariants();
    assert_prefix(c[2], off, 8); // a full spill still needs all 32 bytes
    for dst in 0..=8 {
        assert!(g.remove_edge(0, dst));
    }
    assert_eq!(g.num_edges(), 0);
    // cap170 leaves a stateless log row; cap0 leaves slab space. Both
    // can reuse the pool, but cap0 needs a different, unplaced row.
    let src = if cap == 170 { 0 } else { 63 };
    g.insert_edge(src, 42);
    check_scan(&g, &r, &format!("{codec}_cap{cap}_reuse"), src, &[42]);
    assert_eq!(g.get_row_pub(src).log_head() as u64, off);
    assert_eq!(g.tiny_memory_bytes(), 32);
    assert_eq!(g.tiny_pool_depth(), 0);
    assert_eq!(g.num_edges(), 1);
}

#[test]
fn tiny_full_spill_preserves_neighbors_and_reuse() {
    for cap in [170, 0] {
        check_spill::<Edge>("default", cap);
        check_spill::<XorEdge>("xor", cap);
    }
}

#[test]
fn tiny_nested_scans_preserve_prefix_and_order() {
    // Multi-byte IDs exercise the LE decode across byte boundaries; duplicates
    // and a self-loop stay in the sequence. Degree 0 is checked before any insert.
    let values: [u32; 8] = [0, 5, 5, 0x0102_0304, 0x0000_00ff, 0x1234_5678, 41, 42];
    for cap in [0, 170] {
        for order in [
            lara_graph::graph::AdjacencyOrder::Unordered,
            lara_graph::graph::AdjacencyOrder::Insertion,
        ] {
            let (mut g, r) = fresh::<Edge>(cap);
            g.set_order(order).unwrap();
            assert_eq!(g.order(), order);
            g.insert_edge(1, 7); // sibling tiny row owning a second chunk
            let (empty, c0) = observe(&format!("nested_cap{cap}_{order:?}_degree0"), &r, || {
                g.out_neighbors_vec(0)
            });
            assert!(empty.is_empty());
            assert!(c0.iter().all(|c| *c == Ops::default())); // degree 0: no read at all
            for (i, &dst) in values.iter().enumerate() {
                g.insert_edge(0, dst);
                let want = values[..=i].to_vec(); // append-only fixture: known sequence
                let tag = format!("nested_cap{cap}_{order:?}_n{}", i + 1);
                // 1) single scan: freeze the eight region counters, then check prefix/order.
                let (got, c) = observe(&tag, &r, || g.out_neighbors_vec(0));
                assert_eq!(got, want);
                let off = g.get_row_pub(0).log_head() as u64;
                assert_prefix(c[2], off, i + 1);
                assert_eq!((c[2].writes, c[2].grows), (0, 0));
                assert!(
                    c.iter()
                        .enumerate()
                        .all(|(k, x)| k == 2 || *x == Ops::default())
                );
                // 2) exactly two nested read-only scans per element. Results are
                //    collected inside the frozen interval and compared afterwards.
                let mut nested_same = Vec::new();
                let mut nested_sib = Vec::new();
                let (outer, cn) = observe(&format!("{tag}_nested"), &r, || {
                    let mut outer = Vec::new();
                    g.for_each_neighbor(0, |x| {
                        outer.push(x);
                        g.for_each_neighbor(0, |y| nested_same.push(y));
                        g.for_each_neighbor(1, |y| nested_sib.push(y));
                    });
                    outer
                });
                let n = i as u64 + 1;
                assert_eq!(outer, want);
                assert_eq!(nested_same, want.repeat(n as usize));
                assert_eq!(nested_sib, [7].repeat(n as usize));
                assert_eq!(cn[2].reads, 1 + 2 * n);
                assert_eq!(cn[2].bytes, 4 * n + n * (4 * n + 4));
                assert_eq!((cn[2].writes, cn[2].grows), (0, 0));
                assert!(
                    cn.iter()
                        .enumerate()
                        .all(|(k, x)| k == 2 || *x == Ops::default())
                );
                // 3) ownership / geometry after the counters were frozen.
                assert_eq!(g.out_degree(0), n);
                assert_eq!(g.num_edges(), n + 1);
                assert!(g.row_is_tiny(0) && g.row_is_tiny(1));
                assert_eq!(g.tiny_row_count(), 2);
                assert_eq!(g.tiny_memory_bytes(), 64);
                g.verify_invariants();
            }
        }
    }
}

#[test]
fn vec_read_short_and_fallback_ranges_preserve_bytes() {
    let big: usize = 70_000; // larger than one page, so >65536 offsets exist
    let pattern: Vec<u8> = (0..big).map(|i| (i * 31 + 7) as u8).collect();
    let mem = VecMemory::new();
    mem.write(0, &pattern);
    let clone = mem.clone();
    let size_before = mem.size();
    let clone_size_before = clone.size();
    let pages_bytes = mem.size() * PAGE_SIZE;

    // Independent per-index oracle: destination index j of a read at absolute
    // offset `off` is the fixture byte at `off + j`, else zero when that absolute
    // index lies beyond the written content. No start/end/min arithmetic here.
    let expect = |off: u64, j: usize| -> u8 {
        let idx = off as usize + j;
        if idx < pattern.len() { pattern[idx] } else { 0 }
    };
    let check = |off: u64, n: usize, pad: usize| {
        let mut buf = vec![0xA5u8; pad + n + pad];
        clone.read(off, &mut buf[pad..pad + n]);
        for j in 0..n {
            assert_eq!(buf[pad + j], expect(off, j), "off={off} n={n} j={j}");
        }
        assert!(
            buf[..pad].iter().all(|&b| b == 0xA5),
            "off={off} n={n} head sentinel"
        );
        assert!(
            buf[pad + n..].iter().all(|&b| b == 0xA5),
            "off={off} n={n} tail sentinel"
        );
    };

    // (a) requested lengths 0..=32 plus the larger fallback lengths.
    let lens: Vec<usize> = (0..=32)
        .chain([33, 63, 64, 65, 511, 512, 513, 32768, 32769, 65536, 65537])
        .collect();
    let offsets: [u64; 8] = [0, 1, 2, 3, 7, 4095, PAGE_SIZE - 1, PAGE_SIZE + 5];
    for &n in &lens {
        for &off in &offsets {
            check(off, n, 8);
        }
    }
    // (b) actual copied lengths 0..=32 at the backing end, requested buffer larger,
    //     with at-end and beyond-end offsets.
    for actual in 0..=32usize {
        for extra in [0u64, 1, 17] {
            let off = (big - actual) as u64 + extra;
            check(off, 64, 4);
        }
    }
    // (c) wholly unmapped offsets: inside the last page past the content and beyond it.
    for off in [
        pages_bytes - 3,
        pages_bytes,
        pages_bytes + 1,
        big as u64,
        big as u64 + 9,
        1 << 20,
    ] {
        check(off, 24, 0);
    }
    // (d) boundary-crossing live prefix: destination runs past the written content.
    check((big - 3) as u64, 64, 0);
    // (e) source bytes and size unchanged; a shared clone sees the same storage.
    assert_eq!(mem.size(), size_before);
    assert_eq!(clone.size(), clone_size_before);
    let mut whole = vec![0u8; big];
    mem.read(0, &mut whole);
    assert_eq!(whole, pattern);
    let probe = [0x5Au8; 9];
    mem.write(1000, &probe);
    let mut via_clone = [0u8; 9];
    clone.read(1000, &mut via_clone);
    assert_eq!(via_clone, probe);
}
