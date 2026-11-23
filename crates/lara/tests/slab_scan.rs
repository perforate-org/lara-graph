//! Slab-scan requests through public traversal and strict VecMemory backing.
//! Snapshot requests before semantic/count/ownership oracles; no timing claims.
#![cfg(feature = "std")]

use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::labeled::{LabelId, LabeledLaraGraph};
use lara_graph::memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
use lara_graph::traits::{CsrEdge, CsrEdgeTombstone, CsrVertex, FixedWidth};
use std::cell::RefCell;
use std::rc::Rc;

const SRC: u32 = 8;
const SIBLING: u32 = 9;

#[derive(Default, Debug)]
struct Trace {
    reads: Vec<(u64, usize, bool)>, // offset, bytes, mapped
    writes: usize,
    write_bytes: usize,
    grows: usize,
}

#[derive(Clone, Default)]
struct Probe {
    memory: VecMemory,
    active: Rc<RefCell<Option<Trace>>>,
}

impl Probe {
    fn record_read(&self, offset: u64, len: usize, mapped: bool) {
        assert!(offset + len as u64 <= self.memory.size() * PAGE_SIZE);
        if let Some(trace) = self.active.borrow_mut().as_mut() {
            trace.reads.push((offset, len, mapped));
        }
    }
}

impl Memory for Probe {
    fn read(&self, offset: u64, buf: &mut [u8]) {
        self.record_read(offset, buf.len(), false);
        self.memory.read(offset, buf);
    }
    fn read_mapped(&self, offset: u64, buf: &mut [u8]) {
        self.record_read(offset, buf.len(), true);
        self.memory.read_mapped(offset, buf);
    }
    fn write(&self, offset: u64, buf: &[u8]) {
        assert!(offset + buf.len() as u64 <= self.memory.size() * PAGE_SIZE);
        if let Some(trace) = self.active.borrow_mut().as_mut() {
            trace.writes += 1;
            trace.write_bytes += buf.len();
        }
        self.memory.write(offset, buf);
    }
    fn size(&self) -> u64 {
        self.memory.size()
    }
    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        if let Some(trace) = self.active.borrow_mut().as_mut() {
            trace.grows += 1;
        }
        self.memory.grow(pages)
    }
}

type Graph<E = Edge> = LaraGraph<Vertex, E, Probe>;

fn memories(edges: &Probe) -> LaraMemories<Probe> {
    LaraMemories {
        vertices: Probe::default(),
        edges: edges.clone(),
        logs: Probe::default(),
        counts: Probe::default(),
        spans: Probe::default(),
        free_spans: Probe::default(),
        blocks: Probe::default(),
        buckets: Probe::default(),
    }
}

fn observe<T>(tag: &str, edges: &Probe, scan: impl FnOnce() -> T) -> (T, Trace) {
    let backing = (edges.size(), edges.memory.capacity_bytes());
    assert!(edges.active.replace(Some(Trace::default())).is_none());
    let value = scan();
    let trace = edges.active.take().unwrap();
    let mapped = trace.reads.iter().filter(|r| r.2).count();
    let bytes: usize = trace.reads.iter().map(|r| r.1).sum();
    println!(
        "OBS,{tag},{},{mapped},{bytes},{},{},{}",
        trace.reads.len() - mapped,
        trace.writes,
        trace.write_bytes,
        trace.grows
    );
    for (i, (off, len, mapped)) in trace.reads.iter().enumerate() {
        println!("RANGE,{tag},{i},{off},{len},{mapped}");
    }
    assert_eq!((trace.writes, trace.write_bytes, trace.grows), (0, 0, 0));
    assert_eq!((edges.size(), edges.memory.capacity_bytes()), backing);
    (value, trace)
}

fn core<E: CsrEdgeTombstone>(values: &[u32], sibling: &[u32]) -> (Graph<E>, Probe) {
    let edges = Probe::default();
    let mut g = Graph::new_with_segments(memories(&edges), 64, 8).unwrap();
    let pairs: Vec<_> = [0, 1, 1]
        .into_iter()
        .map(|dst| (0, dst))
        .chain(values.iter().map(|&dst| (SRC, dst)))
        .chain(sibling.iter().map(|&dst| (SIBLING, dst)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    assert_eq!(g.tree_bucket_count(), 0); // bulk does not promote
    assert!(!g.row_is_tiny(SRC));
    (g, edges)
}

fn values(n: usize) -> Vec<u32> {
    (0..n).map(|i| (i % 32) as u32).collect()
}

fn assert_span(
    reads: &[(u64, usize, bool)],
    base: u64,
    bytes: usize,
    lengths: &[usize],
    mapped: bool,
) {
    assert_eq!(reads.iter().map(|r| r.1).collect::<Vec<_>>(), lengths);
    let mut next = base;
    for &(off, len, mode) in reads {
        assert_eq!((off, mode), (next, mapped));
        next += len as u64;
    }
    assert_eq!(next, base + bytes as u64);
}

#[test]
fn core_slab_scans_keep_bounded_chunks_across_cutoff() {
    let cases: &[(usize, &[usize], bool)] = &[
        (8193, &[32772], false),
        (0, &[], false),
        (1, &[4], false),
        (127, &[508], false),
        (128, &[512], false),
        (129, &[516], false),
        (8191, &[32764], false),
        (8192, &[32768], false),
        (16384, &[65536], false),
        (16385, &[32768, 32768, 4], true),
    ];
    for &(n, lengths, mapped) in cases {
        let want = values(n);
        let (g, edges) = core::<Edge>(&want, &[9; 17]);
        let base = g.get_row_pub(SRC).base_slot_start() * 4;
        assert!(base > 0);
        let (got, trace) = observe(&format!("core{n}"), &edges, || g.out_neighbors_vec(SRC));
        assert_eq!(got, want);
        assert_eq!(g.out_degree(SRC), n as u64);
        assert_eq!(g.num_edges(), n as u64 + 20);
        assert_eq!(g.out_neighbors_vec(0), [0, 1, 1]);
        assert_eq!(g.out_neighbors_vec(SIBLING), [9; 17]);
        g.verify_invariants();
        assert_span(&trace.reads, base, n * 4, lengths, mapped);
        if n == 8193 {
            let (all, trace) = observe("core_all8193", &edges, || {
                let mut all = Vec::new();
                g.scan_all(|dst| all.push(dst));
                all
            });
            let expected: Vec<_> = [0, 1, 1].into_iter().chain(want).chain([9; 17]).collect();
            assert_eq!(all, expected);
            assert_eq!(
                trace.reads[0],
                (g.get_row_pub(0).base_slot_start() * 4, 12, false)
            );
            assert_span(&trace.reads[1..2], base, 32772, &[32772], false);
            assert_eq!(
                trace.reads[2],
                (g.get_row_pub(SIBLING).base_slot_start() * 4, 68, false)
            );
            assert_eq!(trace.reads.len(), 3);
        }
    }
}

#[test]
fn core_dense_prefix_and_logs_keep_the_stored_scan_extent() {
    let mut want = values(8195);
    want[0] = 58;
    want[8191..].copy_from_slice(&[59, 60, 61, 62]);
    let (mut g, edges) = core::<Edge>(&want, &[9; 17]);
    g.rebalance_leaf(1); // removes bulk slack, without promoting
    assert_eq!(g.row_allocated(SRC), 8195);
    g.set_promote_threshold(None); // keep the full slab and use its log
    g.insert_edge(SRC, 61);
    g.insert_edge(SRC, 63);
    assert_ne!(g.get_row_pub(SRC).log_head(), -1);
    for dst in [58, 59, 60] {
        assert!(g.remove_edge(SRC, dst));
        want.swap_remove(want.iter().position(|&v| v == dst).unwrap());
    }
    assert_eq!(g.get_row_pub(SRC).stored_degree(), 8192);
    assert_eq!(g.out_degree(SRC), 8194);
    let base = g.get_row_pub(SRC).base_slot_start() * 4;
    let (got, trace) = observe("core_holes_and_logs", &edges, || g.out_neighbors_vec(SRC));
    let expected: Vec<_> = want.iter().copied().chain([63, 61]).collect();
    assert_eq!(got, expected); // slab ascending, log newest-first
    assert_eq!(g.out_neighbors_vec(SIBLING), [9; 17]);
    assert_eq!(g.num_edges(), 8214);
    g.verify_invariants();
    assert_span(&trace.reads, base, 32768, &[32768], false);

    g.rebalance_leaf(1);
    assert_eq!(g.get_row_pub(SRC).log_head(), -1);
    assert_eq!(g.get_row_pub(SRC).stored_degree(), 8194);
    let base = g.get_row_pub(SRC).base_slot_start() * 4;
    let (got, trace) = observe("core_after_fold", &edges, || g.out_neighbors_vec(SRC));
    want.extend([61, 63]); // folding appends logs oldest-first
    assert_eq!(got, want);
    assert_eq!(g.num_edges(), 8214);
    g.verify_invariants();
    assert_span(&trace.reads, base, 32776, &[32776], false);
}

#[derive(Clone, Copy, Default)]
struct Encoded<const W: usize>(u32);

impl<const W: usize> FixedWidth for Encoded<W> {
    const BYTES: usize = W;
    fn read_from(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), W);
        let dst = u32::from_le_bytes(bytes[..4].try_into().unwrap()) ^ 0x8000_0000;
        if W >= 8 {
            assert_eq!(
                u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                dst.rotate_left(7)
            );
            assert!(bytes[8..].iter().all(|&b| b == 0xa5));
        }
        Self(dst)
    }
    fn write_to(&self, bytes: &mut [u8]) {
        bytes.fill(0xa5);
        bytes[..4].copy_from_slice(&(self.0 ^ 0x8000_0000).to_le_bytes());
        if W >= 8 {
            bytes[4..8].copy_from_slice(&self.0.rotate_left(7).to_le_bytes());
        }
    }
}
impl<const W: usize> CsrEdge for Encoded<W> {
    fn neighbor(&self) -> u32 {
        self.0
    }
    fn with_neighbor(self, dst: u32) -> Self {
        Self(dst)
    }
}
impl<const W: usize> CsrEdgeTombstone for Encoded<W> {
    fn tombstone_edge() -> Self {
        Self(u32::MAX)
    }
    fn is_tombstone(&self) -> bool {
        self.0 == u32::MAX
    }
}

fn check_codec<const W: usize>(cases: &[(usize, &[usize], bool)]) {
    for &(n, lengths, mapped) in cases {
        let mut want = values(n);
        if n >= 2 {
            want[n - 2..].copy_from_slice(&[62, 63]);
        }
        let (mut g, edges) = core::<Encoded<W>>(&want, &[9; 17]);
        let base = g.get_row_pub(SRC).base_slot_start() * W as u64;
        assert!(base > 0);
        let (got, trace) = observe(&format!("codec{W}_n{n}"), &edges, || {
            g.out_neighbors_vec(SRC)
        });
        assert_eq!(got, want);
        assert_eq!(g.num_edges(), n as u64 + 20);
        assert_eq!(g.out_neighbors_vec(SIBLING), [9; 17]);
        g.verify_invariants();
        assert_span(&trace.reads, base, n * W, lengths, mapped);
        if W == 12 && n == 2731 {
            g.rebalance_leaf(1);
            assert!(g.remove_edge(SRC, 62));
            want.swap_remove(n - 2);
            assert_eq!(g.get_row_pub(SRC).stored_degree(), 2730);
            let (got, trace) =
                observe("codec12_boundary_hole", &edges, || g.out_neighbors_vec(SRC));
            assert_eq!(got, want);
            assert_eq!(g.out_degree(SRC), 2730);
            g.verify_invariants();
            assert_span(
                &trace.reads,
                g.get_row_pub(SRC).base_slot_start() * 12,
                32760,
                &[32760],
                false,
            );
        }
    }
}

#[test]
fn slab_chunks_preserve_whole_record_codecs() {
    check_codec::<12>(&[
        (2731, &[32772], false),
        (0, &[], false),
        (1, &[12], false),
        (42, &[504], false),
        (43, &[516], false),
        (2730, &[32760], false),
        (5461, &[65532], false),
    ]);
    check_codec::<4>(&[
        (8193, &[32772], false),
        (128, &[512], false),
        (129, &[516], false),
    ]);
    check_codec::<512>(&[
        (1, &[512], false),
        (2, &[1024], false),
        (64, &[32768], false),
        (65, &[33280], false),
    ]);
    check_codec::<516>(&[
        (1, &[516], false),
        (63, &[32508], false),
        (64, &[33024], false),
        (127, &[65532], false),
    ]);
    check_codec::<32768>(&[(1, &[32768], false), (2, &[65536], false)]);
}

#[test]
fn oversized_records_keep_per_record_fallback() {
    check_codec::<32772>(&[
        (0, &[], false),
        (1, &[32772], false),
        (2, &[32772, 32772], false),
    ]);
}

#[test]
fn nested_core_scans_keep_independent_chunk_buffers() {
    let want = values(8193);
    let sibling: Vec<_> = (0..8193).map(|i| [7, 8, 7, 31][i % 4]).collect();
    let (g, edges) = core::<Edge>(&want, &sibling);
    let a = g.get_row_pub(SRC).base_slot_start() * 4;
    let b = g.get_row_pub(SIBLING).base_slot_start() * 4;
    let ((got, nested), trace) = observe("core_nested", &edges, || {
        let mut got = Vec::new();
        let mut nested = Vec::new();
        g.for_each_neighbor(SRC, |dst| {
            got.push(dst);
            if got.len() == 1 || got.len() == 8193 {
                nested.push(g.out_neighbors_vec(SIBLING));
            }
        });
        (got, nested)
    });
    assert_eq!(got, want);
    assert_eq!(nested, [sibling.clone(), sibling]);
    assert_eq!(g.num_edges(), 16389);
    g.verify_invariants();
    assert_eq!(
        trace.reads,
        [(a, 32772, false), (b, 32772, false), (b, 32772, false)]
    );
}

#[test]
fn labeled_scans_preserve_siblings_dense_prefixes_and_nested_callbacks() {
    let edges = Probe::default();
    let l0 = LabelId::new(0);
    let l1 = LabelId::new(1);
    let mut want = values(8193);
    want[8191..].copy_from_slice(&[60, 62]);
    let sibling: Vec<_> = (0..8193).map(|i| [17, 17, 8, 3][i % 4]).collect();
    let pairs: Vec<_> = [(0, 1, l0)]
        .into_iter()
        .chain(want.iter().map(|&dst| (SRC, dst, l0)))
        .chain(sibling.iter().map(|&dst| (SRC, dst, l1)))
        .collect();
    let mut g = LabeledLaraGraph::<Probe>::with_edge_list(memories(&edges), &pairs, 64).unwrap();
    assert_eq!(g.segment_size(), 8);
    assert!(!g.bucket_is_tree(SRC, l0) && !g.bucket_is_tree(SRC, l1));
    // SRC is the first vertex of its leaf, and these are its only buckets.
    let a = g.leaf_geometry(1).0 * 4;
    let b = a + g.bucket_allocated(SRC, l0) * 4;
    assert!(a > 0);
    let (got, trace) = observe("label0_8193", &edges, || {
        let mut got = Vec::new();
        g.for_each_edge_of_label(SRC, l0, |dst| got.push(dst));
        got
    });
    assert_eq!(got, want);
    g.verify_invariants();
    assert_span(&trace.reads, a, 32772, &[32772], false);
    let (all, trace) = observe("all_labels8193", &edges, || g.out_labeled_vec(SRC));
    let expected: Vec<_> = want
        .iter()
        .map(|&dst| (dst, l0))
        .chain(sibling.iter().map(|&dst| (dst, l1)))
        .collect();
    assert_eq!(all, expected);
    assert_span(&trace.reads[..1], a, 32772, &[32772], false);
    assert_span(&trace.reads[1..], b, 32772, &[32772], false);

    assert!(g.remove_edge(SRC, 60, l0));
    want.swap_remove(8191);
    let ((got, nested), trace) = observe("labels_nested_after_hole", &edges, || {
        let mut got = Vec::new();
        let mut nested = Vec::new();
        g.for_each_edge_of_label(SRC, l0, |dst| {
            got.push(dst);
            if got.len() == 1 {
                g.for_each_edge_of_label(SRC, l1, |v| nested.push(v));
            }
        });
        (got, nested)
    });
    assert_eq!(got, want);
    assert_eq!(nested, sibling);
    assert_eq!(g.num_edges(), 16386);
    assert_eq!(g.out_labeled_vec(0), [(1, l0)]);
    g.verify_invariants();
    assert_eq!(trace.reads, [(a, 32768, false), (b, 32772, false)]);
}
