//! Sorted-run write requests through public mutation and strict VecMemory backing.
//! Freeze all eight regions before semantic, payload and ownership checks.
#![cfg(feature = "std")]

use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::ingest::IngestBuffer;
use lara_graph::memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
use lara_graph::traits::{CsrEdge, CsrEdgeTombstone, CsrVertex, FixedWidth, NO_BUCKET};
use lara_graph::tree_csr::{B, R_MAX};
use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

const SRC: u32 = 8;
const SIBLING: u32 = 9;
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
    reads: usize,
    mapped: usize,
    read_bytes: usize,
    writes: Vec<(u64, Vec<u8>)>,
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
        if let Some(c) = self.active.borrow_mut().as_mut() {
            if mapped {
                c.mapped += 1;
            } else {
                c.reads += 1;
            }
            c.read_bytes += len;
        }
    }

    fn backing(&self) -> (u64, usize) {
        (self.size(), self.memory.capacity_bytes())
    }

    fn bytes(&self) -> Vec<u8> {
        let mut out = vec![0; (self.size() * PAGE_SIZE) as usize];
        self.memory.read(0, &mut out);
        out
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
        if let Some(c) = self.active.borrow_mut().as_mut() {
            c.writes.push((offset, buf.to_vec()));
        }
        self.memory.write(offset, buf);
    }
    fn size(&self) -> u64 {
        self.memory.size()
    }
    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        if let Some(c) = self.active.borrow_mut().as_mut() {
            c.grows += 1;
        }
        self.memory.grow(pages)
    }
    fn fence(&self) {
        self.memory.fence();
    }
}

type Graph<E = Edge> = LaraGraph<Vertex, E, Probe>;

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
    for region in r {
        assert!(region.active.replace(Some(Trace::default())).is_none());
    }
    let result = op();
    let traces = std::array::from_fn(|i| r[i].active.take().unwrap());
    for (i, c) in traces.iter().enumerate() {
        let region = REGIONS[i];
        let bytes: usize = c.writes.iter().map(|w| w.1.len()).sum();
        let (pages, capacity) = r[i].backing();
        println!(
            "OBS,{tag},{region},{},{},{},{},{bytes},{},{pages},{capacity}",
            c.reads,
            c.mapped,
            c.read_bytes,
            c.writes.len(),
            c.grows
        );
        for (k, (off, data)) in c.writes.iter().enumerate() {
            println!("RANGE,{tag},{region},{k},{off},{}", data.len());
        }
    }
    (result, traces)
}

fn core<E: CsrEdgeTombstone>(initial: u32, cap: usize) -> (Graph<E>, [Probe; 8]) {
    let r = std::array::from_fn(|_| Probe::default());
    let mut g =
        Graph::<E>::new_with_segments_and_max_log(bundle(&r), initial.max(512), 8, cap).unwrap();
    let pairs: Vec<_> = [0, 1, 1]
        .into_iter()
        .map(|dst| (0, dst))
        .chain((0..initial).map(|dst| (SRC, dst)))
        .chain((0..128).map(|dst| (SIBLING, dst)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    assert!(!g.row_is_tiny(SRC));
    assert_eq!(g.tree_bucket_count(), 0);
    (g, r)
}

fn assert_rows<E: CsrEdgeTombstone>(g: &Graph<E>, expected: &[u32]) {
    assert_eq!(g.out_neighbors_vec(SRC), expected);
    assert_eq!(g.out_degree(SRC), expected.len() as u64);
    assert_eq!(g.out_neighbors_vec(0), [0, 1, 1]);
    assert_eq!(g.out_neighbors_vec(SIBLING), (0..128).collect::<Vec<_>>());
    assert_eq!(g.num_edges(), expected.len() as u64 + 131);
    g.verify_invariants();
}

fn assert_only_payload(c: &[Trace; 8], region: usize) {
    for (i, trace) in c.iter().enumerate() {
        assert_eq!(trace.grows, 0);
        if i != region {
            assert!(
                trace.writes.is_empty(),
                "{} unexpectedly written",
                REGIONS[i]
            );
        }
    }
    assert_eq!(
        (c[region].reads, c[region].mapped, c[region].read_bytes),
        (0, 0, 0)
    );
}

fn assert_span(c: &Trace, base: u64, payload: &[u8], lengths: &[usize]) {
    // Byte/order checks precede the new write-count contract.
    let mut next = base;
    let mut actual = Vec::new();
    for (off, bytes) in &c.writes {
        assert_eq!(*off, next);
        actual.extend_from_slice(bytes);
        next += bytes.len() as u64;
    }
    assert_eq!(actual, payload);
    assert_eq!(
        c.writes.iter().map(|w| w.1.len()).collect::<Vec<_>>(),
        lengths
    );
}

#[test]
fn slab_runs_batch_only_the_reserved_tail() {
    let cases: &[(usize, &[usize])] = &[
        (4, &[16]),
        (0, &[]),
        (1, &[4]),
        (15, &[60]),
        (16, &[64]),
        (17, &[64, 4]),
        (32, &[64, 64]),
        (33, &[64, 64, 4]),
        (129, &[64, 64, 64, 64, 64, 64, 64, 64, 4]),
        (197, &[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 20]),
    ];
    for cap in [170, 0] {
        for &(n, lengths) in cases {
            let (mut g, r) = core::<Edge>(64, cap);
            let row = g.get_row_pub(SRC);
            assert_eq!(g.row_allocated(SRC), 261); // 197 slots of tail room
            let base = (row.base_slot_start() + 64) * 4;
            assert!(base > 0);
            let backing = r.each_ref().map(Probe::backing);
            let before = r[1].bytes();
            let geometry = g.leaf_geometry(1);
            let maintenance = (g.rebalance_count(), g.promote_count());
            let mut dsts: Vec<u32> = [8, 8, 500, 501].into_iter().cycle().take(n).collect();
            dsts.sort_unstable();
            let (_, c) = observe(&format!("slab_cap{cap}_n{n}"), &r, || {
                g.insert_sorted_run(SRC, &dsts)
            });
            assert_rows(&g, &(0..64).chain(dsts.iter().copied()).collect::<Vec<_>>());
            assert_eq!(
                g.get_row_pub(SRC),
                row.with_live_degree(64 + n as u32)
                    .with_stored_degree(64 + n as u32)
            );
            assert_eq!(
                g.leaf_geometry(1),
                (geometry.0, geometry.1, geometry.2 + n as u64)
            );
            assert_eq!((g.rebalance_count(), g.promote_count()), maintenance);
            assert_eq!(r.each_ref().map(Probe::backing), backing);
            assert_only_payload(&c, 1);
            let after = r[1].bytes();
            assert_eq!(before[..base as usize], after[..base as usize]);
            assert_eq!(
                before[base as usize + 4 * n..],
                after[base as usize + 4 * n..]
            );
            let payload: Vec<_> = dsts.iter().flat_map(|d| d.to_le_bytes()).collect();
            assert_span(&c[1], base, &payload, lengths);
            if n == 197 {
                assert!(!g.row_has_slab_room(SRC));
                assert_eq!(g.get_row_pub(SRC).log_head(), -1);
            }
        }
    }

    // The first 64-byte write crosses a backed page boundary. Neither
    // adjacent rows nor the unused tail may be included in that write.
    let r = std::array::from_fn(|_| Probe::default());
    let mut g: Graph = Graph::new_with_segments(bundle(&r), 512, 8).unwrap();
    let pairs: Vec<_> = vec![(0, 0); 4071]
        .into_iter()
        .chain((0..64).map(|dst| (SRC, dst)))
        .chain((0..128).map(|dst| (SIBLING, dst)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    let base = (g.get_row_pub(SRC).base_slot_start() + 64) * 4;
    assert_eq!(base, PAGE_SIZE - 16);
    let before = r[1].bytes();
    let backing = r.each_ref().map(Probe::backing);
    let dsts: Vec<u32> = (1..18).collect();
    let (_, c) = observe("slab_crosses_page", &r, || g.insert_sorted_run(SRC, &dsts));
    assert_eq!(
        g.out_neighbors_vec(SRC),
        (0..64).chain(1..18).collect::<Vec<_>>()
    );
    assert_eq!(g.out_neighbors_vec(0), vec![0; 4071]);
    assert_eq!(g.out_neighbors_vec(SIBLING), (0..128).collect::<Vec<_>>());
    assert_eq!((g.num_edges(), g.out_degree(SRC)), (4280, 81));
    g.verify_invariants();
    assert_only_payload(&c, 1);
    assert_eq!(r.each_ref().map(Probe::backing), backing);
    let after = r[1].bytes();
    assert_eq!(before[..base as usize], after[..base as usize]);
    assert_eq!(before[base as usize + 68..], after[base as usize + 68..]);
    let payload: Vec<_> = dsts.iter().flat_map(|d| d.to_le_bytes()).collect();
    assert_span(&c[1], base, &payload, &[64, 4]);

    // The pre-existing eligible path promotes after the entire run, not
    // at the intermediate threshold. Only its append writes are batched.
    let (mut g, r) = core::<Edge>(64, 170);
    g.set_promote_threshold(Some(66));
    let row = g.get_row_pub(SRC);
    let dsts = [8u32, 8, 500, 501];
    let rb = g.rebalance_count();
    let (_, c) = observe("slab_crosses_promotion", &r, || {
        g.insert_sorted_run(SRC, &dsts)
    });
    assert_rows(&g, &(0..64).chain(dsts).collect::<Vec<_>>());
    assert_eq!((g.rebalance_count(), g.promote_count()), (rb, 1));
    assert_ne!(g.get_row_pub(SRC).tree_bucket(), NO_BUCKET);
    assert_eq!(g.get_row_pub(SRC).stored_degree(), 0);
    let writes = &c[1].writes;
    // Promotion retires the slab extent without marker writes.
    assert_eq!(
        writes,
        &[(
            (row.base_slot_start() + 64) * 4,
            dsts.iter().flat_map(|d| d.to_le_bytes()).collect()
        )]
    );
}

// A non-raw four-byte codec, plus wider records whose even destinations
// intentionally leave zero padding untouched. Chunk reuse must reset it.
#[derive(Clone, Copy, Default)]
struct Encoded<const W: usize>(u32);

impl<const W: usize> FixedWidth for Encoded<W> {
    const BYTES: usize = W;
    fn read_from(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), W);
        let raw = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        let dst = if raw == u32::MAX {
            raw
        } else {
            raw ^ 0x8000_0000
        };
        if W >= 8 {
            assert_eq!(&bytes[4..8], &dst.rotate_left(7).to_le_bytes());
            assert!(
                bytes[8..]
                    .iter()
                    .all(|&b| b == if dst & 1 == 1 { 0xa5 } else { 0 })
            );
        }
        Self(dst)
    }
    fn write_to(&self, bytes: &mut [u8]) {
        assert_eq!(bytes.len(), W);
        let raw = if self.0 == u32::MAX {
            self.0
        } else {
            self.0 ^ 0x8000_0000
        };
        bytes[..4].copy_from_slice(&raw.to_le_bytes());
        if W >= 8 {
            bytes[4..8].copy_from_slice(&self.0.rotate_left(7).to_le_bytes());
            if self.0 & 1 == 1 {
                bytes[8..].fill(0xa5);
            }
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

fn codec_run<const W: usize>(dsts: &[u32], lengths: &[usize]) {
    let (mut g, r) = core::<Encoded<W>>(64, 170);
    let base = (g.get_row_pub(SRC).base_slot_start() + 64) * W as u64;
    let before = r[1].bytes();
    let backing = r.each_ref().map(Probe::backing);
    let (_, c) = observe(&format!("codec{W}_n{}", dsts.len()), &r, || {
        g.insert_sorted_run(SRC, dsts)
    });
    assert_rows(&g, &(0..64).chain(dsts.iter().copied()).collect::<Vec<_>>());
    assert_only_payload(&c, 1);
    assert_eq!(r.each_ref().map(Probe::backing), backing);
    // Independent expected bytes, not the codec's write_to implementation.
    let mut payload = Vec::new();
    for &dst in dsts {
        payload.extend_from_slice(&(dst ^ 0x8000_0000).to_le_bytes());
        if W >= 8 {
            payload.extend_from_slice(&dst.rotate_left(7).to_le_bytes());
            payload.resize(payload.len() + W - 8, if dst & 1 == 1 { 0xa5 } else { 0 });
        }
    }
    let after = r[1].bytes();
    assert_eq!(before[..base as usize], after[..base as usize]);
    assert_eq!(
        before[base as usize + payload.len()..],
        after[base as usize + payload.len()..]
    );
    assert_span(&c[1], base, &payload, lengths);
}

#[test]
fn run_encoding_preserves_wire_bytes_and_zeroed_padding() {
    codec_run::<4>(&[1; 17], &[64, 4]);
    codec_run::<12>(&[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3], &[60, 60, 12]);
    codec_run::<64>(&[1, 2, 3], &[64, 64, 64]);
    codec_run::<68>(&[1, 2, 3], &[68, 68, 68]); // unchanged heap/per-record fallback
}

#[test]
fn slab_tail_append_preserves_dense_deletion_and_fold() {
    let (mut g, r) = core::<Edge>(80, 170);
    g.rebalance_leaf(1);
    assert_eq!(g.row_allocated(SRC), 160);
    assert!(g.remove_edge(SRC, 0));
    let row = g.get_row_pub(SRC);
    assert_eq!((row.stored_degree(), row.degree()), (79, 79));
    let rb = g.rebalance_count();
    let (_, c) = observe("slab_tail_with_hole", &r, || {
        g.insert_sorted_run(SRC, &[500, 501])
    });
    let expected: Vec<_> = [79].into_iter().chain(1..79).chain([500, 501]).collect();
    assert_rows(&g, &expected);
    assert_eq!(
        (g.get_row_pub(SRC).stored_degree(), g.out_degree(SRC)),
        (81, 81)
    );
    assert_eq!(g.rebalance_count(), rb);
    let mut replacement = [0; 4];
    r[1].read(row.base_slot_start() * 4, &mut replacement);
    assert_eq!(replacement, 79u32.to_le_bytes());
    assert_only_payload(&c, 1);
    let payload: Vec<_> = [500u32, 501].iter().flat_map(|d| d.to_le_bytes()).collect();
    assert_span(&c[1], (row.base_slot_start() + 79) * 4, &payload, &[8]);
    g.rebalance_leaf(1);
    assert_eq!(g.get_row_pub(SRC).stored_degree(), 81);
    assert_rows(&g, &expected);
}

#[test]
fn resident_and_fallback_runs_keep_ownership_and_trigger_paths() {
    let r = std::array::from_fn(|_| Probe::default());
    let mut tiny: Graph = Graph::new_with_segments(bundle(&r), 512, 8).unwrap();
    tiny.insert_edge(SRC, SRC);
    let (_, c) = observe("tiny_four", &r, || {
        tiny.insert_sorted_run(SRC, &[8, 9, 9, 10])
    });
    assert_eq!(tiny.out_neighbors_vec(SRC), [8, 8, 9, 9, 10]);
    assert!(tiny.row_is_tiny(SRC));
    assert_eq!(tiny.num_edges(), 5);
    tiny.verify_invariants();
    assert_only_payload(&c, 2);
    assert_eq!(
        c[2].writes.iter().map(|w| w.1.len()).collect::<Vec<_>>(),
        [16]
    );
    tiny.insert_sorted_run(SRC, &[11, 12, 13]); // fill K8 outside the observation
    let (_, c) = observe("tiny_spill", &r, || tiny.insert_sorted_run(SRC, &[14, 15]));
    assert_eq!(tiny.out_degree(SRC), 10);
    assert!(!tiny.row_is_tiny(SRC));
    let mut got = tiny.out_neighbors_vec(SRC);
    got.sort_unstable();
    assert_eq!(got, [8, 8, 9, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(tiny.num_edges(), 10);
    assert_eq!(c[2].read_bytes, 32);
    tiny.verify_invariants();

    let (mut tree, r) = core::<Edge>(1023, 170);
    tree.insert_edge(SRC, 500); // promote with one full block
    tree.insert_edge(SRC, 501); // mint the second block before observation
    let (_, c) = observe("tree_four", &r, || {
        tree.insert_sorted_run(SRC, &[8, 9, 9, 10])
    });
    assert_rows(
        &tree,
        &(0..1023).chain([500, 501, 8, 9, 9, 10]).collect::<Vec<_>>(),
    );
    assert_ne!(tree.get_row_pub(SRC).tree_bucket(), NO_BUCKET);
    assert_only_payload(&c, 6);
    assert_eq!(
        c[6].writes.iter().map(|w| w.1.len()).collect::<Vec<_>>(),
        [16]
    );
    // A block-crossing run still falls back in its entirety; headers and
    // payloads must remain disjoint.
    tree.insert_sorted_run(SRC, &vec![11; 2047 - 1029]);
    let (_, c) = observe("tree_crosses_block", &r, || {
        tree.insert_sorted_run(SRC, &[12, 13, 13, 14])
    });
    assert_rows(
        &tree,
        &(0..1023)
            .chain([500, 501, 8, 9, 9, 10])
            .chain(vec![11; 1018])
            .chain([12, 13, 13, 14])
            .collect::<Vec<_>>(),
    );
    assert!(c[1].writes.is_empty());
    assert!(c[6].writes.iter().filter(|w| w.1.len() == 4).count() >= 4);

    let (mut g, r) = core::<Edge>(80, 170);
    g.rebalance_leaf(1);
    for dst in 80..160 {
        g.insert_edge(SRC, dst);
    }
    for dst in 500..503 {
        g.insert_edge(SRC, dst);
    }
    assert!(g.remove_edge(SRC, 0));
    assert!(g.row_has_slab_room(SRC)); // one free tail slot, even with three logs
    let rb = g.rebalance_count();
    let (_, c) = observe("full_log_backed_hole", &r, || {
        g.insert_sorted_run(SRC, &[503, 504])
    });
    assert_rows(
        &g,
        &[159]
            .into_iter()
            .chain(1..159)
            .chain([503, 504, 502, 501, 500])
            .collect::<Vec<_>>(),
    );
    assert_eq!(g.get_row_pub(SRC).stored_degree(), 160);
    assert_eq!(g.rebalance_count(), rb);
    assert_eq!(c[1].reads, 0);
    assert_eq!(
        c[1].writes.iter().map(|w| w.1.len()).collect::<Vec<_>>(),
        [4]
    );
    assert_eq!(
        c[2].writes.iter().map(|w| w.1.len()).collect::<Vec<_>>(),
        [16]
    );
    for (i, region) in c.iter().enumerate() {
        assert_eq!(region.grows, 0);
        if i != 1 && i != 2 {
            assert!(region.writes.is_empty());
        }
    }

    let (mut g, r) = core::<Edge>(16, 170);
    assert_eq!(g.row_allocated(SRC), 67);
    let dsts = vec![500; 52]; // tail room is 51, so the entire run falls back
    let (_, c) = observe("tail_room_plus_one", &r, || g.insert_sorted_run(SRC, &dsts));
    assert_rows(&g, &(0..16).chain(dsts).collect::<Vec<_>>());
    assert_eq!(g.get_row_pub(SRC).stored_degree(), 67);
    assert_ne!(g.get_row_pub(SRC).log_head(), -1);
    assert_eq!(
        c[1].writes.iter().map(|w| w.1.len()).collect::<Vec<_>>(),
        [4; 51]
    );
    assert_eq!(g.rebalance_count(), 0);

    let r = std::array::from_fn(|_| Probe::default());
    let mut g: Graph = Graph::new_with_segments(bundle(&r), 64, 1).unwrap();
    g.build_pairs(&(0..16).map(|dst| (SRC, dst)).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(g.row_allocated(SRC), 68);
    let (_, total, actual) = g.leaf_geometry(SRC as u64);
    assert_eq!((actual, total), (16, 68));
    let (_, c) = observe("density_equality", &r, || {
        g.insert_sorted_run(SRC, &[8; 52])
    });
    let mut got = g.out_neighbors_vec(SRC);
    got.sort_unstable();
    let mut expected: Vec<_> = (0..16).chain([8; 52]).collect();
    expected.sort_unstable();
    assert_eq!(got, expected);
    assert_eq!(
        (g.num_edges(), g.rebalance_count(), g.promote_count()),
        (68, 1, 0)
    );
    g.verify_invariants();
    assert_eq!(
        c[1].writes[..52]
            .iter()
            .map(|w| w.1.len())
            .collect::<Vec<_>>(),
        [4; 52]
    );
    assert!(c[2].writes.is_empty());
}

fn assert_idle(c: &[Trace; 8]) {
    for trace in c {
        assert_eq!(
            (
                trace.reads,
                trace.mapped,
                trace.read_bytes,
                trace.writes.len(),
                trace.grows
            ),
            (0, 0, 0, 0, 0)
        );
    }
}

#[test]
fn run_preflight_rejects_before_mutation() {
    let (mut g, r) = core::<Edge>(64, 170);
    let row = g.get_row_pub(SRC);
    let backing = r.each_ref().map(Probe::backing);
    let (_, c) = observe("empty_invalid_source", &r, || {
        g.insert_sorted_run(u32::MAX, &[])
    });
    assert_idle(&c);
    let (result, c) = observe("invalid_source", &r, || {
        catch_unwind(AssertUnwindSafe(|| g.insert_sorted_run(u32::MAX, &[1])))
    });
    assert!(result.is_err());
    assert_idle(&c);
    assert_eq!(g.get_row_pub(SRC), row);
    assert_eq!(r.each_ref().map(Probe::backing), backing);
    assert_rows(&g, &(0..64).collect::<Vec<_>>());

    let r = std::array::from_fn(|_| Probe::default());
    let n = B * R_MAX - 2;
    let mut g: Graph = Graph::with_edge_list(bundle(&r), &vec![(0, 0); n], 4).unwrap();
    let row = g.get_row_pub(0);
    let backing = r.each_ref().map(Probe::backing);
    let (result, c) = observe("depth_one_limit", &r, || {
        catch_unwind(AssertUnwindSafe(|| g.insert_sorted_run(0, &[1, 2, 3])))
    });
    assert!(result.is_err());
    assert_idle(&c);
    assert_eq!(g.get_row_pub(0), row);
    assert_eq!(g.num_edges(), n as u64);
    assert_eq!(g.out_neighbors_vec(0), vec![0; n]);
    assert_eq!(
        (
            g.tree_bucket_count(),
            g.rebalance_count(),
            g.promote_count()
        ),
        (0, 0, 0)
    );
    assert_eq!(r.each_ref().map(Probe::backing), backing);
    g.verify_invariants();
}

#[test]
fn explicit_ingest_flush_batches_only_visible_runs() {
    let (mut g, r) = core::<Edge>(64, 170);
    let a = (g.get_row_pub(SRC).base_slot_start() + 64) * 4;
    let b = (g.get_row_pub(SIBLING).base_slot_start() + 128) * 4;
    let backing = r.each_ref().map(Probe::backing);
    let mut ingest = IngestBuffer::new(0, 4);
    for (src, dst) in [(9, 501), (8, 500), (8, 8), (9, 9), (8, 8), (8, 501)] {
        ingest.push(src, dst);
    }
    assert_eq!(ingest.len(), 6); // capacity hint does not auto-flush
    assert_rows(&g, &(0..64).collect::<Vec<_>>());
    let (_, c) = observe("ingest_two_runs", &r, || ingest.flush(&mut g));
    assert!(ingest.is_empty());
    assert_eq!(
        g.out_neighbors_vec(SRC),
        (0..64).chain([8, 8, 500, 501]).collect::<Vec<_>>()
    );
    assert_eq!(
        g.out_neighbors_vec(SIBLING),
        (0..128).chain([9, 501]).collect::<Vec<_>>()
    );
    assert_eq!(g.out_neighbors_vec(0), [0, 1, 1]);
    assert_eq!(
        (g.num_edges(), g.out_degree(SRC), g.out_degree(SIBLING)),
        (201, 68, 130)
    );
    g.verify_invariants();
    assert_only_payload(&c, 1);
    assert_eq!(r.each_ref().map(Probe::backing), backing);
    assert_eq!(
        c[1].writes,
        [
            (
                a,
                [8u32, 8, 500, 501]
                    .iter()
                    .flat_map(|d| d.to_le_bytes())
                    .collect()
            ),
            (
                b,
                [9u32, 501].iter().flat_map(|d| d.to_le_bytes()).collect()
            ),
        ]
    );
    let (_, c) = observe("ingest_empty_flush", &r, || ingest.flush(&mut g));
    assert_idle(&c);
}

#[test]
fn tiny_runs_stay_in_owned_chunks() {
    for cap in [170, 0] {
        let r = std::array::from_fn(|_| Probe::default());
        let mut fresh: Graph =
            Graph::new_with_segments_and_max_log(bundle(&r), 512, 8, cap).unwrap();
        let (_, c) = observe(&format!("fresh_tiny_cap{cap}"), &r, || {
            fresh.insert_sorted_run(SRC, &[8, 8, 9, 10])
        });
        assert_eq!(fresh.out_neighbors_vec(SRC), [8, 8, 9, 10]);
        assert_eq!(fresh.num_edges(), 4);
        fresh.verify_invariants();
        assert_eq!(
            c[2].writes.iter().map(|w| w.1.len()).collect::<Vec<_>>(),
            [1, 4, 4, 4, 4]
        );
        fresh.insert_edge(SRC, 11);
        let off = fresh.get_row_pub(SRC).log_head() as u64 + 20;
        let (_, c) = observe(&format!("partial_tiny_spill_cap{cap}"), &r, || {
            fresh.insert_sorted_run(SRC, &[12, 13, 14, 15])
        });
        let mut got = fresh.out_neighbors_vec(SRC);
        got.sort_unstable();
        assert_eq!(got, [8, 8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(fresh.num_edges(), 9);
        assert!(!fresh.row_is_tiny(SRC));
        fresh.verify_invariants();
        for (k, (at, bytes)) in c[2].writes[..3].iter().enumerate() {
            assert_eq!(
                (*at, bytes.as_slice()),
                (off + k as u64 * 4, &(12 + k as u32).to_le_bytes()[..])
            );
        }
        assert_eq!(c[2].read_bytes, 32);
        for (initial, add) in [
            (1, 4),
            (0, 0),
            (0, 8),
            (1, 7),
            (2, 6),
            (3, 5),
            (4, 4),
            (5, 3),
            (6, 2),
            (7, 1),
            (8, 0),
        ] {
            let r = std::array::from_fn(|_| Probe::default());
            let mut g: Graph =
                Graph::new_with_segments_and_max_log(bundle(&r), 512, 8, cap).unwrap();
            for dst in [9, 9, 42] {
                g.insert_edge(SIBLING, dst);
            }
            g.insert_edge(SRC, SRC);
            if initial == 0 {
                assert!(g.remove_edge(SRC, SRC));
            }
            for i in 1..initial {
                g.insert_edge(SRC, SRC + i);
            }
            let row = g.get_row_pub(SRC);
            let base = row.log_head() as u64 + u64::from(initial) * 4;
            let before = r[2].bytes();
            let backing = r.each_ref().map(Probe::backing);
            let geometry = g.leaf_geometry(1);
            let maintenance = (g.rebalance_count(), g.promote_count());
            let mut dsts: Vec<u32> = [8, 8, 500, 501].into_iter().cycle().take(add).collect();
            dsts.sort_unstable();
            let (_, c) = observe(
                &format!("owned_tiny_cap{cap}_n{initial}_add{add}"),
                &r,
                || g.insert_sorted_run(SRC, &dsts),
            );
            let n = initial + add as u32;
            assert_eq!(
                g.out_neighbors_vec(SRC),
                (SRC..SRC + initial)
                    .chain(dsts.iter().copied())
                    .collect::<Vec<_>>()
            );
            assert_eq!(g.out_neighbors_vec(SIBLING), [9, 9, 42]);
            assert_eq!(g.num_edges(), u64::from(n) + 3);
            assert_eq!(
                g.get_row_pub(SRC),
                row.with_live_degree(n)
                    .with_stored_degree(row.stored_degree() + add as u32)
            );
            assert!(g.row_is_tiny(SRC));
            assert_eq!(
                (
                    g.tiny_row_count(),
                    g.tiny_memory_bytes(),
                    g.tiny_pool_depth()
                ),
                (2, 64, 0)
            );
            assert_eq!(g.leaf_geometry(1), geometry);
            assert_eq!((g.rebalance_count(), g.promote_count()), maintenance);
            g.verify_invariants();
            assert_only_payload(&c, 2);
            assert_eq!(r.each_ref().map(Probe::backing), backing);
            let after = r[2].bytes();
            assert_eq!(before[..base as usize], after[..base as usize]);
            assert_eq!(
                before[base as usize + add * 4..],
                after[base as usize + add * 4..]
            );
            let payload: Vec<_> = dsts.iter().flat_map(|d| d.to_le_bytes()).collect();
            let lengths = if add == 0 { vec![] } else { vec![add * 4] };
            assert_span(&c[2], base, &payload, &lengths);
        }
    }
}

#[test]
fn tree_runs_stay_in_owned_blocks() {
    let (mut g, r) = core::<Edge>(1023, 170);
    g.insert_edge(SRC, 500); // exactly one full block, no second root yet
    let (_, c) = observe("missing_tree_block", &r, || {
        g.insert_sorted_run(SRC, &[8, 8, 500, 501])
    });
    assert_rows(
        &g,
        &(0..1023).chain([500, 8, 8, 500, 501]).collect::<Vec<_>>(),
    );
    assert!(c[1].writes.is_empty());
    let writes = &c[6].writes;
    assert_eq!(
        writes[writes.len() - 4..]
            .iter()
            .map(|w| w.1.len())
            .collect::<Vec<_>>(),
        [4; 4]
    );
    assert_eq!(writes[writes.len() - 4].0, 4192);
    let cases = [
        (4, vec![16]),
        (0, vec![]),
        (1, vec![4]),
        (15, vec![60]),
        (16, vec![64]),
        (17, vec![64, 4]),
        (1023, [vec![64; 63], vec![60]].concat()),
    ];
    for enabled in [true, false] {
        for (add, lengths) in &cases {
            let (mut g, r) = core::<Edge>(1023, 170);
            g.insert_edge(SRC, 500);
            g.insert_edge(SRC, 501);
            if !enabled {
                g.set_promote_threshold(None);
            }
            let row = g.get_row_pub(SRC);
            let idx = row.tree_bucket() as usize;
            let base = 4196; // second minted block, after its first stored slot
            let before = r[6].bytes();
            let backing = r.each_ref().map(Probe::backing);
            let geometry = g.leaf_geometry(1);
            let maintenance = (g.rebalance_count(), g.promote_count());
            let dsts: Vec<u32> = [8, 8, 500, 501].into_iter().cycle().take(*add).collect();
            let (_, c) = observe(&format!("owned_tree_promote{enabled}_add{add}"), &r, || {
                g.insert_sorted_run(SRC, &dsts)
            });
            assert_rows(
                &g,
                &(0..1023)
                    .chain([500, 501])
                    .chain(dsts.iter().copied())
                    .collect::<Vec<_>>(),
            );
            assert_eq!(g.tree_state_stored(idx), 1025 + *add as u32);
            assert_eq!(g.get_row_pub(SRC), row.with_live_degree(1025 + *add as u32));
            assert_eq!(g.leaf_geometry(1), geometry);
            assert_eq!((g.rebalance_count(), g.promote_count()), maintenance);
            assert_only_payload(&c, 6);
            assert_eq!(r.each_ref().map(Probe::backing), backing);
            let after = r[6].bytes();
            assert_eq!(before[..base as usize], after[..base as usize]);
            assert_eq!(
                before[base as usize + add * 4..],
                after[base as usize + add * 4..]
            );
            let payload: Vec<_> = dsts.iter().flat_map(|d| d.to_le_bytes()).collect();
            assert_span(&c[6], base, &payload, lengths);
        }
    }
}

#[test]
fn resident_runs_reuse_retained_noncontiguous_blocks() {
    for add in [424usize, 425] {
        let r = std::array::from_fn(|_| Probe::default());
        let mut g: Graph = Graph::new_with_segments(bundle(&r), 2048, 8).unwrap();
        let pairs: Vec<_> = [0, 1, 1]
            .into_iter()
            .map(|d| (0, d))
            .chain((0..1023).map(|d| (SRC, d)))
            .chain((1024..2047).map(|d| (SIBLING, d)))
            .collect();
        g.build_pairs(&pairs).unwrap();
        g.insert_edge(SRC, 1023); // root block 0
        g.insert_edge(SIBLING, 2047); // root block 1
        g.insert_edge(SRC, 1024); // SRC's second root entry is block 2
        for dst in (600..1025).rev() {
            assert!(g.remove_edge(SRC, dst));
        }
        assert_eq!(g.out_neighbors_vec(SRC), (0..600).collect::<Vec<_>>());
        let row = g.get_row_pub(SRC);
        let backing = r.each_ref().map(Probe::backing);
        let before = r[6].bytes();
        let dsts = vec![7; add];
        let (_, c) = observe(&format!("retained_add{add}"), &r, || {
            g.insert_sorted_run(SRC, &dsts)
        });
        let mut expected: Vec<_> = (0..600).chain(dsts).collect();
        assert_eq!(g.out_neighbors_vec(SRC), expected);
        assert_eq!(
            g.out_neighbors_vec(SIBLING),
            (1024..2048).collect::<Vec<_>>()
        );
        assert_eq!(g.out_neighbors_vec(0), [0, 1, 1]);
        assert_eq!(g.num_edges(), 1627 + add as u64);
        assert_eq!(
            g.tree_state_stored(row.tree_bucket() as usize),
            600 + add as u32
        );
        g.verify_invariants();
        assert_only_payload(&c, 6);
        assert_eq!(r.each_ref().map(Probe::backing), backing);
        let after = r[6].bytes();
        assert_eq!(before[..2480], after[..2480]); // payload 0 + 600 slots
        assert_eq!(before[4176..8304], after[4176..8304]); // headers and sibling block
        let lengths = if add == 424 {
            [vec![64; 26], vec![32]].concat()
        } else {
            vec![4; 425]
        };
        if add == 424 {
            assert_eq!(before[8304..], after[8304..]);
            let payload: Vec<_> = vec![7u32; 424]
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect();
            assert_span(&c[6], 2480, &payload, &lengths);
            // At a block boundary, an already-owned retained entry is usable.
            let before = r[6].bytes();
            let (_, c) = observe("retained_second_block", &r, || {
                g.insert_sorted_run(SRC, &[8, 9, 9, 10])
            });
            expected.extend([8, 9, 9, 10]);
            assert_eq!(g.out_neighbors_vec(SRC), expected);
            assert_eq!(
                g.out_neighbors_vec(SIBLING),
                (1024..2048).collect::<Vec<_>>()
            );
            assert_eq!(g.num_edges(), 2055);
            assert_eq!(g.tree_state_stored(row.tree_bucket() as usize), 1028);
            g.verify_invariants();
            assert_only_payload(&c, 6);
            assert_eq!(r.each_ref().map(Probe::backing), backing);
            let after = r[6].bytes();
            assert_eq!(before[..8304], after[..8304]);
            assert_eq!(before[8320..], after[8320..]);
            let payload: Vec<_> = [8u32, 9, 9, 10]
                .iter()
                .flat_map(|d| d.to_le_bytes())
                .collect();
            assert_span(&c[6], 8304, &payload, &[16]);
        } else {
            // Whole-run fallback, even though both destination blocks exist.
            assert_eq!(before[8308..], after[8308..]);
            assert_eq!(
                c[6].writes.iter().map(|w| w.1.len()).collect::<Vec<_>>(),
                lengths
            );
            for (i, (off, bytes)) in c[6].writes.iter().enumerate() {
                assert_eq!(*off, if i < 424 { 2480 + i as u64 * 4 } else { 8304 });
                assert_eq!(bytes.as_slice(), &7u32.to_le_bytes());
            }
        }
    }
}

#[test]
fn resident_tree_depth_guard_precedes_all_writes() {
    let r = std::array::from_fn(|_| Probe::default());
    let limit = B * R_MAX;
    let n = limit - 2;
    let mut g: Graph = Graph::with_edge_list(bundle(&r), &vec![(0, 0); n - 1], 4).unwrap();
    g.insert_edge(0, 0); // bulk was not eagerly promoted
    g.set_promote_threshold(None); // a live tree must still enforce depth-1
    let row = g.get_row_pub(0);
    assert_ne!(row.tree_bucket(), NO_BUCKET);
    assert_eq!(g.tree_state_stored(row.tree_bucket() as usize), n as u32);
    let backing = r.each_ref().map(Probe::backing);
    let (result, c) = observe("resident_depth_over", &r, || {
        catch_unwind(AssertUnwindSafe(|| g.insert_sorted_run(0, &[1, 2, 3])))
    });
    assert!(result.is_err());
    assert_idle(&c);
    assert_eq!(g.get_row_pub(0), row);
    assert_eq!(g.num_edges(), n as u64);
    assert_eq!(g.out_neighbors_vec(0), vec![0; n]);
    assert_eq!(r.each_ref().map(Probe::backing), backing);
    g.verify_invariants();
    let before = r[6].bytes();
    let (_, c) = observe("resident_depth_fill", &r, || {
        g.insert_sorted_run(0, &[1, 1])
    });
    assert_eq!(g.num_edges(), limit as u64);
    assert_eq!(g.get_row_pub(0), row.with_live_degree(limit as u32));
    assert_eq!(
        g.tree_state_stored(row.tree_bucket() as usize),
        limit as u32
    );
    assert_eq!(g.out_neighbors_vec(0), [vec![0; n], vec![1, 1]].concat());
    g.verify_invariants();
    assert_only_payload(&c, 6);
    assert_eq!(r.each_ref().map(Probe::backing), backing);
    let base = 80 + (R_MAX - 1) as u64 * 4112 + (B - 2) as u64 * 4;
    let after = r[6].bytes();
    assert_eq!(before[..base as usize], after[..base as usize]);
    assert_eq!(before[base as usize + 8..], after[base as usize + 8..]);
    assert_span(&c[6], base, &[1, 0, 0, 0, 1, 0, 0, 0], &[8]);
    let (result, c) = observe("resident_depth_full", &r, || {
        catch_unwind(AssertUnwindSafe(|| g.insert_sorted_run(0, &[2])))
    });
    assert!(result.is_err());
    assert_idle(&c);
    assert_eq!(g.num_edges(), limit as u64);
    g.verify_invariants();
}

#[derive(Clone, Copy, Default)]
struct CanonicalEdge(u32);
impl FixedWidth for CanonicalEdge {
    const BYTES: usize = 4;
    fn read_from(bytes: &[u8]) -> Self {
        Self(Encoded::<4>::read_from(bytes).0)
    }
    fn write_to(&self, bytes: &mut [u8]) {
        Encoded::<4>(self.0).write_to(bytes);
    }
}
impl CsrEdge for CanonicalEdge {
    fn neighbor(&self) -> u32 {
        self.0
    }
    fn with_neighbor(self, dst: u32) -> Self {
        Self(dst & 511)
    }
}
impl CsrEdgeTombstone for CanonicalEdge {
    fn tombstone_edge() -> Self {
        Self(u32::MAX)
    }
    fn is_tombstone(&self) -> bool {
        self.0 == u32::MAX
    }
}

#[test]
fn resident_runs_apply_builders_but_store_raw_neighbors() {
    let incoming = [520u32, 1032, 4159, 0x0100_01ff];
    let expected = [8u32, 8, 63, 511];
    let payload: Vec<_> = expected.iter().flat_map(|d| d.to_le_bytes()).collect();
    let r = std::array::from_fn(|_| Probe::default());
    let mut tiny: Graph<CanonicalEdge> = Graph::new_with_segments(bundle(&r), 512, 8).unwrap();
    tiny.insert_edge(SRC, SRC);
    let base = tiny.get_row_pub(SRC).log_head() as u64 + 4;
    let (_, c) = observe("builder_tiny", &r, || {
        tiny.insert_sorted_run(SRC, &incoming)
    });
    assert_eq!(tiny.out_neighbors_vec(SRC), [8, 8, 8, 63, 511]);
    assert_eq!(tiny.num_edges(), 5);
    assert!(tiny.row_is_tiny(SRC));
    tiny.verify_invariants();
    assert_only_payload(&c, 2);
    assert_span(&c[2], base, &payload, &[16]);

    let (mut tree, r) = core::<CanonicalEdge>(1023, 170);
    tree.insert_edge(SRC, 500);
    tree.insert_edge(SRC, 501);
    let (_, c) = observe("builder_tree", &r, || {
        tree.insert_sorted_run(SRC, &incoming)
    });
    assert_rows(
        &tree,
        &(0..1023)
            .map(|d| d & 511)
            .chain([500, 501])
            .chain(expected)
            .collect::<Vec<_>>(),
    );
    assert_only_payload(&c, 6);
    assert_span(&c[6], 4196, &payload, &[16]);
}

#[test]
fn ingest_flush_batches_mixed_resident_rows() {
    let r = std::array::from_fn(|_| Probe::default());
    let mut g: Graph = Graph::new_with_segments(bundle(&r), 2048, 8).unwrap();
    for dst in 0..1025 {
        g.insert_edge(SRC, dst);
    }
    let tiny_src = 16;
    g.insert_edge(tiny_src, tiny_src); // reuses the chunk retired by SRC's spill
    assert!(g.row_is_tiny(tiny_src));
    assert_eq!(g.tiny_memory_bytes(), 32);
    let off = g.get_row_pub(tiny_src).log_head() as u64 + 4;
    let backing = r.each_ref().map(Probe::backing);
    let logs_before = r[2].bytes();
    let blocks_before = r[6].bytes();
    let maintenance = (g.rebalance_count(), g.promote_count());
    let mut ingest = IngestBuffer::new(0, 4);
    for (src, dst) in [
        (16, 500),
        (8, 9),
        (16, 16),
        (8, 500),
        (8, 8),
        (16, 20),
        (8, 9),
        (16, 16),
    ] {
        ingest.push(src, dst);
    }
    assert_eq!(ingest.len(), 8);
    assert_eq!(g.out_neighbors_vec(SRC), (0..1025).collect::<Vec<_>>());
    assert_eq!(g.out_neighbors_vec(tiny_src), [16]);
    let (_, c) = observe("resident_ingest", &r, || ingest.flush(&mut g));
    assert!(ingest.is_empty());
    assert_eq!(
        g.out_neighbors_vec(SRC),
        (0..1025).chain([8, 9, 9, 500]).collect::<Vec<_>>()
    );
    assert_eq!(g.out_neighbors_vec(tiny_src), [16, 16, 16, 20, 500]);
    assert_eq!(g.num_edges(), 1034);
    assert_eq!((g.rebalance_count(), g.promote_count()), maintenance);
    g.verify_invariants();
    assert_eq!(r.each_ref().map(Probe::backing), backing);
    for (i, t) in c.iter().enumerate() {
        assert_eq!((t.reads, t.mapped, t.grows), (0, 0, 0));
        if i != 2 && i != 6 {
            assert!(t.writes.is_empty());
        }
    }
    let logs_after = r[2].bytes();
    let blocks_after = r[6].bytes();
    assert_eq!(logs_before[..off as usize], logs_after[..off as usize]);
    assert_eq!(
        logs_before[off as usize + 16..],
        logs_after[off as usize + 16..]
    );
    assert_eq!(blocks_before[..4196], blocks_after[..4196]);
    assert_eq!(blocks_before[4212..], blocks_after[4212..]);
    let tiny_payload: Vec<_> = [16u32, 16, 20, 500]
        .iter()
        .flat_map(|d| d.to_le_bytes())
        .collect();
    let tree_payload: Vec<_> = [8u32, 9, 9, 500]
        .iter()
        .flat_map(|d| d.to_le_bytes())
        .collect();
    assert_span(&c[2], off, &tiny_payload, &[16]);
    assert_span(&c[6], 4196, &tree_payload, &[16]);
    let (_, c) = observe("resident_ingest_empty", &r, || ingest.flush(&mut g));
    assert_idle(&c);
}
