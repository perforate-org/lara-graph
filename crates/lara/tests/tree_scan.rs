//! Stored-prefix I/O contracts through public tree traversal and Memory.
//! Setup/mutation and semantic oracles are outside each observation interval.
#![cfg(feature = "std")]

use lara_graph::block_store::{BLOCK_STRIDE, LtbRawBlockStore};
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::labeled::{LabelId, LabeledLaraGraph};
use lara_graph::memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
use lara_graph::tree_csr::TreeCsrBucket;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default, Debug)]
struct Trace {
    reads: Vec<(u64, usize, bool)>, // offset, requested bytes, mapped fast path
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

type Graph = LaraGraph<Vertex, Edge, Probe>;

fn memories(blocks: &Probe) -> LaraMemories<Probe> {
    LaraMemories {
        vertices: Probe::default(),
        edges: Probe::default(),
        logs: Probe::default(),
        counts: Probe::default(),
        spans: Probe::default(),
        free_spans: Probe::default(),
        blocks: blocks.clone(),
        buckets: Probe::default(),
    }
}

fn observe<T>(tag: &str, blocks: &Probe, scan: impl FnOnce() -> T) -> (T, Trace) {
    let backing = (blocks.size(), blocks.memory.capacity_bytes());
    assert!(blocks.active.replace(Some(Trace::default())).is_none());
    let result = scan();
    let trace = blocks.active.take().unwrap(); // snapshot before all oracles
    let mapped = trace.reads.iter().filter(|r| r.2).count();
    let bytes: usize = trace.reads.iter().map(|r| r.1).sum();
    println!(
        "OBS,{tag},{},{mapped},{bytes},{},{},{}",
        trace.reads.len() - mapped,
        trace.writes,
        trace.write_bytes,
        trace.grows
    );
    for (i, (offset, len, mapped)) in trace.reads.iter().enumerate() {
        println!("RANGE,{tag},{i},{offset},{len},{mapped}");
    }
    assert_eq!((trace.writes, trace.grows), (0, 0), "scans are read-only");
    assert_eq!((blocks.size(), blocks.memory.capacity_bytes()), backing);
    (result, trace)
}

fn assert_lengths(trace: &Trace, lengths: &[usize]) {
    assert_eq!(trace.reads.iter().map(|r| r.1).collect::<Vec<_>>(), lengths);
}

#[test]
fn core_shrink_scan_reads_only_stored_prefix() {
    let blocks = Probe::default();
    let pairs: Vec<_> = (0..4095).map(|dst| (0, dst)).collect();
    let mut g = Graph::with_edge_list(memories(&blocks), &pairs, 8192).unwrap();
    g.insert_edge(0, 4095); // bulk alone does not promote
    assert_eq!(g.tree_bucket_count(), 1);
    let original_slots = g.memory_slots();
    for dst in (600..4096).rev() {
        assert!(g.remove_edge(0, dst));
    }
    let (got, trace) = observe("core_shrink600", &blocks, || g.out_neighbors_vec(0));
    assert_eq!(got, (0..600).collect::<Vec<_>>());
    assert_eq!(g.num_edges(), 600);
    assert_eq!(g.tree_bucket_count(), 1); // above default demotion at 512
    g.verify_invariants();
    assert_lengths(&trace, &[2400]);
    assert_eq!(g.memory_slots(), original_slots);
    let (all, trace) = observe("core_full_graph600", &blocks, || {
        let mut got = Vec::new();
        g.scan_all(|dst| got.push(dst));
        got
    });
    assert_eq!(all, got);
    assert_lengths(&trace, &[2400]);

    let growth: &[(u32, &[usize])] = &[
        (1023, &[4092]),
        (1024, &[4096]),
        (1025, &[4096, 4]),
        (2048, &[4096, 4096]),
        (2049, &[4096, 4096, 4]),
        (4096, &[4096, 4096, 4096, 4096]),
        (4097, &[4096, 4096, 4096, 4096, 4]),
    ];
    let mut live = 600;
    for &(n, lengths) in growth {
        for dst in live..n {
            g.insert_edge(0, dst);
        }
        live = n;
        check_core(&g, &blocks, &format!("core_regrow{n}"), n, lengths);
        if n <= 4096 {
            assert_eq!(g.memory_slots(), original_slots);
        }
    }
    let shrink: &[(u32, &[usize])] = &[
        (2049, &[4096, 4096, 4]),
        (2048, &[4096, 4096]),
        (1025, &[4096, 4]),
        (1024, &[4096]),
        (1023, &[4092]),
        (513, &[2052]),
    ];
    for &(n, lengths) in shrink {
        for dst in (n..live).rev() {
            assert!(g.remove_edge(0, dst));
        }
        live = n;
        check_core(&g, &blocks, &format!("core_shrink{n}"), n, lengths);
    }
    assert!(g.remove_edge(0, 512));
    assert_eq!(g.tree_bucket_count(), 0);
    check_core(&g, &blocks, "core_demoted512", 512, &[]);
    for dst in 512..1024 {
        g.insert_edge(0, dst);
    }
    assert_eq!(g.tree_bucket_count(), 1);
    check_core(&g, &blocks, "core_repromoted1024", 1024, &[4096]);
}

fn check_core(g: &Graph, blocks: &Probe, tag: &str, n: u32, lengths: &[usize]) {
    let (got, trace) = observe(tag, blocks, || g.out_neighbors_vec(0));
    assert_eq!(got, (0..n).collect::<Vec<_>>());
    assert_eq!(g.num_edges(), u64::from(n));
    assert_eq!(g.out_degree(0), u64::from(n));
    g.verify_invariants();
    assert_lengths(&trace, lengths);
    assert!(trace.reads.iter().all(|r| r.2 == (r.1 == 4096)));
}

#[test]
fn core_nested_scan_preserves_noncontiguous_roots() {
    let blocks = Probe::default();
    let pairs: Vec<_> = std::iter::repeat_n((0, 0), 1023)
        .chain(std::iter::repeat_n((1, 1), 1023))
        .collect();
    let mut g = Graph::with_edge_list(memories(&blocks), &pairs, 64).unwrap();
    g.insert_edge(0, 0); // first root gets block 0
    g.insert_edge(1, 1); // second root gets block 1
    g.insert_edge(0, 7); // first root extends with block 2
    assert_eq!(g.tree_bucket_count(), 2);
    let ((got, nested), trace) = observe("core_nested1025_and1024", &blocks, || {
        let mut got = Vec::new();
        let mut nested = Vec::new();
        g.for_each_neighbor(0, |dst| {
            if got.is_empty() {
                nested = g.out_neighbors_vec(1);
            }
            got.push(dst);
        });
        (got, nested)
    });
    assert_eq!(nested, vec![1; 1024]);
    let mut want = vec![0; 1024];
    want.push(7);
    assert_eq!(got, want);
    assert_eq!(g.num_edges(), 2049);
    g.verify_invariants();
    assert_lengths(&trace, &[4096, 4096, 4]);
    assert_eq!(trace.reads[2].0 - trace.reads[0].0, 2 * BLOCK_STRIDE);
    assert!(trace.reads[0].2 && trace.reads[1].2 && !trace.reads[2].2);
}

#[test]
fn labeled_tree_scans_preserve_shrink_regrowth_and_sibling() {
    let blocks = Probe::default();
    let label = LabelId::new(0);
    let other = LabelId::new(1);
    let pairs: Vec<_> = (0..4095)
        .map(|dst| (0, dst, label))
        .chain(std::iter::repeat_n((0, 1, other), 1023))
        .collect();
    let mut g = LabeledLaraGraph::with_edge_list(memories(&blocks), &pairs, 4096).unwrap();
    g.insert_edge(0, 4095, label);
    g.insert_edge(0, 1, other);
    assert_eq!(g.tree_bucket_count(), 2);
    for dst in (600..4096).rev() {
        assert!(g.remove_edge(0, dst, label));
    }
    check_labeled(&g, &blocks, "shrink600", 600, &[2400]);
    for dst in 600..1025 {
        g.insert_edge(0, dst, label);
    }
    check_labeled(&g, &blocks, "regrow1025", 1025, &[4096, 4]);
    for dst in (513..1025).rev() {
        assert!(g.remove_edge(0, dst, label));
    }
    assert!(g.bucket_is_tree(0, label));
    check_labeled(&g, &blocks, "shrink513", 513, &[2052]);
    assert!(g.remove_edge(0, 512, label));
    assert!(!g.bucket_is_tree(0, label));
    assert_eq!(g.tree_bucket_count(), 1);
    check_labeled(&g, &blocks, "demoted512", 512, &[]);
    for dst in 512..1024 {
        g.insert_edge(0, dst, label);
    }
    assert!(g.bucket_is_tree(0, label));
    assert_eq!(g.tree_bucket_count(), 2);
    check_labeled(&g, &blocks, "repromoted1024", 1024, &[4096]);
}

fn check_labeled(
    g: &LabeledLaraGraph<Probe>,
    blocks: &Probe,
    tag: &str,
    n: u32,
    lengths: &[usize],
) {
    let (got, trace) = observe(&format!("label0_{tag}"), blocks, || {
        let mut got = Vec::new();
        g.for_each_edge_of_label(0, LabelId::new(0), |dst| got.push(dst));
        got
    });
    assert_eq!(got, (0..n).collect::<Vec<_>>());
    assert_lengths(&trace, lengths);
    let (got, trace) = observe(&format!("labels_{tag}"), blocks, || {
        g.out_labeled_vec(0)
            .into_iter()
            .map(|(dst, label)| (dst, label.as_u16()))
            .collect::<Vec<_>>()
    });
    let want: Vec<_> = (0..n)
        .map(|dst| (dst, 0))
        .chain(std::iter::repeat_n((1, 1), 1024))
        .collect();
    assert_eq!(got, want);
    let mut all_lengths = lengths.to_vec();
    all_lengths.push(4096);
    assert_lengths(&trace, &all_lengths);
    assert!(trace.reads.iter().all(|r| r.2 == (r.1 == 4096)));
    assert_eq!(g.num_edges(), u64::from(n) + 1024);
    g.verify_invariants();
}

#[test]
fn detached_empty_tree_skips_retained_block_and_can_regrow() {
    let blocks = Probe::default();
    let mut g = Graph::new(memories(&blocks), 64).unwrap();
    let id = g.mint_block().unwrap();
    let tree = g.tree_state_push(vec![id], 1) as usize;
    // Public staging-state operations, not a mutation of an attached row.
    g.tree_state_remove_shift(tree, 0);
    assert_eq!(g.tree_state_stored(tree), 0);
    let (got, trace) = observe("detached_empty", &blocks, || {
        let mut got = Vec::new();
        g.for_each_in_tree(tree, |dst| got.push(dst));
        got
    });
    assert!(got.is_empty());
    assert_lengths(&trace, &[]);
    g.tree_state_insert(tree, 42);
    let (got, trace) = observe("detached_regrow1", &blocks, || {
        let mut got = Vec::new();
        g.for_each_in_tree(tree, |dst| got.push(dst));
        got
    });
    assert_eq!(got, vec![42]);
    assert_lengths(&trace, &[4]);
    g.tree_state_release(tree);
    assert_eq!(g.num_edges(), 0);
    g.verify_invariants();
}

#[test]
fn standalone_scans_preserve_ordinals_and_trim_the_tail() {
    let cases: &[(usize, &[usize])] = &[
        (0, &[]),
        (1, &[4]),
        (1023, &[4092]),
        (1024, &[4096]),
        (1025, &[4096, 4]),
        (2048, &[4096, 4096]),
        (2049, &[4096, 4096, 4]),
    ];
    for &(n, lengths) in cases {
        let blocks = Probe::default();
        let mut bucket = TreeCsrBucket::new(LtbRawBlockStore::new(blocks.clone()).unwrap());
        let mut want: Vec<u32> = [0, 7, 7, u32::MAX, 0x0102_0304]
            .into_iter()
            .cycle()
            .take(n)
            .collect();
        bucket.promote_from_slice(&want);
        let (got, trace) = observe(&format!("standalone{n}"), &blocks, || {
            let mut got = Vec::new();
            bucket.for_each_ascending(|slot, value| got.push((slot, value)));
            got
        });
        let expected: Vec<_> = want
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        assert_eq!(got, expected);
        assert_eq!(bucket.stored_slots() as usize, n);
        assert_lengths(&trace, lengths);
        assert!(trace.reads.iter().all(|r| !r.2)); // existing normal-read path
        if n == 2049 {
            assert_eq!(bucket.remove_at(1023), Some(want.remove(1023)));
            let shrink_cases: &[(usize, &[usize], usize)] = &[
                (2048, &[4096, 4096], 2),
                (1025, &[4096, 4], 2),
                (1024, &[4096], 1),
                (1, &[4], 1),
                (0, &[], 0),
            ];
            for &(remaining, lengths, roots) in shrink_cases {
                while want.len() > remaining {
                    let pos = want.len() - 1;
                    assert_eq!(bucket.remove_at(pos as u32), want.pop());
                }
                let (got, trace) =
                    observe(&format!("standalone_shrink{remaining}"), &blocks, || {
                        let mut got = Vec::new();
                        bucket.for_each_ascending(|slot, value| got.push((slot, value)));
                        got
                    });
                let expected: Vec<_> = want
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as u32, v))
                    .collect();
                assert_eq!(got, expected);
                assert_eq!(bucket.root_len(), roots); // pre-existing release policy
                assert_lengths(&trace, lengths);
            }
            bucket.insert(42);
            let (got, trace) = observe("standalone_regrow1", &blocks, || {
                let mut got = Vec::new();
                bucket.for_each_ascending(|slot, value| got.push((slot, value)));
                got
            });
            assert_eq!(got, vec![(0, 42)]);
            assert_lengths(&trace, &[4]);
        }
    }
}
