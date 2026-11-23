//! Insert-only backend API counts, split by region. Setup and verification
//! are excluded; these are not durations or a decomposition of insert cost.
//! Time the uninstrumented adapters with Criterion instead.
//! `cargo test -p benches --test roundtrip_probe -- --nocapture`.
use benches::coo_uniform_stream;
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::memory::{GrowFailed, Memory, VecMemory};
use std::cell::Cell;
use std::rc::Rc;

/// Per-region op counters. Clone copies values, not shared counter handles.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct RegionCounters {
    reads: Cell<u64>,
    read_bytes: Cell<u64>,
    mapped_reads: Cell<u64>,
    writes: Cell<u64>,
    write_bytes: Cell<u64>,
    sizes: Cell<u64>,
    grows: Cell<u64>,
}

impl RegionCounters {
    fn reset(&self) {
        for counter in [
            &self.reads,
            &self.read_bytes,
            &self.mapped_reads,
            &self.writes,
            &self.write_bytes,
            &self.sizes,
            &self.grows,
        ] {
            counter.set(0);
        }
    }
}

/// Counting shell around any [`Memory`]; behavior verbatim, counts only.
#[derive(Clone)]
struct CountingMemory<M: Memory> {
    inner: M,
    c: Rc<RegionCounters>,
}

impl<M: Memory> CountingMemory<M> {
    fn wrap(inner: M) -> (Self, Rc<RegionCounters>) {
        let c = Rc::new(RegionCounters::default());
        (
            Self {
                inner,
                c: Rc::clone(&c),
            },
            c,
        )
    }
}

impl<M: Memory> Memory for CountingMemory<M> {
    fn read(&self, offset: u64, buf: &mut [u8]) {
        self.c.reads.set(self.c.reads.get() + 1);
        self.c
            .read_bytes
            .set(self.c.read_bytes.get() + buf.len() as u64);
        self.inner.read(offset, buf);
    }
    fn read_mapped(&self, offset: u64, buf: &mut [u8]) {
        self.c.mapped_reads.set(self.c.mapped_reads.get() + 1);
        self.c
            .read_bytes
            .set(self.c.read_bytes.get() + buf.len() as u64);
        self.inner.read_mapped(offset, buf);
    }
    fn write(&self, offset: u64, buf: &[u8]) {
        self.c.writes.set(self.c.writes.get() + 1);
        self.c
            .write_bytes
            .set(self.c.write_bytes.get() + buf.len() as u64);
        self.inner.write(offset, buf);
    }
    fn size(&self) -> u64 {
        self.c.sizes.set(self.c.sizes.get() + 1);
        self.inner.size()
    }
    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        self.c.grows.set(self.c.grows.get() + 1);
        self.inner.grow(pages)
    }
    fn fence(&self) {
        self.inner.fence();
    }
}

type CM = CountingMemory<VecMemory>;

fn fresh_counted() -> (LaraMemories<CM>, Vec<Rc<RegionCounters>>) {
    let (vertices, cv) = CountingMemory::wrap(VecMemory::new());
    let (edges, ce) = CountingMemory::wrap(VecMemory::new());
    let (logs, cl) = CountingMemory::wrap(VecMemory::new());
    let (counts, cc) = CountingMemory::wrap(VecMemory::new());
    let (spans, cs) = CountingMemory::wrap(VecMemory::new());
    let (free_spans, cf) = CountingMemory::wrap(VecMemory::new());
    let (blocks, cb) = CountingMemory::wrap(VecMemory::new());
    let (buckets, ck) = CountingMemory::wrap(VecMemory::new());
    let mem = LaraMemories {
        vertices,
        edges,
        logs,
        counts,
        spans,
        free_spans,
        blocks,
        buckets,
    };
    (mem, vec![cv, ce, cl, cc, cs, cf, cb, ck])
}

fn report(tag: &str, m: u64, names: &[&str], cs: &[RegionCounters], extra: &str) {
    println!("{tag}: insert-only backend calls, edges={m} {extra}");
    println!(
        "  {:>10} {:>8} {:>10} {:>8} {:>10} {:>7} {:>6}",
        "region", "reads/e", "rbytes/e", "writes/e", "wbytes/e", "size/e", "grow"
    );
    for (nm, c) in names.iter().zip(cs.iter()) {
        let pe = |v: u64| v as f64 / m as f64;
        println!(
            "  {:>10} {:>8.3} {:>10.1} {:>8.3} {:>10.1} {:>7.3} {:>6}",
            nm,
            pe(c.reads.get() + c.mapped_reads.get()),
            pe(c.read_bytes.get()),
            pe(c.writes.get()),
            pe(c.write_bytes.get()),
            pe(c.sizes.get()),
            c.grows.get()
        );
    }
    let tot_r: u64 = cs
        .iter()
        .map(|c| c.reads.get() + c.mapped_reads.get())
        .sum();
    let tot_w: u64 = cs.iter().map(|c| c.writes.get()).sum();
    let tot_s: u64 = cs.iter().map(|c| c.sizes.get() + c.grows.get()).sum();
    println!(
        "  {:>10} {:>8.3} {:>10} {:>8.3} {:>10} {:>7.3}",
        "TOTAL",
        tot_r as f64 / m as f64,
        "",
        tot_w as f64 / m as f64,
        "",
        tot_s as f64 / m as f64
    );
    println!(
        "  API calls/e (reads+writes+sizes+grows) = {:.3}",
        (tot_r + tot_w + tot_s) as f64 / m as f64
    );
}

fn run(tag: &str, stream: &[(u32, u32)], n: u32) {
    let (mem, cs) = fresh_counted();
    let mut g: LaraGraph<Vertex, Edge, CM> = LaraGraph::new(mem, n).unwrap();
    for c in &cs {
        c.reset();
    }
    for &(a, b) in stream {
        g.insert_edge(a, b);
    }
    let snapshot: Vec<_> = cs.iter().map(|c| c.as_ref().clone()).collect();
    assert_eq!(g.num_edges(), stream.len() as u64);
    g.verify_invariants();
    let extra = format!(
        "tiny_rows={} non_tiny_touched={} rb={} pr={} log_chunks={}",
        g.tiny_row_count(),
        stream
            .iter()
            .map(|&(a, _)| a)
            .collect::<std::collections::HashSet<_>>()
            .len() as u64
            - g.tiny_row_count(),
        g.rebalance_count(),
        g.promote_count(),
        g.log_allocated_chunks()
    );
    let names = [
        "vertices", "edges", "logs", "counts", "spans", "free", "blocks", "buckets",
    ];
    report(tag, stream.len() as u64, &names, &snapshot, &extra);
}

#[test]
fn counter_interval_excludes_setup_and_verification() {
    let (mem, cs) = fresh_counted();
    let mut g: LaraGraph<Vertex, Edge, CM> = LaraGraph::new(mem, 64).unwrap();
    assert!(cs.iter().map(|c| c.writes.get()).sum::<u64>() > 0);
    for c in &cs {
        c.reset();
    }
    g.insert_edge(0, 7);
    let snapshot: Vec<_> = cs.iter().map(|c| c.as_ref().clone()).collect();
    assert_eq!(snapshot.iter().map(|c| c.writes.get()).sum::<u64>(), 2);
    assert_eq!(snapshot[2].write_bytes.get(), 5); // tiny reservation byte + neighbor
    g.verify_invariants();
    assert!(
        cs.iter()
            .zip(&snapshot)
            .any(|(current, old)| current.reads.get() > old.reads.get())
    );
    assert_eq!(snapshot.iter().map(|c| c.reads.get()).sum::<u64>(), 0);
}

#[test]
fn roundtrip_probe() {
    // A. tiny-pure: strided degree exactly 4 (spills=0 expected).
    let tiny: Vec<(u32, u32)> = (0..40_000u32)
        .map(|i| ((i * 7 + 3) % 10_000, (i * 13 + 5) % 10_000))
        .collect();
    run("tiny_strided_deg4", &tiny, 10_000);
    // B. slab-pure: strided degree exactly 16 (all rows spill once, no promote).
    let slab: Vec<(u32, u32)> = (0..160_000u32)
        .map(|i| ((i * 7 + 3) % 10_000, (i * 13 + 5) % 160_000))
        .collect();
    run("slab_strided_deg16", &slab, 10_000);
    // C. dispersed uniform-random headline anchor (mixed tiny + spills).
    let disp = coo_uniform_stream(10_000, 40_000, 1234);
    run("dispersed_uniform10k", &disp, 10_000);
    // D. hub-10k growth (log/promote/rebalance mixed; aggregate only).
    let hub: Vec<(u32, u32)> = (0..10_000u32).map(|d| (1, d)).collect();
    run("hub10k", &hub, 20_480);
}
