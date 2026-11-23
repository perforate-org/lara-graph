//! Logical memory-traffic comparison through the Memory seam (roadmap §3-B):
//! LARA vs the DGAP port on the same fixed workloads and phases.
//!
//! NOT wall time, NOT cache behavior, NOT a PMEM evaluation. Distinct-byte
//! sets are the logical union of seam offsets, not dirty pages or misses.
//!
//! Output: one CSV line per (workload, arm, phase) plus per-region lines.
//! Phases: build = arrival-order insertion of the full stream,
//! churn = fixed pairs, endpoint removal then reinsertion each,
//! scan = one requested-row pass (reads only).

use benches::compare::{WORKLOADS, model_for, stream, stream_fingerprint, update_k, update_pairs};
use benches::probe::{Probe, ProbeHandle, ProbedDgapBridge, RegionTraffic};
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::memory::VecMemory;
use lara_graph::traits::DynamicCsr;

const REGION_NAMES: [&str; 8] = [
    "vertices",
    "edges",
    "logs",
    "counts",
    "spans",
    "free_spans",
    "blocks",
    "buckets",
];

struct LaraArm {
    graph: LaraGraph<Vertex, Edge, Probe<VecMemory>>,
    handles: Vec<(&'static str, ProbeHandle)>,
}

/// Slab-only LARA ablation (no tree promotion), free spans on.
/// Kept fixed for comparison with the first run; not the default/full LARA.
fn lara_arm(n: u32) -> Result<LaraArm, lara_graph::GrowFailed> {
    let probes: Vec<Probe<VecMemory>> = REGION_NAMES
        .iter()
        .map(|_| Probe::new(VecMemory::new()))
        .collect();
    let handles: Vec<(&'static str, ProbeHandle)> = REGION_NAMES
        .iter()
        .zip(&probes)
        .map(|(name, p)| (*name, p.handle()))
        .collect();
    let mut it = probes.into_iter();
    let [
        vertices,
        edges,
        logs,
        counts,
        spans,
        free_spans,
        blocks,
        buckets,
    ] = [(); 8].map(|_| it.next().expect("8 regions"));
    let graph = LaraGraph::new(
        LaraMemories {
            vertices,
            edges,
            logs,
            counts,
            spans,
            free_spans,
            blocks,
            buckets,
        },
        n,
    )?;
    let mut graph = graph;
    graph.set_promote_threshold(None);
    graph.set_use_free_spans(true);
    Ok(LaraArm { graph, handles })
}

fn lara_traffic(lara: &LaraArm) -> RegionTraffic {
    let refs: Vec<(&'static str, &ProbeHandle)> =
        lara.handles.iter().map(|(n, h)| (*n, h)).collect();
    RegionTraffic::from_handles(&refs)
}

struct DgapArm {
    graph: dgap::DgapGraph<u32, ProbedDgapBridge<Probe<VecMemory>>>,
    handle: ProbeHandle,
}

impl DynamicCsr for DgapArm {
    fn num_nodes(&self) -> u64 {
        self.graph.num_nodes()
    }
    fn num_edges(&self) -> u64 {
        self.graph.num_edges() as u64
    }
    fn out_degree(&self, v: u32) -> u64 {
        self.graph.out_degree(v) as u64
    }
    fn insert_edge(&mut self, src: u32, dst: u32) {
        self.graph.insert_edge(src, dst)
    }
    fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, f: F) {
        self.graph.for_each_neighbor(v, f)
    }
    fn scan_all<F: FnMut(u32)>(&self, f: F) {
        self.graph.scan_all(f)
    }
}

fn dgap_arm(n: u32) -> DgapArm {
    let probe = Probe::new(VecMemory::new());
    let handle = probe.handle();
    let graph = dgap::DgapGraph::new(ProbedDgapBridge(probe), n, true).expect("dgap init");
    DgapArm { graph, handle }
}

fn reset_lara(lara: &LaraArm) {
    for (_, h) in &lara.handles {
        h.reset();
    }
}

fn verify_rows<G: DynamicCsr>(g: &G, s: &[(u32, u32)], n: u32, tag: &str) {
    let model = model_for(s, n);
    assert_eq!(g.num_edges(), model.edges, "{tag}: edges");
    for v in 0..n {
        let mut got = Vec::new();
        g.for_each_neighbor(v, |x| got.push(x));
        got.sort_unstable();
        let mut want = model.rows[v as usize].clone();
        want.sort_unstable();
        assert_eq!(got, want, "{tag}: row {v}");
    }
}

fn row(workload: &str, arm: &str, phase: &str, fp: u64, traffic: &RegionTraffic, checksum: u64) {
    let t = &traffic.total;
    println!(
        "TRAFFIC,{workload},{arm},{phase},{fp},reads={},read_bytes={},mapped_reads={},mapped_read_bytes={},writes={},write_bytes={},distinct_read={},distinct_write={},grows={},checksum={checksum}",
        t.reads,
        t.read_bytes,
        t.mapped_reads,
        t.mapped_read_bytes,
        t.writes,
        t.write_bytes,
        t.distinct_read_bytes,
        t.distinct_write_bytes,
        t.grows
    );
    for (name, t) in &traffic.regions {
        println!(
            "REGION,{workload},{arm},{phase},{name},reads={},read_bytes={},writes={},write_bytes={},distinct_read={},distinct_write={},grows={},mapped_reads={},mapped_read_bytes={}",
            t.reads,
            t.read_bytes,
            t.writes,
            t.write_bytes,
            t.distinct_read_bytes,
            t.distinct_write_bytes,
            t.grows,
            t.mapped_reads,
            t.mapped_read_bytes
        );
    }
}

fn scan_checksum<G: DynamicCsr>(g: &G, n: u32) -> u64 {
    let mut sum = 0u64;
    for v in 0..n {
        g.for_each_neighbor(v, |x| sum = sum.wrapping_add(x as u64));
    }
    sum
}

fn main() {
    println!(
        "HEADER,traffic_compare,seam observation per phase; not wall time, not cache, not PMEM"
    );
    for w in WORKLOADS.iter() {
        let s = stream(w);
        let fp = stream_fingerprint(&s);
        let pairs = update_pairs(&s, update_k(w.m));
        println!(
            "INPUT,{},vertices={},edges={},pairs={},fingerprint={fp}",
            w.name,
            w.n,
            s.len(),
            pairs.len()
        );
        let model = model_for(&s, w.n);

        // LARA arm: build (observed) -> churn (observed) -> scan (observed) -> verify.
        let mut lara = lara_arm(w.n).expect("lara init");
        reset_lara(&lara);
        for &(src, dst) in &s {
            lara.graph.insert_edge(src, dst);
        }
        let t = lara_traffic(&lara);
        row(w.name, "lara-slab-only", "build", fp, &t, 0);
        assert_eq!(lara.graph.num_edges(), model.edges, "lara build edges");

        reset_lara(&lara);
        for &(src, dst) in &pairs {
            assert!(lara.graph.remove_edge(src, dst), "lara delete must succeed");
            lara.graph.insert_edge(src, dst);
        }
        let t = lara_traffic(&lara);
        row(w.name, "lara-slab-only", "churn", fp, &t, 0);
        assert_eq!(lara.graph.num_edges(), model.edges, "lara churn edges");

        reset_lara(&lara);
        let checksum = scan_checksum(&lara.graph, w.n);
        let t = lara_traffic(&lara);
        row(w.name, "lara-slab-only", "scan", fp, &t, checksum);
        assert_eq!(
            checksum,
            s.iter().map(|&(_, dst)| u64::from(dst)).sum::<u64>()
        );
        verify_rows(&lara.graph, &s, w.n, "lara");
        drop(lara);

        // DGAP arm: same phases on the same stream.
        let mut dgap = dgap_arm(w.n);
        dgap.handle.reset();
        for &(src, dst) in &s {
            dgap.insert_edge(src, dst);
        }
        let t = RegionTraffic::from_handles(&[("dgap", &dgap.handle)]);
        row(w.name, "dgap-rust", "build", fp, &t, 0);
        assert_eq!(
            dgap.graph.num_edges() as u64,
            model.edges,
            "dgap build edges"
        );

        dgap.handle.reset();
        for &(src, dst) in &pairs {
            assert!(dgap.graph.remove_edge(src, dst), "dgap delete must succeed");
            dgap.insert_edge(src, dst);
        }
        let t = RegionTraffic::from_handles(&[("dgap", &dgap.handle)]);
        row(w.name, "dgap-rust", "churn", fp, &t, 0);
        assert_eq!(
            dgap.graph.num_edges() as u64,
            model.edges,
            "dgap churn edges"
        );

        dgap.handle.reset();
        let checksum = scan_checksum(&dgap, w.n);
        let t = RegionTraffic::from_handles(&[("dgap", &dgap.handle)]);
        row(w.name, "dgap-rust", "scan", fp, &t, checksum);
        assert_eq!(
            checksum,
            s.iter().map(|&(_, dst)| u64::from(dst)).sum::<u64>()
        );
        verify_rows(&dgap, &s, w.n, "dgap");
        dgap.graph.verify_invariants();
        drop(dgap);
    }
    println!("TRAFFIC_OK,workloads={}", WORKLOADS.len());
}
