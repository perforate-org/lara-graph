//! LARA Orkut subset/full byte counts through the Probe seam.
//! Same input and procedure as the C++ EL build (compare-count.cpp).
//! Usage: cargo run -p benches --release --example lara_orkut -- <input> [warmup_edges]
//! Prints LARA_COUNT lines (phase,category,bytes).
use benches::probe::{Probe, ProbeHandle, RegionTraffic};
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::memory::VecMemory;
use std::io::{BufRead, BufReader};

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("input path");
    let warmup: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let f = std::fs::File::open(path).expect("open input");
    let mut lines = BufReader::new(f).lines();
    let header = lines.next().expect("header").expect("read header");
    let mut hit = header.split_whitespace();
    let n: u32 = hit.next().expect("n").parse().expect("parse n");
    let m: usize = hit.next().expect("m").parse().expect("parse m");
    assert!(warmup <= m, "warmup exceeds edges");
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
    let mut g: LaraGraph<Vertex, Edge, Probe<VecMemory>> = LaraGraph::new(
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
    )
    .expect("lara init");
    g.set_promote_threshold(Some(lara_graph::graph::T_PROMOTE));
    g.set_use_free_spans(true);
    let refs: Vec<(&'static str, &ProbeHandle)> = handles.iter().map(|(n, h)| (*n, h)).collect();
    let mut inserted = 0usize;
    let t0 = std::time::Instant::now();
    for (i, line) in lines.enumerate() {
        let line = line.expect("read edge");
        let mut sp = line.split_whitespace();
        let src: u32 = sp.next().expect("src").parse().expect("parse src");
        let dst: u32 = sp.next().expect("dst").parse().expect("parse dst");
        g.insert_edge(src, dst);
        inserted += 1;
        if inserted == warmup {
            let t = RegionTraffic::from_handles(&refs);
            println!(
                "LARA_COUNT,orkut,warmup,write_bytes={},edge_write={},log_write={},rebalance=N/A",
                t.total.write_bytes,
                t.regions
                    .iter()
                    .find(|(n, _)| *n == "edges")
                    .map_or(0, |(_, t)| t.write_bytes),
                t.regions
                    .iter()
                    .find(|(n, _)| *n == "logs")
                    .map_or(0, |(_, t)| t.write_bytes),
            );
            for (_, h) in &handles {
                h.reset();
            }
        }
        let _ = i;
    }
    assert_eq!(inserted, m, "edge count mismatch");
    let build_secs = t0.elapsed().as_secs_f64();
    assert_eq!(g.num_edges(), m as u64, "build edges");
    g.verify_invariants();
    let t = RegionTraffic::from_handles(&refs);
    println!(
        "LARA_COUNT,orkut,build,write_bytes={},edge_write={},log_write={},read_bytes={},reads={},mapped_reads={},wall_secs={:.1}",
        t.total.write_bytes,
        t.regions
            .iter()
            .find(|(n, _)| *n == "edges")
            .map_or(0, |(_, t)| t.write_bytes),
        t.regions
            .iter()
            .find(|(n, _)| *n == "logs")
            .map_or(0, |(_, t)| t.write_bytes),
        t.total.read_bytes + t.total.mapped_read_bytes,
        t.total.reads,
        t.total.mapped_reads,
        build_secs,
    );
    for (name, rt) in &t.regions {
        println!(
            "LARA_REGION,orkut,build,{name},write_bytes={},read_bytes={},mapped_bytes={}",
            rt.write_bytes, rt.read_bytes, rt.mapped_read_bytes,
        );
    }
    for (_, h) in &handles {
        h.reset();
    }
    let mut checksum = 0u64;
    for v in 0..n {
        g.for_each_neighbor(v, |x| checksum += x as u64);
    }
    let t = RegionTraffic::from_handles(&refs);
    println!(
        "LARA_COUNT,orkut,scan,read_bytes={},reads={},mapped_reads={},checksum={checksum}",
        t.total.read_bytes + t.total.mapped_read_bytes,
        t.total.reads,
        t.total.mapped_reads,
    );
    println!("LARA_COUNT_OK");
}
