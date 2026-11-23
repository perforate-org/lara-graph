//! Full-default LARA build/scan byte counts through the Probe seam.
//! Counterpart to the C++ counting build (compare-count.cpp).
//! Usage: cargo run -p benches --release --example dgap_byte_count
//! Prints LARA_COUNT,<workload>,<phase>,<category>,<bytes> lines.
use benches::compare::{WORKLOADS, model_for, stream, stream_fingerprint};
use benches::probe::{Probe, ProbeHandle, RegionTraffic};
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::memory::VecMemory;

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
const ARM: &str = "full-default";

fn main() {
    for w in WORKLOADS.iter() {
        let s = stream(w);
        let fp = stream_fingerprint(&s);
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
            w.n,
        )
        .expect("lara init");
        // full-default: promotion on, free spans on.
        g.set_promote_threshold(Some(lara_graph::graph::T_PROMOTE));
        g.set_use_free_spans(true);
        for &(src, dst) in &s {
            g.insert_edge(src, dst);
        }
        let refs: Vec<(&'static str, &ProbeHandle)> =
            handles.iter().map(|(n, h)| (*n, h)).collect();
        let t = RegionTraffic::from_handles(&refs);
        let edge_write = t
            .regions
            .iter()
            .find(|(n, _)| *n == "edges")
            .map_or(0, |(_, t)| t.write_bytes);
        let log_write = t
            .regions
            .iter()
            .find(|(n, _)| *n == "logs")
            .map_or(0, |(_, t)| t.write_bytes);
        println!(
            "LARA_COUNT,{},{ARM},build,edge_write={edge_write},log_write={log_write},total_write={},edge_read={},fp={fp}",
            w.name,
            t.total.write_bytes,
            t.total.read_bytes + t.total.mapped_read_bytes,
        );
        for (name, rt) in &t.regions {
            println!(
                "LARA_REGION,{},{ARM},build,{name},writes={},write_bytes={},reads={},read_bytes={},mapped={},mapped_bytes={}",
                w.name,
                rt.writes,
                rt.write_bytes,
                rt.reads,
                rt.read_bytes,
                rt.mapped_reads,
                rt.mapped_read_bytes,
            );
        }
        let model = model_for(&s, w.n);
        assert_eq!(g.num_edges(), model.edges, "build edges");
        g.verify_invariants();
        // scan phase
        for (_, h) in &handles {
            h.reset();
        }
        let mut checksum = 0u64;
        for v in 0..w.n {
            g.for_each_neighbor(v, |x| checksum += x as u64);
        }
        assert_eq!(
            checksum,
            s.iter().map(|&(_, dst)| u64::from(dst)).sum::<u64>()
        );
        let t = RegionTraffic::from_handles(&refs);
        println!(
            "LARA_COUNT,{},{ARM},scan,read_bytes={},checksum={checksum}",
            w.name,
            t.total.read_bytes + t.total.mapped_read_bytes,
        );
        drop(g);
    }
    println!("LARA_COUNT_OK");
}
