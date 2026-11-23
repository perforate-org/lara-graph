//! Export fixed streams and independently verified LARA insertion-only rows.
//! Usage: cargo run -p benches --release --example dgap_reference_input -- OUTPUT_DIR
use benches::adapters::{LaraAdapter, LaraConfig};
use benches::compare::{WORKLOADS, model_for, stream, stream_fingerprint};
use lara_graph::traits::DynamicCsr;
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("OUTPUT_DIR"));
    std::fs::create_dir_all(&dir)?;
    let selected = std::env::args().nth(2);
    if let Some(name) = &selected {
        assert!(WORKLOADS.iter().any(|w| w.name == name), "unknown workload");
    }
    for w in WORKLOADS
        .iter()
        .filter(|w| selected.as_ref().is_none_or(|name| w.name == name))
    {
        let edges = stream(w);
        let mut input = BufWriter::new(std::fs::File::create(
            dir.join(format!("{}.input", w.name)),
        )?);
        writeln!(input, "{} {}", w.n, edges.len())?;
        let mut graph = LaraAdapter::new(w.n, LaraConfig::default())?;
        for &(src, dst) in &edges {
            writeln!(input, "{src} {dst}")?;
            graph.insert_edge(src, dst);
        }
        input.flush()?;
        graph.verify();
        assert_eq!(graph.num_edges(), edges.len() as u64);
        let model = model_for(&edges, w.n);
        let mut rows = BufWriter::new(std::fs::File::create(
            dir.join(format!("{}.lara.rows", w.name)),
        )?);
        // Include padded vertices: both implementations round n to whole segments.
        for v in 0..graph.num_nodes() as u32 {
            let mut got = Vec::new();
            graph.for_each_neighbor(v, |dst| got.push(dst));
            got.sort_unstable();
            let mut want = model.rows.get(v as usize).cloned().unwrap_or_default();
            want.sort_unstable();
            assert_eq!(got, want, "{} row {v}", w.name);
            assert_eq!(graph.out_degree(v), got.len() as u64);
            write!(rows, "{v}")?;
            for dst in got {
                write!(rows, " {dst}")?;
            }
            writeln!(rows)?;
        }
        rows.flush()?;
        println!(
            "LARA_OK,{},config=full-default,fingerprint={},nodes={},edges={}",
            w.name,
            stream_fingerprint(&edges),
            graph.num_nodes(),
            graph.num_edges()
        );
    }
    Ok(())
}
