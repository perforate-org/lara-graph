//! Orkut log-residency probe: row-state histogram after insertion.
//! Usage: cargo run -p benches --release --example orkut_residency -- <input>
//! Prints RESIDENCY lines (read-only row inspection via public APIs).
use benches::adapters::{LaraAdapter, LaraConfig};
use lara_graph::traits::DynamicCsr;
use std::io::{BufRead, BufReader};

fn main() {
    let path = std::env::args().nth(1).expect("input path");
    let f = std::fs::File::open(path).expect("open input");
    let mut lines = BufReader::new(f).lines();
    let header = lines.next().expect("header").expect("read header");
    let mut hit = header.split_whitespace();
    let n: u32 = hit.next().expect("n").parse().expect("parse n");
    let m: usize = hit.next().expect("m").parse().expect("parse m");
    let mut g = LaraAdapter::new(n, LaraConfig::default()).expect("lara init");
    let mut inserted = 0usize;
    for line in lines {
        let line = line.expect("read edge");
        let mut sp = line.split_whitespace();
        let src: u32 = sp.next().expect("src").parse().expect("parse src");
        let dst: u32 = sp.next().expect("dst").parse().expect("parse dst");
        g.insert_edge(src, dst);
        inserted += 1;
    }
    assert_eq!(inserted, m, "edge count mismatch");
    let mut log_rows = 0u64;
    let mut log_edges = 0u64;
    let mut tree_rows = 0u64;
    let mut tree_edges = 0u64;
    let mut tiny_rows = 0u64;
    let mut tiny_edges = 0u64;
    let mut slab_rows = 0u64;
    let mut slab_edges = 0u64;
    let mut slab_stored = 0u64;
    for v in 0..g.num_nodes() as u32 {
        let deg = g.out_degree(v);
        if deg == 0 {
            continue;
        }
        if g.row_is_tree(v) {
            tree_rows += 1;
            tree_edges += deg;
        } else if g.row_is_tiny(v) {
            tiny_rows += 1;
            tiny_edges += deg;
        } else if g.row_has_log(v) {
            log_rows += 1;
            log_edges += deg;
        } else {
            slab_rows += 1;
            slab_edges += deg;
            slab_stored += g.row_stored(v);
        }
    }
    println!("RESIDENCY,rows={n},edges={m}");
    println!("RESIDENCY,tree,rows={tree_rows},edges={tree_edges}");
    println!("RESIDENCY,tiny,rows={tiny_rows},edges={tiny_edges}");
    println!("RESIDENCY,log,rows={log_rows},edges={log_edges}");
    println!("RESIDENCY,slab,rows={slab_rows},edges={slab_edges},stored={slab_stored}");
    println!("RESIDENCY_OK");
}
