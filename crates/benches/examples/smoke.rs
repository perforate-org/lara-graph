//! Quick smoke run printing MEPS numbers outside criterion.

use benches::{DgapAdapter, LaraAdapter, LaraConfig, coo_uniform_stream};
use lara_graph::traits::DynamicCsr;

fn time_inserts<G: DynamicCsr>(g: &mut G, edges: &[(u32, u32)]) -> f64 {
    let t0 = std::time::Instant::now();
    for &(u, v) in edges {
        g.insert_edge(u, v);
    }
    t0.elapsed().as_secs_f64()
}

fn meps(edges: usize, secs: f64) -> f64 {
    if secs > 0.0 {
        edges as f64 / secs / 1e6
    } else {
        f64::INFINITY
    }
}

fn main() {
    let n = 10_000u32;
    let m = (n as u64) * 4;
    let stream = coo_uniform_stream(n, m, 1234);

    let mut g = DgapAdapter::new(n).unwrap();
    let t = time_inserts(&mut g, &stream);
    println!(
        "dgap       n={n}: {:8.2} MEPS ({} edges)",
        meps(stream.len(), t),
        g.num_edges()
    );

    let mut g = LaraAdapter::new(
        n,
        LaraConfig {
            promote_threshold: None,
            ..LaraConfig::default()
        },
    )
    .unwrap();
    let t = time_inserts(&mut g, &stream);
    println!(
        "lara_slab  n={n}: {:8.2} MEPS ({} edges)",
        meps(stream.len(), t),
        g.num_edges()
    );

    let mut g = LaraAdapter::new(n, LaraConfig::default()).unwrap();
    let t = time_inserts(&mut g, &stream);
    println!(
        "lara_tree  n={n}: {:8.2} MEPS ({} edges)",
        meps(stream.len(), t),
        g.num_edges()
    );
}
