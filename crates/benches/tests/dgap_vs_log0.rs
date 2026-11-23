//! Cross-system correctness: DGAP port vs no-log LARA return identical
//! neighbor sets on a mixed hub + dispersed fixture.

use benches::{DgapAdapter, LaraAdapter, LaraConfig, coo_grouped, coo_power_law_stream};
use lara_graph::traits::DynamicCsr;

fn neighbors_of<G: DynamicCsr>(g: &G, v: u32) -> Vec<u32> {
    let mut out = Vec::new();
    g.for_each_neighbor(v, |t| out.push(t));
    out.sort_unstable();
    out
}

#[test]
fn rank_zipf_stream_and_bulk_match_the_same_multisets() {
    for n in [10_000u32, 100_000] {
        let edges = coo_power_law_stream(n, u64::from(n) * 4, 99, 1.0);
        let grouped = coo_grouped(edges.clone());
        let mut expected = vec![Vec::new(); n as usize];
        let mut dgap = DgapAdapter::new(n).unwrap();
        let mut log0 = LaraAdapter::new(
            n,
            LaraConfig {
                max_log_entries: 0,
                ..LaraConfig::default()
            },
        )
        .unwrap();
        let mut lara = LaraAdapter::new(n, LaraConfig::default()).unwrap();
        for &(src, dst) in &edges {
            expected[src as usize].push(dst);
            dgap.insert_edge(src, dst);
            log0.insert_edge(src, dst);
            lara.insert_edge(src, dst);
        }
        let dgap_bulk = DgapAdapter::with_edges(n, &grouped).unwrap();
        let lara_bulk = LaraAdapter::with_edges(n, &grouped, LaraConfig::default()).unwrap();
        dgap.verify();
        log0.verify();
        lara.verify();
        dgap_bulk.verify();
        lara_bulk.verify();
        for (v, want) in expected.iter_mut().enumerate() {
            want.sort_unstable();
            let v = v as u32;
            assert_eq!(&neighbors_of(&dgap, v), want, "DGAP stream vertex {v}");
            assert_eq!(&neighbors_of(&log0, v), want, "LARA log0 vertex {v}");
            assert_eq!(&neighbors_of(&lara, v), want, "LARA log170 vertex {v}");
            assert_eq!(&neighbors_of(&dgap_bulk, v), want, "DGAP bulk vertex {v}");
            assert_eq!(&neighbors_of(&lara_bulk, v), want, "LARA bulk vertex {v}");
        }
    }
}

#[test]
fn dgap_vs_log0_same_neighbors() {
    let n = 512u32;
    let mut edges: Vec<(u32, u32)> = Vec::new();
    // Hub: vertex 1 -> 300 unique targets.
    for t in 0..300u32 {
        edges.push((1, t));
    }
    // Dispersed: 200 pseudo-random pairs.
    for i in 0..200u32 {
        edges.push(((i * 31 + 7) % n, (i * 17 + 3) % n));
    }
    assert_eq!(edges.len(), 500);

    let mut dgap = DgapAdapter::new(n).unwrap();
    let mut log0 = LaraAdapter::new(
        n,
        LaraConfig {
            promote_threshold: Some(1024),
            max_log_entries: 0,
            ..LaraConfig::default()
        },
    )
    .unwrap();
    for &(u, v) in &edges {
        dgap.insert_edge(u, v);
        log0.insert_edge(u, v);
    }
    assert_eq!(dgap.num_edges(), log0.num_edges());
    dgap.verify();
    log0.verify();
    for v in 0..n {
        assert_eq!(
            neighbors_of(&dgap, v),
            neighbors_of(&log0, v),
            "neighbor mismatch at vertex {v}"
        );
    }
}
