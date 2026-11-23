//! petgraph vs LARA neighbor-set equivalence (100v/500e).
use benches::{LaraAdapter, LaraConfig, coo_uniform_stream};
use lara_graph::traits::DynamicCsr;
use petgraph::{Directed, Graph};
fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v
}
#[test]
fn petgraph_lara_same_neighbors() {
    let n = 100u32;
    let stream = coo_uniform_stream(n, 500, 7);
    let mut pg: Graph<(), (), Directed> = Graph::with_capacity(n as usize, 500);
    let nodes: Vec<_> = (0..n).map(|_| pg.add_node(())).collect();
    for &(u, v) in &stream {
        pg.add_edge(nodes[u as usize], nodes[v as usize], ());
    }
    let mut lara = LaraAdapter::new(n, LaraConfig::default()).unwrap();
    for &(u, v) in &stream {
        lara.insert_edge(u, v);
    }
    assert_eq!(pg.edge_count() as u64, lara.num_edges());
    for v in 0..n {
        let mut a: Vec<u32> = pg
            .neighbors(nodes[v as usize])
            .map(|i| i.index() as u32)
            .collect();
        a = sorted(a);
        let mut b = Vec::new();
        lara.for_each_neighbor(v, |x| b.push(x));
        assert_eq!(a, sorted(b), "neighbor mismatch v={v}");
    }
}
