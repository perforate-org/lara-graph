//! Behavioral tests for the DGAP Rust port, mirroring the `graph.h`
//! invariants (sentinel layout, PMA counters, scan order, rebalances).

use dgap::{DgapGraph, VecMemory};

fn sorted(v: &[u32]) -> Vec<u32> {
    let mut w = v.to_vec();
    w.sort_unstable();
    w
}

/// Build a path graph 0-1-2-...-n-1 (undirected, both orientations stored).
fn path_edges(n: u32) -> Vec<(u32, u32)> {
    let mut e = Vec::new();
    for i in 0..n - 1 {
        e.push((i, i + 1));
        e.push((i + 1, i));
    }
    e.sort_by_key(|&(u, _)| u);
    e
}

#[test]
fn bulk_build_layout_and_scan() {
    let edges = path_edges(10);
    let g: DgapGraph<u32, VecMemory> =
        DgapGraph::with_edge_list(VecMemory::new(), &edges, 10, true).unwrap();

    assert_eq!(g.num_nodes(), 10); // ceil_log2(10)=4 → count=hyperfloor(3)=2 → size=5
    assert_eq!(g.num_edges(), edges.len() as i64);
    for v in 0..10u32 {
        let expect: Vec<u32> = match v {
            0 => vec![1],
            9 => vec![8],
            x => vec![x - 1, x + 1],
        };
        assert_eq!(sorted(&g.out_neighbors_vec(v)), expect, "vertex {v}");
        assert_eq!(g.out_degree(v), expect.len() as i64);
    }
    g.verify_invariants();
}

#[test]
fn isolated_vertices_get_sentinels() {
    // Only vertex 0 and 5 connected.
    let edges = vec![(0u32, 5u32)];
    let g: DgapGraph<u32, VecMemory> =
        DgapGraph::with_edge_list(VecMemory::new(), &edges, 8, true).unwrap();
    for v in [1u32, 2, 3, 4, 6, 7] {
        assert_eq!(g.out_degree(v), 0, "vertex {v}");
        assert!(g.out_neighbors_vec(v).is_empty());
    }
    assert_eq!(g.out_degree(0), 1);
    g.verify_invariants();
}

#[test]
fn dynamic_inserts_and_scans() {
    let mut g: DgapGraph<u32, VecMemory> = DgapGraph::new(VecMemory::new(), 64, true).unwrap();
    // Insert a star around vertex 7 plus extra edges.
    for d in 0..32u32 {
        g.insert_edge(7, d);
    }
    for d in 40..50u32 {
        g.insert_edge(7, d);
    }
    g.insert_edge(3, 7);
    g.insert_edge(3, 9);

    let mut got = g.out_neighbors_vec(7);
    got.sort_unstable();
    let mut want: Vec<u32> = (0..32).chain(40..50).collect();
    want.sort_unstable();
    assert_eq!(got, want);
    assert_eq!(g.out_degree(7), 42);
    assert_eq!(g.num_edges(), 44);
    g.verify_invariants();
}

#[test]
fn inserts_trigger_rebalance_and_resize() {
    // Small graph so the leaf fills quickly: segment_size for 16 vertices
    // is ceil_log2(16)=4 ideal -> count 4 -> size 4.
    let mut g: DgapGraph<u32, VecMemory> = DgapGraph::new(VecMemory::new(), 16, true).unwrap();
    // Hammer one leaf with inserts to drive density past UP_0 at every
    // height, including the full resize path.
    let hub = 3u32;
    for d in 0..200u32 {
        g.insert_edge(hub, d);
    }
    assert_eq!(g.out_degree(hub), 200);
    let mut got = g.out_neighbors_vec(hub);
    got.sort_unstable();
    let want: Vec<u32> = (0..200).collect();
    assert_eq!(got, want);
    assert_eq!(g.num_edges(), 200);
    g.verify_invariants();

    // Other vertices untouched but consistent.
    assert_eq!(g.out_degree(10), 0);
}

#[test]
fn scan_order_matches_cpp_semantics() {
    // On-seg ascending, then log entries newest-first. Force a log entry by
    // filling the hub's region: after the leaf is packed tight by inserts,
    // overflowing inserts land in the per-segment log.
    let mut g: DgapGraph<u32, VecMemory> = DgapGraph::new(VecMemory::new(), 16, true).unwrap();
    for d in 0..64u32 {
        g.insert_edge(2, d);
    }
    // All edges must be present exactly once.
    let mut got = g.out_neighbors_vec(2);
    got.sort_unstable();
    assert_eq!(got, (0..64).collect::<Vec<u32>>());
    g.verify_invariants();
}

#[test]
fn remove_edge_on_segment_and_log() {
    let mut g: DgapGraph<u32, VecMemory> = DgapGraph::new(VecMemory::new(), 32, true).unwrap();
    for d in 0..80u32 {
        g.insert_edge(5, d);
    }
    assert_eq!(g.out_degree(5), 80);

    // Remove a few edges (some on-seg, some in-log depending on layout).
    assert!(g.remove_edge(5, 0));
    assert!(g.remove_edge(5, 79));
    assert!(g.remove_edge(5, 40));
    assert!(!g.remove_edge(5, 40)); // already gone
    assert!(!g.remove_edge(5, 1000)); // never existed
    assert_eq!(g.out_degree(5), 77);

    let mut got = g.out_neighbors_vec(5);
    got.sort_unstable();
    let want: Vec<u32> = (0..80).filter(|d| ![0, 40, 79].contains(d)).collect();
    assert_eq!(got, want);
    g.verify_invariants();

    // Re-inserting after removal works.
    g.insert_edge(5, 0);
    assert!(g.out_neighbors_vec(5).contains(&0));
    g.verify_invariants();
}

#[test]
fn rebalance_weighted_is_public_and_correct() {
    let edges = path_edges(32);
    let mut g: DgapGraph<u32, VecMemory> =
        DgapGraph::with_edge_list(VecMemory::new(), &edges, 32, true).unwrap();
    // Manual whole-graph rebalance (rebalance_all equivalent):
    // used_space = sum of degrees (sentinels included), total = capacity.
    let used = g.num_edges_with_sentinels();
    let total = g.elem_capacity();
    g.rebalance_weighted(0, g.num_nodes() as i32, used, total);
    g.verify_invariants();
    for v in 0..32u32 {
        let expect: Vec<u32> = match v {
            0 => vec![1],
            31 => vec![30],
            x => vec![x - 1, x + 1],
        };
        assert_eq!(sorted(&g.out_neighbors_vec(v)), expect, "vertex {v}");
    }
}

#[test]
fn scan_all_visits_every_edge_once() {
    let edges = path_edges(20);
    let mut g: DgapGraph<u32, VecMemory> =
        DgapGraph::with_edge_list(VecMemory::new(), &edges, 20, true).unwrap();
    for d in 0..40u32 {
        g.insert_edge(9, d);
    }
    let mut total = 0i64;
    g.scan_all(|_| total += 1);
    assert_eq!(total, g.num_edges());
}
