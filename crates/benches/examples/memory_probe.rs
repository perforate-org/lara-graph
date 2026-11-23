//! Retained Rust allocation requests, NOT RSS/allocator usable bytes/peak memory.
//! Run alone: cargo run -p benches --release --example memory_probe -- 100000
//! Fixtures and verification buffers are outside the graph allocation interval.
//! Every graph must return the allocator to its starting value on drop.

use benches::{coo_power_law_stream, coo_uniform_stream};
use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
use lara_graph::memory::{Memory, PAGE_SIZE, VecMemory};
use petgraph::{Directed, Graph, graph::NodeIndex};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct CountedSystem;
static LIVE: AtomicUsize = AtomicUsize::new(0);

// SAFETY: allocations are delegated to System with unchanged layouts/pointers.
// Accounting never allocates, panics, or calls back into the allocator.
unsafe impl GlobalAlloc for CountedSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            System.dealloc(ptr, layout);
        }
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, size) };
        if !p.is_null() {
            if size >= layout.size() {
                LIVE.fetch_add(size - layout.size(), Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - size, Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: CountedSystem = CountedSystem;

type Lara = LaraGraph<Vertex, Edge, VecMemory>;
type Pg = Graph<(), (), Directed>;

#[derive(Default)]
struct Observation {
    vertices: u64,
    heap: usize,
    buffers: usize,
    mapped: u64,
    stack: usize,
    partial: u64,
    free_slab: u64,
    pooled_log: u64,
    pooled_tiny: u64,
}

fn lara_snapshot(g: &Lara, regions: &[VecMemory; 8], baseline: usize, cap: usize) -> Observation {
    let heap = LIVE.load(Relaxed) - baseline;
    let buffers: usize = regions.iter().map(VecMemory::capacity_bytes).sum();
    let mapped = regions.iter().map(|m| m.size() * PAGE_SIZE).sum();
    let tiny_chunks = g.tiny_row_count() + g.tiny_pool_depth() as u64;
    let pooled_tiny =
        g.tiny_memory_bytes().checked_div(tiny_chunks).unwrap_or(0) * g.tiny_pool_depth() as u64;
    assert!(heap >= buffers && buffers as u64 >= mapped);
    Observation {
        vertices: g.num_nodes(),
        heap,
        buffers,
        mapped,
        stack: std::mem::size_of_val(g),
        partial: g.memory_bytes_total(),
        free_slab: g.free_span_stats().1 * 4,
        pooled_log: g.log_pool_depth() as u64 * cap as u64 * 16,
        pooled_tiny,
    }
}

fn measure_lara(
    n: u32,
    edges: &[(u32, u32)],
    expected: &[Vec<u32>],
    cap: usize,
) -> [Observation; 2] {
    let baseline = LIVE.load(Relaxed);
    let regions: [VecMemory; 8] = std::array::from_fn(|_| VecMemory::new());
    let mem = LaraMemories {
        vertices: regions[0].clone(),
        edges: regions[1].clone(),
        logs: regions[2].clone(),
        counts: regions[3].clone(),
        spans: regions[4].clone(),
        free_spans: regions[5].clone(),
        blocks: regions[6].clone(),
        buckets: regions[7].clone(),
    };
    let mut g = Lara::new_with_max_log_entries(mem, n, cap).unwrap();
    for &(src, dst) in edges {
        g.insert_edge(src, dst);
    }
    let before = lara_snapshot(black_box(&g), &regions, baseline, cap);
    verify_lara(&g, expected);
    for leaf in 0..g.segment_count() {
        g.rebalance_leaf(leaf);
    }
    let after = lara_snapshot(black_box(&g), &regions, baseline, cap);
    verify_lara(&g, expected);
    drop(g);
    drop(regions);
    assert_eq!(
        LIVE.load(Relaxed),
        baseline,
        "LARA retained allocation after drop"
    );
    [before, after]
}

fn verify_lara(g: &Lara, expected: &[Vec<u32>]) {
    g.verify_invariants();
    for (v, want) in expected.iter().enumerate() {
        let mut got = g.out_neighbors_vec(v as u32);
        got.sort_unstable();
        assert_eq!(&got, want);
    }
}

fn measure_petgraph(
    n: u32,
    edges: &[(u32, u32)],
    expected: &[Vec<u32>],
    preallocate: bool,
) -> Observation {
    let baseline = LIVE.load(Relaxed);
    let mut g = Pg::with_capacity(n as usize, if preallocate { edges.len() } else { 0 });
    for _ in 0..n {
        g.add_node(());
    }
    for &(src, dst) in edges {
        g.add_edge(
            NodeIndex::new(src as usize),
            NodeIndex::new(dst as usize),
            (),
        );
    }
    black_box(&g);
    let heap = LIVE.load(Relaxed) - baseline;
    let (nc, ec) = g.capacity();
    let buffers = nc * std::mem::size_of::<petgraph::graph::Node<()>>()
        + ec * std::mem::size_of::<petgraph::graph::Edge<()>>();
    assert_eq!(
        heap, buffers,
        "petgraph allocations must match actual node/edge capacities"
    );
    if preallocate {
        assert_eq!((nc, ec), (n as usize, edges.len()));
    }
    for (v, want) in expected.iter().enumerate() {
        let mut got: Vec<_> = g
            .neighbors(NodeIndex::new(v))
            .map(|nb| nb.index() as u32)
            .collect();
        got.sort_unstable();
        assert_eq!(&got, want);
    }
    let out = Observation {
        vertices: g.node_count() as u64,
        heap,
        buffers,
        stack: std::mem::size_of_val(&g),
        ..Observation::default()
    };
    drop(g);
    assert_eq!(
        LIVE.load(Relaxed),
        baseline,
        "petgraph retained allocation after drop"
    );
    out
}

fn allocator_self_check() {
    let before = LIVE.load(Relaxed);
    let mut v = black_box(Vec::<u64>::with_capacity(17));
    v.resize(17, 42);
    assert_eq!(LIVE.load(Relaxed) - before, v.capacity() * 8);
    v.reserve(100);
    black_box(&v);
    assert_eq!(LIVE.load(Relaxed) - before, v.capacity() * 8);
    v.shrink_to_fit();
    black_box(&v);
    assert_eq!(LIVE.load(Relaxed) - before, v.capacity() * 8);
    drop(v);
    let zeroed = black_box(vec![0u64; 19]);
    assert_eq!(LIVE.load(Relaxed) - before, zeroed.capacity() * 8);
    drop(zeroed);
    assert_eq!(LIVE.load(Relaxed), before);
}

fn emit(
    fixture: &str,
    n: u32,
    m: usize,
    fingerprint: u64,
    model: &str,
    phase: &str,
    o: &Observation,
) {
    println!(
        "{fixture},{n},{m},{fingerprint:016x},{model},{phase},{},{},{},{},{},{},{},{},{},{},{}",
        o.vertices,
        m * 4,
        o.stack,
        o.mapped,
        o.buffers,
        o.heap - o.buffers,
        o.heap,
        o.partial,
        o.free_slab,
        o.pooled_log,
        o.pooled_tiny
    );
}

fn main() {
    let n: u32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("vertex count"))
        .unwrap_or(100_000);
    assert!(
        (1..=262_144).contains(&n),
        "4*n must fit the depth-1 hub limit"
    );
    let m = u64::from(n) * 4;
    println!(
        "# allocator=System; metric=live requested Rust heap bytes; excludes fixture, stack, allocator overhead, RSS and peak"
    );
    println!(
        "# uniform seed=1234; rank_zipf seed=99 beta=1 IDs=ranks; hub src=0 dst=i%n; directed, id-only, parallel edges allowed"
    );
    println!(
        "# buffer_capacity is backing Vec capacity for LARA, node+edge capacity for petgraph; mapped=0 means not applicable for petgraph"
    );
    println!(
        "# partial_estimate and pool/free columns are diagnostic subsets, NOT additive; graph_stack is separate from heap"
    );
    println!(
        "fixture,n,m,fnv1a64,model,phase,graph_vertex_count,neighbor_id_bytes,graph_stack_bytes,mapped_backend_bytes,buffer_capacity_bytes,other_heap_bytes,total_requested_heap_bytes,legacy_partial_estimate_bytes,free_slab_bytes,pooled_log_bytes,pooled_tiny_bytes"
    );
    allocator_self_check();
    for fixture in ["uniform", "rank_zipf", "hub"] {
        let edges = match fixture {
            "uniform" => coo_uniform_stream(n, m, 1234),
            "rank_zipf" => coo_power_law_stream(n, m, 99, 1.0),
            _ => (0..m).map(|i| (0, (i % u64::from(n)) as u32)).collect(),
        };
        let mut expected = vec![Vec::new(); n as usize];
        let mut fingerprint = 0xcbf29ce484222325u64;
        for &(src, dst) in &edges {
            expected[src as usize].push(dst);
            for byte in src.to_le_bytes().into_iter().chain(dst.to_le_bytes()) {
                fingerprint = (fingerprint ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        for row in &mut expected {
            row.sort_unstable();
        }
        let distinct = expected.iter().filter(|row| !row.is_empty()).count();
        let max_degree = expected.iter().map(Vec::len).max().unwrap();
        println!("# {fixture}: distinct_sources={distinct}, max_degree={max_degree}");
        for (model, cap) in [("lara_log0", 0), ("lara_log170", 170)] {
            let [before, after] = measure_lara(n, &edges, &expected, cap);
            emit(
                fixture,
                n,
                edges.len(),
                fingerprint,
                model,
                "after_insert",
                &before,
            );
            emit(
                fixture,
                n,
                edges.len(),
                fingerprint,
                model,
                "after_settle",
                &after,
            );
        }
        for (model, reserve) in [("petgraph_growing", false), ("petgraph_preallocated", true)] {
            let o = measure_petgraph(n, &edges, &expected, reserve);
            emit(
                fixture,
                n,
                edges.len(),
                fingerprint,
                model,
                "after_insert",
                &o,
            );
        }
    }
}
