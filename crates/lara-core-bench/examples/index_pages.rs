//! Space the target index takes in real hubs, against the hub's 4-byte slots.
//!
//! Builds hubs of 65536 out-edges of several shapes through the graph API and prints the index's
//! pages and bytes per edge. Run with `cargo run --release --example index_pages -p
//! lara-core-bench`.

use lara_core::bench_internals::index_pages;
use lara_core::{Graph, Label, Placement, Side, VecMemory, VertexId};
use lara_core_bench::heap_graph;

const EDGES: u32 = 65536;
const LABEL: Label = Label::new(2);
const PAGE: f64 = 4096.0;

/// splitmix64: deterministic randomness.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) % n
    }
}

/// A hub with an edge to each of `targets` (by index into `pool`), in order.
fn hub(pool: u32, targets: impl Iterator<Item = u32>) -> (Graph<VecMemory>, VertexId) {
    let mut g = heap_graph();
    let hub = g.add_vertex().unwrap_or_else(|e| panic!("vertex: {e}"));
    let leaves: Vec<VertexId> = (0..pool)
        .map(|_| g.add_vertex().unwrap_or_else(|e| panic!("vertex: {e}")))
        .collect();
    for t in targets {
        let leaf = leaves
            .get(usize::try_from(t).unwrap_or(usize::MAX))
            .copied()
            .unwrap_or_else(|| panic!("target {t} outside the pool"));
        g.insert_directed(hub, LABEL, leaf, Placement::ReuseHoles, &[])
            .unwrap_or_else(|e| panic!("insert: {e}"));
    }
    (g, hub)
}

fn report(shape: &str, g: &Graph<VecMemory>, hub: VertexId) {
    let pages = u32::try_from(index_pages(g, hub, LABEL, Side::Out)).unwrap_or(u32::MAX);
    let per_edge = f64::from(pages) * PAGE / f64::from(EDGES);
    println!(
        "{shape:<40} {pages:>4} pages  {per_edge:5.2} B/edge  {:4.2}x slots",
        per_edge / 4.0
    );
}

fn main() {
    let (g, h) = hub(EDGES, 0..EDGES);
    report("distinct targets, ascending", &g, h);
    let mut rng = Rng(7);
    let (g, h) = hub(
        EDGES,
        (0..EDGES).map(|_| u32::try_from(rng.below(u64::from(EDGES))).unwrap_or(0)),
    );
    report("random targets (some repeated)", &g, h);
    let (g, h) = hub(64, (0..EDGES).map(|i| i % 64));
    report("64 targets, round robin", &g, h);
    let (g, h) = hub(64, (0..EDGES).map(|i| i / 1024));
    report("64 targets, each in one run", &g, h);
    let mut rng = Rng(11);
    let (g, h) = hub(
        1_000_000,
        (0..EDGES).map(|_| u32::try_from(rng.below(1_000_000)).unwrap_or(0)),
    );
    report("random targets among 1M vertices", &g, h);
}
