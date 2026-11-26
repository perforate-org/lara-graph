//! Space the target index takes in real hubs, against the hub's 4-byte slots (a measurement).
//!
//! Builds hubs of 65536 out-edges of several shapes and prints the index's pages and bytes per
//! edge. Then removes fifteen edges in sixteen from two of them, and prints the pages before and
//! after compacting. Run at full size (see `scale`) with
//! `RUSTFLAGS="--cfg lara_full_size" cargo test -p lara-graph --release -- --ignored --nocapture
//! index_pages`.

extern crate std;

use super::model::Rng;
use super::*;
use crate::outcome::Progress;
use crate::store::scale::Scale;
use std::println;

const EDGES: u32 = 65536;
const LABEL: EdgeLabel = EdgeLabel::new(2);
const PAGE: f64 = 4096.0;

/// A hub with an edge to each of `targets` (by index into a pool of `pool` nodes), in order, with
/// each edge's halves.
fn hub(pool: u32, targets: impl Iterator<Item = u32>) -> (Graph<VecMemory>, NodeId, Vec<Halves>) {
    let mut g = Graph::format(Memories::from_fn(|_| VecMemory::new()));
    let hub = g.add_node(&[]).unwrap();
    let leaves: Vec<NodeId> = (0..pool).map(|_| g.add_node(&[]).unwrap()).collect();
    let edges = targets
        .map(|t| {
            let leaf = leaves[usize::try_from(t).unwrap()];
            g.add_edge(Edge::directed(hub, LABEL, leaf)).unwrap()
        })
        .collect();
    (g, hub, edges)
}

/// Pages of the target index of `hub`'s out bucket.
fn pages(g: &Graph<VecMemory>, hub: NodeId) -> usize {
    let key = LabelBucketKey {
        label: LABEL,
        direction: Direction::Outgoing,
    };
    g.find_label_bucket(hub, key)
        .ok()
        .flatten()
        .map_or(0, |found| {
            target_index::extents(&g.edge_view().pages(), found.bucket.index.tree()).len()
        })
}

fn report(shape: &str, (g, hub, _): &(Graph<VecMemory>, NodeId, Vec<Halves>)) {
    let pages = u32::try_from(pages(g, *hub)).unwrap();
    let per_edge = f64::from(pages) * PAGE / f64::from(EDGES);
    println!(
        "{shape:<40} {pages:>4} pages  {per_edge:5.2} B/edge  {:4.2}x slots",
        per_edge / 4.0
    );
}

/// Removes fifteen edges in sixteen, reporting the index's pages then and after compacting.
fn thin(shape: &str, (mut g, hub, edges): (Graph<VecMemory>, NodeId, Vec<Halves>)) {
    for (i, halves) in edges.iter().enumerate() {
        if i % 16 != 0 {
            g.remove_edge(halves.canonical()).unwrap();
        }
    }
    let removed = pages(&g, hub);
    while g.compact(hub, LABEL.outgoing()).unwrap().progress() == Progress::Remaining {}
    println!(
        "{shape:<40} {removed:>4} pages after removing, {:>4} after compacting",
        pages(&g, hub)
    );
}

#[test]
#[ignore = "a measurement: run at full size with --ignored --nocapture"]
fn index_pages() {
    assert_eq!(
        Scale::BUILD,
        Scale::Full,
        "build with RUSTFLAGS=\"--cfg lara_full_size\""
    );
    let mut rng = Rng(7);
    let random = |rng: &mut Rng, among| {
        (0..EDGES)
            .map(move |_| rng.below(among))
            .collect::<Vec<_>>()
    };
    report("distinct targets, ascending", &hub(EDGES, 0..EDGES));
    let targets = random(&mut rng, EDGES);
    report(
        "random targets (some repeated)",
        &hub(EDGES, targets.into_iter()),
    );
    report(
        "64 targets, round robin",
        &hub(64, (0..EDGES).map(|i| i % 64)),
    );
    report(
        "64 targets, each in one run",
        &hub(64, (0..EDGES).map(|i| i / 1024)),
    );
    let among_1m = random(&mut rng, 1_000_000);
    report(
        "random targets among 1M nodes",
        &hub(1_000_000, among_1m.iter().copied()),
    );
    println!("\nKeeping one edge in sixteen (4096):");
    thin("distinct targets, ascending", hub(EDGES, 0..EDGES));
    thin(
        "random targets among 1M nodes",
        hub(1_000_000, among_1m.into_iter()),
    );
}
