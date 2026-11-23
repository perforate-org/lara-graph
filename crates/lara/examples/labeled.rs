//! Labeled edges: user-defined enum labels and a string-name catalog.
//!
//! Two styles of label:
//! 1. A user-defined `Copy` enum implementing [`Label`] — the graph hands
//!    back the same enum it was given.
//! 2. A [`LabelCatalog`] mapping string names to dense [`LabelId`]s — you
//!    register a name, get its id, and insert directly (no enum).
//!
//! Run with: `cargo run --example labeled -p lara-graph`

use lara_graph::catalog::LabelCatalog;
use lara_graph::graph::LaraMemories;
use lara_graph::labeled::{Label, LabeledLaraGraph};
use lara_graph::memory::VecMemory;

/// User-defined enum label (style 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Label)]
enum Rel {
    Follows,
    Blocks,
}

fn fresh_mem() -> LaraMemories<VecMemory> {
    LaraMemories {
        vertices: VecMemory::new(),
        edges: VecMemory::new(),
        logs: VecMemory::new(),
        counts: VecMemory::new(),
        spans: VecMemory::new(),
        free_spans: VecMemory::new(),
        blocks: VecMemory::new(),
        buckets: VecMemory::new(),
    }
}

fn main() {
    // Style 1: enum labels.
    let mut g: LabeledLaraGraph<VecMemory, Rel> = LabeledLaraGraph::new(fresh_mem(), 3).unwrap();
    g.insert_edge(0, 1, Rel::Follows);
    g.insert_edge(0, 2, Rel::Blocks);
    g.insert_edge(1, 2, Rel::Follows);
    println!("enum labels of 0: {:?}", g.out_labeled_vec(0));

    // Style 2: string-name catalog → dense LabelId, inserted directly.
    let mut cat = LabelCatalog::new();
    let follows = cat.register("FOLLOWS").unwrap();
    let blocks = cat.register("BLOCKS").unwrap();
    let mut h: LabeledLaraGraph<VecMemory> = LabeledLaraGraph::new(fresh_mem(), 3).unwrap();
    h.insert_edge(0, 1, follows);
    h.insert_edge(0, 2, blocks);
    println!("catalog ids: FOLLOWS={:?} BLOCKS={:?}", follows, blocks);
    println!("catalog name_of(FOLLOWS): {:?}", cat.name_of(follows));
    println!("catalog labels of 0: {:?}", h.out_labeled_vec(0));
}
