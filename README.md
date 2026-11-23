# lara-graph

LARA (Localized Adjacency Relocation Array) is a labeled multigraph store on paged memory. It runs
on any memory that reads, writes and grows by 64 KiB pages; its first target is Internet Computer
stable memory, beside [ic-stable-structures](https://crates.io/crates/ic-stable-structures). It
keeps two promises, in this order:

1. **CSR-speed scans.** A node's edges with one label and direction are read as contiguous slots:
   a node read, a label bucket list read and one data read.
2. **Bounded updates.** An insert or removal copies at most one run below a page or one page of
   index entries, however large the node.

Edges are directed or undirected, carry a `u16` label and, if the label declares a width, a
fixed-width value. Positions are stable: removing an edge leaves a hole that a later insert may
fill, and only an explicit `compact` moves edges. Large nodes keep a target index, so pairing
an edge's two halves and refilling holes never scan them. The crate is `no_std` + `alloc`.

## Use in a canister

Enable the `ic-stable-structures` feature, give the graph five memories from a `MemoryManager`,
and open it with `init`, as ic-stable-structures' own structures are opened. `init` formats empty
memories and reopens a graph written before an upgrade, so one call serves `#[init]` and
`#[post_upgrade]`.

```rust
use std::cell::RefCell;

use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
use ic_stable_structures::DefaultMemoryImpl;
use lara_graph::{Direction, EdgeKind, Graph, Label, Memories, Placement};

type Memory = VirtualMemory<DefaultMemoryImpl>;

thread_local! {
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

    // Memory ids 0 to 4, one per role.
    static GRAPH: RefCell<Graph<Memory>> = RefCell::new(Graph::init(Memories::from_fn(|role| {
        MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(role.index())))
    })));
}

const FOLLOWS: Label = Label::new(0);

GRAPH.with(|graph| {
    let mut graph = graph.borrow_mut();
    let alice = graph.add_node().unwrap();
    let bob = graph.add_node().unwrap();
    graph
        .add_edge(EdgeKind::Directed, alice, FOLLOWS, bob, Placement::Append, &[])
        .unwrap();

    let mut out = Vec::new();
    graph.neighbors_directed(alice, FOLLOWS, Direction::Outgoing, &mut out).unwrap();
    assert_eq!(out.iter().map(|e| e.target()).collect::<Vec<_>>(), [bob]);
});
```

`Graph::new` formats (overwriting what the memories held) and `Graph::load` reopens; `try_new`,
`try_load` and `try_init` return an `InitError` instead of panicking. Off the IC, `VecMemory`
keeps a graph on the heap.

## Memories

Each memory starts, as ic-stable-structures' do, with a 3-byte magic and a version byte.

| Role | Holds |
| --- | --- |
| `Nodes` | one 8-byte record per node |
| `EdgePages`, `EdgeDirectory` | label buckets: edges, their values and target indexes |
| `LabelBucketListPages`, `LabelBucketListDirectory` | each node's label bucket list, and the width table |

The byte layouts are drawn in the modules' documentation. The design and its measurements are in
[`crates/lara-graph/DESIGN.md`](crates/lara-graph/DESIGN.md).

## Repository

```text
crates/lara-graph/        the crate
crates/lara-graph-bench/  canbench instruction counts on the IC
docs/                     design notes and measurements
```

The original lara-graph, which ported DGAP's PMA to relocatable spans, is kept under the tag
`lara-graph-legacy` (branch `legacy`), with its DGAP port, benchmarks, paper, experiments and
design notes.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
