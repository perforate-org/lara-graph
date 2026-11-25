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
an edge's two halves and refilling holes never scan them. Nodes carry any set of `u16` labels,
and each label lists its nodes. The crate is `no_std` + `alloc`.

## Use in a canister

Enable the `ic-stable-structures` feature, give the graph six memories from a `MemoryManager`,
and open it with `init`, as ic-stable-structures' own structures are opened. `init` formats empty
memories and reopens a graph written before an upgrade, so one call serves `#[init]` and
`#[post_upgrade]`.

```rust
use std::cell::RefCell;

use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
use ic_stable_structures::DefaultMemoryImpl;
use lara_graph::{Direction, Graph, Label, Memories, NewEdge, Placement};

type Memory = VirtualMemory<DefaultMemoryImpl>;

thread_local! {
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

    // Memory ids 0 to 5, one per role.
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
        .add_edge(NewEdge::directed(alice, FOLLOWS, bob, Placement::Append, &[]))
        .unwrap();

    let mut out = Vec::new();
    graph.neighbors_directed(alice, FOLLOWS, Direction::Outgoing, &mut out).unwrap();
    assert_eq!(out.iter().map(|e| e.target()).collect::<Vec<_>>(), [bob]);
});
```

`Graph::new` formats (overwriting what the memories held) and `Graph::load` reopens; `try_new`,
`try_load` and `try_init` return an `InitError` instead of panicking. Off the IC, `VecMemory`
keeps a graph on the heap.

## Node labels

A node's labels are a set of `Label`s, kept apart from its edges' labels. `set_node_labels`
replaces the set and is the one writer of both directions; `node_labels` lists it and
`has_node_label` checks one label, with one 5-byte read for a node with at most two labels, so
filtering neighbours by a label during a traversal stays cheap. `visit_nodes_with_label` gives
a label's nodes in ascending id order, resuming after any node and stopping when the callback
breaks, and `node_label_count` their number. A deleted node loses its labels.

| Operation | Instructions in stable memory (canbench) |
| --- | --- |
| `has_node_label`, a node with one label | about 270 |
| `visit_nodes_with_label`, 16 of 10,000 nodes | about 2.1 K in all |
| `set_node_labels`, one label on a new node | about 3.8 K |

## Batch operations

Single operations return `Result<_, Error>` and are all-or-none. Batch operations take a slice
and return `Result<_, BatchError>`; the error carries the failing index and whether the graph
was changed.

| Operation | On failure |
| --- | --- |
| `add_edges` | Ordered bulk. The failing edge is all-or-none; earlier edges remain. |
| `set_edge_values` | Checks every node and value first; a later reservation failure restores earlier values. |
| `remove_edges` | Checks every node first, then removes all or nothing. |
| `detach_many` | Checks every node first, then marks all or nothing. |

For an empty graph with nodes and widths already declared, `bulk_load` builds every bucket and
its target index in one pass. Within a bucket, positions follow input order; the `placement` of
each `NewEdge` is ignored because there are no holes yet. It returns `Error::NotEmpty` if the
graph already holds edges. On the stable-memory hub benchmark, bulk load is about 51 % cheaper
than `add_edges`.

`BatchError::Unchanged` means no element was applied. `BatchError::PartiallyApplied` means the
elements before `index` were applied; only `add_edges` can return it. A canister can turn a partial batch into an all-or-none
update by trapping on `PartiallyApplied`; the IC then discards the call's state changes:

```rust,ignore
match graph.add_edges(&edges) {
    Ok(halves) => halves,
    Err(BatchError::Unchanged { error, .. }) => return Err(error.to_string()),
    Err(BatchError::PartiallyApplied { index, error }) => {
        ic_cdk::trap(&format!("batch edge {index}: {error}"));
    }
}
```

The trap rolls back the whole update call, including any other writes made earlier in the same
call, and does not roll back other canisters called before the trap. Put a batch that must be
all-or-none in its own update call. Batches are not a transaction API; `ReadBatch` is only a
read-side context that holds node-record and list-page caches.

## Memories

Each memory starts, as ic-stable-structures' do, with a 3-byte magic and a version byte.

| Role | Holds |
| --- | --- |
| `Nodes` | one 6-byte record per node |
| `EdgePages`, `EdgeDirectory` | label buckets: edges, their values and target indexes |
| `LabelBucketListPages`, `LabelBucketListDirectory` | each node's label bucket list, the width table, larger node label sets and each label's nodes |
| `NodeLabels` | one 5-byte label cell per node, and the root of the label postings |

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
