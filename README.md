# lara-graph

LARA (Labeled Adjacency Rooted Array) is a labeled multigraph store on paged memory. It runs
on any memory that reads, writes and grows by 64 KiB pages; its first target is Internet Computer
stable memory, beside [ic-stable-structures](https://crates.io/crates/ic-stable-structures). It
keeps two promises, in this order:

1. **CSR-speed scans.** A node's edges with one label and direction are read as contiguous slots:
   a node read, a label bucket list read and one data read.
2. **Bounded updates.** An insert or removal copies at most one run below a page or one page of
   index entries, however large the node.

Edges are directed or undirected, carry an `EdgeLabel` and, if the label declares a width, a
fixed-width value. Positions are stable: removing an edge leaves a hole that a later insert may
fill, and only an explicit `compact` moves edges. Large nodes keep a target index, so pairing
an edge's two halves and refilling holes never scan them. Nodes carry any set of `NodeLabel`s,
and each label lists its nodes. The crate is `no_std` + `alloc`.

## Use in a canister

Enable the `ic-stable-structures` feature, give the graph four memories from a `MemoryManager`,
and open it with `init`, as ic-stable-structures' own structures are opened. `init` formats empty
memories and reopens a graph written before an upgrade, so one call serves `#[init]` and
`#[post_upgrade]`.

```rust
use std::cell::RefCell;

use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
use ic_stable_structures::DefaultMemoryImpl;
use lara_graph::{Edge, EdgeLabel, Graph, Memories, NodeLabel};

type Memory = VirtualMemory<DefaultMemoryImpl>;

thread_local! {
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

    // Memory ids 0 to 3, one per role. The proxies memory (3) stays empty until a proxy is added.
    static GRAPH: RefCell<Graph<Memory>> = RefCell::new(Graph::init(Memories::from_fn(|role| {
        MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(role.index())))
    })));
}

const PERSON: NodeLabel = NodeLabel::new(0);
const FOLLOWS: EdgeLabel = EdgeLabel::new(0);

GRAPH.with(|graph| {
    let mut graph = graph.borrow_mut();
    let alice = graph.add_node(&[PERSON]).unwrap();
    let bob = graph.add_node(&[PERSON]).unwrap();
    graph.add_edge(Edge::directed(alice, FOLLOWS, bob)).unwrap();

    let mut out = Vec::new();
    graph.neighbors(alice, FOLLOWS.outgoing(), &mut out).unwrap();
    assert_eq!(out.iter().map(|n| n.node()).collect::<Vec<_>>(), [bob]);
    assert_eq!(graph.remove_edges_between(alice, FOLLOWS.outgoing(), bob), Ok(1));
});
```

`Graph::format` formats (overwriting what the memories held) and `Graph::load` reopens;
`try_format`, `try_load` and `try_init` return an `InitError` instead of panicking. Off the IC,
`VecMemory` keeps a graph on the heap.

Each memory may be of its own type: `Graph<Memories<N, EP, ED, ..>>` takes them by role, and a
graph that never adds a proxy may leave `proxies` as `None`. Where nothing else names the graph's
type, name it at the call: `Graph::<VecMemory>::format(..)`.

The costs in `crates/lara-graph-bench/canbench_results.yml` are measured with
`codegen-units = 1` in the release profile, as a canister's wasm is often built. With the default
of sixteen, how rustc splits the crate follows its symbol hashes, and the split decides what is
inlined: costs then move by a few percent with unrelated code, or with where the build runs.

## Edges

An edge names its ends, its label and its kind: `Edge::directed(a, FOLLOWS, b)`, with
`.value(&bytes)` for a label that declares a width. A read names one adjacency of a node, a label
in one direction: `FOLLOWS.outgoing()`, `.incoming()` or `.undirected()`. `add_edge` puts each
half in the first hole its bucket can reuse, else at the end; `add_edge_with(edge,
Placement::Append)` always appends, so that position order is insertion order.

## Node labels

A node's labels are a set of `NodeLabel`s, kept apart from its edges' `EdgeLabel`s. `add_node`
and `add_nodes` give new nodes theirs; `add_node_label` and `remove_node_label` change one, and
`set_node_labels` replaces the set. `node_labels` lists it and `has_node_label` checks one label,
with one 6-byte read for a node with at most two labels, so filtering neighbours by a label
during a traversal stays cheap. `visit_nodes_with_label` gives a label's nodes in ascending id
order, resuming after any node and stopping when the callback breaks, and `node_label_count`
their number. A deleted node loses its labels.

## Deleting nodes

`begin_delete` marks a node: it takes no new edges, and `delete_step` removes a bounded number of
its edges per call until it is deleted. Deleting many nodes that share a neighbour is cheaper
through `deleting_set` and `sweep`, which clear their edges from the shared bucket in one pass.

## Batch operations

Single operations return `Result<_, Error>` and are all-or-none. Batch operations take a slice
and return `Result<_, BatchError>`; the error carries the failing index and whether the graph
was changed.

| Operation | On failure |
| --- | --- |
| `add_edges` | Ordered bulk. The failing edge is all-or-none; earlier edges remain. |
| `set_edge_values` | Checks every node and value first; a later reservation failure restores earlier values. |
| `remove_edges` | Checks every node first, then removes all or nothing. |
| `begin_delete_many` | Checks every node first, then marks all or nothing. |

For an empty graph with nodes and widths already declared, `bulk_load` builds every bucket and
its target index in one pass. Within a bucket, positions follow input order. It returns
`Error::NotEmpty` if the graph already holds edges. It is much cheaper than `add_edges` for the
same edges.

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
| `Nodes` | one 8-byte record per node |
| `Pages` | label buckets (edges, their values and target indexes), each node's label bucket list, the width table, larger node label sets and each label's nodes; before every 128 of them, a page of 32-byte allocation records |
| `NodeLabels` | one 6-byte label cell per node, and the root of the label postings |
| `Proxies` | one 8-byte record per proxy; optional, and empty until the first proxy |

## Where things are written down

The code is the one source of truth. Each module's documentation (`cargo doc
--document-private-items`) draws its persistent layout and gives the reasons for its choices;
`crates/lara-graph/tests/public-api.txt` pins the public API; the measured instruction counts
of every operation are in
[`crates/lara-graph-bench/canbench_results.yml`](crates/lara-graph-bench/canbench_results.yml).

## Repository

```text
crates/lara-graph/        the crate
crates/lara-graph-bench/  canbench instruction counts on the IC
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
