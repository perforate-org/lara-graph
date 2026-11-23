# lara-graph

Experimental mutable CSR-style adjacency storage with relocatable slab spans,
overflow logs, pooled small rows, and a block-based representation for large rows.
It provides directed, id-only topology, including an optional labeled interface.
Parallel edges and self-loops are allowed. The default `Unordered` mode makes
no adjacency-order guarantee: deletion swaps the last occupied tiny/slab/tree
entry into a removed position, or unlinks a log entry. Mutation and traversal
require only `CsrEdge`. The nonlabeled core also supports explicit `Insertion`
ordering; marker capability alone does not select it. `LabeledLaraGraph` selects
the same policy per label across all vertices. Raw tree shifting and standalone
ordinal removal retain their separate order-preserving contracts.

Requires Rust 1.85 or newer.

## Features

- `std` (default): heap-backed `VecMemory` and a string-to-label catalog.
- `derive` (default): re-exports the `Label` derive macro.
- `mmap`: file-backed byte storage through `memmap2`; enables `std`.
- With default features disabled, the core uses `no_std` + `alloc`. Supply a
  backend implementing `Memory`. The `derive` feature can be enabled separately.

## Example

```rust
use lara_graph::{Edge, LaraGraph, VecMemory, Vertex};
use lara_graph::graph::LaraMemories;

// Each region must have its own backing store.
let mem = LaraMemories {
    vertices: VecMemory::new(),
    edges: VecMemory::new(),
    logs: VecMemory::new(),
    counts: VecMemory::new(),
    spans: VecMemory::new(),
    free_spans: VecMemory::new(),
    blocks: VecMemory::new(),
    buckets: VecMemory::new(),
};
let mut graph: LaraGraph<Vertex, Edge, VecMemory> = LaraGraph::new(mem, 4).unwrap();
graph.insert_edge(0, 1);
graph.insert_edge(0, 2);
assert_eq!(graph.num_edges(), 2);
graph.remove_edge(0, 1);
assert_eq!(graph.out_neighbors_vec(0), vec![2]);
```

The packaged `examples/` directory includes `quickstart`, `labeled`, and a
`custom_backend` with boundary and growth tests. The labeled example requires
both `std` and `derive`.

## Core insertion order

Configure an empty core graph before inserting or calling its existing
`build_pairs` method:

```rust
use lara_graph::{AdjacencyOrder, Edge, LaraGraph, VecMemory, Vertex};

fn populate_empty(graph: &mut LaraGraph<Vertex, Edge, VecMemory>) {
    graph.set_order(AdjacencyOrder::Insertion).unwrap();
    graph.build_pairs(&[(0, 3), (1, 2), (0, 1)]).unwrap();
    assert_eq!(graph.out_neighbors_vec(0), vec![3, 1]);
}
```

- `set_order` requires `E: CsrEdgeTombstone`. Ordinary operations and `order()`
  retain the `CsrEdge` bound. Actual changes in either direction require zero
  canonical live edges across the whole core graph; same-mode calls are no-ops.
  Empty marker/log extents are retired before changing interpretation, without
  erasing unused backing bytes.
- `Insertion` means successful graph-application order within each source,
  including surviving duplicates. It does not promise global chronology,
  label order, stable edge IDs or stable ordinals.
- Ordered slabs use typed tombstones and never fill old holes with newer edges.
  Tiny/tree deletion shifts the suffix left; ordered tree deletion is therefore
  linear. Logs are visited oldest-first, and pending logs cannot be overtaken
  by newer slab appends. Repacking and storage transitions preserve local order.
- `insert_sorted_run` retains its historical name but accepts unsorted
  destinations. Ordered runs preserve slice order. Configured `build_pairs`
  replaces old adjacency and preserves relative input order per source.
- Ordered ingest stably groups sources rather than sorting destinations. A
  buffered `push(A)` is invisible until flush: `push(A), direct insert(B), flush`
  yields B,A for that source. Capacity remains a hint, not automatic flushing.
- Slabs preserve full records; tiny/tree/log storage remains neighbor-only.
  Ordered entry points reject records recognized by the configured predicate,
  including records whose default/neighbor reconstruction becomes a marker.
  With standard `Edge`, this excludes `u32::MAX` as an ordered graph edge, not
  from the separate raw tree domain. No marker's neighbor/wire word is treated
  as a universal sentinel. Trait implementations must supply a consistent codec
  and marker predicate.
- Ordered scans/removal may allocate temporary log-chain vectors; ordered
  ingest uses stable sorting. No latency or allocation advantage is promised.
  Tested marker/source preflight is not arbitrary-failure transactional rollback.

## Per-label insertion order

Configure a label before populating it, then use the instance's `build_edges`
for bulk replacement or ordinary `insert_edge` calls:

```rust
use lara_graph::{AdjacencyOrder, LabelId, LabeledLaraGraph, VecMemory};

fn populate_labels(graph: &mut LabeledLaraGraph<VecMemory>) {
    let feed = LabelId::new(1);
    let other = LabelId::new(0);
    graph.set_label_order(feed, AdjacencyOrder::Insertion).unwrap();
    graph.build_edges(&[(0, 3, feed), (0, 2, other), (0, 1, feed)]).unwrap();
    let mut neighbors = Vec::new();
    graph.for_each_edge_of_label(0, feed, |d| neighbors.push(d));
    assert_eq!(neighbors, vec![3, 1]);
    assert_eq!(graph.label_order(other), AdjacencyOrder::Unordered);
}
```

- One setting applies to a label at every vertex, including future buckets.
  Actual changes require zero canonical live degree for that label everywhere;
  other labels may remain populated. Same-mode calls are no-ops. No live
  unordered history is converted into an insertion-history guarantee.
- Order is local to `(source, label)`. Whole-vertex scans retain ascending
  label-ID groups, not cross-label application chronology. DynamicCsr and
  RemovableCsr mutation still address label ID0 and honor its selected policy.
- Slab/log/tree mutation, scans, repacking and promotion/demotion use each
  bucket's policy. Ordered trees use linear suffix shifts, not tree markers.
  Labeled storage has no tiny rows. Its concrete `Edge` admits `u32::MAX` in
  Unordered labels but rejects it in Insertion labels before routing; markers
  in one label do not hide live MAX neighbors in another.
- `build_edges` replaces adjacency while retaining settings and graph
  configuration. Each `(source, label)` follows input-relative order, including
  interleaved input and duplicates. Source/marker rejection precedes destructive
  writes; later allocation/backend failures are not transactional. The existing
  `with_edge_list` convenience constructor still starts Unordered.
- Settings occupy one sorted vector of enabled label IDs on the graph, not
  fields on edges, vertices or bucket descriptors. The vector header is part of
  object size; backing capacity may be retained after disabling a label.
  Ordered log scans may allocate temporary vectors. Empty-label switching
  retires its extents but leaves shared leaf-log allocation/cursor reclamation
  to ordinary maintenance so other owners remain safe.
- LabelCatalog remains an optional name-to-ID registry, not the policy owner.
  The existing IngestBuffer is core-only; labeled buffered ingest is not provided.

## Limits and storage semantics

- `Memory::grow(delta)` adds 64 KiB pages and returns the new page count.
  Vertex rows and leaf counts are authoritative heap vectors. The `vertices`
  and `counts` handles in `LaraMemories` are retained for interface compatibility
  but are not read, written, or grown by the graph, even during construction.
  A byte backend does not by itself persist the graph's heap metadata.
- There is no graph reopen or upgrade-recovery API. `MmapMemory::create`
  truncates its file; mmap is not an upgrade-safe graph store.
- Only depth-1 Tree-CSR is implemented: at most 1,048,576 neighbors per promoted
  row/bucket. Unsupported promotion-enabled additions panic. Bulk-built slab
  rows are a distinct path and do not eagerly promote.
- Some storage addresses have narrower limits than `u64`; large virtual-address
  or platform-capacity claims do not follow from the backend interface.
- Small degree does not guarantee pooled tiny-row residency. Empty rows may
  reuse tiny storage; there is no lifetime "spill once" guarantee.
- Vertex counts may include layout padding. Each labeled vertex owns an
  independently located, label-sorted descriptor run with power-of-two capacity.
  Creating a bucket shifts or relocates only that run; leaf rebalancing updates
  only the leaf's active descriptors. A full run first reuses an available run
  of the required capacity. Otherwise it can extend in place at the arena's
  allocated frontier; non-frontier runs still relocate. Runs need not appear
  in vertex order in backing memory. Retired runs are reused, not erased or
  returned to the OS. Slack, retired runs, backing growth and the store's free-head
  vector can increase retained memory; there is no per-vertex capacity field.
  Edge repacking and backing growth still have their own costs, and tree-state
  release may still require global descriptor ID fixups.
- Labeled bulk/rebalance reserves no slab slots for a leaf without any buckets.
  Leaves with retained empty buckets or only tree buckets still keep the
  1024-slot minimum, as do other populated leaves. A later first bucket uses
  ordinary layout allocation. Zero reserved width does not shrink retained
  backing or remove constructor allocations.
- `memory_bytes_total()` is a legacy partial storage estimate, not total heap,
  resident memory, or peak usage. No workload-independent performance or memory
  advantage is promised.

## License

MIT OR Apache-2.0; see `LICENSE-MIT` and `LICENSE-APACHE`.
