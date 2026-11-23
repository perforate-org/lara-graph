# LARA-graph: Localized Adjacency Relocation Array

A mutable CSR-style adjacency store for dynamic graphs, generalized from the
DGAP reference implementation (arXiv:2403.02665). LARA reuses the PMA update
logic — segment tree, per-leaf overflow logs, degree-weighted rebalancing — and
changes parts of the physical layout: relocatable leaf spans, a free-span store
for retired ranges, pooled small rows, and a block-based representation for
high-degree rows.

The workspace is an experimental implementation in this repository; whether the
crate is published to a registry is not verified here. The planned release
order is `lara-graph-macros` first, then `lara-graph`.

## What it is

- **Ownership.** Canonical vertex rows, leaf counts, and tree roots are
  heap-resident vectors. The byte backend is a six-method `Memory` seam
  (`read`, `read_mapped`, `write`, `size`, `grow`, `fence`). `grow` adds 64 KiB
  pages and returns the new page count; `read_mapped` copies into the caller's
  buffer and is not a borrowed view. Edge payloads live in several storage
  modes (slab records, tiny chunks, log entries, and tree blocks), so there is
  no single physical packed array of all edges; "one mutable CSR" refers to
  one logical graph representation. The `vertices` and `counts` region handles
  are retained for interface compatibility but are not read, written, or grown
  by the graph.
- **Free-span store.** Retired physical ranges are indexed by length and by
  start position in ordered maps and reused best-fit; releasing a range first
  coalesces it with adjacent free ranges. Its standalone `init` rebuilds the
  store index from its records; this is not a graph reopen path.
- **Leaf spans.** Span metadata separates logical leaf identity from physical
  slab placement. Rebalancing may allocate a replacement span and return the
  old range to the free-span store; canonical rows and labeled descriptors are
  adjusted to the new geometry. This is not a transactional publication or
  failure-recovery protocol. Edge spans and the labeled descriptor arena are
  separate regions.
- **Overflow logs.** Up to 170 entries per leaf by default, allocated lazily
  and recycled through a free stack when logs fold.
- **Small rows.** Small rows may live in pooled 32-byte chunks in the log
  region: reads use the live chunk prefix, and the insert past the chunk spills
  the row into the usual slab/log path. Small degree alone does not determine
  residency, an emptied chunk may stay resident, and there is no lifetime
  "spill once" guarantee. The labeled layer has no tiny rows.
- **Tree representation.** Depth-1 only: at most 2^20 slots per promoted row or
  label bucket, addressed through heap-resident root arrays over fixed 4 KiB
  blocks. Unsupported promotion-enabled additions panic. Leaf rebalancing does
  not rewrite promoted payloads; that is not a constant-time guarantee for
  root fixups, promotion/demotion, or ordered tree deletion.
- **Ordering.** `Unordered` (default) makes no adjacency-order guarantee:
  deletion swaps the last occupied slab or tiny entry, swaps within a tree
  bucket, or unlinks a log position. Explicit `Insertion` ordering requires
  tombstone-capable edge types (`CsrEdgeTombstone`); tiny and tree deletion
  shifts suffixes left, slab storage uses typed markers, logs are visited
  oldest-first, and application order is preserved per source (or per
  `(source, label)`)---not global chronology, not stable edge ids. Scanning all
  labels of one vertex visits ascending label-ID groups.
- **Labels.** Each labeled vertex owns one descriptor run in the separate
  buckets region: the run is label-sorted with power-of-two capacity, and each
  descriptor points at that bucket's slab range or its tree bucket. Retired
  runs are reused. Tree-bucket payloads are outside the slab relocation path.
- **Features.** With default features disabled the core is `no_std` + `alloc`.
  `std` (default) provides the heap-backed `VecMemory` and the label-name
  catalog; `derive` (default) provides the `Label` derive macro; `mmap` adds a
  file-backed backend (and enables `std`).

Dynamic allocation is headroom, not a fixed width: for a bucket or row with x
live edges the dynamic path allocates `max(x, min(max(2x, 1), cap))`, where the
cap equals the promotion threshold (or no cap when promotion is disabled). The
cap limits headroom and never removes live slots. Bulk build uses a separate
builder layout and does not eagerly promote. In the labeled layer, a leaf with
any bucket---including retained empty buckets and tree-only leaves---keeps a
1024-slot floor, while a leaf with neither slab nor tree buckets reserves no
slots.

The fixed defaults are promotion 1024, demotion 512, log cap 170, and tiny
width 8. They are constants without an optimality claim; disabling promotion
also changes the capacity policy.

## Using it locally

```toml
[dependencies]
lara-graph = { path = "crates/lara" }
```

```rust
use lara_graph::{Edge, LaraGraph, VecMemory, Vertex};
use lara_graph::graph::LaraMemories;

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

See [`crates/lara/examples/`](crates/lara/examples/) (`quickstart`, `labeled`,
`custom_backend`) and [`crates/lara/README.md`](crates/lara/README.md) for the
ordering and label contracts.

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Workspace

```text
crates/dgap/               Rust port of the DGAP reference implementation
crates/lara/               LARA (lara-graph crate)
crates/lara-graph-macros/  Label derive macro
crates/benches/            criterion harnesses, DynamicCsr adapters, datasets
experiments/canbench-stable/  0036 IC instruction-pilot fixture
experiments/pocketic-contract/  0037 message-boundary contract fixture
paper/                     paper in Typst (main.typ English, main_ja.typ Japanese)
docs/                      evidence and research records
```

The Rust DGAP port keeps the reference placement algorithms but omits the
original's PMDK persistence path and thread-local undo logs, so it does not
provide the original's recovery guarantees, and its measurements are not the
original system's. The last petgraph comparison in `crates/benches` used the
pre-0034 baseline and is not a current result.

## Limits

- No graph reopen or upgrade-recovery API, and no graph-wide durable commit.
  `Memory::fence` has a default no-op implementation; `MmapMemory::flush()`
  returns a `Result` for checked persistence, and its `fence` override calls
  flush best-effort and discards the error. `MmapMemory::create` truncates its
  file.
- Depth-1 Tree-CSR only (2^20 slots per promoted row/bucket). No ~500 GiB
  capacity claim and no concurrent-execution claim.
- Slab storage keeps full records; tiny, tree, and log storage keeps
  neighbor-only payloads.
- `IngestBuffer::flush` is a host-side API; it does not aggregate external
  update calls.
- No workload-independent performance or memory advantage is claimed. The old
  benchmark log in `docs/` is a historical record, not current canonical
  numbers.

## Evidence

Three scope-limited records support the current claims; none of them
establishes a general ranking among graph systems:

- [0034 adoption record](docs/short-copy-adoption-2026-09-15/report-v4.md) —
  native `VecMemory` backend copy comparison (paired ratios; one adverse case).
- [0036 canbench pilot record](docs/stable-memory-adapter-contract-2026-09-15/canbench-v5/report.md) —
  six-kernel instruction pilot on the stable-memory adapter; the stable side
  costs more instructions in that pilot.
- [0037 update/trap record](docs/update-trap-contract-2026-09-16/v5/report.md) —
  normal-update retention and single-message trap rollback contract test.
- [Locality roadmap](docs/locality-roadmap-2026-09-15.md) — how future
  evaluation and persistence work is sequenced.

## Paper

[`paper/main.typ`](paper/main.typ) (English) and
[`paper/main_ja.typ`](paper/main_ja.typ) (Japanese), Typst with charged-ieee
0.1.4. The draft is scoped to the current implementation and the records above;
It reports the limited message-boundary observations above,
not general performance superiority, durable graph recovery, or upgrade
guarantees.

## License

MIT OR Apache-2.0.
