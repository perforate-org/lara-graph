# Aligning lara-core's API with ic-stable-structures and graph libraries (2026-11-23)

lara-core is meant to sit in canisters beside ic-stable-structures, and to read like a graph
library. This note does three things:
- records what `new`, `init` and `load` do in ic-stable-structures 0.7.2, and how canisters use
  them (read from the crate's source and README);
- compares that, and the graph vocabulary of petgraph 0.8.3, with lara-core at 9cafcd4;
- proposes changes for a decision.

## ic-stable-structures: what the constructors do

| Structure | `new(memory…)` | `init(memory…)` | `load(memory)` | Gives back |
| --- | --- | --- | --- | --- |
| `BTreeMap`, `BTreeSet` | writes a fresh header, **overwriting** whatever the memory held | empty memory, or magic missing: `new`; magic present: `load` | reads the header (migrates v1 to v2) | `into_memory` |
| `Vec`, `MinHeap` | fresh header, overwriting | empty: `new`; otherwise **expects** the structure; panics on another magic, version or element type | — | `into_memory` |
| `Cell` | `new(memory, value)`: writes the value, overwriting | empty or magic missing: `new(default)`; version mismatch panics | — | `into_memory` |
| `Log` | `new(index, data)`: two memories, overwriting | data memory empty or magic missing: `new`; version mismatch or a bad index panics | — | `into_memories` |
| `MemoryManager` | (internal `new`) | `init(memory)` and `init_with_bucket_size`: same rule as `BTreeMap` | (internal) | `into_memory` (an `Option`, as it is shared through `Rc`) |

Common rules:
- **`init` returns `Self`, not a `Result`.** An incompatible version panics with an `InitError`
  message. Only the internal `BaseVec::init` returns a `Result`.
- **`init` both creates and reopens.** One code path serves `#[init]` (empty memory) and
  `#[post_upgrade]` (a memory written before the upgrade).
- **`new` never fails for a non-empty memory.** It overwrites. Its only failure is growing
  memory, and that panics.
- **Constant time.** Only headers are read or written.
- **The memory is given by value** and can be had back with `into_memory`.

How a canister uses them (the README's example, and how Gleaph does it today): state lives in
`thread_local!` statics behind `RefCell`, initialized lazily on first access.

```rust
thread_local! {
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
    static MAP: RefCell<StableBTreeMap<u128, u128, Memory>> = RefCell::new(
        StableBTreeMap::init(MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(0)))));
}
```

- The initializer cannot return an error, which is why `init` panics instead.
- It runs on the first message after install and after every upgrade, so `init` must be cheap and
  must reopen whatever is there.
- Gleaph today calls ic-stable-lara's `init_with_config` the same way, and panics with "a
  stable-memory migration is required" on a mismatch. It forces the initialization in
  `post_upgrade`, so that a layout mismatch traps the upgrade rather than a later message.

**The memory trait.**
- `ic_stable_structures::Memory` has `size`, `grow(pages) -> i64` (−1 on failure), `read`,
  `write`, and an `unsafe read_unsafe` with a default.
- A page is 64 KiB, the same as lara-core's `PAGE_SIZE`.
- Implementations: `DefaultMemoryImpl` (stable memory on wasm, a vector elsewhere),
  `VectorMemory` (`Rc<RefCell<Vec<u8>>>`), `RestrictedMemory`, `VirtualMemory`, `FileMemory`.

## lara-core today

| Concern | lara-core | ic-stable-structures |
| --- | --- | --- |
| Fresh structure | `Graph::create(Memories) -> Result<Self, CreateError>`; `NotEmpty` if a memory holds data | `new`: overwrites, returns `Self` |
| Reopen | `Graph::open(Memories) -> Result<Self, OpenError>`; `NotAGraph`, `UnsupportedVersion` | `load` (BTreeMap), or through `init` |
| Create or reopen | — (the caller tries `open`, then `create`) | `init`, returns `Self` and panics on a mismatch |
| Memories | `Memories::new(m0, m1, m2, m3, m4)`, five, by value, unnamed at the call | one per structure (`Log`: two, named `index_memory`, `data_memory`) |
| Getting them back | — | `into_memory` / `into_memories` |
| Memory trait | its own `Memory`, with `grow -> Result<u64, GrowFailed>`; `VecMemory` | `ic_stable_structures::Memory`, with `grow -> i64`; `VectorMemory`, `DefaultMemoryImpl` |
| Glue | each canister writes a newtype adapter (as `lara-core-bench` does) | — |
| Constructor cost | headers only | headers only |

## Graph vocabulary: petgraph 0.8.3 and GQL

petgraph's `StableGraph` is the closest match: indices stay valid when other nodes and edges are
removed, as lara-core's positions do.

| Concept | petgraph `StableGraph` | GQL (Gleaph's language) | lara-core |
| --- | --- | --- | --- |
| Vertex | node, `NodeIndex` | node | vertex, `VertexId` |
| Edge handle | `EdgeIndex` | edge | `EdgeRef` (vertex, label, side, position) |
| Direction | `Direction::{Outgoing, Incoming}` | directed / undirected edges | `Side::{Out, In, Undirected}` |
| Add a vertex | `add_node(weight)`, `try_add_node` | `INSERT (n)` | `add_vertex()` |
| Add an edge | `add_edge(a, b, weight)`, `try_add_edge` | `INSERT ()-[e]->()` | `insert_directed(from, label, to, placement, value)`, `insert_undirected` |
| Remove an edge | `remove_edge(e) -> Option<E>` | `DELETE e` | `remove_edge(EdgeRef) -> Removal` |
| Remove a vertex and its edges | `remove_node(a) -> Option<N>`, in one call | `DETACH DELETE n` | `detach(v)`, then `detach_step(v, budget)`; `sweep` |
| Counts | `node_count()`, `edge_count()` | — | `vertex_count()`; no edge count; `degree(v, label, side)` |
| Neighbours | `neighbors(a)`, `neighbors_directed(a, dir)`, `neighbors_undirected(a)`, returning iterators | — | `neighbors(v, label, side, &mut Vec)`, `visit_neighbors(…, Order, f)`, `visit_window`, `targets_at` |
| Edges of a vertex | `edges(a)`, `edges_directed(a, dir)` | — | the same reads, with `Edge` (position, target) |
| Edges between two vertices | `find_edge(a, b)`, `edges_connecting(a, b)`, `contains_edge(a, b)` | — | — (the target index could answer it) |
| Endpoints | `edge_endpoints(e)` | — | `edge(EdgeRef) -> Option<VertexId>`, `counterpart(EdgeRef)` |
| Edge data | `edge_weight(e)`, `edge_weight_mut(e)` | property | `value(EdgeRef)`, `set_value(EdgeRef, bytes)`, `declare_width` |
| Labels | — | label | `Label`, `labels(v, side)` |
| Fallible forms | `try_` prefix next to a panicking form | — | every mutation returns `Result` |

## Proposal

### Lifecycle (the part a canister cannot work without)

1. **`Graph::init(memories) -> Self`.** Reopens when the vertices memory holds the graph's magic;
   otherwise formats a fresh graph. It panics, with the error's message, on an unsupported version
   or memories that hold something else. Constant time. This is the call for
   `thread_local!` statics, and it serves both `#[init]` and `#[post_upgrade]`.
2. **`Graph::new(memories) -> Self`.** Formats a fresh graph. For parity with ic-stable-structures
   it **overwrites** what the memories held, where `create` refuses today. This needs checking:
   the page stores assume pages past their minted count are zero, which memory that held something
   else does not give. If formatting over old data cannot be made sound in constant time, `new`
   should keep refusing non-empty memories, and say so in its documentation.
3. **`Graph::load(memories) -> Self`.** Reopens and panics if the memories do not hold a graph.
4. **Fallible forms keep the typed errors:** `try_init`, `try_new` and `try_load`, returning
   `InitError` (merging `CreateError` and `OpenError`). They follow petgraph's `try_` naming, and
   keep the project's rule of errors as types. The panicking forms call them.
5. **`into_memories(self) -> Memories<M>`.**
6. **Name the five memories at the call.** For example `Memories { vertices, edge_pages,
   edge_directory, list_pages, list_directory }` with public fields, or a builder, instead of five
   positional arguments.

### The memory trait

7. Behind a cargo feature `ic-stable-structures`, implement `lara_core::Memory` for every
   `ic_stable_structures::Memory`. Then `VirtualMemory<DefaultMemoryImpl>` from a `MemoryManager`
   goes straight into `Memories`, and canisters drop their adapters.
   - The lara-core crate stays `no_std` without the feature.
   - Coherence with `impl Memory for &M` and `VecMemory` needs a compile check. If they clash, a
     single newtype in lara-core, `IcMemory<M>`, does the same job.

### Graph vocabulary

8. Rename towards petgraph and GQL, where the meaning is the same:

   | Today | Proposed | Why |
   | --- | --- | --- |
   | `VertexId` | `NodeId` | petgraph and GQL say node |
   | `add_vertex` | `add_node` | petgraph |
   | `vertex_count`, `vertex_state`, `VertexState`, `VertexSet` | `node_count`, `node_state`, `NodeState`, `NodeSet` | follows the rename |
   | `insert_directed`, `insert_undirected` | `add_edge`, `add_undirected_edge` | petgraph's `add_edge` |
   | `Side::{Out, In, Undirected}` | `Direction::{Outgoing, Incoming, Undirected}` | petgraph's `Direction`, with the undirected case lara-core needs |
   | `neighbors(v, label, side, &mut Vec)` | `neighbors_directed(v, label, dir, &mut Vec)` | petgraph names the directed form so |
   | `edge(EdgeRef)` | `edge_target(EdgeRef)` | says what it returns, without implying petgraph's `edge_endpoints` |
   | `value`, `set_value` | `edge_value`, `set_edge_value` | petgraph's `edge_weight`, `edge_weight_mut`, but values are bytes |

   Kept as they are, with no counterpart elsewhere or already the same: `remove_edge`,
   `counterpart`, `degree`, `labels`, `Label`, `Width`, `Placement`, `compact`, `holes`, and
   `detach`/`detach_step` (GQL's `DETACH DELETE`, in bounded steps).
9. Not adopted: iterator-returning `neighbors`. A lazy iterator would hold a borrow of the graph
   and read pages as it goes. The visitor (`visit_neighbors`) already does that without the
   borrow, and is lara-core's cheapest read (~26 instructions per edge).

### Not changed

- `Result` on every mutation (petgraph's `try_` forms are the same idea).
- Positions in `EdgeRef`: they are what makes edges addressable without an id table.

## Outcome (2026-11-23)

Adopted as proposed, with no data to keep, so with no compatibility or migration:
1. `new` overwrites. Formatting over old data is sound in constant time: every header field is
   written, and allocation zeroes what it hands out.
2. `IcMemory<M>`, a newtype behind the `ic-stable-structures` feature. A blanket impl clashes with
   `impl Memory for &M`.
3. The vocabulary rename (8) in full.
4. The memories are named by `MemoryRole` and built with `Memories::from_fn`, not public fields,
   which the `exhaustive_structs` lint forbids.
5. Later the same day: the bucket vocabulary says **label bucket** throughout, so it cannot be
   read as `MemoryManager`'s buckets beside it in a canister: `MemoryRole::LabelBucketListPages`
   and `LabelBucketListDirectory`, `Error::TooManyLabelBuckets`, and inside the crate
   `LabelBucket`, `LabelBucketKey`, `LabelBucketList` and the modules `label_bucket` and
   `label_bucket_list`. The leftover `side` names became `direction` (`EdgeRef::direction`).

**Headers.** ic-stable-structures starts each memory with a 3-byte magic and a version byte (both
of `Log`'s memories included). lara-core's nodes memory and directories already did; the pages
memories had no header and now have one, with pages starting at byte 64. Every version is 1.
