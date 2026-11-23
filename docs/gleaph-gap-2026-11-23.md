# What lara-core still lacks for Gleaph (2026-11-23)

This survey compares the uses Gleaph makes of ic-stable-lara today with lara-core's public API at
793820f. The inventory of Gleaph's uses covers every raw call into
`DeferredBidirectionalLabeledLaraGraph`. All of them sit in `gleaph/crates/graph/src/facade/`, and
the rest of Gleaph reaches them through `GraphStore` wrappers.

Below, `F/` stands for `gleaph/crates/graph/src/facade/`.

## Summary

- **The graph operations themselves are covered.** Inserts, removals, reads, values, pairing and
  detach all have a lara-core counterpart. Several are simpler or cheaper than what Gleaph uses
  today:
  - an edge is looked up by handle in O(1), where Gleaph scans the bucket;
  - a removal moves no other edge;
  - compaction returns its moves as data, instead of calling observers back inside a borrow.
- **Two gaps blocked a switch; both are now settled.** In order of weight:
  1. **Deferred maintenance.** Gleaph drains a durable queue under an instruction budget, from a
     timer. lara-core has only the explicit steps (`compact`, `detach_step`). Someone has to keep
     the list of pending work across messages and upgrades.
  2. **Unordered buckets do not heal themselves.** lara-core's `ReuseHoles` refills holes only in
     a paged bucket's last page. Gleaph never compacts Unordered buckets, and relies on inserts to
     refill their holes.
- **The rest is adapter work or small API additions.** Each is listed below.
- **No migration is needed.** Gleaph has no deployed data to carry over (decided 2026-11-23), so
  lara-core starts empty and needs no loader that preserves slot numbers.

## Coverage, need by need

Status key:
- **Covered**: lara-core has it.
- **Adapter**: the Gleaph adapter can build it from lara-core's API.
- **Gap**: something has to be added or decided.

| Gleaph need | Today (ic-stable-lara) | lara-core | Status |
| --- | --- | --- | --- |
| Create and reopen, failing on layout skew | `init_with_config` over 33 virtual memories | `Graph::create` / `Graph::open` over 5 memories; `OpenError::{NotAGraph, UnsupportedVersion}` | Covered. Five new memory ids are needed. |
| Dense vertex ids, never reused | `push_vertex_row`, `push_vertices`, `vertex_count` (tombstones included) | `add_vertex`, `vertex_count` (deleted vertices included) | Covered. Bulk add is a loop. |
| Vertex liveness | `vertex_row(v).is_tombstone()` | `vertex_state` → `Live` / `Detaching` / `Deleted` | Covered |
| Edge-free check before a plain DELETE | `has_incident_edges` | `has_edges(v)`: any side, past buckets of holes | Covered |
| Delete an edge-free vertex | `delete_vertex_deferred` | `detach` followed by one `detach_step` | Covered |
| Labels and directedness | `BucketLabelKey` u16 with a directed bit | `Label(u16)` and `Side::{Out, In, Undirected}` | Adapter: a fixed mapping |
| Buckets per vertex | 1024 per orientation | tiered lists, ~110 000 | Covered |
| Paired insert with exact positions | `insert_*_with_locations` returning `ScalarInsertPair` | `insert_directed` / `insert_undirected` returning `Halves` | Covered |
| Insertion or Unordered placement | `EdgePlacementPolicy` per label | `Placement::{Append, ReuseHoles}` per insert | Covered. Unordered differs in hole reuse (gap 2). |
| Inline value width | per bucket, `ensure_*_width` before an insert | per label, `declare_width`; a different width is `WidthConflict` | Covered, as long as a label's width never changes. See question 3. |
| Remove at a slot, with reported moves | `remove_*_at_slot_with_move` | `remove_edge(EdgeRef)`: both halves, the value returned, no moves | Covered. The adapter reports no moves. |
| Remove-side compaction trigger | `maybe_enqueue_remove_side_compaction` with a policy closure | `holes(v, label, side)` to decide, `compact` to act | Adapter, plus gap 1 |
| Counterpart and canonical half | `counterparts().of` / `.canonical_handle`; `canonical_undirected_owner` (the larger id) | `counterpart`; `Halves::canonical()`, the larger id for undirected edges | Covered. Same rule. |
| Stale handles fail closed | `SourceNotFound` | `counterpart` / `edge` return `None` for a hole | Covered while the slot stays a hole. A reused hole answers with the new edge, as today. |
| Paired value update | `update_directed_edge_inline_property` | `set_value`: both halves, returns the old value | Covered |
| Scans of one label, either order | `for_each_*_for_label_ordered`, topology-only variants | `visit_neighbors(.., Order, ..)`, `neighbors` | Covered |
| Scans across all labels | `for_each_directed_out_edges` and the like | `labels(v, side)`, then one visit per label | Adapter |
| OFFSET skip, then visit (insertion DESC) | `skip_then_visit_each_*` | `visit_neighbors` in `Descending` order, counting the skip in the closure | Adapter. It skips edge by edge, not by page counts. |
| Reads of chosen positions | `read_*_edge_slots_for_label` | `targets_at`, `values_at` | Covered |
| Value batches, values first | `visit_*_inline_property_batches_for_label` | `visit_values` (per value; the adapter can batch) | Covered |
| Edges with values | `visit_*_edge_inline_property_batches_for_label` | `visit_neighbors_with_values` | Covered |
| Eligibility probe for values-first expand | `*_inline_property_bytes_first_predicate_eligible` | — | Not needed: every bucket with a width can be read values-first. |
| Resumable window for export | `visit_forward_edges_from_slot_with_inline_property` → `(next, exhausted)` | `visit_window(.., Cursor, ..)` → next `Cursor` or none | Covered |
| Edge by handle | linear scan for the slot | `edge(EdgeRef)`, O(1) | Covered |
| Label of an edge | `find_forward_edge_label(src, &E)` | the handle carries the label | Adapter |
| DETACH: purge without blocking | `begin_vertex_delete_deferred`, then the queue with `DeleteEdgeObserver` | `detach`, then `detach_step(v, budget)` returning `RemovedEdge`s with values | Covered, but who drives the steps is gap 1 |
| DETACH pre-compaction of spans | `mark_compact_vertex_edge_span` | — | Not needed: detach does not depend on compaction. |
| Compaction moves | `EdgeSlotMoveObserver` callbacks inside the graph borrow | `compact` returns `Step<Move>`; the caller re-keys after the call | Covered, and simpler |
| Budgeted drain | `maintenance_with_observers(MaintenanceBudget)` → report | `compact` and `detach_step` steps, sized by edge count | Gap 1 |
| Recoverable errors before the first write | relied on after the index fence | inserts are all or nothing; removals take no memory | Covered |
| Remote targets | `VertexRef` remote bit | — | Not needed: Gleaph's remote builders have no callers. |
| Reverse-mirror repair | `repair_*`, `remove_reverse_edge_*` | — | Not needed: dead in production. One graph cannot diverge. |

**Fixed since:** lara-core did not export `RemovedEdge`, so callers could call its `halves()` and
`value()` but not name the type. It is re-exported now.

## The gaps

### 1. Deferred maintenance

Gleaph's maintenance today works like this:
- **A durable queue** in memory 30.
- **Work items** that capture a label's policy.
- **A timer** that drains the queue under an instruction budget and re-arms.
- **Message-end drains.**

lara-core's work is explicit and bounded per call:
- `compact(v, label, side)` takes one step;
- `detach_step(v, budget)` removes up to `budget` edges.

What is missing is the list of pending work: buckets to compact, and vertices being detached.

- **In the adapter.** A small stable map holds `(vertex, label, side)` entries to compact and the
  vertices in `Detaching`. The adapter's timer loop checks the IC's performance counter between
  steps. This keeps lara-core free of scheduling policy.
- **In lara-core.** A worklist kept in the graph's memories. Detaching vertices are already known
  from their records, but finding them means a scan.

**Decided (2026-11-23): the Gleaph adapter keeps the list.**
- It records a vertex when it calls `detach`, and a bucket when `holes()` passes its threshold.
- Its timer runs the steps against the IC's instruction counter.
- lara-core stays free of scheduling. Since the adapter records detaching vertices itself,
  lara-core does not need a way to find them.

### 2. Unordered buckets do not heal themselves: resolved

**Decided (2026-11-23): extend hole reuse.** This keeps positions stable, so nothing keyed by
position is ever re-keyed, and Unordered buckets need no compaction.

What `ReuseHoles` now does:
- It takes the first hole after the last live edge to the same target, anywhere in the bucket.
- A paged bucket finds that last edge through its target index. The index is skipped when the
  other end's bucket shows no edge between the two.
- The page index leads straight to the first page with a hole. Its entries are now 8 bytes: each
  node counts its children with holes, and each data page records its first hole.

Measured on canbench, on a 65536-edge hub with every other edge removed:

| Refill | Per insert |
| --- | ---: |
| Edges to new vertices | ~29K |
| A second edge to each remaining neighbour | ~34K |
| Append, for comparison | ~12K |

- The target index's +1 is written in place without re-coding the segment: a held posting's
  count bits, or a new posting's bits with the rest of the code shifted along.
  - Through a plan, the +1 cost ~25K, and refills ~40K and ~44K. Re-coding the segment in its
    slot brought them to ~33K and ~38K.
- Other benches moved by +2 to +5 %: the 8-byte page-index entries, and the check for holes on
  every insert.

## Widths: answered from Gleaph's design

- **Gleaph fixes a label's width once its edges exist.** ADR 0008, "Risks", says: if a LARA bucket
  already holds edges at a different width from the router's schema, refuse the DML. A width
  change is not carried over to existing edges.
- **lara-core's per-label width fits this.** It is one width per label, and a different one is
  `WidthConflict`.
- **One difference remains.** A label whose schema is redefined after all its edges are gone
  still keeps its old width in lara-core. Gleaph's per-bucket width would let new buckets take the
  new width.
  - If Gleaph can redefine a label's schema that way, lara-core needs a way to clear a label's
    width once no bucket uses it.

## Small additions to lara-core

- Done (2026-11-23):
  - `RemovedEdge` is re-exported.
  - `Graph::has_edges(v)` answers Gleaph's edge-free check in one call.
- Optional: page-level skipping for an OFFSET of live edges, using the per-page live counts.
  Gleaph's insertion-DESC fast path skips edge by edge today as well.
