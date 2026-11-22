# Vertex detach: survey and design (2026-11-22)

Goal: remove a vertex's edges (both halves of each) with work per call bounded, because a hub may
have millions of edges. Sources: ic-stable-lara and Gleaph's facade (code reading), and a survey
of published systems.

## ic-stable-lara (Gleaph ADR 0021)

`DeleteVertexV1 { vid, removed_edges }` is a maintenance-queue item driven under a budget. The
vertex's shrinking buckets are the cursor; each step removes the top live slot of its first
non-empty bucket, then the matching half at the neighbour. Problems found:

| Problem | Effect |
| --- | --- |
| Step 0 compacts both orientations of the vertex in one step | O(degree), not bounded |
| `begin_vertex_delete_deferred` visits every incident edge and discards the result | O(degree) where O(1) is documented |
| The neighbour half is found by a linear scan for "first entry to vid", not by rank | the wrong parallel copy is reported to sidecars; O(k²) for k parallel edges |
| Each step re-reads every label bucket of the vertex | O(#labels) per edge |
| A separate pending-purge bitmap hides half-removed edges from reads | write-order hazards (ADR 0021 addendum), a filter on read paths |
| Span releases are not batched in the stepped path | ~26-32K instructions per emptied neighbour bucket |

## Other systems

Almost no dynamic-graph research store bounds the cost of deleting a hub. Either vertex deletion
is missing or it is eager and O(degree). Production databases batch the work or tolerate dangling
halves.

| System | Mechanism |
| --- | --- |
| Neo4j | Relationship records are in doubly-linked chains at both ends, so unlinking the far half is O(1). One DETACH DELETE is unbounded (transaction memory); the advice is to batch edge deletes first. |
| Memgraph | Empties the vertex's own lists, groups the far ends, and makes one partition pass per neighbour. Unbounded; GC reclaims later. |
| Teseo, STINGER, LLAMA, Kùzu, RapidStore | Walk the vertex's own out- and in-lists and remove each pair. O(degree), in one call. |
| GraphOne | Sets a flag and writes tombstoned edges to the log; archiving applies them per vertex in batches. |
| JanusGraph | Tolerates half edges and "ghost vertices"; a full-scan job removes them. |
| NebulaGraph, Dgraph | Leave dangling or inbound edges unless the user removes them. |
| LiveGraph, GTX, Sortledton, DGAP, PCSR family | No vertex deletion. Sortledton notes it cannot be O(log V). |
| RadixGraph, RisGraph | Reuse an id only after its stale references are gone (time-gated queue or recycle pool). |

The techniques that fit LARA's slot layout:

1. **Use the vertex's own buckets as the to-do list** and remove one pair at a time.
2. **Group work by neighbour** when several vertices are detached at once: one pass over a hub
   removes the entries of every detaching vertex.
3. **Lazy purge**: hide the vertex on read and drop its entries at compaction. This puts a filter
   on every scan, which is against priority 1. Rejected.
4. **Gate id reuse** on stale references being gone. Gleaph never reuses ids, so this is not
   needed here.

## Design for lara-core

`Graph::detach(v)` is O(1): it marks the vertex record *detaching* (a spare bit of the record).
While a vertex is detaching, inserts touching it fail with `Error::Detaching`. Removing edges and
compaction stay allowed.

`Graph::detach_step(v, budget)` removes up to `budget` edges. Each edge goes like this:

1. Take v's first bucket in list order. Take its first live entry `(p, u)`; paged buckets skip
   fully dead pages through the index, so this costs at most one page of slots.
2. The other half is the **first live entry to v** in u's opposite bucket: `In` for `Out`, `Out`
   for `In`, `Undirected` for `Undirected`.
   - Because v's entries go in position order, the earlier ones are already gone, so this is the
     true partner by rank. Sidecars therefore get the right `Halves`, unlike ic-stable-lara.
   - An in-step hint per neighbour skips entries already passed.
3. Tombstone both halves and report the `Halves`.
4. A bucket of v that has become all holes is compacted away within the same budget. A paged
   bucket's dead pages are hollowed one per step.

When v has no buckets left, the mark is cleared and the step returns `Progress::Done`.

Properties:

- **Consistent at every step boundary.** Every edge is either whole or gone, so the graph passes
  its invariants after each step. Reads need no filter and no pending-purge set.
- **No persisted cursor.** The vertex's buckets are the cursor, and compaction between steps
  cannot invalidate anything.
- **Cost of a step.** For each edge removed:
  - finding v's next live entry costs at most one page of slots, plus index entries to skip dead
    pages;
  - finding the other half scans u's bucket up to it, which is the worst case when a leaf hangs
    off a hub.

  The sorted target index for paged buckets would make the second part O(log deg u).
  Detaching many vertices that share a hub can later group the hub scans (technique 2).
