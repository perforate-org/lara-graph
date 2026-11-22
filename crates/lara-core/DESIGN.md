# lara-core design

This note records the decisions behind the code. The goals come from Gleaph's
`design/storage/lara.md` and its LARA ADRs; the layout differs on purpose, as decided below. The
evidence is in `docs/` (layout, page source and storage comparisons, the ic-stable-lara review).

## Priorities

1. **CSR-level scans.** A bucket is read as contiguous slots, and a read never consults
   allocation metadata.
2. **Bounded updates.** An insert or removal copies at most one run below a page or one page of
   index entries. Every update finishes inside the call: there is no deferred maintenance.
3. **Support.** Inline (tiny) buckets, per-label buckets that narrow a scan, constant-time
   allocation that shares freed space across every use.

## How the rules map to code

| Rule | Mechanism |
| --- | --- |
| Behaviour in types | Newtypes for ids and units (`VertexId`, `Label`, `Position`, `PageId`, `ByteClass`, `ByteOffset`); one enum per exclusive state (`Side`, `Halves`, `Descriptor`, `Slot`, `PageRecord`, `Lookup`, `TopChange`, `TombstoneChange`, `Removal`); no sentinel values in the model — they exist only inside wire codecs. |
| Scan contract | `bucket::read` and `bucket::get` take a `Pages` view: page data only, no directory. A read *cannot* consult allocation metadata. |
| Encapsulation | Every module is private; `lib.rs` re-exports the public surface and `tests/public-api.txt` snapshots it. Fields are private. |
| RAII | `SpanTxn` scopes updates that allocate more than once: extents allocated through it are returned on drop unless committed. |
| All or nothing | An edge insert checks every half and reserves memory for all of them before writing any (see "Both ends"). |
| DRY | One page store serves edges and per-vertex label lists; one bucket layer serves every label; one index walk serves reads and checks. |
| Lints | The workspace `Cargo.toml` and `clippy.toml`; the gate is `just gate`. |

## Memories

A graph uses five memories (Gleaph wires each to a stable-memory id):

| Memory | Holds |
| --- | --- |
| vertices | a header and one 8-byte record per vertex: the extent of its bucket list, its bucket count, and whether it is detaching |
| edge pages / edge directory | the edge page store |
| list pages / list directory | the bucket-list page store |

Every piece of state lives behind the `Memory` seam, so a graph reopens by reading three headers.
`Memory` mirrors `ic_stable_structures::Memory`; it is the one public trait taking raw `u64`.

## The page store

Everything is a 4096-byte page. Pages hold data only, at `page_id × 4096`; a separate directory
holds one 32-byte record per page (free, chunk, or whole). An allocation is an `Extent` of a
power-of-two byte class from 32 to 4096. Classes below a page are carved from chunk pages: one
class per page, a 128-bit slot bitmap, and a per-class list of chunk pages with room. Freed pages
go to one free list and serve any class or use.

Chosen over a TLSF span allocator with boundary tags (32 bytes of metadata per span) because
per-bucket runs are mostly 8-16 slots: counted with metadata, the page store used less memory
than both TLSF with chunks and PMA leaves, with fewer memory calls and the same copying
(`docs/storage-compare-2026-11-22.md`). The TLSF allocator is kept under the `bench-internals`
feature for those comparisons.

Allocation fails only when a memory cannot grow, and grows before changing anything. Release
never fails, which is what lets `SpanTxn` roll back in `Drop`.

## Buckets

Each `(vertex, label, side)` has a bucket with a 16-byte `Descriptor`:

- **Tiny**: up to 3 slots inside the descriptor.
- **Run**: one extent of up to a page (1024 slots), at its front.
- **Paged**: more than 1024 slots in data pages, listed by a page index.

A slot is an edge or a tombstone (`u32::MAX`, never a vertex id). **Positions count tombstones**
and are stable: removal writes a tombstone in place; a run grows by copying its slots, tombstones
included; a full one-page run becomes data page 0 without copying. Only compaction renumbers
positions, and it reports each move.

**Each insert says where it may go** (`Placement`). `Append` always goes after every stored
position, so position order is insertion order: this is Gleaph's `Insertion` order, and
compaction keeps it. `ReuseHoles` refills a hole when one is safe, without waiting for
compaction; this is Gleaph's `Unordered`. Appending never reads the bucket's slots.

**Equal-target edges stay in insertion order** either way. `ReuseHoles` takes the first hole
after the last live edge to the same target, and appends when there is none. Gleaph pairs forward
and reverse entries of parallel edges by their rank among equal-target edges (ADR 0048), so this
invariant keeps both sides aligned without coordinating them.

**Reused holes keep positions stable.** An `Unordered` bucket heals itself through inserts and
never needs compaction, so nothing keyed by position is ever re-keyed.
- In a paged bucket, the last edge to the target comes from the target index. When the other end's
  bucket is absent, or tiny without an edge back, there is none and the index is not read.
- The page index then leads straight to the first page with a hole after it, and the page's entry
  names its first hole.
- A fill costs ~40-45K instructions on a 65536-edge hub, against ~12K for an append. Most of it is
  the target index's new posting: its segment is decoded and coded again.

The descriptor size is set by memory: on sparse graphs per-vertex descriptors dominate, and 16
bytes with 3 inline slots used 8-17% less memory than 24 bytes with 5
(`docs/storage-compare-2026-11-22.md`).

## Both ends

Every edge is stored at both ends, as two **halves** in one graph: `u → v` is an entry to `v` in
`u`'s `Out` bucket and an entry to `u` in `v`'s `In` bucket; `u — v` is an entry in each end's
`Undirected` bucket, and an undirected self-loop is one entry. One graph (not a forward and a
reverse graph) keeps one set of memories and one bucket list per vertex, and lets an insert check
both halves before writing either.

**Halves pair by rank.** The k-th live entry to `v` in `u`'s out bucket is the same edge as the
k-th live entry to `u` in `v`'s in bucket (Gleaph ADR 0048). Inserts place an entry after every
live entry to the same vertex, removal takes both halves, and compaction keeps order, so the
ranks stay aligned without links between halves. `Graph::counterpart` first looks at the smaller of the two
buckets. If that bucket holds exactly one live entry back to the half's vertex, that entry is the
other half. Otherwise it counts in one bucket and selects in the other, and each scan stops at the
edge it needs. Removing a hub's edge to a leaf therefore never scans the hub.

**The canonical half** is where a caller keys what belongs to the edge (properties, index
postings): the out half of a directed edge, and the half at the **larger** vertex id of an
undirected one. Ids grow in creation order, and hubs tend to be old. So the canonical halves of a
hub's undirected edges sit in its newer neighbours' small buckets: the keyed data spreads out,
and compacting a hub moves few canonical halves. `Halves` lists the canonical half first.

**Inserts are all or nothing.** First every check (vertices, bucket count, position overflow),
then the page stores grow far enough that the inserts cannot fail: an edge-bucket insert takes at
most `PAGES_PER_INSERT` (4) pages, a bucket-list change three. Then both halves are written.
Removal and the reads never allocate.

## Detaching a vertex

A vertex is `Live`, `Detaching` or `Deleted` (`Graph::vertex_state`; two spare bits of its record).
Ids are never reused. `Graph::detach(v)` marks the vertex `Detaching` in constant time; from then
on inserts touching it fail with `Error::Detaching`. `Graph::detach_step(v, budget)` removes up to
`budget` of its edges and reports each one's `Halves`. For each edge:

1. Take v's first live entry in its first bucket. Paged buckets skip dead pages through the index,
   so this reads at most one page of slots.
2. Find the other half: the first live entry to v in the neighbour's opposite bucket. v's earlier
   entries to that neighbour are already gone, so this is the partner by rank, and the reported
   `Halves` are the true ones.
3. Tombstone both halves. A bucket of v left with only holes is compacted away.

When v has no buckets left, it is `Deleted` for good: inserts touching it fail with
`Error::Deleted`. The vertex's own buckets are the cursor: nothing
else is persisted, so compaction or reopening between steps cannot invalidate a step. Every edge
is whole or gone at each step boundary, so reads need no filter for half-removed edges.

Compared with ic-stable-lara's `DeleteVertexV1`:
- there is no unbounded first step and no O(degree) start;
- the partner is matched by rank, not by "first entry to v";
- there is no separate pending-purge set.

The survey is in `docs/vertex-detach-2026-11-22.md`.

The remaining cost is the neighbour scan: detaching a leaf of a hub scans the hub's bucket up to
the leaf's entry.

## The page index

A paged bucket's index lists its data pages. Each 8-byte entry is a page id, a count and a slot.
- At the bottom level, the count is the page's tombstones, so a scan skips fully dead pages
  without reading them. The slot is the page's first hole, so a reusing insert finds one without
  reading the page.
- Above it, the count is how many of the node's children have a fillable hole, so the search for
  one goes straight down. A hollow page's holes are not filled.

The index is a tree of fanout 512 (one page of entries) whose depth follows from the page count and
is never stored. The top is an extent sized to its entries: it grows by copying at most one page,
and a full one-page top becomes the first child of a new top without copying (as in
ic-stable-lara's Tree-CSR). Three levels cover every position a `u32` can name.

## Compaction

`Graph::compact(v, label, side)` reclaims one bucket's holes in one bounded step and returns every edge
that moved (old position, new position, target) and whether more remains. It is explicit so the
caller decides when to pay for moves: Gleaph re-keys sidecar properties and index postings for
each one. `Graph::holes` tells the caller how much there is to reclaim.

Every step packs live edges toward the front, so their relative order holds: insertion order and
the rank pairing of parallel edges survive without coordinating forward and reverse buckets.

| Bucket | One step | Moves at most |
| --- | --- | --- |
| tiny or run | pack; shrink to the smallest class, to tiny at 3 or fewer, remove the label at 0 | 1023 |
| paged, at most 512 live | demote: pack into a run or tiny bucket, release pages and index | 512 |
| paged, more live | release a dead page (not the last) and leave its index entry hollow; or merge the next page with content into a page with holes when both fit one page | 1024 |

Hollow entries let a page go without renumbering any later position. The last page is never
hollowed or merged: inserts reuse its holes and append after it.

## Bucket lists

A vertex's buckets are listed in the list page store: 19-byte entries (label, side and
descriptor) sorted by `(label, side)`.

- **Flat**, up to 215 entries (one page). A lookup reads the vertex record and the list once each
  and searches in memory. Adding a bucket shifts entries or moves the list to the next class.
- **Tiered**, past that. A top extent lists leaf pages, each with its first key and entry count
  (8 bytes per leaf), and each leaf holds up to a page of entries.
  - A lookup reads the top and one leaf.
  - A full leaf splits into two half-full pages.
  - An emptied leaf is released, and a list down to one leaf turns flat again.
  - Every change copies at most one page. The top holds 511 leaves: at least ~55 000 buckets
    when every leaf is half full, and up to ~110 000. Lists are independent per vertex, unlike ic-stable-lara, which packs 16 vertices' lists
together and rewrites all of them when one outgrows its slack.

Reading one bucket costs a vertex read, a list read and one data read (index and page reads past
1024 edges).
`Graph::visit_neighbors` reads without collecting, at about 30 instructions per edge on the IC.
`neighbors`, which fills a `Vec`, costs about twice that (`docs/ic-measurements-2026-11-22.md`).

## Edge values

A label may carry a fixed-width value on every edge (`Graph::declare_width`, 1 to 4096 bytes,
the same on every side). Both halves hold the value. The design and its reasons are in
`docs/inline-properties-design-2026-11-22.md`.

- **A column beside the slots, addressed by position.**
  - A bucket's **value plane** (`values`) holds the value of position `p` in cell `p`.
  - Slot pages and scans of targets are untouched.
  - A delete writes no value, and a hole refill writes one.
  - Only compaction moves values, one cell per moved edge. It also releases the value pages of
    positions that no longer hold edges: past a shrunken bucket's end, or in a slot page it
    released.
- **Values read as zeros until written.** A plane is empty, a run, or a sparse radix table of
  value pages, where missing pages read as zeros.
  - Declaring a width costs O(1) and writes nothing.
  - A bucket gets its plane on its first value write.
  - Any write allocates at most the path to its page: `PAGES_PER_WRITE`, 9 pages.
- **Where a plane is found.** A bucket with values or a target index keeps a 32-byte header (slot
  descriptor, plane root, width, index root) and its list entry says `Extended(header)`
  (`stored`).
  - Buckets with neither are unchanged.
  - A bucket with values is never tiny.
- **The width table.** Widths live in a table in the list store, referenced from the vertices
  header and mirrored in the heap. Only 0 → w is allowed.
- **Reads.**
  - `value` reads one edge; `values_at` reads chosen positions.
  - `visit_values` scans the value plane alone, as the first phase of a values-first traversal.
    It skips slot pages that are all dead; holes come with stale bytes.
  - `visit_neighbors_with_values` joins a slot page's live edges with one read of their values.
- **Writes and reports.**
  - `set_value` rewrites both halves and returns the old value.
  - Removal and detach report the canonical half's value (`RemovedEdge`).

## Reads

Every read walks a range of positions in either `Order`, and reads pages through the index,
skipping dead pages.

- **`visit_neighbors`** visits a whole bucket, ascending or descending. Descending is newest
  first under `Placement::Append`.
- **`visit_window`** reads a `Cursor` window: `span` positions from a start, holes included. Each
  call therefore reads at most `span` slots, however many edges were removed, and returns the next
  cursor. Positions are stable, so a cursor holds until the bucket is compacted. A descending
  start past the end is clamped to the last position.
- **`targets_at`** returns the targets at chosen positions, in the order given. It reads each
  touched run or page once, from its first wanted position to its last. Gleaph uses this for the
  second phase of property-first expansion.

## Target index

A paged bucket keeps a **counted B+ tree of postings** (`targets`, `graph/index`). A posting
`(target, block, count)` says that `count` live entries to `target` sit among the 64 positions of
`block`. Rank and selection of a target's entries are therefore logarithmic, followed by one block
scan. This matches halves in `counterpart`, removal and detach without scanning a hub. A target's
postings are usually reached in one descent, by walking the leaf where they start.

- **Coverage.** The index covers the bucket's complete blocks; the last, partial block is scanned
  instead.
  - An append that completes a block brings the block's live entries in as one batch, sized
    beforehand by a counting pass.
  - A removal lowers its posting's count in place in one descent, and takes no memory. Counts are
    fixed-width per segment, and a posting that reaches zero stays until its segment is
    rewritten. Batches of decrements are lowered in place too.
  - Batches are planned once in memory, reserved exactly and committed. A block's batch is planned
    before the insert that completes the block; compaction plans its moves.
- **Leaves are slotted pages of coded segments.** Each segment holds 16-32 postings, coded as
  Rice target gaps, fixed-width counts and fixed-width blocks.
  - A change reads and writes only the segments it touches.
  - Internal nodes keep keys, pages and counts in separate arrays, so a lookup reads only what it
    probes.
- **A detaching vertex drops its buckets' indexes**, since all their edges are going. Lookups into
  such a bucket scan it.
- **Costs.** Measurements and the space it takes are in
  `docs/target-index-design-2026-11-22.md`:
  - pairing through a hub costs ~17-28K instructions instead of a scan;
  - hub inserts cost about 15-20 % more than before the index;
  - removals from a hub's side cost one in-place index change each, ~6K;
  - the index takes 0.8-1.4× the slots for hubs of distinct targets.

## Not yet

- A cheaper index change for a filled hole: a new posting re-codes its whole segment (~25K
  instructions). Bumping an existing posting's count in place, and appending a posting past a
  segment's last key, would avoid it.
- Reclaiming zero-count postings and index pages of hubs whose edges were mostly removed,
  short of demotion.
- Detaching many vertices at once with one pass over each shared hub.
- Recovering the ~7% values added to removal with compaction and to detach: compaction still
  allocates its buffers (`docs/ic-measurements-2026-11-22.md`).
- Descending scans of a paged bucket read its whole leaf list; a reverse index walk would read
  only the pages wanted.
- Settling the constants (descriptor size, inline slots, page threshold) by measuring on the IC
  with canbench (`crates/lara-core-bench`).
