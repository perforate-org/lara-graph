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

**Equal-target edges stay in insertion order.** An insert reuses a hole only if no live edge to
the same target lies after it (otherwise it appends); paged buckets reuse holes in their last page
only. Gleaph pairs forward and reverse entries of parallel edges by their rank among equal-target
edges (ADR 0048), so this invariant keeps both sides aligned without coordinating them.

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
ranks stay aligned without links between halves. `Graph::counterpart` finds the other half by
counting in one bucket and selecting in the other: each scan stops at the edge it needs.

**Inserts are all or nothing.** First every check (vertices, bucket count, position overflow),
then the page stores grow far enough that the inserts cannot fail: an edge-bucket insert takes at
most `PAGES_PER_INSERT` (4) pages, a new bucket-list entry one. Then both halves are written.
Removal and the reads never allocate.

## Detaching a vertex

`Graph::detach(v)` marks the vertex in constant time (a spare bit of its record); from then on
inserts touching it fail with `Error::Detaching`. `Graph::detach_step(v, budget)` removes up to
`budget` of its edges and reports each one's `Halves`. For each edge:

1. Take v's first live entry in its first bucket. Paged buckets skip dead pages through the index,
   so this reads at most one page of slots.
2. Find the other half: the first live entry to v in the neighbour's opposite bucket. v's earlier
   entries to that neighbour are already gone, so this is the partner by rank, and the reported
   `Halves` are the true ones.
3. Tombstone both halves. A bucket of v left with only holes is compacted away.

When v has no buckets left, the mark is cleared. The vertex's own buckets are the cursor: nothing
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

A paged bucket's index lists its data pages; each 6-byte entry is a page id and, at the bottom
level, that page's tombstone count, so a scan skips fully dead pages without reading them. The
index is a tree of fanout 682 (one page of entries) whose depth follows from the page count and
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

A vertex's buckets are listed in its own extent in the list page store: 19-byte entries (label,
side and descriptor) sorted by `(label, side)`, up to 215 (one page). A lookup reads the vertex record and the
list once each and searches in memory; adding a label shifts entries or moves the list to the next
class. Lists are independent per vertex, unlike ic-stable-lara, which packs 16 vertices' lists
together and rewrites all of them when one outgrows its slack.

Reading one bucket costs a vertex read, a list read and one data read (index and page reads past
1024 edges).

## Not yet

- A sorted target index for paged buckets, so `counterpart`, detaching a hub's neighbour and
  joins on a target need not scan the bucket.
- Detaching many vertices at once with one pass over each shared hub.
- Inline edge properties.
- Instruction measurements on the IC (canbench) to settle constants.
