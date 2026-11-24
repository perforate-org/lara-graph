# lara-graph design

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
| Behaviour in types | Newtypes for ids and units (`NodeId`, `Label`, `Position`, `PageId`, `ByteClass`, `ByteOffset`); one enum per exclusive state (`Direction`, `Halves`, `Descriptor`, `Slot`, `PageRecord`, `Lookup`, `TopChange`, `TombstoneChange`, `Removal`); no sentinel values in the model — they exist only inside wire codecs. |
| Scan contract | `bucket::read` and `bucket::get` take a `Pages` view: page data only, no directory. A read *cannot* consult allocation metadata. |
| Encapsulation | Every module is private; `lib.rs` re-exports the public surface and `tests/public-api.txt` snapshots it. Fields are private. |
| RAII | `SpanTxn` scopes updates that allocate more than once: extents allocated through it are returned on drop unless committed. |
| All or nothing | An edge insert checks every half and reserves memory for all of them before writing any (see "Both ends"). |
| DRY | One page store serves edges and per-node label bucket lists; one label bucket layer serves every label; one index walk serves reads and checks. |
| Lints | The workspace `Cargo.toml` and `clippy.toml`; the gate is `just gate`. |

## Memories

A graph uses five memories, each named by a `MemoryRole` (Gleaph wires each to a stable-memory
id; `Memories::from_fn` builds the set from a function of the role):

| Memory | Holds |
| --- | --- |
| nodes | a header and one 8-byte record per node: the extent of its label bucket list, its label bucket count, and whether it is detaching |
| edge pages / edge directory | the edge page store |
| label bucket list pages / directory | the label bucket list page store, and the width table |

**Headers.** As in ic-stable-structures, every memory starts with a 3-byte magic and a version
byte: `LGN` (nodes), `LPS` (a directory) and `LPD` (a pages memory), all at version 1. A page
store's headers take 64 bytes, and a pages memory's pages start after its header; the nodes
memory's header takes 16 (with the node count and the width table's reference).

**Layout diagrams.** As in ic-stable-structures, each wire format is drawn in its module's
documentation: the nodes memory (`graph`), a node record (`node`), the page store's memories and
`Extent` (`page_store`) and directory record (`page_store::record`), label bucket lists
(`label_bucket_list`, `tiered`), descriptors (`descriptor`), extended entries and headers
(`stored`), value planes (`values`), page index entries (`paged`), the target index's nodes and
leaves (`targets::tree`, `targets::leaf`), and the width table (`graph::widths`).

**Lifecycle**, following ic-stable-structures:
- `Graph::new` formats all five memories, overwriting what they held, in constant time: every
  header field is written, and allocation zeroes what it hands out, so nothing old is read.
- `Graph::load` reopens, reading the five headers.
- `Graph::init` loads when the nodes memory holds the graph's magic, and calls `new` otherwise. It
  is the call for a canister's `thread_local!` statics and serves `#[init]` and `#[post_upgrade]`
  alike.
- They panic on a failure; `try_new`, `try_load` and `try_init` return `InitError`
  (`OutOfMemory`, `NotAGraph`, `UnsupportedVersion`). `into_memories` gives the memories back.

Every piece of state lives behind the `Memory` seam. `Memory` mirrors
`ic_stable_structures::Memory`; it is the one public trait taking raw `u64`. With the
`ic-stable-structures` feature, every memory of that crate is one (a `VirtualMemory` from a
`MemoryManager`, say), with a failed `grow` (−1) mapped to `GrowFailed`. A shared reference is
not a memory, as ic-stable-structures could make it one of its own and the two would clash;
`VecMemory` clones share their bytes instead, as its `VectorMemory` does.

## The page store

Everything is a 4096-byte page. Pages hold data only, at `64 + page_id × 4096` (after the header); a
separate directory holds one 32-byte record per page (free, chunk, or whole). An allocation is an
`Extent` of a power-of-two byte class from 32 to 4096. Classes below a page are carved from chunk
pages: one class per page, a 128-bit slot bitmap, and a per-class list of chunk pages with room.
Freed pages go to one free list and serve any class or use.

Chosen over a TLSF span allocator with boundary tags (32 bytes of metadata per span) because
per-bucket runs are mostly 8-16 slots: counted with metadata, the page store used less memory
than both TLSF with chunks and PMA leaves, with fewer memory calls and the same copying
(`docs/storage-compare-2026-11-22.md`). The TLSF allocator is gone from the crate; it is in the
history under the tag `lara-graph-legacy`.

Allocation fails only when a memory cannot grow, and grows before changing anything. Release
never fails, which is what lets `SpanTxn` roll back in `Drop`.

## Buckets

Each `(node, label, direction)` has a **label bucket** (a bucket, for short, in the rest of this
document) with a 16-byte `Descriptor`:

- **Tiny**: up to 3 slots inside the descriptor.
- **Run**: one extent of up to a page (1024 slots), at its front.
- **Paged**: more than 1024 slots in data pages, listed by a page index.

A slot is an edge or a tombstone (`u32::MAX`, never a node id). **Positions count tombstones**
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
- A fill costs ~29-34K instructions on a 65536-edge hub, against ~12K for an append.
- The target index's +1 goes in place (`targets::raise`), without re-coding the segment.
  - A posting the segment holds gets its count bits rewritten.
  - A new posting is coded after the ones before it, then the next posting's gap is coded again,
    and the bits after it move along unchanged. Table entries record a code's length in bits, so
    the end is known without decoding.
  - A segment that cannot take it (a count or block past its width, a full segment or slot, a new
    first key) goes through a plan instead.

The descriptor size is set by memory: on sparse graphs per-node descriptors dominate, and 16
bytes with 3 inline slots used 8-17% less memory than 24 bytes with 5
(`docs/storage-compare-2026-11-22.md`).

## Both ends

Every edge is stored at both ends, as two **halves** in one graph: `u → v` is an entry to `v` in
`u`'s `Outgoing` bucket and an entry to `u` in `v`'s `Incoming` bucket; `u — v` is an entry in each
end's `Undirected` bucket, and an undirected self-loop is one entry. One graph (not a forward and a
reverse graph) keeps one set of memories and one label bucket list per node, and lets an insert
check both halves before writing either.

**Halves pair by rank.** The k-th live entry to `v` in `u`'s out bucket is the same edge as the k-th
live entry to `u` in `v`'s in bucket (Gleaph ADR 0048). Inserts place an entry after every live
entry to the same node, removal takes both halves, and compaction keeps order, so the ranks stay
aligned without links between halves. `Graph::counterpart` first looks at the smaller of the two
buckets. If that bucket holds exactly one live entry back to the half's node, that entry is the
other half. Otherwise it counts in one bucket and selects in the other, and each scan stops at the
edge it needs. Removing a hub's edge to a leaf therefore never scans the hub.

**The canonical half** is where a caller keys what belongs to the edge (properties, index
postings): the out half of a directed edge, and the half at the **larger** node id of an
undirected one. Ids grow in creation order, and hubs tend to be old. So the canonical halves of a
hub's undirected edges sit in its newer neighbours' small buckets: the keyed data spreads out,
and compacting a hub moves few canonical halves. `Halves` lists the canonical half first.

**Inserts are all or nothing.** First every check (nodes, bucket count, position overflow),
then the page stores grow far enough that the inserts cannot fail: an edge-bucket insert takes at
most `PAGES_PER_INSERT` (4) pages, a label bucket list change three. Then both halves are written.
Removal and the reads never allocate.

## Detaching a node

A node is `Live`, `Detaching` or `Deleted` (`Graph::node_state`; two spare bits of its record).
Ids are never reused. `Graph::detach(v)` marks the node `Detaching` in constant time; from then
on inserts touching it fail with `Error::Detaching`. `Graph::detach_step(v, budget)` removes up to
`budget` of its edges and reports each one's `Halves`. For each edge:

1. Take v's first live entry in its first bucket. Paged buckets skip dead pages through the index,
   so this reads at most one page of slots.
2. Find the other half: the first live entry to v in the neighbour's opposite bucket. v's earlier
   entries to that neighbour are already gone, so this is the partner by rank, and the reported
   `Halves` are the true ones.
3. Tombstone both halves. A bucket of v left with only holes is compacted away.

When v has no buckets left, it is `Deleted` for good: inserts touching it fail with
`Error::Deleted`. The node's own buckets are the cursor: nothing
else is persisted, so compaction or reopening between steps cannot invalidate a step. Every edge
is whole or gone at each step boundary, so reads need no filter for half-removed edges.

**Many nodes at once.** `Graph::sweep(v, (label, direction), (from, budget), &detaching)` removes
the edges of one shared bucket (a hub's, say) whose other ends are in a `DetachingSet` of detaching
nodes.
- It reads the positions `from..from + budget` in one pass, then removes what it found, both
  halves of each. The other half is the first live entry to `v` in the detaching node's bucket.
- It returns the edges and where to go on. The detaching nodes' own `detach_step`s then find
  those edges gone.
- `DetachingSet` is built once by `Graph::detaching_set`, which checks each member is not live:
  a node never becomes live again, so the check holds for the set's lifetime and a sweep's cost
  follows its budget, not the set's size. It keeps a bit per id when the ids are close together,
  else a sorted list.
- Against `detach_step` one node at a time, which pairs each edge through the hub's target
  index, detaching 1024 / 4096 / 16384 of a 65536-edge hub's neighbours costs 11 / 23 / 23 % less.
  The rest is fixed per edge: the hub's index change and each node's label bucket list going away.

Compared with ic-stable-lara's `DeleteVertexV1`:
- there is no unbounded first step and no O(degree) start;
- the partner is matched by rank, not by "first entry to v";
- there is no separate pending-purge set.

The survey is in `docs/node-detach-2026-11-22.md`.

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

`Graph::compact(v, label, direction)` reclaims one bucket's holes in one bounded step and returns
every edge that moved (old position, new position, target) and whether more remains. It is explicit
so the caller decides when to pay for moves: Gleaph re-keys sidecar properties and index postings
for each one. `Graph::holes` tells the caller how much there is to reclaim.

Every step packs live edges toward the front, so their relative order holds: insertion order and
the rank pairing of parallel edges survive without coordinating forward and reverse buckets.

| Bucket | One step | Moves at most |
| --- | --- | --- |
| tiny or run | pack; shrink to the smallest class, to tiny at 3 or fewer, remove the label at 0 | 1023 |
| paged, at most 512 live | demote: pack into a run or tiny bucket, release pages and index | 512 |
| paged, more live | release a dead page (not the last) and leave its index entry hollow; or merge the next page with content into a page with holes when both fit one page | 1024 |

Hollow entries let a page go without renumbering any later position. The last page is never
hollowed or merged: inserts reuse its holes and append after it.

A step is planned, then applied (`compact::plan`, `compact::apply`). The plan reads the pages
once and lists the moves. The target index's changes are planned from those moves, with their
pages reserved, before anything is written.

## Label bucket lists

A node's label buckets are listed in its **label bucket list**, in the label bucket list page
store: 19-byte entries (label, direction and descriptor) sorted by `(label, direction)`.

- **Flat**, up to 215 entries (one page). A lookup reads the node record and the list once each
  and searches in memory. Adding a bucket shifts entries or moves the list to the next class.
- **Tiered**, past that. A top extent lists leaf pages, each with its first key and entry count
  (8 bytes per leaf), and each leaf holds up to a page of entries.
  - A lookup reads the top and one leaf.
  - A full leaf splits into two half-full pages.
  - An emptied leaf is released, and a list down to one leaf turns flat again.
  - Every change copies at most one page. The top holds 511 leaves: at least ~55 000 buckets when
    every leaf is half full, and up to ~110 000. Lists are independent per node, unlike
    ic-stable-lara, which packs 16 nodes' lists together and rewrites all of them when one outgrows
    its slack.

Reading one bucket costs a node read, a label bucket list read and one data read (index and page
reads past 1024 edges). `Graph::visit_neighbors` reads without collecting, at about 30 instructions
per edge on the IC. `neighbors_directed`, which fills a `Vec`, costs about twice that
(`docs/ic-measurements-2026-11-22.md`).

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
  descriptor, plane root, width, index root) and its label bucket list entry says `Extended(header)`
  (`stored`).
  - Buckets with neither are unchanged.
  - A bucket with values is never tiny.
- **The width table.** Widths live in a table in the label bucket list store, referenced from the
  nodes header and mirrored in the heap. Only 0 → w is allowed.
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
- **`visit_edges_between`** and **`contains_edge`** find the edges between two given nodes. Every
  edge has a half at each end, so they read the smaller of `u`'s bucket and `v`'s opposite one:
  the cost follows the smaller end. A bucket without an index is read whole; an indexed one only
  in the blocks its index lists for the other end (`targets::visit_blocks`, one descent per leaf),
  adjacent blocks merged into one read. Each edge comes as the half at the end that was read.
  Measured: about 3.5K instructions per `contains_edge` between small buckets, present or not;
  1024 parallel edges listed for 39K; 256 edges between two hubs for 120K, as much as reading the
  sender's whole bucket since the target is in every block (`crates/lara-graph-bench/src/between.rs`).

## Target index

A paged bucket keeps a **counted B+ tree of postings** (`targets`, `graph/index`). A posting
`(target, block, count)` says that `count` live entries to `target` sit among the 128 positions of
`block`. Rank and selection of a target's entries are therefore logarithmic, followed by one block
scan. This matches halves in `counterpart`, removal and detach without scanning a hub. A target's
postings are usually reached in one descent, by walking the leaf where they start.

- **Coverage.** The index covers the bucket's complete blocks; the last, partial block is scanned
  instead.
  - An append that completes a block brings the block's live entries in as one batch, sized
    beforehand by a counting pass.
  - A removal lowers its posting's count in place in one descent, and takes no memory. Counts are
    fixed-width per segment. A posting whose count reaches zero leaves its segment: the bits after
    it move along. A segment it empties leaves the leaf's table. Zeros stay only at a segment's
    first posting (see below) and a leaf's last, since a leaf is never emptied in place. Batches
    of decrements (compaction's) lower counts in place and leave zeros until the segment is next
    rewritten.
  - **Sparse leaves merge.** Once a paged bucket's slots need no compaction, each `compact` step
    merges one pair of neighbours, found by going down from the root: leaves whose counts sum to
    512 or less, or nodes whose children fit one node. A root with one child gives way to it.
    - Keeping one edge in sixteen of a 65536-edge hub, then compacting, brings its index from 53 to
      13 pages (distinct targets) and from 89 to 11 (random targets among a million nodes).
  - Batches are planned once in memory, reserved exactly and committed. A block's batch is planned
    before the insert that completes the block; compaction plans its moves.
- **Leaves are slotted pages of coded segments.** Each segment holds 24-48 postings, coded as
  Rice target gaps, fixed-width counts (none when every count is one) and fixed-width block
  offsets.
  - A block is coded as its offset from the segment's first block, zigzagged so that small
    offsets either way are small. Targets inserted about together have blocks near each other,
    so the offsets take a few bits where the blocks took ten or more.
  - The segment's first posting is what the others' blocks are coded against. When its count
    reaches zero it stays, at zero, until the segment is next rewritten; the others leave. In a
    segment without count bits, a flag in its table entry says the first posting is at zero.
  - Without count bits, a posting can go to two only by re-coding its segment, through a plan.
    Removals stay in place and take no memory; a batch of decrements removes postings one at a
    time.
  - A change reads and writes only the segments it touches.
  - Internal nodes keep keys, pages and counts in separate arrays, so a lookup reads only what it
    probes.
- **A detaching node drops its buckets' indexes**, since all their edges are going. Lookups into
  such a bucket scan it.
- **Costs.** Measurements and the space it takes are in
  `docs/target-index-design-2026-11-22.md`:
  - pairing through a hub costs ~17-28K instructions instead of a scan;
  - hub inserts cost about 15-20 % more than before the index;
  - removals from a hub's side cost one in-place index change each, ~6K;
  - the index takes 0.34× the slots for a hub of distinct targets added in id order, about 1.0-1.2×
    for random ones, and 0.52× for parallel edges to 64 targets in turn (the `index_pages`
    measurement, run at full size as `scale` describes).

## Constants

Each was chosen by measuring the alternatives with canbench and the space examples; the numbers
are in `docs/ic-measurements-2026-11-22.md` (eighth round).

| Constant | Value | Why |
| --- | --- | --- |
| Descriptor size, inline slots | 16 bytes, 3 slots | A bucket's 4th edge moves it to a run, once, for ~3.5K instructions. Five slots would need 24 bytes per bucket; on the IC, 8 bytes stored cost that much within about six weeks. |
| Run to paged, paged back to run | above 1024 slots, at 512 live or fewer | A run is at most one page. The gap between the two keeps a bucket near the line from changing form back and forth. |
| Index block | 128 positions | Against 64: inserts −2 to −10 %, index pages −6 % for distinct targets and −49 % for parallel edges; a pairing through a hub +15 % (~2.6K instructions). 256 made that +40 %. |
| Postings per segment | 24, at most 48 | Against 16: index pages −16 % (distinct) and −8 % (random); a removal from the leaf's side +7 %. 32 saved 17 % more space but made pairing and detaching a satellite 16-23 % dearer. |
| Leaf merge | counts summing to 512 or less | Leaves hold 700 to 1200 postings, so merged ones fit one. After merging, every pair of neighbours indexes more than 512 edges. |
| Search windows | 32 slots, then 128 | Most searches stop within a few slots of where they start. 16 slots first made a block scan take two reads. |

**Build canisters with wasm SIMD** (`-C target-feature=+simd128`; the IC runs it). With no code
change, auto-vectorization makes most operations 2-7 % cheaper, and the benches are measured that
way.

Segments whose counts are all one carry no count bits. This saves 7-8 % of a hub's index, and no
instructions moved by 2 % or more. A flag in the table entry marks the first posting at zero.

## Not yet

