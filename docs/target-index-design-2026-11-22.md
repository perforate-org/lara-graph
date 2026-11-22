# A target index for hub buckets: design draft (2026-11-22, revised)

## The problem

Positions are in insertion order, so finding the entries *to a given vertex* in a bucket means
scanning it. Small buckets (at most 1024 slots, one read) do not care. Paged buckets do:

| Operation | Today, in a hub of d edges | Measured on the IC |
| --- | --- | --- |
| `counterpart` / `remove_edge` from a leaf's half into the hub: select the k-th entry to the leaf | O(d) scan | 138 K at 4096 edges |
| Detaching a leaf of a hub: the leaf's entry in the hub | O(d) | 2.1 M at 65536 edges |
| `remove_edge` from the hub's half when the leaf has parallel edges: the rank count | O(d) | — |
| "Is there an edge u → v?"; intersections (WCOJ) | O(d) | not offered |

**The requirement (user, 2026-11-22).** The index need not map each edge exactly. A coarse index
that points at blocks, combined with scanning them, is fine as long as cost stays **bounded**, and
it should keep **space** small.

## What the survey found

The full survey is in this session's notes. Primary sources:
- LiveGraph: VLDB 13(7) p1020
- RisGraph: arXiv 2004.00803
- Sortledton: VLDB 15 p1173
- binary fuse filters: arXiv 2201.01174
- Aleph filter: arXiv 2404.04703
- A+ indexes: arXiv 2004.00130

| Approach | Space per edge | Gives select / rank? | Worst case | Verdict |
| --- | --- | --- | --- | --- |
| Per-page filters (Bloom, cuckoo, quotient, XOR / fuse, ribbon, prefix, Aleph) | ~7-14 bits | **No**: membership only; rank needs exact counts in every earlier page | expected only. LiveGraph's per-block Bloom falls back to a full scan on a hit. Cuckoo inserts fail on many duplicates. XOR / fuse / ribbon are static. | at most an optional pre-check |
| Zone maps / BRIN (min-max per block) | a few bytes per page | no | no pruning on unsorted targets | no |
| Sorted pages with a per-page permutation (10 bits per slot) | ~1.25 B | yes, by binary search in every page | bounded but **linear in hub size** (pages × log) | no |
| Hash map from target to offset (RisGraph above degree 512) | ~3.25× the raw data | membership; no order | expected, adversarial ids | no |
| Exact (target, position) B+ tree | 8-11 B | yes, O(log d) | worst case | too big: 2-3× the slots |
| Sorted immutable run, compressed (Elias-Fano), plus a mutable buffer and incremental rebuild | ~3.3-4.3 B | yes | worst case if the rebuild is spread across messages | the right idea, but complex |

**Lessons.**
- Membership filters cannot answer rank or select, the queries that pair halves. A bounded answer
  needs an index **ordered by target** with **counts**.
- What saves space is coarsening the location (block, not position), collapsing parallel edges
  into one entry with a count, and delta-compressing the sorted targets.

## Second survey (three parallel reviews)

### Recent dynamic graph stores, 2019-2025

| System | Hub structure | Space | Lookup bound | Order kept |
| --- | --- | --- | --- | --- |
| GTX (PVLDB 18) | per-label append block; targets hashed onto delta chains | 64 B per edge-delta | expected; the paper notes it degrades on large neighbourhoods | **insertion** |
| LiveGraph | append-only log plus Bloom filter | ~2.5× CSR | O(d) | **insertion** |
| Terrace | per-vertex B-tree above a degree threshold | 2-3× LSGraph | O(log_B d) | sorted |
| Aspen / PaC-trees (CPAM) | blocked, delta-coded leaves; augmented nodes (e.g. subtree counts) | PaC 0.35 GB on LiveJournal, the smallest | O(log d + B) | sorted |
| RapidStore (PVLDB 18) | compressed ART with sorted leaves of 64-256 ids | up to 56 % less than competitors | O(w + log B) | sorted |
| LSGraph | learned index over gapped arrays | 0.61 GB on LiveJournal | model plus search, no worst case | sorted |
| CuckooGraph | chained cuckoo tables; duplicates collapse into counts | 68 % of Spruce | expected O(1) | none |
| Spruce, LSMGraph, BACH, DGAP | buffer or log merged into sorted form | — | amortized merges | sorted after merge |

Sources:
- GTX: arXiv 2405.01418
- Terrace: MIT
- CPAM: arXiv 2204.06077
- RapidStore: arXiv 2507.00839
- LSGraph: EuroSys 2024
- CuckooGraph: arXiv 2405.15193
- LSMGraph: arXiv 2411.06392
- Benchmark comparison: arXiv 2502.10959

Lessons from the stores:
- Only GTX and LiveGraph keep insertion order and index by target, and neither bounds the worst
  case.
- Nobody pairs an insertion-ordered primary with a sorted, *counted* secondary index.
- Counts per target are the established way to make parallel edges cheap (CuckooGraph).
- Deterministic trees (B-tree, ART, PaC-tree) fit a per-message budget. Hash, learned and skip-list
  structures give only expected bounds.

### Coarse secondary indexes for unsorted data

**Pruning by value range does not work on unsorted targets.** Zone maps, column imprints, column
sketches, BF-Tree, Hippo and SuRF all assume clustering. Two families fit:

- **Value → blocks with counts**, i.e. a block-level inverted list. The Cuckoo Index's stripe
  bitmaps are a coarse, immutable form of it.
- **Retrieval structures** (BuRR, SEA 2022) store `target → block` without storing the target, at
  about `log2(blocks)` bits per key. A one-block scan checks the answer and rejects absent targets.
  - A hybrid sends targets with parallel edges to block-and-count posting lists.
  - About **1.3-1.65 B per edge** for a hub of distinct targets, about 0.5 B with heavy parallel
    edges.
  - The base is **static**: appends go to a hash tail and the base is rebuilt. Bounding that per
    message needs incremental rebuilding.
  - It gives no sorted order, so it cannot serve intersections.

### Dynamic succinct structures

- **Dynamic wavelet trees and matrices do not fit.**
  - With mostly distinct 32-bit targets they hold ~32 levels, about as much space as the slots
    again.
  - Each update and query touches ~32 dynamic bitvectors; a published BFS ran 155× slower than a
    plain vector.
  - An append is a mid-sequence insert in most levels.
- **Alphabet partitioning** reduces to a dictionary from target to location when most targets
  occur once.
- **PGM-index** updates are amortized and degrade under deletes.
- **Partitioned Elias-Fano** is static, so it suits only frozen runs.
- The practical optimum under explicit 4 KB page reads is a **counted B-tree** over
  `(target, block)` keys with compressed leaves: about 20-26 bits per distinct pair.

## Evaluation

| Candidate | Space, distinct hub | Space, parallel edges | Worst case per operation | Intersections | Complexity |
| --- | --- | --- | --- | --- | --- |
| **A. Counted B+ tree of (target, block, count), delta-coded leaves** | ~2.5-4 B per edge (~0.6-1× the slots) | small (one posting per run in a block) | **O(log d) + one block scan, deterministic** | yes (sorted) | moderate: CPAM-style leaves and augmented counts |
| B. Retrieval (BuRR) + heavy-target postings | ~1.3-1.65 B | ~0.5 B | one probe + one block scan, but the static base needs an incremental rebuild to stay bounded | no | high: ribbon construction and a staged rebuild |
| C. Hash-partitioned chains (GTX) | small | small | expected only | no | low |
| D. Exact (target, position) B+ tree | 8-11 B | 8-11 B | O(log d) | yes | moderate |

**Recommendation: A.**
- It is the only candidate that is deterministic, local to one leaf per update, and sorted (so it
  can serve intersections later), at about the size of the slots or less.
- B saves a further ~2× in space but needs a static structure, a hash tail and incremental
  rebuilds. Keep it as a later option if measurement shows the index dominating hub memory.
- **Validate A's space with a prototype first.** Encode leaves for synthetic hubs before building
  the tree, and measure bytes per edge for:
  - distinct targets with sequential or random ids;
  - power-law parallel edges;
  - blocks of 64, 128 and 256.

## A property of lara-core that makes this simpler

Per target, a bucket's entries are **append-only in position order**.
- An insert lands after every live entry to the same target, because of rank-safe hole reuse
  (always so under `Append`).
- Removal deletes entries.
- Compaction moves entries but keeps their relative order.

So a target's postings never splice into the middle of its own list.

## Proposal: a counted B+ tree of block postings, with compressed leaves

For each **paged** bucket:

- **Postings.** A posting is `(target, block, count)`: `count` live entries to `target` among the
  positions of `block`.
  - A block is `B` consecutive positions, with `B` a power of two from 64 to 1024, to be chosen by
    measurement.
  - Postings are sorted by `(target, block)`.
- **The tree.** A B+ tree over postings. Each internal entry keeps the **sum of counts** below it
  (an order-statistic tree), so rank and select within one target cost O(log) even when the target
  has millions of parallel edges.
- **Compressed leaves.**
  - A leaf (one page) stores its postings delta-coded: the target gap as a varint (0 for the same
    target), the block as a varint (a delta when the target repeats), and the count as a varint.
  - A posting usually takes 3-5 bytes; a run of parallel edges in one block takes one posting.
  - An update decodes, changes and re-encodes one leaf (at most 4 KB). A full leaf splits in two.

| Query | How | Cost |
| --- | --- | --- |
| Select the k-th entry to `t` | descend by counted entries to the posting holding the k-th, then scan its block for the j-th entry to `t` | O(log d) reads + one block scan (`B` slots) |
| Rank of position `p` among entries to `t` | sum the counts of `t`'s postings before `block(p)` (counted descent), then scan `block(p)` up to `p` | same |
| Does `t` occur? / `edges_to` | seek `t`; scan the blocks its postings name | O(log d) + `B` per posting |
| Insert at `p` (always in the last page) | increment `(t, block(p))`, or add it | one leaf rewrite, plus counts up the path |
| Remove at `p` | decrement, dropping the posting at zero | same |
| Compaction move `p → q` | decrement `(t, block(p))`, increment `(t, block(q))`, merging runs of moves in one block | at most 1024 per step, each O(log d) |
| Promotion (a full run becomes paged) / demotion | build from the 1024 slots in memory / release the tree | bounded, once |

**Space.**
- A hub of distinct leaves needs one posting per edge, about 3-5 bytes against the slots' 4 (the
  target gaps are small and the block varint is 2-3 bytes). That is roughly 1-1.25× the slots,
  where an exact B+ tree costs 2-3×.
- Parallel edges cost almost nothing.
- Buckets below a page carry no index.

**Choosing `B`.**
- A larger `B` shortens block ids and merges more parallel edges, but every lookup scans more
  slots. One slot costs about 20-30 instructions on the IC, so a scan of 1024 slots is about
  25 K instructions.
- `B` = 64 to 256 keeps a lookup at roughly 2-8 K instructions plus the tree descent.
- Measure 64, 128 and 256 with canbench before fixing it.

**Where the tree is referenced.** The paged bucket's 32-byte header, already used for values, also
holds the tree's root page and depth (31 bytes in all). A paged bucket always has a header; hubs
are few.

**Not chosen, and why.**
- **Per-page filters** cannot count, and their bounds hold only in expectation. One may still serve
  as a cheap negative pre-check for existence queries later.
- **The Elias-Fano run with a buffer and incremental rebuild** is close in space but much harder to
  keep bounded per message. The compressed-leaf B+ tree gets most of the space saving with plain,
  local updates.

## Questions to settle

1. **Index structure.** The counted B+ tree over `(target, block, count)` with delta-compressed
   leaves, for paged buckets only.
2. **Block size `B`.** Chosen by measurement among 64, 128 and 256.
3. **Scope.** Faster `counterpart`, removal and detach, plus `edges_to` (existence). Target-ordered
   iteration for intersections comes later.

## Prototype measurement (space)

`cargo run --release --example index_space -p lara-core-bench` builds the postings of synthetic
1M-edge hubs and packs them into 4 KB leaves. The figures are bytes per edge of the whole tree,
leaves and internal nodes; the slots alone take 4.00.

| Hub | Block | Varint leaves, 70 % fill | Bit-packed, full | Bit-packed, 70 % | Bit-packed, 81 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| distinct, ids rising with time | 256 | 5.71 | 2.12 | 3.03 | 2.62 |
| distinct, ids rising with time | 1024 | 5.57 | 1.87 | 2.67 | 2.31 |
| distinct, random among 10M ids | 256 | 5.71 | 2.25 | 3.22 | 2.78 |
| distinct, random among 1G ids | 256 | 6.97 | 3.09 | 4.41 | 3.81 |
| parallel, Zipf over 50k targets, interleaved | 256 | 3.40 | 1.69 | 2.40 | 2.07 |
| parallel, bursts of 1-200 | 256 | 0.09 | 0.07 | 0.09 | 0.08 |

The full table, with blocks of 64 and 128 as well, is in the example's output.

**The two layouts.**
- **Varint leaves** code each posting as a target gap, a block and a count, each in whole bytes.
  They are **too big**: a distinct-target hub gets an index 1.4-1.75× the slots. The block number
  costs 2-3 bytes as a varint, and the count costs 1 byte though it is almost always 1.
- **Bit-packed leaves** store, per leaf, the first target, a Rice parameter and a block width.
  Each posting is then a Rice-coded target gap, one bit for "count is 1" (else an Elias-gamma
  count), and the block in the leaf's fixed width. This roughly halves the space: an index of
  **~0.5-0.8× the slots** for distinct hubs at full leaves, and ~0.65-1.1× at realistic fill.

**What drives the size.**
- **Leaf fill matters as much as the encoding.** 70 % is a B-tree's steady state under random
  inserts. Splitting 2-to-3 (B*-tree, ~81 %) or repacking index leaves during compaction steps
  recovers most of the gap.
- **Hubs whose targets arrive in id order** (new vertices joining an old hub) insert at the right
  end, so splitting the right edge can keep those leaves full.
- **Coarser blocks** save about one bit per doubling of the block size, but double the scan.

**Still to measure: decode cost.** Rice codes decode sequentially, so a whole 4 KB leaf is about
1000 postings to decode. Leaves need a few sync points, for example every 64 postings, so that a
lookup decodes at most one segment. Its instruction cost on the IC is still unmeasured.

## Implementation (2026-11-22)

The index is in `crates/lara-core/src/targets` (the tree and its leaves) and
`crates/lara-core/src/graph/index.rs` (keeping it in step with the slots). It departs from the
proposal in four places, each settled by measurement on the IC (canbench).

**Leaves are slotted pages of segments.** The first version decoded and re-encoded a whole leaf
per change. That cost 100-200K instructions per hub insert, mostly copying 4 KB pages. A leaf is
now a table of segment entries plus slots for coded segments:
- Each table entry is 20 bytes: first key, total count, and slot offset, capacity, code length and
  coding parameters packed in 8 bytes.
- Each segment holds 16-32 postings, coded as in the prototype.
- A change reads the table entries it probes and one slot, then writes that slot and its entry.
- A segment that outgrows its slot moves into free space with some slack. Rewriting the leaf
  reclaims the garbage.
- A rewrite whose changes all landed in the last segment fills the earlier leaves to 7/8; any
  other rewrite splits evenly. Appends therefore leave full leaves behind.

Segments of 16 measured best for a single removal: 8 gave 70K instructions, 16 gave 69K, 32 gave
79K.

**Internal nodes keep keys, pages and counts in three arrays.** A lookup reads only the keys it
probes and the counts it sums. A change that keeps every node on its page writes only the
entries it alters.

**The last, partial block is not indexed.** The index covers a paged bucket's complete blocks;
lookups scan the partial block instead.
- An append that completes a block brings the block's live entries in as one batch.
- A counting pass sizes the reservation for that batch before the insert changes anything.
- Hub inserts therefore touch the index once per block.

**A detaching vertex's buckets drop their indexes.** All their edges are going, so keeping the
index in step only costs. The header records the drop, and lookups into such a bucket scan it.

**Blocks are 64 positions, not 256.** A lookup's block scan fell from about 19K instructions to
about 5K.

### Costs on the IC

Measured with canbench against the commit before the index (eb02e34). Every edge is stored at
both ends.

| Bench | Before | With the index | Change |
| --- | ---: | ---: | ---: |
| `counterpart_into_hub_4096` (find the hub half) | 138K | 27K | −80 % |
| `detach_satellite_65536` (its one edge into a hub) | 2.1M | 46K | −98 % |
| `hub_remove_one_65536` (remove one hub edge) | 975K | 48K | −95 % |
| `hub_insert_4096` | 39.9M | 60.2M | +51 % |
| `hub_insert_65536` | 641M | 1.03B | +61 % |
| `hub_remove_half_compact_65536` | 287M | 967M | +237 % |
| `detach_hub_65536` | 978M | 1.21B | +24 % |
| `edge_lookup_65536` | 1.9K | 2.5K | +31 % |
| `hub_visit_65536`, `hub_scan_65536` | | | ±0 % |

These figures include two later changes (2026-11-23):
- **Plan, then commit.** A batch of changes is planned in memory once. The plan's exact page count
  is reserved, and the same plan is written: no second, counting pass.
  - A block's batch is planned before the insert that completes the block, and committed after it
    when the insert appended where planned. Compaction plans its moves the same way.
  - The plan travels boxed through the insert path. Passing it by value made every insert about 20
    % slower, even those with no batch.
- **Counts patched in place.** Counts are fixed-width per segment. A batch of decrements to postings
  that exist rewrites their count bits and the segment's total, without re-encoding the segment.
  - A posting whose count reaches zero stays until its segment is next rewritten. So a tree may
    outlive its last indexed edge, until the bucket is demoted or detached.

Against the tree before these changes: inserts are 6-19 % cheaper, a single removal 21 %,
removing half a hub 26 %, and compaction 33 %.

**Where the remaining costs come from.**
- **Lookups and removals.** Anything that pairs halves through a hub costs tens of thousands of
  instructions instead of a scan of the hub. Most of it is the tree descents and segment decodes.
- **Inserts.** They pay about 6K per edge: the block batches, once per 64 inserts, and reading the
  hub's header on every insert.
- **Removals from the hub's side.** These were cheap before: the leaf's one-entry bucket answers
  the pairing. Each now pays one in-place index change.
- **Single lookups.** Paged buckets now have a header, so `edge_lookup` and similar reads pay one
  more 32-byte read.

### Cheaper removals and pairing (2026-11-23)

canbench's instruction tracing (`--instruction-tracing`, a flame graph per bench) showed where a
removal from a hub's side went, at ~24K instructions per edge:
- **About 11K was the index decrement**, sent through the batch machinery: changes sorted and
  combined, a plan, then a commit. Most of it was memory reads (~150 instructions each on the
  heap memory) and short-lived `Vec`s (~200 each).
- **About 3.4K was writing the bucket header**, on every removal and insert into a hub: the
  descriptor, plane and header were encoded through iterator chains collected into `Vec`s.

Changes:
- **A removal lowers its posting in one descent** (`targets::lower`). It walks to the leaf, rewrites
  the count bits and the segment's total there, and lowers each parent's count on the way back.
  - It writes in place and allocates nothing, so a removal takes no memory: `remove_edge` and
    `detach_step` no longer reserve, and `remove_edge` no longer fails for memory.
  - Batches of decrements, such as compaction's, still go through the plan.
- **Headers, descriptors and plane roots encode into fixed arrays**, without allocating.
- **Rank and selection by target take one descent** (`targets::nth`, `targets::rank`). They walk the
  target's postings in the leaf where they start. Before, they took two or three counted descents
  from the root.
  - A target whose postings run on into the next leaf falls back to the counted descents.
- **Segment walks are inlined** (`#[inline]`). Without the hint, the new callers stopped the
  compiler from inlining them into rank and selection, which cost ~900 instructions per walk.

| Bench | Before | After | Change |
| --- | ---: | ---: | ---: |
| `hub_remove_half_compact_65536` | 915M | 629M | −31 % |
| — of which removals | 791M | 504M | −36 % (~15.4K per edge) |
| — of which compaction | 124M | 125M | ±0 % |
| `hub_remove_one_65536` (from the leaf's side) | 48.1K | 28.3K | −41 % |
| `counterpart_into_hub_4096` | 27.0K | 17.3K | −36 % |
| `detach_satellite_65536` | 46.3K | 27.4K | −41 % |
| `detach_hub_65536` | 1.21B | 949M | −22 % |
| `hub_insert_4096` / `_65536` | 60.5M / 1.03B | 46.0M / 775M | −24 % / −25 % |
| `hub_insert_parallel_4096` | 56.5M | 40.8M | −28 % |
| `set_value_4096` | 21.5K | 14.7K | −32 % |
| scans, visits, `edge_lookup_65536` | | | ±0 % |

**Against ic-stable-lara (one side), per half:**
- inserts: 1.25× at 4096 edges, 1.9× at 65536;
- removal with compaction: 2.3×;
- hub detach at 4096 edges: 1.5× its synchronous detach.

**Where a removal's ~15K go now.**
- Finding both halves: ~5K. That is two bucket lookups, the half's slot, and the leaf's one-entry
  bucket.
- The index decrement: ~6K, about 20 memory reads and writes.
- Tombstones and the two buckets' state: the rest.
- Compaction's sort of its changes is ~1.1M of its 4.0M at 4096 edges. A stable sort and a sort on
  a packed `u64` key measured the same.

### Space

`cargo run --release --example index_pages -p lara-core-bench` builds 65536-edge hubs through the
graph API and counts the index's pages:

| Hub | Pages | Bytes per edge | Against the slots |
| --- | ---: | ---: | ---: |
| distinct targets, ascending | 53 | 3.31 | 0.83× |
| random targets (some repeated) | 77 | 4.81 | 1.20× |
| 64 targets, round robin | 65 | 4.06 | 1.02× |
| 64 targets, each in one run | 1 | 0.06 | 0.02× |
| random targets among 1M vertices | 89 | 5.56 | 1.39× |

This is above the prototype's 0.5-1.1×, for three reasons:
- the table entries: 20 bytes per 16-32 postings;
- the slack in rewritten slots;
- splits at 7/8 fill.

Blocks of 64 also make round-robin parallel edges cost a posting per edge, where blocks of 256
would hold about four.

### Alternatives measured

The prototypes in `docs/target-index-alternatives-2026-11-22.md` were measured against this tree:
- a plain hash;
- a directory over posting leaves;
- a compact hash, the best of them, whose lookups and removals cost about 40 % less than this tree.

The tree stays, for three reasons:
- **Small differences on the IC.** An update message costs a base fee of 5M cycles, so tens of
  thousands of instructions are a fraction of a percent of a single-edge update.
- **Comparable space.**
- **Bounds that hold for every hub shape.** The tree stays logarithmic even with heavy parallel
  edges.
