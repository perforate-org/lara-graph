# Pairing halves through a hub: is there a fundamentally cheaper design? (2026-11-22)

The counted B+ tree of block postings (`docs/target-index-design-2026-11-22.md`) made lookups
through hubs logarithmic. Its costs on the IC, measured with canbench:

- **Lookups.** 29-61K instructions.
- **Removals.** About 20-25K per hub-side removal, spent re-encoding a 16-32 posting segment.
- **Inserts.** About 7K per hub edge, batched per block.
- **Space.** 0.8-1.4× the slots.

Three surveys looked for a different cost class. They covered dynamic graph stores, write-optimised
counted indexes, and ways to pair the two copies of an edge without rank and select. Every
instruction count below is an estimate from our cost model unless it says "measured".

## What the field does

- **Almost no system has our problem.** Most use set semantics, or merge duplicate edges into a
  count, so locating an edge is a key search:
  - sorted per vertex: Teseo, Sortledton, Aspen, Terrace, Spruce, LSGraph;
  - hashed per vertex: CuckooGraph, DegAwareRHH.
- **Some never locate at write time.** They append tombstones and settle them at compaction:
  DGAP, GraphOne, XPGraph, LSMGraph, BACH.
- **Neo4j stores each edge once.** The record links into both endpoints' chains, so a delete is
  O(1), but scans chase pointers.
- **Both directions are kept consistent as two directed operations in one transaction.** Each copy
  is located on its own; none pairs copies by rank.
- **Insertion-ordered adjacency with a side index on hubs** is the closest match to lara-core:
  - RisGraph: an array per vertex, and for degree above 512 a hash `(dst, weight) → offset`. Space
    is 3.25× the raw data; its B-tree variant takes 1.15× and runs 22 % slower.
  - GraphTango: lists for scans plus a hash for hubs, about 4.5× the inserts and 3.2× the deletes of
    the next best format.
- Sources: RisGraph https://arxiv.org/abs/2004.00803, GraphTango https://arxiv.org/pdf/2212.11935,
  GTX https://arxiv.org/html/2405.01418v2, CuckooGraph https://arxiv.org/html/2405.15193v3, Neo4j
  store formats https://neo4j.com/docs/operations-manual/current/database-internals/store-formats/

## Candidates

### A. Twin offsets: each half stores its partner's position

Locating the partner becomes one read, and no index is needed. It does not fit lara-core:

- **Inline buckets.** Tiny buckets live inside a 16-byte list entry with no room, so every edge into
  a hub would need a side plane. That turns the leaf's bucket from inline into a run with a header.
- **Inserts.** Writing two cells per edge costs about 15K: the measured difference between
  `hub_insert_valued_4096` and `hub_insert_4096`. That is about twice today's index.
- **Compaction.** Every move must patch a back-pointer in another vertex's bucket, which is a random
  remote write.
- **Space.** 4 bytes per half on every edge, 2× the slots, against 0.8-1.4× of hub slots today.

### B. A hash from target to hub positions (RisGraph and GraphTango, adapted)

A paged bucket keeps a hash table keyed by target. Its entries hold a position and a short
fingerprint, and a match is confirmed by reading the slot, which already holds the target. Two
choices make it fit:

- **Rank comes from positions, not from the table's order.** Every live entry to `t` sits in the
  same probe region, so the rank of `p` is the number of `t`'s positions below `p`, and select
  sorts those few positions. Hole reuse and compaction need no ordering invariant.
- **Extendible hashing over 4 KiB pages.** A small directory addresses bucket pages, and a full
  page splits on its own. Growth costs one page split at a time, never a whole rehash; a page
  holds about 700-800 entries of 5 bytes.

Expected costs: about 2-4K per lookup, 2-3K per insert, removal or compaction move. That is 5-10×
below the tree, and a different cost class: O(1) expected against O(log n) with a heavy constant.
Space: about 1.2-1.5× the hub's slots at 80-85 % load, close to today.

There are two weak points.

- **Heavy parallel edges.** All of a target's entries hash to one page. A target with more parallel
  edges than a page holds cannot be split by hashing, and its rank costs O(multiplicity). Such a
  target needs a run-length or counted entry, such as one entry holding a target and a small
  posting list of (block, count). That is the current postings, used only for heavy targets.
- **Nothing measured yet.** Probe lengths, page-split spikes and how often slots are read to
  confirm a match all need canbench before this is decided.

### C. Keep the tree, update it in place

The write-optimised dynamic rank/select literature agrees on one point. The systems avoid
re-encoding a leaf per update, either with per-leaf buffers (Dönges, Puglisi and Raman, DCC 2022)
or with fixed-width fields patched in place (Pibiri and Kanda 2020).

- **For us:** counts become a fixed-width column per segment, and a count that reaches zero stays
  until the segment is next rewritten. Ancestors' subtree counts are updated directly, and leaves
  get a small buffer for changes that do not fit in place.
- **Estimated cost:** a removal of about 5-12K. Lookups stay at two descents plus a block scan,
  about 25K.
- **Space:** unchanged, or 0.6-1.1× with smaller table entries.
- **Sources:** https://users.dcc.uchile.cl/~gnavarro/ps/spe25.pdf, https://arxiv.org/abs/2009.12809,
  https://github.com/xxsds/DYNAMIC

### Rejected

- **Bε message buffers in internal nodes.** Counted aggregates force the ancestors' counts to be
  updated directly anyway. Appends are already batched.
- **LSM tiers of postings.** Every query pays a delta lookup, and select becomes a fixed-point loop.
- **GTX-style hashed chains through the slots.** A lookup walks d/k slots, about 80K instructions
  for a 100K-edge hub.
- **Tombstones that keep their target, settled at compaction.** Removals become free, but heavy
  parallel edges degrade lookups to many block scans, losing the logarithmic bound.

## Comparison

| | Tree today (measured) | C: tree, in place (est.) | B: page hash (est.) |
| --- | ---: | ---: | ---: |
| Lookup through a hub | 29-61K | ~25K | ~2-4K |
| Hub-side removal | ~20-25K | ~5-12K | ~2-3K |
| Hub insert (per edge) | ~7K | ~5K | ~2-3K |
| Compaction move | batched, ~8K | batched | ~2-3K |
| Space against hub slots | 0.8-1.4× | 0.6-1.4× | 1.2-1.5× |
| Heavy parallel edges | logarithmic | logarithmic | needs counted entries |

## Recommendation

**B is the only candidate that changes the cost class.** It has precedent, since RisGraph and
GraphTango index hubs with a hash beside an insertion-ordered list, and it touches nothing outside
paged buckets.

1. Prototype it behind the same interface: `rank_to`, `select_to`, and the insert, removal and move
   hooks.
2. Measure it with the existing canbench benches, `hub_remove_one_65536` and `index_pages`.
3. Add a multigraph bench, many parallel edges to a few targets, before deciding.

If heavy parallel edges or split spikes disappoint, C is the fallback and keeps the logarithmic
bound.

## Prototype B: measured (2026-11-22)

The prototype is in a separate worktree, on the branch `hash-index-prototype`, which has no
commits yet; nothing is merged here. It works as follows:
- **Layout.** An extendible hash with a one-page directory of up to 1024 buckets. Each bucket page
  is a linear-probing table of `(hash u32, position u32)` entries, splitting at 80 % load. Buckets
  that cannot be split chain extra pages.
- **Hash.** A bijective mix of the target, so an entry's hash identifies its target.
- **Coverage.** It indexes every live entry, so no block is left out and none is scanned.
- **Testing.** The lib tests pass, including a model test of the hash; the gate was not run.

canbench on the IC, against the counted tree's persisted results:

| Bench | Tree | Hash | Change |
| --- | ---: | ---: | ---: |
| `counterpart_into_hub_4096` | 28.5K | 6.5K | −77 % |
| `hub_remove_one_65536` | 61K | 15.5K | −75 % |
| `detach_satellite_65536` | 48K | 25K | −49 % |
| `hub_remove_half_compact_65536` | 1.33B | 526M | −61 % |
| — of which removals | 1.08B | 477M | −56 % |
| — of which compaction | 255M | 50M | −81 % |
| `hub_insert_65536` | 1.12B | 1.14B | +2 % |
| `hub_insert_parallel_4096` | 69M | 98M | +42 % |

Space, from the `index_pages` example (65536-edge hubs):

| Hub | Tree | Hash |
| --- | ---: | ---: |
| distinct targets, ascending | 0.83× | 4.02× |
| random targets | 1.20-1.39× | 4.02× |
| 64 targets, round robin | 1.02× | 2.97× |
| 64 targets, each in one run | 0.05× | 2.56× |

**What the numbers say.**
- **Lookups, removals and compaction moves** are 2-5× cheaper, as predicted: one directory read and
  one probed page.
- **Inserts gain nothing.** The tree already batches appends once per block. The hash pays about
  the same per edge: several small reads and writes, plus a split about every 200 inserts that
  rewrites two pages. The extra cost against the commit before any index stays at about +75 %.
- **Parallel edges hurt the hash.** Entries with one hash form a single probe cluster, so each
  insert walks it.
- **Space is the decisive cost.** Entries are 8 bytes. The pages split in step: uniform hashes
  fill them evenly, so they split together and sit near half full. Where postings compress
  sorted gaps to about 2 bytes, a hash pays for a key and a position per edge.

**What would narrow the space gap (estimates).**
- Quotiented entries, storing the hash above the bucket's depth plus the position, take 5-6 bytes.
- Spreading the splits, for example by linear hashing, raises the average fill to about 75 %.

Together that is about 2× the slots: still well above the tree's 0.8-1.4×.

## Prototype D: a directory over posting leaves (2026-11-22)

**The idea.** Take the hash's navigation and the tree's leaves:
- An extendible directory on the targets' low bits replaces the tree's internal nodes.
- Each bucket is a leaf of the same compressed postings, keyed `target >> depth` so that gaps stay
  small.
- A bucket that a split cannot thin (one target's many postings) chains leaves in key order.
- A lookup reads one directory entry and one leaf, and gets all of a target's postings at once.

It is in a separate worktree, on the branch `dir-index-prototype`, which has no commits yet. The
lib tests pass there, including the 680K self-loop hub, which exercises chains.

canbench against the tree:

| Bench | Tree | Directory | Change |
| --- | ---: | ---: | ---: |
| `counterpart_into_hub_4096` | 28.5K | 21.7K | −24 % |
| `hub_remove_one_65536` | 61K | 48K | −21 % |
| `detach_satellite_65536` | 48K | 34K | −28 % |
| `hub_insert_4096` | 64M | 115M | +78 % |
| `hub_insert_65536` | 1.12B | 3.62B | +224 % |
| `hub_remove_half_compact_65536` | 1.33B | 2.02B | +52 % |
| — of which compaction | 255M | 850M | +233 % |

Space: 1.02× the slots for every random-target shape. Splits by hash bit leave the leaves about
half full.

**Why it fails.**
- **Lookups.** Navigation was not where the tree's lookups spent their time. Opening a leaf,
  searching its table and decoding a segment cost the same under either top layer.
- **Inserts and compaction.** Hashing destroys the locality that makes the tree's batches cheap.
  A block of appends to new vertices has ascending targets: the tree re-encodes one or two
  segments per batch, while the directory scatters the batch over up to 64 leaves. Compaction
  moves are scattered the same way.

**What the three prototypes show together.**
- **CPU cost comes from the coding.** Navigation is minor. Compressed postings must be decoded to
  read and re-encoded to change, and that is the cost of their compactness.
- **The hash's speed comes from storing plain entries.** One probe reads them, and changing one is
  a single write.
- **A true middle ground has to change what is stored.** The candidate is a compact hash: entries
  keep only the bits that slot addresses cannot imply. The literature does this with Cleary-style
  quotienting or quotient filters (https://link.springer.com/content/pdf/10.1007/s00453-022-00996-y.pdf,
  https://www.fsl.cs.stonybrook.edu/docs/vldb12qfflash/vldb12qfflash.pdf). Estimated space is
  1.1-1.3× the slots with a block instead of an exact position, at a cost of considerable
  complexity, since entries become bit fields that shift on insert.

## Prototype E: a compact hash (2026-11-22)

**The idea.** Keep the hash's plain, directly addressable entries, but store only the bits that
cannot be implied:
- **Directory.** An extendible directory on the targets' own low bits, with no hash function.
- **Buckets.** Each bucket page is a sorted array of fixed-width entries `(target >> depth,
  position)`. The widths fit the page's largest values and are re-encoded when a larger value
  arrives; entries usually take 4 bytes.
- **Lookups.** A target's entries are contiguous and sorted by position, so rank and select are
  binary searches in one page.
- **Changes.** Inserts and removals shift one page's tail; compaction moves rewrite one position.
- **Growth.** Full pages split on the next target bit. Buckets that cannot split (one target,
  or the deepest directory) chain pages, with a tail pointer for appends.

It is in a separate worktree, on the branch `compact-index-prototype`, which has no commits yet;
the lib tests pass there.

canbench (the tree's persisted results as the baseline; the hash and directory columns repeat
earlier runs):

| Bench | Tree | Hash (B) | Directory (D) | Compact hash (E) |
| --- | ---: | ---: | ---: | ---: |
| `counterpart_into_hub_4096` | 28.5K | 6.5K | 21.7K | 10.9K |
| `hub_remove_one_65536` | 61K | 15.5K | 48K | 26K |
| `detach_satellite_65536` | 48K | 25K | 34K | 31K |
| `hub_remove_half_compact_65536` | 1.33B | 526M | 2.02B | 788M |
| `hub_insert_65536` | 1.12B | 1.14B | 3.62B | 1.21B |
| `hub_insert_parallel_4096` | 69M | 98M | 72M | 72M |

Space against the slots, from the `index_pages` example (65536-edge hubs):

| Hub | Tree | Hash (B) | Compact hash (E) |
| --- | ---: | ---: | ---: |
| distinct targets, ascending | 0.83× | 4.02× | 2.02× |
| random targets | 1.20× | 4.02× | 1.45× |
| random targets among 1M vertices | 1.39× | 4.02× | 1.56× |
| 64 targets, round robin | 1.02× | 2.97× | 1.02× |
| 64 targets, each in one run | 0.05× | 2.56× | 1.02× |

**Reading the results.**
- **CPU.** Lookups and removals cost 1.7× the plain hash's and 0.4× the tree's. Inserts are within
  a few percent of both.
- **Space.** Half the plain hash's, from 1.02 to 2.02× the slots.
  - **Lockstep splits.** Ascending distinct targets peak at 2.0×. Targets' low bits spread evenly,
    so every page splits at the same moment and sits half full at powers of two.
  - **Parallel edges.** The compact hash stores one entry per edge, while the tree merges a block's
    parallel edges into one posting, hence 1.02× against 0.05× for runs.

**What could narrow the gap further (estimates).**
- **Fill.** Lockstep splits cost up to 2×. Buckets sized by byte class (1-4 KiB extents) or
  redistributing into a neighbour before splitting would keep the fill near 75 %.
- **Entry width.** Bit-packing instead of byte-rounding, and fewer spare position bits, save about
  10 %.
- **Parallel edges.** Merging runs of one target's consecutive positions into one entry with a
  count would make runs cheap, at some cost to the binary searches.

### E2: chunked buckets (2026-11-22)

**What changed.**
- **The cause.** Uniform targets fill every page at the same rate, so pages split together and sit
  half full at powers of two. Power-of-two size classes would not help: the buckets would cross
  each class boundary together just as they cross a page.
- **Chunks.** A bucket is now a 256-byte record listing up to 12 chunks of 512 bytes, which hold
  its sorted entries under one entry width shared by the record.
- **Repacks.** A full chunk repacks the record's chunks to 90 %, by moving bytes, without decoding.
- **Splits.** Only a full record splits its bucket. The waste left is a part of a chunk per bucket.
- **Header.** The index root is now a sub-page extent, so bucket headers grew from 32 to 64 bytes
  in this prototype.

Space against the slots (65536-edge hubs):

| Hub | Tree | E (pages) | E2 (chunks) |
| --- | ---: | ---: | ---: |
| distinct targets, ascending | 0.83× | 2.02× | 1.20× |
| random targets | 1.20× | 1.45× | 1.22× |
| random targets among 1M vertices | 1.39× | 1.56× | 1.20× |
| 64 targets, round robin | 1.02× | 1.02× | 0.95× |
| 64 targets, each in one run | 0.05× | 1.02× | 0.83× |

canbench (against the tree):

| Bench | Tree | E2 | Change |
| --- | ---: | ---: | ---: |
| `counterpart_into_hub_4096` | 28.5K | 16.4K | −42 % |
| `hub_remove_one_65536` | 61K | 36K | −41 % |
| `detach_satellite_65536` | 48K | 44K | −8 % |
| `hub_remove_half_compact_65536` | 1.33B | 1.01B | −24 % |
| `hub_insert_4096` | 64M | 104M | +61 % |
| `hub_insert_65536` | 1.12B | 1.93B | +73 % |
| `hub_insert_parallel_4096` | 69M | 93M | +35 % |

**Status.**
- **Space.** The gap to the tree is closed, except for long runs of one target: 0.83-1.22× the
  slots against 0.05-1.39×.
- **Lookups and removals.** Still clearly cheaper than the tree, though above E's single page:
  E took 10.9K for `counterpart_into_hub_4096` and 26K for `hub_remove_one_65536`.
- **Inserts.** These are where E2 loses. Each pays its index update, while the tree batches appends
  once per 64. The per-insert cost is reading the 256-byte record, the chunk's tail and the record
  write, plus a repack every dozen or so inserts into one chunk.

**Next steps if this direction is kept (estimates).**
- Write only the record entry a change touches.
- Repack only the full chunk and a neighbour instead of the whole record.
- Batch appends per block as the tree does. That helps only if a batch's targets share buckets;
  with ids partitioned by low bits they do not.

### E3: cheaper inserts (2026-11-23)

**Changes.**
- A change writes only the record entry it touches (17 bytes), not the whole record.
- A full chunk spills into a neighbour with room. Otherwise a new chunk takes the overflow: an
  append starts the new chunk alone, so the full one stays full. The whole record is repacked only
  when it holds its 12 chunks.
- Entries are encoded into stack buffers and spliced by rotation: no allocation per entry, and none
  to grow a chunk's buffer.

canbench, against the tree:

| Bench | Tree | E2 | E3 |
| --- | ---: | ---: | ---: |
| `hub_insert_4096` | 64M | 104M (+61 %) | 83M (+29 %) |
| `hub_insert_65536` | 1.12B | 1.93B (+73 %) | 1.51B (+35 %) |
| `hub_insert_parallel_4096` | 69M | 93M (+35 %) | 76M (+10 %) |
| `counterpart_into_hub_4096` | 28.5K | 16.4K | 16.4K (−43 %) |
| `hub_remove_one_65536` | 61K | 36K | 36K (−41 %) |
| `hub_remove_half_compact_65536` | 1.33B | 1.01B | 1.00B (−25 %) |

Space: 0.83-1.27× the slots, against E2's 0.83-1.22×. Spilling leaves random-target buckets
slightly less full.

**Where a hub insert's instructions go.** Against the commit before any index, at
`hub_insert_65536`:

| | Instructions per edge | Index's share |
| --- | ---: | ---: |
| no index | 9.8K | — |
| tree | 17K | 7K, batched once per 64 appends |
| E3 | 23K | 13K |

The rest of E3's share, from a native profile:
- reading the 256-byte record;
- reading one chunk and searching it;
- writing the chunk's tail and one record entry;
- the 64-byte bucket header, written on every insert. This prototype doubled it from 32 bytes so
  that it can hold a sub-page root.

## Decision (2026-11-23)

**The tree stays.**
- **Cost.** On the IC an update message costs a 5M-cycle base fee at 1 cycle per instruction, so
  the index's per-edge differences (thousands to tens of thousands of instructions) are a fraction
  of a percent of a single-edge update.
- **Where instructions do matter.** They count only in bulk messages, where the tree's batched
  appends win, and in compaction and detach steps.
- **Space.** It costs about $0.45 per GiB per month, and is comparable between the tree and the
  compact hash.
- **Bounds.** The tree's logarithmic bounds hold for every hub shape, heavy parallel edges
  included.

**Improvements adopted from the comparison.** Removals rewrite counts in place, and batches are
planned once and committed, without a counting pass. See the design doc's costs.

The prototypes stay on their worktree branches for reference: `hash-index-prototype`,
`dir-index-prototype` and `compact-index-prototype`.
