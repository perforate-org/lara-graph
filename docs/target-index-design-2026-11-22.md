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
