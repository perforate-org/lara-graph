# lara-core on the IC: first canbench measurements (2026-11-22)

`crates/lara-core-bench` (canbench 0.7, PocketIC). The memory is the heap `VecMemory`, as in
ic-stable-lara's benches, unless marked *stable*. Stable means five `MemoryManager` virtual
memories. The ic-stable-lara numbers are from `gleaph/crates/ic-stable-lara/canbench_results.yml`.

**Not like for like.** Every lara-core edge is stored at both ends: an insert or removal writes
two buckets, and every hub edge goes to a fresh leaf, which creates a bucket list there.
ic-stable-lara's single-label hub benches insert, remove and scan **one side only**. Its detach
benches are bidirectional, like ours.

## Results (instructions)

| Scenario | lara-core | ic-stable-lara | Note |
| --- | ---: | ---: | --- |
| Hub insert, 1024 edges | 15.4 M | 8.9 M (`s2_h_ins_1024`, one side) | ~15 K per edge (both halves), flat in degree |
| Hub insert, 4096 | 62.3 M | 18.3 M (`hub_insert_grow_4096`, one side) | |
| Hub insert, 65536 | 998 M | 204 M (`hub_insert_grow_65536`, one side) | |
| Hub insert, 4096, stable | 55.5 M | — | stable ≈ heap |
| Insert, parallel edges to 8 targets, 4096 | 53.9 M | — | no new lists: still ~13 K per edge |
| 51 scans of a 1024-edge hub | 2.86 M | 1.77 M (`s2_h_scn_d_1024`) | ~55 per edge vs ~34 |
| 51 scans, stable | 2.85 M | — | |
| 51 scans of a 65536-edge hub | 177 M | — | ~53 per edge: flat |
| Remove every other edge, compact, 4096 | **202 M** | 8.7 M (one side) | quadratic, see below |
| Same, 65536 | **41.8 B** | 138 M (one side) | ×207 for ×16 edges |
| `counterpart` from the hub's last out half, 4096 | 189 K | — | scans the hub bucket |
| `counterpart` from a leaf's in half into the hub, 4096 | 164 K | — | scans the hub bucket |
| Edge lookup by handle, paged, 65536 | 2.7 K | — | |
| Detach a hub, 1024 | 35.2 M | 9.5 M sync, 13.3 M stepped | ~34 K per edge |
| Detach a hub, 4096 | 145 M | 35.6 M sync, 45.9 M stepped | |
| Detach a hub, 4096, steps of 64 | 138 M | — | stepping costs nothing extra |
| Detach a hub, 65536 | 2.86 B | — | ~44 K per edge: linear |
| Detach a satellite into a hub, 1024 | **62 K** | 434 K | |
| Detach a satellite into a hub, 4096 | **180 K** | 1.28 M | ~7× faster |
| Detach a satellite into a hub, 65536 | 2.5 M | — | ~38 per scanned slot |

## Findings

1. **Removal is O(position) in a hub, which makes bulk removal quadratic.**
   - `remove_edge` calls `counterpart`, which counts entries to the neighbour before the half in
     the hub's own bucket. That count scans the hub up to the half's position.
   - Fix: look at the smaller bucket first. Scan the neighbour's bucket for entries to the
     owner; if there is exactly one, it is the partner and the hub is never scanned.
   - Parallel edges still need the count. From the small side, only the selection in the large
     bucket is left, and that is what a sorted target index would remove.
2. **Detach re-skips the tombstones it just wrote.**
   - Every edge looks for v's first live entry from position 0, so a run or page skips up to
     1023 holes per edge: linear with a large constant.
   - Fix: carry a per-bucket cursor for v within the step.
3. **Per-operation constants are 2-4× ic-stable-lara's.**
   - An insert looks every bucket up twice (`admit`, then `insert_half`). Each lookup reads the
     vertex record and the whole bucket list, decodes every entry into a `Vec`, then binary
     searches it.
   - Removal and detach repeat the lookups per half.
   - Reads allocate a `Vec<u8>` per call.
   - Fixes: search the list bytes in place, pass looked-up descriptors on instead of looking
     them up again, and read into reusable buffers.
4. **Scans cost ~55 instructions per edge**, against ~34, and the cost is flat from 1024 to
   65536 edges.
   - Each scan copies the page bytes into a fresh `Vec`, decodes every slot, and pushes 8-byte
     `Edge`s into the caller's `Vec`.
   - A visitor (callback) read and decoding in place should close most of the gap. This is the
     top priority, so it is worth doing.
5. **Satellite detach is already ~7× faster than ic-stable-lara.** The scan stops at the
   partner, and a page skips without reading when it is all holes. It is still O(hub
   in-degree), as expected.
6. **Stable memory costs the same as the heap** for these access patterns. `MemoryManager`
   grows in 128-page buckets, which shows as the 256 pages of stable growth for 4096 edges.

## After the fixes (same day)

Changes:
- `counterpart` and `remove_edge` look at the smaller bucket first and stop when it holds the
  only entry back.
- Detach keeps a cursor per bucket within a step.
- Searches read 128-slot windows from their start position; full scans read a page at a time
  into one reused buffer.
- List lookups binary-search the raw entries and decode one descriptor.
- An insert reads each vertex record once and does not look a bucket up twice.
- `Graph::visit_neighbors` reads without collecting.

| Scenario | Before | After | ic-stable-lara |
| --- | ---: | ---: | ---: |
| Hub insert, 1024 | 15.4 M | 13.0 M | 8.9 M (one side) |
| Hub insert, 4096 | 62.3 M | 52.4 M | 18.3 M (one side) |
| Hub insert, 65536 | 998 M | 828 M | 204 M (one side) |
| Insert, parallel to 8 targets, 4096 | 53.9 M | 37.5 M | — |
| 51 scans of 1024, `neighbors` | 2.86 M | 2.95 M | 1.77 M |
| 51 scans of 1024, `visit_neighbors` | — | **1.58 M** | 1.77 M |
| 51 scans of 65536, `visit_neighbors` | — | 81.9 M | — |
| Remove half and compact, 4096 | 202 M | **17.5 M** | 8.7 M (one side) |
| Remove half and compact, 65536 | 41.8 B | **285 M** | 138 M (one side) |
| `counterpart` from the hub's last out half, 4096 | 189 K | 4.9 K | — |
| `counterpart` into the hub, 4096 | 164 K | 138 K | — |
| Edge lookup, 65536 | 2.7 K | 2.4 K | — |
| Detach a hub, 1024 | 35.2 M | **10.9 M** | 9.5 M sync, 13.3 M stepped |
| Detach a hub, 4096 | 145 M | 55.5 M | 35.6 M sync, 45.9 M stepped |
| Detach a hub, 65536 | 2.86 B | 1.45 B | — |
| Detach a satellite, 4096 | 180 K | 152 K | 1.28 M |

What the numbers say now:

- **Scans.** The visitor scans at ~30 instructions per edge, below ic-stable-lara's ~34, and
  stays flat up to 65536 edges.
  - Collecting into a `Vec` costs twice that. An `extend` variant measured slower, so the push
    version stays.
- **Removal is linear.** At ~4.3 K per edge, both halves plus compaction, it is about twice
  ic-stable-lara's one-sided removal.
- **Inserts** cost ~12.7 K per edge for both halves when the target is a new vertex, which has
  to create its bucket list, and ~9 K when the target already has one. That is about 4.5-6 K
  per half. ic-stable-lara's one-sided inserts cost ~3-4.5 K at these sizes. The rest of the
  gap is memory calls: reservation, list updates, and the page-store directory for new lists.
- **Hub detach** is level with ic-stable-lara at 1024 edges. Per edge it grows with the page
  count (~22 K at 65536), because each search rebuilds the paged bucket's leaf list. Jumping
  straight to the data page that holds the cursor would remove that.
- **Still open.** `counterpart` and removal into a hub, starting from the small side, still scan
  the hub up to the partner: 138 K at 4096 edges. A sorted target index would remove that scan.

## Second round (same day)

Changes:
- The page store keeps its minted page count and the pages its memories hold in the heap, and
  the graph keeps the vertex count there. Each owns its memories exclusively.
  - An insert's memory calls fall from ~23 to ~13 when the target is a new vertex, and from ~18
    to ~10 otherwise, measured with a counting `Memory`.
- An insert writes the vertex record it already read instead of reading it again.
- Paged searches walk the page index from their start page and stop early. Before, every search
  built the full leaf list.

| Scenario | First round | Now |
| --- | ---: | ---: |
| Hub insert, 1024 / 4096 / 65536 | 13.0 / 52.4 / 828 M | **11.4 / 46.3 / 740 M** |
| Insert, parallel to 8 targets, 4096 | 37.5 M | 34.6 M |
| Detach a hub, 4096 / 65536 | 55.5 M / 1.45 B | **51.6 M / 919 M** |
| Remove half and compact, 65536 | 285 M | 274 M |
| Edge lookup, 65536 | 2.4 K | 2.2 K |
| 51 visits of a 1024-edge run / 65536 paged | 1.58 / 81.9 M | 1.68 / 88.5 M |

**The visits regressed by 6-8%.** Only the codegen of `bucket::scan` explains it. Rebuilding the
committed code reproduces 1.58 M exactly, and the run path's logic did not change. Every variant
of the new paged branch measured 1.68 M:
- a shared closure,
- a reader struct,
- the paged path in a function that is never inlined,
- inline hints.

Visits stay below ic-stable-lara's 1.77 M.

**A fixed 1 KB stack buffer was tried and dropped.** It sped up full scans by ~5%, but slowed
short searches by up to 16%. Zero-filling seems to cost per byte on the IC, so buffers are sized
to each read.

**Insert is about 1.25× ic-stable-lara per half.** That is ~5.7 K for a half going to a new
vertex, against ~4.5 K for one-sided inserts at 4096 edges. What remains is mostly computation,
not memory calls.

## Third round: reads in either order, windows and chosen positions

| Scenario | Before | Now |
| --- | ---: | ---: |
| 51 visits, 1024-edge run / 65536 paged | 1.68 / 88.5 M | **1.35 / 66.9 M** |
| 51 visits of 65536, descending | — | 79.1 M |
| All 65536 in descending windows of 256 positions | — | 5.08 M (~78 per edge: a lookup and leaf list per window) |
| 256 scattered positions of 65536 (`targets_at`) | — | 235 K |
| 51 `neighbors` of 1024 | 2.94 M | 2.84 M |
| `counterpart` into a hub of 4096 / satellite detach of 65536 | 146 K / 2.19 M | 138 K / 2.12 M |

**Three loops over the same bytes.** The visitor, the search (`find`) and the collecting read
(`read`) each run their own decode loop.
- One shared loop was always slower for one of the three uses.
- An `enumerate` + `try_for_each` loop, in a function of its own per direction, is the fastest
  visitor: ~26 instructions per edge, against ic-stable-lara's ~34.
- A `zip` over positions is the fastest search and collecting read.

As in the second round, the IC instruction counts follow codegen more than logic.

**`targets_at` choice.** It reads a page's wanted range in one read when the positions are dense.
When there are more than 16 slots per wanted position, it reads each slot instead: 3.09 M fell to
235 K.

## Fourth round: edge values

| Scenario | Instructions |
| --- | ---: |
| Hub insert of 4096 edges with 8-byte values (both halves) | 109.5 M (~26.7 K per edge) |
| 51 values-first scans of 65536 values (`visit_values`) | 102.5 M (~31 per value) |
| 51 combined scans of 1024 edges with values | 4.37 M (~84 per edge) |
| `set_value` in a 4096-edge hub (both halves) | 19.4 K |

**First attempts.** The values-first scan collected each page into a `Vec` and zero-filled every
chunk (~92 per value). The combined scan read each value on its own (~1040 per edge). Joining a
slot page's live edges with one read of their values brought them to the figures above.

**Paths without values regressed.**

| Scenario | Regression |
| --- | ---: |
| Plain inserts | +7 % |
| Plain inserts, parallel edges | +10 % |
| Remove half and compact | +14 % |
| Detach a hub | +8-10 % |

- Memory calls did not change. The cost is in the bucket resolution and the larger plans the value
  code adds.
- Splitting plain and valued insert paths, resolving headers only for valued buckets, and
  skipping value reads for labels without a width recovered about half of it.
- Removing the heap allocations from the insert path is the next lever.
- Scans and lookups did not change.

## Fifth round: no heap allocation on the insert path

Changes:
- **Admission** uses fixed arrays: an edge has at most two halves and touches at most two
  vertices.
- **Bucket-list reads** go through a 64-byte stack buffer when the list fits, which covers three
  entries (`Pages::with_read`).
- **Adding a list entry** shifts the tail in place and writes the entry. It no longer rebuilds the
  list in a `Vec`.

| Scenario | Before values | With values | Now |
| --- | ---: | ---: | ---: |
| Hub insert, 1024 / 4096 / 65536 | 11.4 / 46.6 / 740 M | 12.2 / 49.3 / 789 M | **9.8 / 39.9 / 641 M** |
| Insert, parallel edges, 4096 | 34.4 M | 37.3 M | **30.8 M** |
| Hub insert with 8-byte values, 4096 | — | 109.5 M | 99.8 M |
| Remove half and compact, 4096 | 16.5 M | 19.0 M | 17.7 M |
| Detach a hub, 4096 | 52.1 M | 56.9 M | 55.6 M |
| Edge lookup, 65536 | 2.2 K | 2.2 K | 1.9 K |

Insert costs ~4.9 K per half; ic-stable-lara's one-sided steady state is ~3.6-4.5 K. Removal with
compaction and detach remain ~7 % above their cost before values. Compaction's own buffers (live
edges, packed bytes, moves) are the next allocations to remove.

## Sixth round (2026-11-23): descending walks, one compaction plan, short first windows

Changes:
- **Descending scans walk the page index backwards.** Before, they collected the whole leaf
  list and reversed it. The walk goes either way: it reads a window of entries from the near end,
  and visits a node's entries in the scan's order.
- **A compaction step is planned once.** Before, `preview` read the pages and listed the moves
  to plan the index changes. `step` then read the pages again and listed them again, and
  `verify` checked that the two lists matched. Now `compact::plan` reads once,
  `compact::apply` writes what the plan says, and no check is needed.
- **Searches read 32 slots first, then 128 at a time.** A detaching vertex's next edge, a pair's
  partner or a hole is usually near where the search starts.
  - 16 slots first made a block scan for a counterpart take two reads (+2 %).

| Scenario | Before | Now |
| --- | ---: | ---: |
| All 65536 in descending windows of 256 positions | 5.13 M | 2.93 M (−43 %) |
| Compaction after removing half, 4096 / 65536 | 4.08 M / 127 M | 3.67 M / 114 M (−10 %) |
| Detach a hub, 1024 / 4096 / 65536 | 10.6 M / 54.6 M / 980 M | 9.48 M / 50.3 M / 911 M (−7 to −11 %) |
| Refill with parallel edges, 65536 | 1.12 B | 1.04 B (−7 %) |

No other scenario moved by 2 % or more.
