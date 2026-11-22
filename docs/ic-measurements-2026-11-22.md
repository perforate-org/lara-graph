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
