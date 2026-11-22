# What to adopt from ic-stable-lara (2026-11-22)

This note comes from reading ic-stable-lara's hot paths and how Gleaph uses it
(`DeferredBidirectionalLabeledLaraGraph` only). "Ref." gives the place in Gleaph to look:
- `L/` is `gleaph/crates/ic-stable-lara/src/labeled/`
- `G/` is `gleaph/crates/graph/src/`
- `D` is `L/bidirectional/deferred.rs`

## A. Gleaph needs these (lara-core lacks them)

| # | Capability | Gleaph use | Ref. | Fit for lara-core |
| --- | --- | --- | --- | --- |
| 1 | **Insertion-order placement**: append only, never refill an interior hole | `ORDER BY INSERTION(e) [DESC]`; set per label | ADR 0052; `G/facade/store/helpers.rs` (`lara_edge_placement`) | lara-core always refills a safe hole, which puts a new edge before older ones. Needs a per-bucket policy (`Insertion` / `Unordered`) stored in the descriptor or label. Compaction already keeps order. |
| 2 | **Descending scans** | insertion DESC; offset skipping walks downward | ADR 0050 §4; `D:2429` | Walk pages and slots in reverse; the per-page tombstone counts still skip dead pages. |
| 3 | **Cursor windows**: `(start position, limit) → (next, exhausted)`, holes counted | durable export cursor | `D:1256`; `G/index/canonical_export.rs:987` | Positions are stable, so the cursor is just a position. |
| 4 | **Batch reads of chosen positions** | phase 2 of property-first expand | `D:2303/2379` | A sorted batch `edges_at(&[Position])`: one read per touched page. |
| 5 | **Inline edge properties**: fixed width per label | ~25 call sites: batch value reads, updates, insert with bytes | ADR 0099 | Store values in a column parallel to each bucket's slots, **addressed by position**. ic-stable-lara's first, ordinal-addressed design cost O(n) per delete (454 K instructions at 128 rows). Moves carry the value. |
| 6 | **Permanent deleted state** for a vertex; ids never reused | `is_vertex_live` in ~10 places | ADR 0017 | Detach currently ends in "attached" again. Add `Deleted`, and a public `vertex_state`. The detaching bit can replace Gleaph's pending-purge bitmap. |
| 7 | **Removed edges reported with their property bytes** | removing index postings during detach and removal | `G/facade/store/helpers.rs:64-128` | Comes with 5: `Halves` plus the value. |
| 8 | **Canonical half documented** | re-keying sidecars on moves | `helpers.rs:33-48` | Directed: the Out half. Undirected: decide and document whether the smaller or larger end is canonical. `Halves` uses the smaller end; Gleaph today uses the larger. |
| 9 | **Bucket limit** | ic-stable-lara allows 1024 per vertex per orientation | `L/graph.rs:97` | lara-core allows 215 across all sides (one page of list). If Gleaph schemas can exceed it, lists need a second level. |

## B. Hot-path techniques (no `unsafe` in ic-stable-lara either)

| Rank | Technique | Ref. | Expected effect |
| --- | --- | --- | --- |
| 1 | **No heap allocation on the insert path.** Point reads into fixed arrays: a 1-2 entry list is read directly, larger ones are binary-searched one entry per probe, and only big ones are bulk-read. | `L/graph/bucket.rs:401-497` | lara-core allocates ~5-7 `Vec`s per half: `places`, `touched`, records, `fresh`, the list read, and `labels::add`'s split. Each costs a few hundred instructions in wasm. This is the likely remainder of the insert gap: 5.7 K per half against ~3.6 K steady-state. |
| 2 | **Heap mirrors of hot headers**, rebuilt on open | `lara/edge.rs:117`, `lara/vertex.rs:462` | lara-core already mirrors minted pages, room and vertex count. The per-class partial-list heads still come from the directory on every sub-page allocation. |
| 3 | **Bucket lookup cache**: the last lookup plus 64 hashed entries, validated on read, cleared on rewrite | `L/graph/bucket.rs:306-551` | For lara-core: cache `(vertex, key) → list index`, check it against the record, and point-read the 16-byte descriptor. Helps vertices with many labels. |
| 4 | **Append without reading the bucket** when there are no holes (an O(1) check) | `L/graph/insert.rs:365-385` | lara-core reads the whole run when it has holes. Scan backwards in short windows instead, or not at all under the `Insertion` policy (A1). |

Not worth adopting:
- The scan path. It zero-fills the same way, and lara-core's visitor is already cheaper.
- The counterpart lookup. It scans linearly, and lara-core's smaller-side-first rule is better.
- The Tree-CSR append, which is equivalent to lara-core's paged append.

**Reading without zero-filling.** The only safe route is a `Memory::read_to_vec` trait method:
- Its default resizes the vector and reads into it.
- `VecMemory` can override it with `extend_from_slice`, with no zero-fill.
- Gleaph's stable adapter can implement it with `read_unsafe`, so the `unsafe` lives in the adapter, not in lara-core.

## C. Skip

These are physical design only, or ic-stable-lara's own documents judged them mistakes.

- Default-label bypass rows (GAP-LARA-MODE-FOOTPRINT, bugs when converting a vertex in or out of bypass).
- PMA leaves, the counts tree (56 % of append time and no reader), spill runs.
- The B-tree hub tier, rejected in ADR 0022: 6.3× slower scans.
- The maintenance queue's layout kinds. lara-core's explicit `compact` and `detach_step` steps cover the two that matter.
- Swap compaction, which breaks the rank pairing.
- One-sided repair inserts: one graph cannot diverge.
- Remote targets: not supported by Gleaph either.

## D. Testing ideas worth copying

- **Allocation failure injection.** Fail at every allocation inside an insert, and assert every memory's bytes are unchanged.
- **Reads leave memory untouched.** Snapshot the memories around reads and errors in the model test.
- **Public invariant check.** Expose `check` as a `debug_verify`, including the rank pairing.
- **Formal codec proofs.** ic-stable-lara has Lean proofs of its codecs, with a `SOURCES.sha256` freshness check.
- **Layout settings in the header.** Record any setting that changes how stored data is read, such as the placement policy, in durable metadata. ADR 0039 shows the cost of not doing so.
