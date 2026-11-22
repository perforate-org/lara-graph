# Inline edge properties in lara-core: design (2026-11-22)

Implemented with the five recommendations below; see `crates/lara-core/DESIGN.md` ("Edge values").

This note covers what Gleaph needs, the options, and a proposal. The requirements come from
Gleaph's ADR 0008, 0051 and 0099, `design/storage/labeled-edge-inline-properties.md`,
`inline-property-bytes-first-traversal.md`, and the graph crate's call sites.

## What Gleaph needs

- **Width.** Each label has one fixed byte width (`u16`, 0 = no value), chosen by Gleaph's
  schema (the Router). Out, In and Undirected buckets of a label share it.
- **Changing the width.** Only 0 → w is allowed. Edges that existed before read as zero bytes.
  There is no null value.
- **Both halves hold the value.** Reverse traversals read values (incoming expansion, detach).
  Index postings key on the canonical half.
- **Insert and update.**
  - An insert writes the bytes to both halves atomically.
  - An update finds the other half, reads the old value (to update indexes), and overwrites both.
- **Values-first traversal.**
  - Phase 1 scans a bucket's **values only**, getting positions and bytes.
  - Phase 2 reads the edges at the matching positions; `targets_at` already does this.
  - Combined edge-and-value scans also exist, for weighted paths and value-aware expansion.
- **Reads at chosen positions**, for one edge or a batch.
- **Removal and detach** must give the caller the canonical half's value, so it can remove index
  postings.
- **Compaction** moves a value with its edge.
- **Widths.** Values wider than 8 bytes needed a blob map in ic-stable-lara only because its spill
  cells were 8 bytes. Fixed widths need no blobs.

ic-stable-lara's lessons (ADR 0099):
- Addressing values by live ordinal cost O(n) per delete, caused a wrong-row update bug, and
  blocked compaction.
- Separate residencies for edges and values drifted out of step.
- Materializing values for existing rows needed retries across messages.

The fix ADR 0099 recommends:
- Address a value **by the edge's position**.
- A delete writes no value, and a hole refill writes one value.
- Only compaction moves values.
- One owner publishes edges and values together.

## Proposal

### 1. Values in a column beside the slots, addressed by position

A bucket with values has a **value plane** next to its slot plane. The value of position `p` is
cell `p` of that plane, `width` bytes long.

| Choice | Target-only scans | Values-first scans | Insert | Page geometry |
| --- | --- | --- | --- | --- |
| Rows: (target, value) together | read `4 + w` bytes per edge | strided | one write | slots per page vary by bucket |
| **Columns: slots and values apart** | **unchanged** | **contiguous** | two writes | **slot pages unchanged** |

Columns, because:
- the CSR scan of targets is the first priority and stays exactly as it is;
- Gleaph's values-first phase reads one contiguous plane;
- every piece of slot-page arithmetic (1024 slots per page, per-page tombstone counts) stays.

The value plane has its own geometry: `cells_per_page = 4096 / w`, at most 4096 bytes per cell.
It reuses the run and paged storage, generalized from 4-byte slots to cells of any width:
- **Run form:** one extent of `capacity × w` bytes. It grows with the slot run, and at most one
  page is copied.
- **Paged form:** value pages listed by the same page-index structure. A **hollow** entry means
  every value in that page is zero.

### 2. Width per label, declared once, and no materialization

The graph keeps a **label width table**: `(label u16, width u16)` entries, sorted, in one
list-store extent. It is referenced from the 8 spare bytes of the vertices header, so it holds up
to 1024 labels that carry values.

- `Graph::declare_width(label, w)` costs O(1). The only changes allowed are 0 → w, or w → w
  unchanged. Anything else fails with `Error::WidthMismatch`.
- Buckets do **not** materialize. A bucket created before the declaration has no value plane, and
  its values read as zeros. A value plane is created on the bucket's first value write. Its pages
  appear when written to; missing and hollow pages read as zeros.
- This removes ic-stable-lara's slowest and most fragile part: zero-filling existing rows,
  deferred to a queue and retried. Declaring a width on a label with a million edges costs
  nothing up front, and each later write allocates at most one page.

### 3. Where a bucket's value plane is referenced

The 16-byte descriptor has no room for a second extent. A bucket with a value plane gets a new
descriptor kind, **`Valued { header }`**. Its header is a 32-byte extent holding:
- the inner slot descriptor (Run or Paged, 16 bytes);
- the value plane's root (kind and extent, 8 bytes);
- the width (2 bytes).

Consequences:
- Buckets without values, and every tiny bucket, are unchanged.
- A bucket with values is never tiny: three inline slots leave no room for values, the same as in
  Gleaph today.
- A lookup of a bucket with values costs one extra 32-byte read.
- Inserts update the header instead of the list entry. That is still one write.

### 4. Operations

| API (sketch) | Behaviour |
| --- | --- |
| `insert_directed(.., placement, value: Value<'_>)`, same for undirected | Writes the slot and the value cell in both halves, in one reservation. The length must match the label's width (`Error::WidthMismatch`). `Value::NONE` is for labels without a width. |
| `set_value(half, value)` | Overwrites both halves' cells and returns the old canonical value (Gleaph needs it to update indexes). |
| `value(half, out)` and `values_at(v, label, side, positions, out)` | Point and batch reads: one cell per position, or one read per touched value page. |
| `visit_values(v, label, side, order, f(Position, &[u8]))` | Phase 1 scan. It reads the value plane only and skips pages the slot index marks fully dead. Other holes are included with their stale bytes, and phase 2 (`targets_at`) drops them. |
| `visit_neighbors_with_values(.., f(Edge, &[u8]))` | Combined scan: one slot page and one value page per step. |
| `remove_edge`, `detach_step` | Report the canonical half's value with the `Halves`. A delete writes no value; a tombstone's cell is never read again. |
| `compact` | Moves each moved edge's value cell. Releasing a dead slot page also releases the value pages wholly inside it. |

### 5. Invariants and tests

- **Invariant:** every live position reads the value last written to it, or zeros if none was
  written since the width was declared.
- **Model test:** extend the paired model so every logical edge carries a random value. Mix in:
  - deletes, hole refills (`ReuseHoles`) and compactions, since only that mix tells position
    addressing from ordinal addressing;
  - width declarations on labels that already have edges;
  - detach.
  Check that both halves read the same value, and that removal reports it.
- **Failure test:** an insert with a value that runs out of memory changes nothing, value plane
  included.

## Questions to settle

1. **Columns, not rows.** Recommended.
2. **A label width table with lazy zeros** instead of per-bucket materialization. Recommended.
   The table lives in lara-core, so Gleaph's Router declares each label once.
3. **Maximum width: 4096 bytes**, one cell per value page at most. Gleaph allows up to 64 KiB,
   but no realistic inline property is that wide. Wider values would move to Gleaph's sidecar
   store.
4. **Holes in values-only scans.** Reporting them, with phase 2 dropping them, keeps the scan to
   one plane. The alternative is to also read the slot page to skip holes, which costs about
   twice the reads in buckets with holes. Recommended: report them.
5. **Removal reports carry the value** (an allocation per removed edge) instead of Gleaph reading
   the value before removing. Recommended: carry it. Detach needs it anyway, and it saves a
   second lookup.
