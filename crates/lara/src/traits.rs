//! Shared CSR row traits and the common mutable-CSR interfaces.
//!
//! The row traits are split into a **core minimal** layer ([`FixedWidth`],
//! [`CsrVertex`], [`CsrEdge`]) that knows nothing about liveness, and
//! **optional marker capabilities** ([`CsrEdgeTombstone`], [`CsrVertexTombstone`]).
//! Physical deletion needs no marker: LARA's unordered graph operations
//! require only [`CsrEdge`]. [`RemovableCsr`] remains separate from
//! [`DynamicCsr`] so insert-only graph implementations need not support deletion.

/// One fixed-width wire row (vertex or edge).
///
/// `Default` is required so callers can construct a fresh row and then
/// mutate it with the `with_*` builders (e.g. `E::with_neighbor(E::default(),
/// dst)`).
pub trait FixedWidth: Copy + Default {
    /// Wire width in bytes.
    const BYTES: usize;
    /// Decodes a row from exactly [`Self::BYTES`] bytes.
    fn read_from(bytes: &[u8]) -> Self;
    /// Encodes the row into exactly [`Self::BYTES`] bytes.
    fn write_to(&self, bytes: &mut [u8]);
}

/// One fixed-width CSR vertex row.
///
/// This is the scan + update core: it exposes the geometry a graph needs to
/// locate and mutate a vertex's neighborhood, with no notion of liveness.
pub trait CsrVertex: FixedWidth {
    /// Global edge-slab slot where this vertex's neighborhood starts.
    fn base_slot_start(&self) -> u64;
    /// Logical (live) out-degree.
    fn degree(&self) -> u32;
    /// Stored slab prefix width. Live degree is this width minus any
    /// policy-selected markers, plus live log entries. Unordered LARA has
    /// no such markers. Concrete rows may tag tiny storage in this field;
    /// tree rows keep their extent separately.
    fn stored_degree(&self) -> u32;
    /// Head index of the per-leaf overflow log chain, or `-1`.
    fn log_head(&self) -> i32;
    /// Tree-CSR bucket id for high-degree vertices, or [`NO_BUCKET`].
    fn tree_bucket(&self) -> u32;

    /// Returns a copy with a new slab base.
    fn with_base_slot_start(self, start: u64) -> Self;
    /// Returns a copy with live and stored degree both set to `n`.
    fn with_degree(self, n: u32) -> Self;
    /// Returns a copy with only the live degree changed.
    fn with_live_degree(self, n: u32) -> Self;
    /// Returns a copy with only the stored width changed.
    fn with_stored_degree(self, n: u32) -> Self;
    /// Returns a copy with a new overflow-log head.
    fn with_log_head(self, head: i32) -> Self;
    /// Returns a copy with a new tree bucket id.
    fn with_tree_bucket(self, bucket: u32) -> Self;
}

/// Sentinel for "no tree bucket".
pub const NO_BUCKET: u32 = u32::MAX;

/// One fixed-width CSR edge record.
///
/// The core trait knows nothing about liveness: every record is a neighbor
/// id. Dense physical deletion needs no additional edge capability.
pub trait CsrEdge: FixedWidth {
    /// Adjacent vertex id.
    fn neighbor(&self) -> u32;
    /// Returns a copy with a new neighbor id.
    fn with_neighbor(self, neighbor: u32) -> Self;
}

/// Derived capability: edge tombstones (logical deletion).
///
/// Describes marker records for algorithms using logical deletion.
/// LARA's default graph uses dense physical deletion and does not inspect
/// this capability. Implementing it does not select an ordering policy.
/// The explicit core Insertion setter uses it for typed slab deletion.
/// A marker and its codec round-trip must be recognized by `is_tombstone`.
/// Ordered insertion rejects both marker inputs and records whose existing
/// default/neighbor reconstruction is a marker, before choosing storage.
pub trait CsrEdgeTombstone: CsrEdge {
    /// The tombstone marker record (a slot that reads as "deleted").
    fn tombstone_edge() -> Self;
    /// `true` when this slot is a logical-delete marker.
    fn is_tombstone(&self) -> bool;
}

/// Derived capability: vertex tombstones (future vertex-deletion support).
///
/// Reserved as the extension point for row-level deletion; the generalized
/// crate does not yet exercise it, so implementations may be trivial.
pub trait CsrVertexTombstone: CsrVertex {
    /// `true` when this row is a logical-delete marker.
    fn is_row_tombstone(&self) -> bool;
    /// Returns a copy with the row-tombstone flag set to `t`.
    fn with_row_tombstone(self, t: bool) -> Self;
}

/// Common interface shared by DGAP-style and LARA graphs so the benchmark
/// harness can drive them uniformly.
///
/// This is the **insert + scan** core. Deletion is a separate capability
/// ([`RemovableCsr`]) so read-only and insert-only graphs can implement
/// this without a deletion path.
pub trait DynamicCsr {
    /// Number of vertices (including rounded-up padding rows).
    fn num_nodes(&self) -> u64;
    /// Number of live edges.
    fn num_edges(&self) -> u64;
    /// Logical out-degree of `v`.
    fn out_degree(&self, v: u32) -> u64;
    /// Inserts one directed edge (no dedup).
    fn insert_edge(&mut self, src: u32, dst: u32);
    /// Visits `v`'s live neighbors.
    fn for_each_neighbor<F: FnMut(u32)>(&self, v: u32, f: F);
    /// Visits every live edge once.
    fn scan_all<F: FnMut(u32)>(&self, f: F);
}

/// Deletion-capable graph: [`DynamicCsr`] plus edge removal.
///
/// Does not prescribe a marker encoding or survivor order. LARA honors
/// its selected policy (Unordered by default); the DGAP port shifts in place.
pub trait RemovableCsr: DynamicCsr {
    /// Removes one directed edge; `false` when absent.
    fn remove_edge(&mut self, src: u32, dst: u32) -> bool;
}
