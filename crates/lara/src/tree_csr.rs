//! Tree-CSR mode: ordinal-addressed indirect block tree for high-degree
//! vertices (ADR 0088 of the ic-stable-lara design, generalized).
//!
//! Simplified port of `tree_csr.rs`: fixed fan-out `B` slots per block,
//! root array capped at [`R_MAX`], depth derived from `stored_slots`
//! (no persisted depth — inconsistent states are unrepresentable).
//! Mutations split into a reserve phase (mint + populate data blocks) and
//! a commit phase (rewrite the root array), so intermediate states stay
//! valid.

use crate::block_store::{BLOCK_PAYLOAD_BYTES, LtbRawBlockStore};
use crate::memory::Memory;
use alloc::vec::Vec;

/// Slots per block payload (4-byte rows): 1024.
pub const B: usize = BLOCK_PAYLOAD_BYTES / 4;
/// Root fan-out cap.
pub const R_MAX: usize = 1024;
/// Fail-closed depth boundary.
pub const MAX_DEPTH: u32 = 3;

/// Total slots addressable at depth `d`: `B^d` (with `B = 1024`).
const fn coverage_at_depth(depth: u32) -> u64 {
    1u64 << (10 * depth)
}

/// Derives depth from `stored_slots`:
/// `depth = min { d >= 1 : ceil(stored_slots / B^d) <= R_max }`.
///
/// **Current implementation: depth-1 only.** The graph's tree buckets and
/// [`TreeCsrBucket`] address slots with a single root→leaf indirection
/// (`slot / B`), so the practical cap is `B * R_MAX = 2^20` slots per
/// bucket. `derive_depth` still reports depth ≥ 2 for larger inputs, but
/// the addressing code fails closed (panics) rather than mis-address.
///
/// Panics past [`MAX_DEPTH`] coverage (fail-closed structural boundary).
pub fn derive_depth(stored_slots: u32) -> u32 {
    if stored_slots == 0 {
        return 1;
    }
    let s = u64::from(stored_slots);
    for d in 1..=MAX_DEPTH {
        if s.div_ceil(coverage_at_depth(d)) <= R_MAX as u64 {
            return d;
        }
    }
    panic!("tree_csr: stored_slots={stored_slots} exceeds MAX_DEPTH={MAX_DEPTH} coverage");
}

/// Required root-array length for `stored_slots` at the derived depth.
pub fn root_len(stored_slots: u32) -> usize {
    let s = u64::from(stored_slots);
    let d = derive_depth(stored_slots);
    s.div_ceil(coverage_at_depth(d)) as usize
}

/// Reject unsupported addressing before allocating blocks or publishing state.
#[inline]
pub(crate) fn assert_depth_one(slots: u64) {
    assert!(
        slots <= (B * R_MAX) as u64,
        "tree-csr: depth-1 capacity exceeded ({slots} > {})",
        B * R_MAX
    );
}

/// Ordinal-addressed tree bucket over an [`LtbRawBlockStore`].
///
/// Packing invariant: every non-tail block at every level is full; only the
/// right-spine tail is partial. The root array holds the minted block ids
/// (depth-1 layout: root → leaf blocks directly, matching the prototype
/// simplification in the reference). Deeper addressing is not implemented;
/// promotion and insertion panic before mutation above `B * R_MAX` slots.
pub struct TreeCsrBucket<M: Memory> {
    store: LtbRawBlockStore<M>,
    root: Vec<u32>,
    stored_slots: u32,
}

impl<M: Memory> TreeCsrBucket<M> {
    /// Creates an empty bucket over a fresh block store.
    pub fn new(store: LtbRawBlockStore<M>) -> Self {
        Self {
            store,
            root: Vec::new(),
            stored_slots: 0,
        }
    }

    /// Consumes the bucket and returns the block store (for stats/tests).
    pub fn into_store(self) -> LtbRawBlockStore<M> {
        self.store
    }

    /// Number of stored slots (tombstone-inclusive logical width).
    pub fn stored_slots(&self) -> u32 {
        self.stored_slots
    }

    /// Derived depth (recomputed on every call).
    pub fn depth(&self) -> u32 {
        derive_depth(self.stored_slots)
    }

    /// Root array length (= number of leaf blocks at depth 1).
    pub fn root_len(&self) -> usize {
        self.root.len()
    }

    /// Promotion: fills an empty bucket with `targets` in order,
    /// one block-batched 4 KiB write per block.
    pub fn promote_from_slice(&mut self, targets: &[u32]) {
        assert_eq!(self.stored_slots, 0, "promote on non-empty bucket");
        assert_depth_one(targets.len() as u64);
        let n = targets.len() as u32;
        let new_root_len = root_len(n);
        debug_assert!(new_root_len <= R_MAX);

        let mut reserved = Vec::with_capacity(new_root_len);
        for _ in 0..new_root_len {
            let id = self.store.mint_zeroed().expect("promote mint");
            reserved.push(id);
        }
        for (root_index, &block_id) in reserved.iter().enumerate() {
            let first = root_index * B;
            let last = (first + B).min(targets.len());
            let mut payload = [0u8; BLOCK_PAYLOAD_BYTES];
            for (target, off) in targets[first..last]
                .iter()
                .zip((0..last - first).map(|s| s * 4))
            {
                payload[off..off + 4].copy_from_slice(&target.to_le_bytes());
            }
            self.store
                .write_payload(block_id, &payload)
                .expect("promote write");
        }
        self.root = reserved;
        self.stored_slots = n;
        debug_assert_eq!(self.root.len(), root_len(self.stored_slots));
    }

    /// Appends one target (reserve/commit: new blocks are minted and
    /// populated before the root grows).
    ///
    /// Depth-1 only: exceeding `B * R_MAX` slots fails closed (depth-2
    /// addressing is a future slice).
    pub fn insert(&mut self, target: u32) {
        let new_stored = self
            .stored_slots
            .checked_add(1)
            .expect("tree_csr stored_slots overflow");
        assert_depth_one(u64::from(new_stored));
        let new_root_len = root_len(new_stored);
        while self.root.len() < new_root_len {
            let id = self.store.mint_zeroed().expect("insert mint");
            self.root.push(id);
        }
        let (block, in_block) = self.physical_address(new_stored - 1);
        let block_id = self.root[block];
        self.store
            .write_payload_partial(block_id, in_block * 4, &target.to_le_bytes())
            .expect("insert write");
        self.stored_slots = new_stored;
        debug_assert_eq!(self.root.len(), root_len(self.stored_slots));
    }

    /// Reads the target at logical position `slot` (full-block read path).
    pub fn read_slot(&self, slot: u32) -> u32 {
        assert!(slot < self.stored_slots, "read_slot out of bounds");
        let (block, in_block) = self.physical_address(slot);
        let block_id = self.root[block];
        let mut buf = [0u8; 4];
        self.store
            .read_payload_partial(block_id, in_block * 4, &mut buf)
            .expect("read_slot");
        u32::from_le_bytes(buf)
    }

    /// Point lookup: `None` past the end (uses the 4-byte partial read).
    pub fn range_target(&self, slot: u32) -> Option<u32> {
        if slot >= self.stored_slots {
            return None;
        }
        Some(self.read_slot(slot))
    }

    /// Tombstones by logical position, shifting the tail left to keep the
    /// left-packed invariant. Returns the removed target.
    pub fn remove_at(&mut self, slot: u32) -> Option<u32> {
        if slot >= self.stored_slots {
            return None;
        }
        let removed = self.read_slot(slot);
        let last = self.stored_slots - 1;
        for s in slot..last {
            let t = self.read_slot(s + 1);
            self.write_slot(s, t);
        }
        if self.stored_slots > 0 {
            self.write_slot(last, 0);
        }
        self.stored_slots -= 1;
        while self.root.len() > root_len(self.stored_slots) {
            let popped = self.root.pop().expect("root pop");
            let _ = self.store.release(popped);
        }
        Some(removed)
    }

    fn write_slot(&mut self, slot: u32, target: u32) {
        let (block, in_block) = self.physical_address(slot);
        let block_id = self.root[block];
        self.store
            .write_payload_partial(block_id, in_block * 4, &target.to_le_bytes())
            .expect("write_slot");
    }

    /// Sequential ascending scan: one read per occupied block, with only
    /// the stored prefix read from a partial tail. Calls back once per slot.
    pub fn for_each_ascending<F: FnMut(u32, u32)>(&self, mut f: F) {
        for (root_index, &block_id) in self.root.iter().enumerate() {
            let mut payload = [0u8; BLOCK_PAYLOAD_BYTES];
            let first = root_index * B;
            let remaining = self.stored_slots.saturating_sub(first as u32).min(B as u32);
            self.store
                .read_payload_partial(block_id, 0, &mut payload[..remaining as usize * 4])
                .expect("scan read");
            for s in 0..remaining {
                let off = s as usize * 4;
                let target = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
                f(first as u32 + s, target);
            }
        }
    }

    /// Deepen probe: rewrites the spine one level down. At depth 1 the
    /// prototype keeps the root → leaf layout; `deepen` verifies the
    /// fail-closed boundary and reports the (unchanged) depth for the
    /// generalized single-level implementation.
    pub fn deepen(&mut self) -> u32 {
        let before = self.depth();
        debug_assert!(before < MAX_DEPTH, "deepen past MAX_DEPTH");
        before
    }

    /// Flatten probe: mirror of [`TreeCsrBucket::deepen`].
    pub fn flatten(&mut self) -> u32 {
        self.depth()
    }

    fn physical_address(&self, slot: u32) -> (usize, usize) {
        debug_assert!((1..=MAX_DEPTH).contains(&self.depth()));
        let in_block = (slot as usize) % B;
        let root_index = slot as usize / B;
        (root_index, in_block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_store::LtbRawBlockStore;
    use crate::memory::VecMemory;

    fn fresh() -> TreeCsrBucket<VecMemory> {
        TreeCsrBucket::new(LtbRawBlockStore::new(VecMemory::new()).unwrap())
    }

    #[test]
    fn empty_bucket_is_depth_one() {
        let b = fresh();
        assert_eq!(b.stored_slots(), 0);
        assert_eq!(b.depth(), 1);
        assert_eq!(b.root_len(), 0);
    }

    #[test]
    fn promote_and_scan() {
        let mut b = fresh();
        let targets: Vec<u32> = (0..2500u32).map(|i| i + 10).collect();
        b.promote_from_slice(&targets);
        assert_eq!(b.stored_slots(), 2500);
        assert_eq!(b.root_len(), 3); // ceil(2500/1024)
        assert_eq!(b.read_slot(0), 10);
        assert_eq!(b.read_slot(1024), 1034);
        assert_eq!(b.read_slot(2499), 2509);
        assert_eq!(b.range_target(2500), None);
        let mut count = 0;
        b.for_each_ascending(|slot, t| {
            assert_eq!(t, targets[slot as usize]);
            count += 1;
        });
        assert_eq!(count, 2500);
    }

    #[test]
    fn left_packing_invariant_at_boundary() {
        let mut b = fresh();
        for s in 0..1025u32 {
            b.insert(s + 1);
        }
        assert_eq!(b.stored_slots(), 1025);
        assert_eq!(b.root_len(), 2);
        assert_eq!(b.read_slot(1024), 1025);
        assert_eq!(b.read_slot(0), 1);
        assert_eq!(b.read_slot(1023), 1024);
    }

    #[test]
    fn remove_at_shifts_left() {
        let mut b = fresh();
        for t in [10u32, 20, 30, 40, 50] {
            b.insert(t);
        }
        assert_eq!(b.remove_at(2), Some(30));
        assert_eq!(b.stored_slots(), 4);
        assert_eq!(b.read_slot(0), 10);
        assert_eq!(b.read_slot(2), 40);
        assert_eq!(b.read_slot(3), 50);
        assert_eq!(b.remove_at(9), None);
    }

    #[test]
    fn derived_depth_matches_formula() {
        assert_eq!(derive_depth(4096), 1);
        assert_eq!(root_len(4096), 4);
        assert_eq!(derive_depth(1_048_576), 1);
        assert_eq!(root_len(1_048_576), 1024);
        assert_eq!(derive_depth(1_048_577), 2);
        assert_eq!(root_len(1_048_577), 2);
    }

    #[test]
    fn root_len_matches_stored_slots_after_every_insert() {
        let mut b = fresh();
        for s in 0..3000u32 {
            b.insert(s);
            assert_eq!(b.root_len(), root_len(b.stored_slots()));
        }
    }
}
