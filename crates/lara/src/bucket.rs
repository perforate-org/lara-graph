//! Labeled-bucket descriptor store.
//!
//! The labeled layer stores slab records as plain 4-byte neighbors (the
//! shared [`crate::graph::Edge`] wire) and keeps the label → sub-span
//! mapping in a per-vertex **descriptor run** here (ADR 0001 parity: the
//! label lives out of band in the descriptor, not in the edge record).
//!
//! Each labeled vertex owns a contiguous, label-sorted run, independently
//! located in this slot array. Capacity is the next power of two of its
//! nonzero bucket count. Full runs prefer exact-class reuse, then extend at
//! the arena frontier or relocate just that run. Global vertex-order
//! concatenation is not required.
//!
//! Dense standalone push/insert/remove operations remain available. The
//! labeled graph instead uses the crate-private run allocator: its slot
//! extent includes active capacity and free runs, not just live descriptors.

use crate::memory::{GrowFailed, Memory, safe_write};
use crate::traits::FixedWidth;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

// Allocator-local link sentinel, outside every checked u32 run-start value.
const NO_RUN: u64 = u64::MAX;

fn run_capacity(count: u32) -> u32 {
    assert!(
        count > 0 && count <= 65_536,
        "bucket count outside label domain"
    );
    count.next_power_of_two()
}

fn checked_run_end(start: u64, capacity: u32) -> u64 {
    start
        .checked_add(u64::from(capacity))
        .filter(|&end| end <= u64::from(u32::MAX) + 1)
        .expect("bucket descriptor address space exhausted")
}

/// One label bucket of one vertex: the mapping from a dense `label` to its
/// sub-span in the shared slab, plus per-bucket liveness/log/tree state.
///
/// Wire layout (little-endian, 24 bytes):
///
/// | offset | field        | width |
/// |--------|--------------|-------|
/// | 0      | `label`      | u16   |
/// | 2      | (padding)    | u16   |
/// | 4      | `degree`     | u32   |
/// | 8      | `stored`     | u32   |
/// | 12     | `log_head`   | i32   |
/// | 16     | `edge_start` | u32   |
/// | 20     | `tree_bucket`| u32   |
///
/// `edge_start` is an **absolute slot** in the shared edge slab (matching
/// ic-stable-lara). This is a narrow address field, not a practical-capacity
/// or recovery guarantee. `degree` is the live count, `stored` the
/// tombstone-inclusive width, `log_head`
/// the head of the vertex's leaf-log chain (`-1` = none) restricted to
/// this bucket, and `tree_bucket` the Tree-CSR state id (`u32::MAX` =
/// not promoted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BucketDescriptor {
    /// Dense label id (wire `u16`).
    pub label: u16,
    /// Live edges in this bucket.
    pub degree: u32,
    /// Tombstone-inclusive stored width in the slab.
    pub stored: u32,
    /// Leaf-log chain head for this bucket (-1 = none).
    pub log_head: i32,
    /// Absolute slab slot where this bucket's sub-span begins.
    pub edge_start: u32,
    /// Tree-CSR state id, or `u32::MAX` when not promoted.
    pub tree_bucket: u32,
}

impl Default for BucketDescriptor {
    fn default() -> Self {
        Self {
            label: 0,
            degree: 0,
            stored: 0,
            log_head: -1,
            edge_start: 0,
            tree_bucket: u32::MAX,
        }
    }
}

impl FixedWidth for BucketDescriptor {
    const BYTES: usize = 24;

    fn read_from(bytes: &[u8]) -> Self {
        Self {
            label: u16::from_le_bytes(bytes[0..2].try_into().unwrap()),
            degree: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            stored: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            log_head: i32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            edge_start: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            tree_bucket: u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
        }
    }

    fn write_to(&self, bytes: &mut [u8]) {
        bytes[0..2].copy_from_slice(&self.label.to_le_bytes());
        bytes[2..4].fill(0);
        bytes[4..8].copy_from_slice(&self.degree.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.stored.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.log_head.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.edge_start.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.tree_bucket.to_le_bytes());
    }
}

/// Descriptor slot store living in the shared `buckets` memory region.
///
/// Slot extent and run free heads are heap state (like vertex rows); the
/// bytes themselves live in `M`. Operations take the region as an explicit
/// argument so the graph can borrow its slab / logs / descriptors from one
/// `LaraMemories` bundle.
pub struct BucketStore {
    len: u64,
    cap: u64,
    free_heads: Vec<u64>,
}

impl BucketStore {
    /// Creates an empty store over a freshly sized `buckets` region.
    pub fn new(initial_cap: u64) -> Self {
        Self {
            len: 0,
            cap: initial_cap,
            free_heads: Vec::new(),
        }
    }

    /// Slot extent. For dense callers this is the descriptor count; the
    /// labeled run allocator also includes reserved slack and free runs.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the allocated slot extent is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Sets the logical descriptor count (used when rebuilding the store in
    /// place; must be `<= cap`).
    pub fn set_len(&mut self, len: u64) {
        assert!(len <= self.cap);
        self.len = len;
    }

    /// Reads the descriptor at global index `idx`.
    pub fn read<M: Memory>(&self, mem: &M, idx: u64) -> BucketDescriptor {
        debug_assert!(idx < self.len);
        let off = idx * BucketDescriptor::BYTES as u64;
        let mut buf = [0u8; 24];
        mem.read(off, &mut buf);
        BucketDescriptor::read_from(&buf)
    }

    /// Writes `d` at global index `idx` (allow `idx == len` for append as
    /// part of [`BucketStore::insert_at`]).
    pub fn write<M: Memory>(&mut self, mem: &mut M, idx: u64, d: &BucketDescriptor) {
        debug_assert!(idx <= self.len);
        let mut buf = [0u8; 24];
        d.write_to(&mut buf);
        mem.write(idx * BucketDescriptor::BYTES as u64, &buf);
    }

    /// Ensures capacity for at least `len` descriptors, growing `mem` and
    /// bumping `cap` as needed. Does not change the logical [`BucketStore::len`].
    pub fn grow<M: Memory>(&mut self, mem: &mut M, len: u64) -> Result<(), GrowFailed> {
        if len <= self.cap {
            return Ok(());
        }
        let new_cap = (self.cap * 2).max(8).max(len);
        safe_write(mem, new_cap * BucketDescriptor::BYTES as u64 - 1, &[0u8])?;
        self.cap = new_cap;
        Ok(())
    }

    /// Inserts `d` at global position `pos`, shifting the tail right.
    /// Caller must `grow` first so `pos < len` (or `pos == len` appends).
    pub fn insert_at<M: Memory>(&mut self, mem: &mut M, pos: u64, d: &BucketDescriptor) {
        assert!(pos <= self.len);
        // Shift [pos, len) right by one.
        let mut cur = self.len;
        while cur > pos {
            let prev = self.read(mem, cur - 1);
            self.write(mem, cur, &prev);
            cur -= 1;
        }
        self.write(mem, pos, d);
        self.len += 1;
    }

    /// Removes the descriptor at `pos`, shifting the tail left.
    pub fn remove_at<M: Memory>(&mut self, mem: &mut M, pos: u64) {
        assert!(pos < self.len);
        let mut cur = pos;
        while cur + 1 < self.len {
            let next = self.read(mem, cur + 1);
            self.write(mem, cur, &next);
            cur += 1;
        }
        self.len -= 1;
    }

    /// Appends `d` at the end of the store.
    pub fn push<M: Memory>(&mut self, mem: &mut M, d: &BucketDescriptor) -> Result<(), GrowFailed> {
        self.grow(mem, self.len + 1)?;
        self.insert_at(mem, self.len, d);
        Ok(())
    }

    /// Allocates a whole run. No active descriptor or row is published here.
    pub(crate) fn alloc_run<M: Memory>(
        &mut self,
        mem: &mut M,
        count: u32,
    ) -> Result<u32, GrowFailed> {
        let capacity = run_capacity(count);
        let class = capacity.trailing_zeros() as usize;
        let head = self.free_heads.get(class).copied().unwrap_or(NO_RUN);
        if head != NO_RUN {
            assert!(
                checked_run_end(head, capacity) <= self.len,
                "free run outside arena"
            );
            let mut next = [0; 8];
            mem.read(head * 24, &mut next);
            self.free_heads[class] = u64::from_le_bytes(next);
            return Ok(head as u32);
        }
        let end = checked_run_end(self.len, capacity);
        self.grow(mem, end)?;
        let start = self.len as u32;
        self.len = end;
        Ok(start)
    }

    /// Called only after the old run is no longer referenced by a vertex.
    pub(crate) fn release_run<M: Memory>(&mut self, mem: &mut M, start: u32, count: u32) {
        if count == 0 {
            return;
        }
        let capacity = run_capacity(count);
        assert!(
            checked_run_end(u64::from(start), capacity) <= self.len,
            "released run outside arena"
        );
        let class = capacity.trailing_zeros() as usize;
        if self.free_heads.len() <= class {
            self.free_heads.resize(class + 1, NO_RUN);
        }
        mem.write(u64::from(start) * 24, &self.free_heads[class].to_le_bytes());
        self.free_heads[class] = u64::from(start);
    }

    /// Writes a sorted insertion, returning the run location. The caller
    /// publishes its row and then releases the old run if the location changed.
    pub(crate) fn insert_run<M: Memory>(
        &mut self,
        mem: &mut M,
        start: u32,
        count: u32,
        pos: u32,
        d: &BucketDescriptor,
    ) -> Result<u32, GrowFailed> {
        assert!(pos <= count && count < 65_536, "invalid bucket insertion");
        if count > 0 {
            let mut capacity = run_capacity(count);
            let end = checked_run_end(u64::from(start), capacity);
            assert!(end <= self.len, "run outside arena");
            // Reuse an existing larger run before reserving any new tail space.
            if count == capacity
                && end == self.len
                && self
                    .free_heads
                    .get(capacity.trailing_zeros() as usize + 1)
                    .is_none_or(|&head| head == NO_RUN)
            {
                capacity = run_capacity(count + 1);
                let next_end = checked_run_end(u64::from(start), capacity);
                self.grow(mem, next_end)?;
                self.len = next_end;
            }
            if count < capacity {
                for i in (pos..count).rev() {
                    let old = self.read(mem, u64::from(start) + u64::from(i));
                    self.write(mem, u64::from(start) + u64::from(i) + 1, &old);
                }
                self.write(mem, u64::from(start) + u64::from(pos), d);
                return Ok(start);
            }
        }
        let next = self.alloc_run(mem, count + 1)?;
        for i in 0..count {
            let old = self.read(mem, u64::from(start) + u64::from(i));
            let at = u64::from(next) + u64::from(i) + u64::from(i >= pos);
            self.write(mem, at, &old);
        }
        self.write(mem, u64::from(next) + u64::from(pos), d);
        Ok(next)
    }

    /// Starts a new bulk epoch without erasing or shrinking retained backing.
    pub(crate) fn clear_runs(&mut self) {
        self.len = 0;
        self.free_heads.clear();
    }

    pub(crate) fn verify_runs<M: Memory>(
        &self,
        mem: &M,
        runs: impl IntoIterator<Item = (u32, u32)>,
    ) {
        assert!(self.len <= self.cap, "descriptor extent exceeds capacity");
        assert!(
            self.len * 24 <= mem.size() * crate::memory::PAGE_SIZE,
            "unbacked descriptor arena"
        );
        let mut ranges = BTreeMap::new();
        for (start, count) in runs {
            if count == 0 {
                continue;
            }
            let start = u64::from(start);
            let end = checked_run_end(start, run_capacity(count));
            assert!(end <= self.len, "active run outside arena");
            assert!(
                ranges.insert(start, end).is_none(),
                "duplicate active descriptor run"
            );
        }
        for (class, &head) in self.free_heads.iter().enumerate() {
            assert!(class <= 16, "invalid descriptor size class");
            let mut current = head;
            while current != NO_RUN {
                let end = checked_run_end(current, 1 << class);
                assert!(end <= self.len, "free run outside arena");
                assert!(
                    ranges.insert(current, end).is_none(),
                    "shared or cyclic descriptor run"
                );
                let mut next = [0; 8];
                mem.read(current * 24, &mut next);
                current = u64::from_le_bytes(next);
            }
        }
        let mut next = 0;
        for (start, end) in ranges {
            assert_eq!(
                start, next,
                "descriptor runs overlap or leave unowned slots"
            );
            next = end;
        }
        assert_eq!(next, self.len, "unowned descriptor arena tail");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::VecMemory;

    #[test]
    fn run_class_and_address_boundaries() {
        assert_eq!(run_capacity(1), 1);
        assert_eq!(run_capacity(3), 4);
        assert_eq!(run_capacity(32_769), 65_536);
        assert_eq!(run_capacity(65_536), 65_536);
        for count in [0, 65_537] {
            assert!(std::panic::catch_unwind(|| run_capacity(count)).is_err());
        }
        // Numeric address guard only: no enormous graph/backing is fabricated.
        assert_eq!(checked_run_end(u64::from(u32::MAX), 1), 1u64 << 32);
        assert!(std::panic::catch_unwind(|| checked_run_end(1u64 << 32, 1)).is_err());
        assert!(std::panic::catch_unwind(|| checked_run_end(u64::MAX, 2)).is_err());
    }

    #[test]
    fn run_verifier_rejects_overlap_and_free_cycles() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let mut mem = VecMemory::new();
        let mut store = BucketStore::new(0);
        let a = store.alloc_run(&mut mem, 1).unwrap();
        let b = store.alloc_run(&mut mem, 2).unwrap();
        store.verify_runs(&mem, [(a, 1), (b, 2)]);
        assert!(
            catch_unwind(AssertUnwindSafe(
                || store.verify_runs(&mem, [(a, 2), (b, 2)])
            ))
            .is_err()
        );
        store.release_run(&mut mem, a, 1);
        store.verify_runs(&mem, [(b, 2)]);
        mem.write(u64::from(a) * 24, &u64::from(a).to_le_bytes());
        assert!(catch_unwind(AssertUnwindSafe(|| store.verify_runs(&mem, [(b, 2)]))).is_err());
        mem.write(u64::from(a) * 24, &NO_RUN.to_le_bytes());
        store.verify_runs(&mem, [(b, 2)]);
        assert_eq!(store.alloc_run(&mut mem, 1).unwrap(), a);
        store.verify_runs(&mem, [(a, 1), (b, 2)]);
        store.clear_runs();
        store.verify_runs(&mem, []);
    }

    #[test]
    fn descriptor_wire_roundtrip() {
        let d = BucketDescriptor {
            label: 7,
            degree: 10,
            stored: 12,
            log_head: 3,
            edge_start: 1234,
            tree_bucket: u32::MAX,
        };
        let mut buf = [0u8; 24];
        d.write_to(&mut buf);
        let r = BucketDescriptor::read_from(&buf);
        assert_eq!(r, d);
    }

    #[test]
    fn insert_shifts_tail() {
        let mut mem = VecMemory::new();
        let mut store = BucketStore::new(8);
        store.grow(&mut mem, 8).unwrap();
        store
            .push(
                &mut mem,
                &BucketDescriptor {
                    label: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        store
            .push(
                &mut mem,
                &BucketDescriptor {
                    label: 3,
                    ..Default::default()
                },
            )
            .unwrap();
        store
            .push(
                &mut mem,
                &BucketDescriptor {
                    label: 5,
                    ..Default::default()
                },
            )
            .unwrap();
        // Insert label 4 at position 2 (between 3 and 5).
        store.grow(&mut mem, store.len() + 1).unwrap();
        store.insert_at(
            &mut mem,
            2,
            &BucketDescriptor {
                label: 4,
                ..Default::default()
            },
        );
        assert_eq!(store.len(), 4);
        assert_eq!(store.read(&mem, 0).label, 1);
        assert_eq!(store.read(&mem, 1).label, 3);
        assert_eq!(store.read(&mem, 2).label, 4);
        assert_eq!(store.read(&mem, 3).label, 5);
    }

    #[test]
    fn remove_shifts_tail() {
        let mut mem = VecMemory::new();
        let mut store = BucketStore::new(8);
        store.grow(&mut mem, 8).unwrap();
        for l in [1u16, 3, 5] {
            store
                .push(
                    &mut mem,
                    &BucketDescriptor {
                        label: l,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        store.remove_at(&mut mem, 1);
        assert_eq!(store.len(), 2);
        assert_eq!(store.read(&mem, 0).label, 1);
        assert_eq!(store.read(&mem, 1).label, 5);
    }
}
