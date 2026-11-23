//! Segment span metadata: explicit `physical_start` per PMA leaf.
//!
//! Ported from `ic-stable-lara/src/lara/edge/span_meta.rs` in simplified
//! form. Localized relocation may place a leaf segment's slab span anywhere;
//! this store records where each leaf's span starts.

use crate::memory::{GrowFailed, Memory, safe_write};
use alloc::vec::Vec;

/// Sentinel meaning "this leaf has no pinned slab span yet". Slot 0 is a
/// legal start, so 0 never means unassigned.
pub const SPAN_UNASSIGNED: u64 = u64::MAX;

/// Placement metadata for one leaf segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentSpanMeta {
    /// Physical slab start slot of this leaf's span.
    pub physical_start: u64,
}

impl Default for SegmentSpanMeta {
    fn default() -> Self {
        Self {
            physical_start: SPAN_UNASSIGNED,
        }
    }
}

const DATA_OFFSET: u64 = 64;
const COUNT_OFFSET: u64 = 0;
const ENTRY_SIZE: u64 = 8;

/// Fixed-stride store of one [`SegmentSpanMeta`] per leaf segment.
pub struct SegmentSpanMetaStore<M: Memory> {
    memory: M,
    len: u64,
}

impl<M: Memory> SegmentSpanMetaStore<M> {
    /// Creates a fresh store over `memory`.
    pub fn new(memory: M) -> Result<Self, GrowFailed> {
        safe_write(&memory, DATA_OFFSET - 1, &[0u8])?;
        Ok(Self { memory, len: 0 })
    }

    /// Number of leaves tracked.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Returns `true` when no leaves are tracked.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Grows the store by `additional` leaves (initialized to `SPAN_UNASSIGNED`).
    pub fn grow_to(&mut self, new_len: u64) -> Result<(), GrowFailed> {
        debug_assert!(new_len >= self.len);
        let need = DATA_OFFSET + new_len * ENTRY_SIZE;
        let cur = DATA_OFFSET + self.len * ENTRY_SIZE;
        if need > cur {
            let unassigned = SPAN_UNASSIGNED.to_le_bytes();
            let entries = (need - cur) / ENTRY_SIZE;
            let mut buf = Vec::with_capacity((need - cur) as usize);
            for _ in 0..entries {
                buf.extend_from_slice(&unassigned);
            }
            safe_write(&self.memory, cur, &buf)?;
        }
        self.len = new_len;
        safe_write(&self.memory, COUNT_OFFSET, &self.len.to_le_bytes())?;
        Ok(())
    }

    /// Reads one leaf's metadata. Panics past `len`.
    pub fn get(&self, leaf: u64) -> SegmentSpanMeta {
        assert!(leaf < self.len, "span_meta leaf out of range");
        let mut buf = [0u8; 8];
        self.memory.read(DATA_OFFSET + leaf * ENTRY_SIZE, &mut buf);
        let physical_start = u64::from_le_bytes(buf);
        SegmentMetaWire { physical_start }.into_meta()
    }

    /// Writes one leaf's metadata. Panics past `len`.
    pub fn set(&mut self, leaf: u64, meta: SegmentSpanMeta) -> Result<(), GrowFailed> {
        assert!(leaf < self.len, "span_meta leaf out of range");
        let bytes = meta.physical_start.to_le_bytes();
        safe_write(&self.memory, DATA_OFFSET + leaf * ENTRY_SIZE, &bytes)?;
        Ok(())
    }
}

/// Wire wrapper applying the unassigned-sentinel decode.
struct SegmentMetaWire {
    physical_start: u64,
}

impl SegmentMetaWire {
    fn into_meta(self) -> SegmentSpanMeta {
        SegmentSpanMeta {
            physical_start: self.physical_start,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::VecMemory;

    #[test]
    fn fresh_leaves_are_unassigned() {
        let store = SegmentSpanMetaStore::new(VecMemory::new()).unwrap();
        let mut store = store;
        store.grow_to(4).unwrap();
        for leaf in 0..4u64 {
            assert_eq!(store.get(leaf).physical_start, SPAN_UNASSIGNED);
        }
    }

    #[test]
    fn set_get_roundtrip_and_zero_start() {
        let mut store = SegmentSpanMetaStore::new(VecMemory::new()).unwrap();
        store.grow_to(8).unwrap();
        store.set(0, SegmentSpanMeta { physical_start: 0 }).unwrap();
        store
            .set(
                7,
                SegmentSpanMeta {
                    physical_start: 4096,
                },
            )
            .unwrap();
        assert_eq!(store.get(0).physical_start, 0); // 0 is legal, not "unassigned"
        assert_eq!(store.get(7).physical_start, 4096);
    }
}
