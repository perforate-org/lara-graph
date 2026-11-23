//! Stable free-span store: retired physical edge ranges available for reuse.
//!
//! Ported from `ic-stable-lara/src/lara/edge/free_span.rs` in simplified
//! form: allocation is best-fit over size-class bins; coalescing merges
//! adjacent spans in start order. Records live in `Memory` at a fixed
//! record stride; the hot path mirrors bin heads into a `BTreeMap` for
//! logarithmic best-fit and start-order neighbor lookup.

use crate::memory::{GrowFailed, Memory, safe_write};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

/// A reusable physical span: `[start_slot, start_slot + len)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FreeSpan {
    /// First slot of the free range.
    pub start: u64,
    /// Number of slots in the free range.
    pub len: u64,
}

/// Errors raised by allocation and release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreeSpanError {
    /// Zero-length span supplied.
    EmptySpan,
    /// `start + len` overflowed.
    SpanOverflow(FreeSpan),
    /// The released span overlaps an existing span.
    Overlap,
    /// The exact span was expected but not found.
    SpanNotFound(FreeSpan),
    /// Backend could not grow for a new record.
    OutOfMemory,
}

impl fmt::Display for FreeSpanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySpan => write!(f, "empty span"),
            Self::SpanOverflow(s) => write!(f, "span {s:?} overflows"),
            Self::Overlap => write!(f, "released span overlaps an existing span"),
            Self::SpanNotFound(s) => write!(f, "span {s:?} not found"),
            Self::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FreeSpanError {}

impl From<GrowFailed> for FreeSpanError {
    fn from(_: GrowFailed) -> Self {
        FreeSpanError::OutOfMemory
    }
}

/// Exact bins for lengths `1..=64`, then power-of-two buckets above
/// (mirrors the ic-stable-lara bin layout).
pub const BIN_COUNT: usize = 128;

/// Size-class bin for `len` (documentation of the size-class scheme; the
/// BTreeMap index orders `(len, start)` directly).
#[allow(dead_code)]
#[inline]
fn bin_index(len: u64) -> usize {
    if len == 0 {
        return 0;
    }
    if len <= 64 {
        (len - 1) as usize
    } else {
        let log = 63 - (len - 1).leading_zeros(); // floor_log2(len-1)
        (64 + log as usize).min(BIN_COUNT - 1)
    }
}

/// Record layout per span in `Memory`: `(start: u64, len: u64)`.
const RECORD_STRIDE: u64 = 16;
const RECORDS_OFFSET: u64 = 4096;
const COUNT_OFFSET: u64 = 0;

/// Free-span allocator over a `Memory` region.
///
/// `Memory` holds the durable record array (`(start, len)` per span slot);
/// a `BTreeMap<(len, start)>` index and a `BTreeMap<start, len>` start-order
/// map are rebuilt from it at open time. The maps are benchmark-era state —
/// the durable truth is the record array.
pub struct FreeSpanStore<M: Memory> {
    memory: M,
    /// Active span count.
    count: u32,
    /// Total free slots across all spans.
    free_slots: u64,
    /// Highest minted record index (records are tombstoned on free).
    records_minted: u32,
    /// Free record-slot stack (indices of tombstoned records).
    free_records: Vec<u32>,
    /// `(len, start) -> ()` best-fit index.
    by_size: BTreeMap<(u64, u64), ()>,
    /// `start -> len` start-order map for coalescing.
    by_start: BTreeMap<u64, u64>,
    /// `start -> record index` map for O(1) tombstoning (benchmark-era
    /// in-heap index; the durable truth stays in the record array).
    /// `alloc` has no `HashMap`, so this is a `BTreeMap` (logarithmic).
    record_by_start: BTreeMap<u64, u32>,
    /// Largest single free span.
    largest: u64,
}

impl<M: Memory> FreeSpanStore<M> {
    /// Creates a fresh store over `memory`.
    pub fn new(memory: M) -> Result<Self, GrowFailed> {
        safe_write(&memory, RECORDS_OFFSET - 1, &[0u8])?;
        Ok(Self {
            memory,
            count: 0,
            free_slots: 0,
            records_minted: 0,
            free_records: Vec::new(),
            by_size: BTreeMap::new(),
            by_start: BTreeMap::new(),
            record_by_start: BTreeMap::new(),
            largest: 0,
        })
    }

    /// Reopens a store (rebuilds indexes from the record array).
    pub fn init(memory: M) -> Result<Self, GrowFailed> {
        let mut buf = [0u8; 4];
        memory.read(COUNT_OFFSET, &mut buf);
        let minted = u32::from_le_bytes(buf);
        let mut store = Self::new(memory)?;
        store.records_minted = minted;
        for idx in 0..minted as u64 {
            let mut rec = [0u8; 8];
            store
                .memory
                .read(RECORDS_OFFSET + idx * RECORD_STRIDE, &mut rec);
            let start = u64::from_le_bytes(rec);
            store
                .memory
                .read(RECORDS_OFFSET + idx * RECORD_STRIDE + 8, &mut rec);
            let len = u64::from_le_bytes(rec);
            if len > 0 {
                store.insert_index(FreeSpan { start, len }, Some(idx as u32));
                store.count += 1;
                store.free_slots += len;
            }
        }
        Ok(store)
    }

    fn insert_index(&mut self, span: FreeSpan, record_idx: Option<u32>) {
        self.by_size.insert((span.len, span.start), ());
        self.by_start.insert(span.start, span.len);
        if let Some(idx) = record_idx {
            self.record_by_start.insert(span.start, idx);
        }
        self.largest = self.largest.max(span.len);
    }

    fn remove_index(&mut self, span: FreeSpan) -> Option<u32> {
        self.by_size.remove(&(span.len, span.start));
        self.by_start.remove(&span.start);
        let record_idx = self.record_by_start.remove(&span.start);
        if self.largest == span.len {
            self.largest = self.by_size.keys().next_back().map_or(0, |k| k.0);
        }
        record_idx
    }

    /// Writes `None` over the record slot and recycles the index.
    fn retire_record(&mut self, record_idx: Option<u32>) {
        if let Some(idx) = record_idx {
            self.write_record(idx, None).expect("retire_record write");
            self.free_records.push(idx);
        }
    }

    fn write_record(&mut self, idx: u32, span: Option<FreeSpan>) -> Result<(), GrowFailed> {
        let off = RECORDS_OFFSET + idx as u64 * RECORD_STRIDE;
        let zero = [0u8; 8];
        match span {
            Some(s) => {
                safe_write(&self.memory, off, &s.start.to_le_bytes())?;
                safe_write(&self.memory, off + 8, &s.len.to_le_bytes())?;
            }
            None => {
                safe_write(&self.memory, off, &zero)?;
                safe_write(&self.memory, off + 8, &zero)?;
            }
        }
        // Persist minted count.
        safe_write(
            &self.memory,
            COUNT_OFFSET,
            &self.records_minted.to_le_bytes(),
        )?;
        Ok(())
    }

    /// Discards obsolete free ranges before a full slab rebuild.
    pub(crate) fn clear(&mut self) {
        self.memory.write(COUNT_OFFSET, &0u32.to_le_bytes());
        self.count = 0;
        self.free_slots = 0;
        self.records_minted = 0;
        self.free_records.clear();
        self.by_size.clear();
        self.by_start.clear();
        self.record_by_start.clear();
        self.largest = 0;
    }

    /// Returns allocator statistics: `(span_count, free_slots, largest)`.
    pub fn stats(&self) -> (u32, u64, u64) {
        (self.count, self.free_slots, self.largest)
    }

    /// Lists all free spans in start order (tests/diagnostics).
    pub fn spans(&self) -> Vec<FreeSpan> {
        self.by_start
            .iter()
            .map(|(&start, &len)| FreeSpan { start, len })
            .collect()
    }

    /// Best-fit allocation: smallest span with `len >= want`, splitting the
    /// remainder back into the store. Returns the allocated start slot.
    pub fn alloc(&mut self, want: u64) -> Option<u64> {
        if want == 0 {
            return None;
        }
        // Best-fit: the first `(len, start)` key at or above `want` is the
        // smallest span that fits (BTreeMap orders by len, then start).
        let candidate = self
            .by_size
            .range((want, 0)..)
            .next()
            .map(|(&(len, start), _)| FreeSpan { start, len });
        let span = candidate?;
        let consumed_record = self.remove_index(span);
        self.count -= 1;
        self.free_slots -= span.len;

        // Retire the durable record for the consumed span so reopen does
        // not resurrect it as a live free span.
        self.retire_record(consumed_record);

        let start = span.start;
        let rest_len = span.len - want;
        if rest_len > 0 {
            let rest = FreeSpan {
                start: start + want,
                len: rest_len,
            };
            self.release_inner(rest).expect("rest split release");
        }
        Some(start)
    }

    /// Returns a retired range to the store, coalescing adjacent spans.
    pub fn release(&mut self, start: u64, len: u64) -> Result<(), FreeSpanError> {
        self.release_inner(FreeSpan { start, len })
    }

    fn release_inner(&mut self, span: FreeSpan) -> Result<(), FreeSpanError> {
        if span.len == 0 {
            return Err(FreeSpanError::EmptySpan);
        }
        let end = span
            .start
            .checked_add(span.len)
            .ok_or(FreeSpanError::SpanOverflow(span))?;

        // Coalesce with predecessor and successor in start order.
        let prev = self
            .by_start
            .range(..span.start)
            .next_back()
            .map(|(&s, &l)| FreeSpan { start: s, len: l });
        let next = self
            .by_start
            .range(span.start..)
            .next()
            .map(|(&s, &l)| FreeSpan { start: s, len: l });

        if let Some(p) = prev {
            if p.start + p.len > span.start {
                return Err(FreeSpanError::Overlap);
            }
        }
        if let Some(n) = next {
            if end > n.start {
                return Err(FreeSpanError::Overlap);
            }
        }

        // Compute the merged span first, then adjust counters exactly once
        // for each index-map change: remove each absorbed neighbor, add the
        // merged span. Net count delta: +1 - (number of absorbed neighbors).
        let mut merged = span;
        let mut to_tombstone: Vec<FreeSpan> = Vec::new();
        let prev_touches = prev.is_some_and(|p| p.start + p.len == span.start);
        let next_touches = next.is_some_and(|n| end == n.start);
        if prev_touches {
            let p = prev.unwrap();
            to_tombstone.push(p);
            merged = FreeSpan {
                start: p.start,
                len: p.len + span.len,
            };
        }
        if next_touches {
            let n = next.unwrap();
            to_tombstone.push(n);
            merged.len += n.len;
        }
        for t in &to_tombstone {
            let record_idx = self.remove_index(*t);
            self.count -= 1;
            self.free_slots -= t.len;
            self.retire_record(record_idx);
        }
        let merged_idx = self.persist_record(merged)?;
        self.insert_index(merged, Some(merged_idx));
        self.count += 1;
        self.free_slots += merged.len;
        Ok(())
    }

    /// Persists `span` as a new (or reused) record; returns the record index.
    fn persist_record(&mut self, span: FreeSpan) -> Result<u32, GrowFailed> {
        let idx = match self.free_records.pop() {
            Some(idx) => idx,
            None => {
                self.records_minted += 1;
                self.records_minted - 1
            }
        };
        self.write_record(idx, Some(span))?;
        Ok(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::VecMemory;

    #[test]
    fn alloc_split_and_stats() {
        let mut store = FreeSpanStore::new(VecMemory::new()).unwrap();
        store.release(0, 100).unwrap();
        let (count, slots, largest) = store.stats();
        assert_eq!((count, slots, largest), (1, 100, 100));

        let a = store.alloc(30).unwrap();
        assert_eq!(a, 0);
        let (count, slots, largest) = store.stats();
        assert_eq!((count, slots, largest), (1, 70, 70));
        assert_eq!(store.spans(), vec![FreeSpan { start: 30, len: 70 }]);
    }

    #[test]
    fn coalesce_prev_next_and_both() {
        let mut store = FreeSpanStore::new(VecMemory::new()).unwrap();
        store.release(10, 10).unwrap();
        store.release(40, 10).unwrap();
        assert_eq!(store.stats().0, 2);
        // Touch only the predecessor: 0..10 + 10..20 -> 0..20.
        store.release(0, 10).unwrap();
        assert_eq!(store.stats().0, 2);
        assert_eq!(store.stats().1, 30);
        assert_eq!(
            store.spans(),
            vec![
                FreeSpan { start: 0, len: 20 },
                FreeSpan { start: 40, len: 10 }
            ]
        );
        // Touch both sides: 30..40 coalesces with next 40..50 → {0..20, 30..50}.
        store.release(30, 10).unwrap();
        assert_eq!(store.stats().0, 2);
        assert_eq!(store.stats().1, 40);
        assert_eq!(
            store.spans(),
            vec![
                FreeSpan { start: 0, len: 20 },
                FreeSpan { start: 30, len: 20 }
            ]
        );
        // Then 20..30 touches both 0..20 (prev) and 30..50 (next) → 0..50.
        store.release(20, 10).unwrap();
        assert_eq!(store.stats().0, 1);
        assert_eq!(store.stats().1, 50);
        assert_eq!(store.spans(), vec![FreeSpan { start: 0, len: 50 }]);
    }

    #[test]
    fn overlap_rejected() {
        let mut store = FreeSpanStore::new(VecMemory::new()).unwrap();
        store.release(0, 10).unwrap();
        assert_eq!(store.release(5, 10), Err(FreeSpanError::Overlap));
        assert_eq!(store.release(9, 2), Err(FreeSpanError::Overlap));
    }

    #[test]
    fn best_fit_prefers_smallest_sufficient_span() {
        let mut store = FreeSpanStore::new(VecMemory::new()).unwrap();
        store.release(0, 100).unwrap();
        store.release(200, 10).unwrap();
        let a = store.alloc(8).unwrap();
        assert_eq!(a, 200); // 10-slot span fits exactly; 100 is bigger.
        // Remainder (208..210) returns to the store.
        assert_eq!(
            store.spans(),
            vec![
                FreeSpan { start: 0, len: 100 },
                FreeSpan { start: 208, len: 2 }
            ]
        );
    }

    #[test]
    fn init_rebuilds_indexes_from_records() {
        let memory = VecMemory::new();
        let mut store = FreeSpanStore::new(memory.clone()).unwrap();
        store.release(0, 50).unwrap();
        store.release(100, 25).unwrap();
        let a = store.alloc(10).unwrap();
        assert_eq!(a, 100);
        drop(store);

        let reopened = FreeSpanStore::init(memory).unwrap();
        assert_eq!(reopened.spans().len(), 2);
        assert_eq!(reopened.stats().1, 65);
    }
}
