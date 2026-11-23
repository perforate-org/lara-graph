//! Byte-addressable backend abstraction for generalized LARA.
//!
//! Mirrors the `ic_stable_structures::Memory` shape (all methods `&self`,
//! interior mutability expected) so the graph code is canister-agnostic.
//! [`VecMemory`] is the test/dev backend.
//!
//! The core ([`Memory`], [`GrowFailed`], [`PAGE_SIZE`], [`safe_write`]) is
//! `no_std` + `alloc` compatible. [`VecMemory`] (a heap-backed backend) and
//! the `std::error::Error` impls are gated behind the default `std` feature.

use core::fmt;

/// Error returned when a backend cannot grow far enough.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowFailed {
    /// Current size in pages.
    pub current_pages: u64,
    /// Requested additional pages.
    pub delta_pages: u64,
}

impl fmt::Display for GrowFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to grow memory: current={} pages, delta={} pages",
            self.current_pages, self.delta_pages
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GrowFailed {}

/// Page size used by [`Memory::grow`].
pub const PAGE_SIZE: u64 = 65536;

/// Byte-addressable memory backend (IC-agnostic LARA contract).
pub trait Memory {
    /// Reads bytes known to lie within the current size, skipping the
    /// trailing zero-fill of [`Memory::read`]. The caller guarantees
    /// `[offset, offset + buf.len())` is mapped; backends must still not
    /// read out of bounds (clamp the copy, leave the tail untouched).
    /// Scan fast path only — all other callers keep using [`Memory::read`]
    /// with its zero-fill contract.
    #[inline]
    fn read_mapped(&self, offset: u64, buf: &mut [u8]) {
        self.read(offset, buf)
    }

    /// Reads bytes at `offset` into `buf`.
    fn read(&self, offset: u64, buf: &mut [u8]);

    /// Writes bytes at `offset`.
    fn write(&self, offset: u64, buf: &[u8]);

    /// Current size in pages.
    fn size(&self) -> u64;

    /// Grows by `pages` pages; returns the new page count or an error.
    fn grow(&self, pages: u64) -> Result<u64, GrowFailed>;

    /// Persistent-memory fence (no-op for DRAM).
    #[inline]
    fn fence(&self) {}
}

/// Writes `buf` at `offset`, growing the backend when needed.
///
/// This is core (not gated behind `std`): it only needs the [`Memory`]
/// trait and is used by the free-span store, span metadata, block store,
/// and graph slab paths.
pub fn safe_write<M: Memory>(memory: &M, offset: u64, buf: &[u8]) -> Result<(), GrowFailed> {
    let last = offset
        .checked_add(buf.len() as u64)
        .expect("address space overflow");
    let size_bytes = memory.size() * PAGE_SIZE;
    if size_bytes < last {
        let pages = last.div_ceil(PAGE_SIZE);
        memory.grow(pages - memory.size())?;
    }
    memory.write(offset, buf);
    Ok(())
}

/// Heap-backed backend for tests and benchmarks (requires the `std`
/// feature).
#[cfg(feature = "std")]
mod vec_memory {
    use super::{GrowFailed, Memory, PAGE_SIZE};
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::vec::Vec;

    /// Heap-backed backend for tests and benchmarks.
    #[derive(Default, Clone)]
    pub struct VecMemory {
        inner: Rc<RefCell<Vec<u8>>>,
    }

    impl VecMemory {
        /// Creates an empty store.
        pub fn new() -> Self {
            Self::default()
        }

        /// Requested capacity of the shared backing Vec in bytes, including
        /// spare capacity beyond the mapped length. Excludes the Rc/RefCell
        /// allocation, allocator overhead, and resident-set accounting.
        /// Cloned handles describe the same allocation; count it only once.
        pub fn capacity_bytes(&self) -> usize {
            self.inner.borrow().capacity()
        }
    }

    impl Memory for VecMemory {
        fn read(&self, offset: u64, buf: &mut [u8]) {
            let inner = self.inner.borrow();
            let start = std::cmp::min(offset as usize, inner.len());
            let end = std::cmp::min(start + buf.len(), inner.len());
            let n = end - start;
            // Branch-only fixed-width copies for the eligible small live prefixes.
            // Every width stays inside [start, end); nothing is rounded up.
            if n > 0 {
                if n > 32 {
                    buf[..n].copy_from_slice(&inner[start..end]);
                } else if n >= 16 {
                    buf[..16].copy_from_slice(&inner[start..start + 16]);
                    if n > 16 {
                        buf[n - 16..n].copy_from_slice(&inner[end - 16..end]);
                    }
                } else if n >= 8 {
                    buf[..8].copy_from_slice(&inner[start..start + 8]);
                    if n > 8 {
                        buf[n - 8..n].copy_from_slice(&inner[end - 8..end]);
                    }
                } else if n >= 4 {
                    buf[..4].copy_from_slice(&inner[start..start + 4]);
                    if n > 4 {
                        buf[n - 4..n].copy_from_slice(&inner[end - 4..end]);
                    }
                } else {
                    // 1..=3: bounded byte accesses, no loop.
                    buf[0] = inner[start];
                    if n > 1 {
                        buf[1] = inner[start + 1];
                    }
                    if n > 2 {
                        buf[2] = inner[start + 2];
                    }
                }
            }
            for b in buf[n..].iter_mut() {
                *b = 0;
            }
        }

        fn read_mapped(&self, offset: u64, buf: &mut [u8]) {
            let inner = self.inner.borrow();
            let start = (offset as usize).min(inner.len());
            let end = (start + buf.len()).min(inner.len());
            let n = end - start;
            debug_assert_eq!(n, buf.len(), "read_mapped range must be mapped");
            buf[..n].copy_from_slice(&inner[start..end]);
        }

        fn write(&self, offset: u64, buf: &[u8]) {
            let mut inner = self.inner.borrow_mut();
            let start = offset as usize;
            let end = start + buf.len();
            if inner.len() < end {
                inner.resize(end, 0);
            }
            inner[start..end].copy_from_slice(buf);
        }

        fn size(&self) -> u64 {
            self.inner.borrow().len().div_ceil(PAGE_SIZE as usize) as u64
        }

        fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
            let mut inner = self.inner.borrow_mut();
            let current = inner.len().div_ceil(PAGE_SIZE as usize) as u64;
            let failed = GrowFailed {
                current_pages: current,
                delta_pages: pages,
            };
            let new_pages = current.checked_add(pages).ok_or(failed)?;
            let target = new_pages
                .checked_mul(PAGE_SIZE)
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or(failed)?;
            let additional = target - inner.len();
            inner.try_reserve(additional).map_err(|_| failed)?;
            inner.resize(target, 0);
            Ok(new_pages)
        }
    }
}

#[cfg(feature = "std")]
pub use vec_memory::VecMemory;

#[cfg(all(test, feature = "std"))]
mod capacity_tests {
    use super::*;

    #[test]
    fn capacity_bytes_tracks_shared_backing_not_handles() {
        let m = VecMemory::new();
        let other = m.clone();
        assert_eq!(m.capacity_bytes(), 0);
        for pages in 1..=3 {
            m.grow(1).unwrap();
            assert_eq!(m.size(), pages);
            assert!(m.capacity_bytes() as u64 >= pages * PAGE_SIZE);
            assert_eq!(m.capacity_bytes(), other.capacity_bytes());
        }
    }
}

/// Shared persist counters behind [`PersistCounterMemory`].
///
/// One `Rc`-shared instance spans all regions of a graph so the total
/// persisted byte count is a single number. Uses `Cell` (not atomics):
/// the [`Memory`] contract is `&self`-based single-threaded interior
/// mutability, matching the rest of the crate.
#[derive(Default, Debug)]
pub struct PersistCounters {
    bytes: core::cell::Cell<u64>,
    ops: core::cell::Cell<u64>,
    fences: core::cell::Cell<u64>,
}

impl PersistCounters {
    /// Total persisted bytes written through this counter set.
    pub fn bytes(&self) -> u64 {
        self.bytes.get()
    }

    /// Total [`Memory::write`] calls through this counter set.
    pub fn ops(&self) -> u64 {
        self.ops.get()
    }

    /// Total [`Memory::fence`] calls through this counter set.
    pub fn fences(&self) -> u64 {
        self.fences.get()
    }

    /// Zeroes all counters in this set.
    pub fn reset(&self) {
        self.bytes.set(0);
        self.ops.set(0);
        self.fences.set(0);
    }
}

/// PMEM-cost-model emulator: a [`Memory`] wrapper that counts persisted
/// writes without changing behavior.
///
/// Every [`Memory::write`] adds `buf.len()` to the byte counter and one
/// to the op counter; every [`Memory::fence`] adds one fence op. Reads,
/// `size`, and `grow` (allocation, not persistence) are uncounted and
/// delegated verbatim. DRAM time makes rebalances look free — this
/// instrument measures the write volume rebalances actually cost, which
/// is the ranking-relevant axis under a persistent-memory cost model.
///
/// Share one `Rc<PersistCounters>` across all regions of a memory bundle
/// (see [`PersistCounterMemory::with_counters`]) so totals aggregate.
#[derive(Clone)]
pub struct PersistCounterMemory<M: Memory> {
    inner: M,
    counters: alloc::rc::Rc<PersistCounters>,
}

impl<M: Memory> PersistCounterMemory<M> {
    /// Wraps `inner` with a fresh private counter set.
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            counters: alloc::rc::Rc::new(PersistCounters::default()),
        }
    }

    /// Wraps `inner`, aggregating into a shared counter set. Clone the
    /// `Rc` (not the whole wrapper) for each region of a bundle.
    pub fn with_counters(inner: M, counters: &alloc::rc::Rc<PersistCounters>) -> Self {
        Self {
            inner,
            counters: alloc::rc::Rc::clone(counters),
        }
    }

    /// Total persisted bytes written through this counter set.
    pub fn persisted_bytes(&self) -> u64 {
        self.counters.bytes.get()
    }

    /// Total [`Memory::write`] calls through this counter set.
    pub fn write_op_count(&self) -> u64 {
        self.counters.ops.get()
    }

    /// Total [`Memory::fence`] calls through this counter set.
    pub fn fence_op_count(&self) -> u64 {
        self.counters.fences.get()
    }

    /// Zeroes all counters in this counter set.
    pub fn reset_counters(&self) {
        self.counters.bytes.set(0);
        self.counters.ops.set(0);
        self.counters.fences.set(0);
    }
}

impl<M: Memory> Memory for PersistCounterMemory<M> {
    fn read(&self, offset: u64, buf: &mut [u8]) {
        self.inner.read(offset, buf)
    }

    fn read_mapped(&self, offset: u64, buf: &mut [u8]) {
        self.inner.read_mapped(offset, buf)
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        self.counters
            .bytes
            .set(self.counters.bytes.get() + buf.len() as u64);
        self.counters.ops.set(self.counters.ops.get() + 1);
        self.inner.write(offset, buf)
    }

    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        self.inner.grow(pages)
    }

    fn fence(&self) {
        self.counters.fences.set(self.counters.fences.get() + 1);
        self.inner.fence()
    }
}
