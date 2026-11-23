//! Persistent-memory abstraction for the DGAP Rust port.
//!
//! The original DGAP implementation uses PMDK (`libpmemobj`) for crash
//! consistency. This crate abstracts the backing store behind [`Memory`]
//! so the same graph algorithms can run over DRAM ([`VecMemory`]) in tests
//! and benchmarks.

use std::fmt;

/// Error returned when a backing store cannot grow far enough.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowFailed {
    /// Current store size in bytes.
    pub current_size: u64,
    /// Number of additional pages requested.
    pub delta_pages: u64,
}

impl fmt::Display for GrowFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to grow memory: current_size={} bytes, delta={} pages",
            self.current_size, self.delta_pages
        )
    }
}

impl std::error::Error for GrowFailed {}

/// Byte-addressable persistent-memory backend.
///
/// Mirrors the `ic_stable_structures::Memory` shape used by LARA so a shared
/// backend can serve both crates in the benchmark harness, but keeps the
/// mutating methods `&mut self` (DGAP rebuilds are single-threaded in this
/// port).
pub trait Memory {
    /// Reads bytes at `offset` into `buf`.
    fn read(&self, offset: u64, buf: &mut [u8]);

    /// Writes bytes at `offset`.
    fn write(&mut self, offset: u64, buf: &[u8]);

    /// Current store size in bytes.
    fn size(&self) -> u64;

    /// Grows the store by `additional_pages` pages. Returns the new size in
    /// bytes, or an error when growth is impossible.
    fn grow(&mut self, additional_pages: u64) -> Result<u64, GrowFailed>;

    /// Persistent-memory fence. No-op for DRAM backends.
    #[inline]
    fn fence(&self) {}
}

/// Page size used by [`Memory::grow`] (matches WASM / stable-memory pages).
pub const PAGE_SIZE: u64 = 65536;

/// Writes `buf` at `offset`, growing the store when needed.
pub fn safe_write<M: Memory>(memory: &mut M, offset: u64, buf: &[u8]) -> Result<(), GrowFailed> {
    let last_byte = offset
        .checked_add(buf.len() as u64)
        .expect("address space overflow");
    if memory.size() < last_byte {
        let diff_bytes = last_byte - memory.size();
        let diff_pages = diff_bytes.div_ceil(PAGE_SIZE);
        memory.grow(diff_pages)?;
    }
    memory.write(offset, buf);
    Ok(())
}

/// Heap-backed memory for tests and development. Mirrors
/// `ic_stable_structures::VectorMemory` (which stores `Rc<RefCell<Vec<u8>>>`
/// and needs `&self` mutation); this variant takes `&mut self` per the
/// [`Memory`] contract above.
#[derive(Default, Clone)]
pub struct VecMemory {
    inner: std::rc::Rc<std::cell::RefCell<Vec<u8>>>,
}

impl VecMemory {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Memory for VecMemory {
    fn read(&self, offset: u64, buf: &mut [u8]) {
        let inner = self.inner.borrow();
        let start = std::cmp::min(offset as usize, inner.len());
        let end = std::cmp::min(start + buf.len(), inner.len());
        let len = end - start;
        buf[..len].copy_from_slice(&inner[start..end]);
        // Zero-fill the unread tail (mirrors fresh stable pages).
        for b in buf[len..].iter_mut() {
            *b = 0;
        }
    }

    fn write(&mut self, offset: u64, buf: &[u8]) {
        let mut inner = self.inner.borrow_mut();
        let start = offset as usize;
        let end = start + buf.len();
        if inner.len() < end {
            inner.resize(end, 0);
        }
        inner[start..end].copy_from_slice(buf);
    }

    fn size(&self) -> u64 {
        self.inner.borrow().len() as u64
    }

    fn grow(&mut self, additional_pages: u64) -> Result<u64, GrowFailed> {
        let mut inner = self.inner.borrow_mut();
        let new_len = inner.len() + (additional_pages * PAGE_SIZE) as usize;
        inner.resize(new_len, 0);
        Ok(inner.len() as u64)
    }
}
