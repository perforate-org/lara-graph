//! The memory seam: the one place bytes enter or leave the store.

use crate::store::units::{ByteOffset, Overflow};
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;
use core::mem::MaybeUninit;

/// Bytes per page; [`Memory::grow`] counts in pages.
pub const PAGE_SIZE: u64 = 65_536;

/// A growable, byte-addressed backend.
///
/// The shape mirrors `ic_stable_structures::Memory` (shared `&self`, interior mutability,
/// page-granular growth); with the `ic-stable-structures` feature, every memory of that crate is
/// one. All graph state lives
/// behind this trait, which is what lets a graph be reopened.
pub trait Memory {
    /// Current size in pages.
    fn size(&self) -> u64;

    /// Grows by `pages` pages and returns the previous size in pages.
    ///
    /// # Errors
    ///
    /// [`GrowFailed`] when the backend cannot provide the pages; the size is then unchanged.
    fn grow(&self, pages: u64) -> Result<u64, GrowFailed>;

    /// Copies `buf.len()` bytes starting at `offset` into `buf`. The range must lie within the
    /// current size.
    fn read(&self, offset: u64, buf: &mut [u8]);

    /// [`Self::read`] into a buffer that need not be initialized, returning it filled. The
    /// default zeroes `buf` first; the IC's stable memory copies into it directly, skipping that,
    /// which was most of the cost of reading a small bucket's slots.
    fn read_uninit<'b>(&self, offset: u64, buf: &'b mut [MaybeUninit<u8>]) -> &'b mut [u8] {
        let buf = zeroed(buf);
        self.read(offset, buf);
        buf
    }

    /// Copies `buf` to `offset`. The range must lie within the current size.
    fn write(&self, offset: u64, buf: &[u8]);
}

/// Every ic-stable-structures memory is a lara-graph memory: `DefaultMemoryImpl`, a
/// `VirtualMemory` from a `MemoryManager`, a `RestrictedMemory`, a `VectorMemory`. Its `grow`
/// reports failure as −1, which becomes [`GrowFailed`].
#[cfg(feature = "ic-stable-structures")]
impl<M: ic_stable_structures::Memory> Memory for M {
    fn size(&self) -> u64 {
        ic_stable_structures::Memory::size(self)
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        let previous = ic_stable_structures::Memory::grow(self, pages);
        u64::try_from(previous)
            .map_err(|_| GrowFailed::new(ic_stable_structures::Memory::size(self), pages))
    }

    fn read(&self, offset: u64, buf: &mut [u8]) {
        ic_stable_structures::Memory::read(self, offset, buf);
    }

    #[expect(
        unsafe_code,
        reason = "a read into uninitialized bytes, as the backend offers it"
    )]
    fn read_uninit<'b>(&self, offset: u64, buf: &'b mut [MaybeUninit<u8>]) -> &'b mut [u8] {
        // SAFETY: `buf` is valid for writes of `buf.len()` bytes and is not the memory's own
        // storage, as `read_unsafe` requires. Its contract is to copy `count` bytes into `dst`
        // (the default zeroes them first; the IC's stable memory writes them with `stable64_read`;
        // a virtual memory forwards each segment), else to trap, so on return every byte is
        // initialized.
        unsafe {
            ic_stable_structures::Memory::read_unsafe(
                self,
                offset,
                buf.as_mut_ptr().cast::<u8>(),
                buf.len(),
            );
            assume_init(buf)
        }
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        ic_stable_structures::Memory::write(self, offset, buf);
    }
}

/// A backend could not grow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowFailed {
    current_pages: u64,
    requested_pages: u64,
}

impl GrowFailed {
    /// Reports a failed request for `requested_pages` more pages on a backend of `current_pages`.
    #[must_use]
    pub const fn new(current_pages: u64, requested_pages: u64) -> Self {
        Self {
            current_pages,
            requested_pages,
        }
    }

    /// Size of the backend, in pages, when growth failed.
    #[must_use]
    pub const fn current_pages(&self) -> u64 {
        self.current_pages
    }

    /// Pages the failed request asked for.
    #[must_use]
    pub const fn requested_pages(&self) -> u64 {
        self.requested_pages
    }
}

impl fmt::Display for GrowFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "memory could not grow by {} pages from {} pages",
            self.requested_pages, self.current_pages
        )
    }
}

impl core::error::Error for GrowFailed {}

/// A heap-backed [`Memory`] for tests, benchmarks and in-process use. Clones share the bytes, as
/// ic-stable-structures' `VectorMemory` does, so a graph can be reopened from a clone.
#[derive(Clone, Debug, Default)]
pub struct VecMemory(Rc<Heap>);

#[derive(Debug, Default)]
struct Heap {
    bytes: RefCell<Vec<u8>>,
    /// Pages past which growth fails (tests of out-of-memory paths).
    #[cfg(test)]
    cap: core::cell::Cell<Option<u64>>,
}

impl VecMemory {
    /// An empty memory of zero pages.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Makes growth past `pages` fail.
    #[cfg(test)]
    pub(crate) fn cap(&self, pages: u64) {
        self.0.cap.set(Some(pages));
    }

    #[cfg(test)]
    fn allows(&self, total: u64) -> bool {
        self.0.cap.get().is_none_or(|cap| total <= cap)
    }

    #[cfg(not(test))]
    #[expect(clippy::unused_self, reason = "only tests cap growth")]
    const fn allows(&self, _total: u64) -> bool {
        true
    }
}

impl Memory for VecMemory {
    fn size(&self) -> u64 {
        let len = u64::try_from(self.0.bytes.borrow().len()).unwrap_or(u64::MAX);
        len / PAGE_SIZE
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        let previous = self.size();
        let failed = GrowFailed::new(previous, pages);
        let new_len = previous
            .checked_add(pages)
            .filter(|&total| self.allows(total))
            .and_then(|total| total.checked_mul(PAGE_SIZE))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(failed)?;
        self.0.bytes.borrow_mut().resize(new_len, 0);
        Ok(previous)
    }

    fn read(&self, offset: u64, buf: &mut [u8]) {
        let bytes = self.0.bytes.borrow();
        let source = byte_window(offset, buf.len()).and_then(|range| bytes.get(range));
        // `Memory::read` requires an in-bounds range; violating it is a caller bug.
        let source = source.unwrap_or_else(|| out_of_bounds(offset, buf.len()));
        buf.copy_from_slice(source);
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        let mut bytes = self.0.bytes.borrow_mut();
        let target = byte_window(offset, buf.len()).and_then(|range| bytes.get_mut(range));
        let target = target.unwrap_or_else(|| out_of_bounds(offset, buf.len()));
        target.copy_from_slice(buf);
    }
}

fn byte_window(offset: u64, len: usize) -> Option<core::ops::Range<usize>> {
    let start = usize::try_from(offset).ok()?;
    Some(start..start.checked_add(len)?)
}

#[cold]
fn out_of_bounds(offset: u64, len: usize) -> ! {
    panic!("memory access out of bounds: {len} bytes at offset {offset}")
}

/// Typed access to a [`Memory`]: internal callers address it with [`ByteOffset`] and grow it on
/// write, so no caller computes page counts.
pub(crate) struct Region<'m, M: Memory> {
    memory: &'m M,
}

/// A write could not be placed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WriteError {
    Grow(GrowFailed),
    Overflow(Overflow),
}

impl From<GrowFailed> for WriteError {
    fn from(error: GrowFailed) -> Self {
        Self::Grow(error)
    }
}

impl From<Overflow> for WriteError {
    fn from(error: Overflow) -> Self {
        Self::Overflow(error)
    }
}

impl<'m, M: Memory> Region<'m, M> {
    pub(crate) const fn new(memory: &'m M) -> Self {
        Self { memory }
    }

    /// Reads `buf.len()` bytes at `offset`.
    pub(crate) fn read(&self, offset: ByteOffset, buf: &mut [u8]) {
        self.memory.read(offset.get(), buf);
    }

    /// Reads `buf.len()` bytes at `offset` into `buf`, which need not be initialized.
    pub(crate) fn read_uninit<'b>(
        &self,
        offset: ByteOffset,
        buf: &'b mut [MaybeUninit<u8>],
    ) -> &'b mut [u8] {
        self.memory.read_uninit(offset.get(), buf)
    }

    /// Reads `N` bytes at `offset`.
    pub(crate) fn read_array<const N: usize>(&self, offset: ByteOffset) -> [u8; N] {
        let mut buf = [0; N];
        self.memory.read(offset.get(), &mut buf);
        buf
    }

    /// Overwrites bytes that are already addressable. Never grows, so it cannot fail; writing
    /// past the current size is a caller bug and panics in the backend.
    pub(crate) fn overwrite(&self, offset: ByteOffset, buf: &[u8]) {
        self.memory.write(offset.get(), buf);
    }

    /// Ensures `[0, end)` is addressable, growing by whole pages.
    pub(crate) fn reserve(&self, end: ByteOffset) -> Result<(), WriteError> {
        let have = self.memory.size();
        let need = end.get().div_ceil(PAGE_SIZE);
        if need > have {
            self.memory.grow(need - have)?;
        }
        Ok(())
    }

    /// Writes `buf` at `offset`, growing the memory first when needed.
    pub(crate) fn write(&self, offset: ByteOffset, buf: &[u8]) -> Result<(), WriteError> {
        let len = u64::try_from(buf.len()).map_err(|_| Overflow)?;
        self.reserve(offset.advance(len)?)?;
        self.memory.write(offset.get(), buf);
        Ok(())
    }
}

/// `buf` zeroed, as initialized bytes.
#[expect(unsafe_code, reason = "viewing bytes just written as initialized")]
pub(crate) fn zeroed(buf: &mut [MaybeUninit<u8>]) -> &mut [u8] {
    buf.fill(MaybeUninit::new(0));
    // SAFETY: every byte of `buf` was just written.
    unsafe { assume_init(buf) }
}

/// `buf` as initialized bytes.
///
/// # Safety
///
/// Every byte of `buf` must have been written.
#[expect(
    unsafe_code,
    reason = "the one conversion of written bytes to initialized ones"
)]
const unsafe fn assume_init(buf: &mut [MaybeUninit<u8>]) -> &mut [u8] {
    // SAFETY: `MaybeUninit<u8>` has the layout of `u8`, and the caller guarantees every byte is
    // initialized; the borrow is carried over, so nothing else can reach `buf` meanwhile.
    unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), buf.len()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn region_grows_on_write_and_reads_back() {
        let memory = VecMemory::new();
        let region = Region::new(&memory);
        let at = ByteOffset::new(PAGE_SIZE - 2);
        region.write(at, &[1, 2, 3, 4]).unwrap();
        assert_eq!(memory.size(), 2);
        let mut buf = [0; 4];
        region.read(at, &mut buf);
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    #[test]
    fn grow_reports_previous_size() {
        let memory = VecMemory::new();
        assert_eq!(memory.grow(3), Ok(0));
        assert_eq!(memory.grow(1), Ok(3));
        assert_eq!(memory.size(), 4);
    }

    #[test]
    fn write_past_address_space_is_an_error() {
        let memory = VecMemory::new();
        let region = Region::new(&memory);
        let error = region.write(ByteOffset::new(u64::MAX), &[0]);
        assert_eq!(error, Err(WriteError::Overflow(Overflow)));
    }

    /// Ranges a read into uninitialized bytes is checked on: empty, within a page, across one
    /// page boundary and across two.
    const RANGES: [(u64, usize); 5] = [
        (0, 0),
        (5, 10),
        (PAGE_SIZE - 3, 7),
        (PAGE_SIZE - 100, 300),
        (PAGE_SIZE - 2, 65_540),
    ];

    /// Fills three pages of `memory` with a pattern, then checks that `read_uninit` returns what
    /// `read` does over [`RANGES`], into a buffer never initialized. Small enough for Miri, which
    /// checks the conversion of the bytes read to initialized ones.
    fn read_uninit_matches_read<M: Memory>(memory: &M) {
        let pattern: Vec<u8> = (0..PAGE_SIZE * 3)
            .map(|i| u8::try_from(i * 7 % 251).unwrap())
            .collect();
        Region::new(memory)
            .write(ByteOffset::new(0), &pattern)
            .unwrap();
        for (offset, len) in RANGES {
            let mut want = vec![0; len];
            memory.read(offset, &mut want);
            let mut buf = vec![MaybeUninit::<u8>::uninit(); len];
            let got = memory.read_uninit(offset, &mut buf);
            assert_eq!(got, want.as_slice(), "{len} bytes at {offset}");
        }
    }

    #[test]
    fn a_default_read_uninit_reads_what_read_does() {
        read_uninit_matches_read(&VecMemory::new());
    }

    #[cfg(feature = "ic-stable-structures")]
    #[test]
    fn an_ic_read_uninit_reads_what_read_does_across_segments() {
        use ic_stable_structures::memory_manager::{MemoryId, MemoryManager};
        use ic_stable_structures::{DefaultMemoryImpl, VectorMemory};
        read_uninit_matches_read(&VectorMemory::default());
        // One-page buckets, grown in turn with another memory: a virtual page's neighbours lie
        // apart in real memory, so a read across pages is several real reads.
        let manager = MemoryManager::init_with_bucket_size(DefaultMemoryImpl::default(), 1);
        let (ours, other) = (manager.get(MemoryId::new(0)), manager.get(MemoryId::new(1)));
        for _ in 0..3 {
            assert!(ic_stable_structures::Memory::grow(&ours, 1) >= 0);
            assert!(ic_stable_structures::Memory::grow(&other, 1) >= 0);
        }
        read_uninit_matches_read(&ours);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn raw_read_out_of_bounds_panics() {
        let mut buf = [0; 1];
        VecMemory::new().read(0, &mut buf);
    }
}
