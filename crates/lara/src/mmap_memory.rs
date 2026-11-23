//! File-backed [`Memory`] backend over `memmap2`.
//!
//! Available with the `mmap` feature (implies `std`). Backs one memory
//! region with a file-backed `MmapMut`, so wall-clock measurements run
//! against OS page-cache / file-backed mappings instead of a heap `Vec`.
//! For write-volume accounting, wrap this in
//! [`PersistCounterMemory`](crate::memory::PersistCounterMemory) — the
//! counter lives in the wrapper, not here.
//!
//! Reads past the current mapping zero-fill (same contract as
//! [`VecMemory`](crate::memory::VecMemory)). Writes auto-extend the file
//! (page-granular, like `safe_write` growth); I/O failure inside
//! [`Memory::write`] panics because the trait carries no `Result`.
//! [`MmapMemory::flush`] exists for checked persistence.

use crate::memory::{GrowFailed, Memory, PAGE_SIZE};
use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

struct Inner {
    file: File,
    mmap: memmap2::MmapMut,
}

/// File-backed memory region.
///
/// Create with [`MmapMemory::create`]; each graph region needs its own
/// file (the benchmark adapter creates one per region in a temp dir).
/// Not `Clone` or `Sync`: single-threaded use, like the rest of the crate.
/// The backing file is left on disk on drop (caller removes it).
pub struct MmapMemory {
    inner: RefCell<Inner>,
    path: PathBuf,
}

impl MmapMemory {
    /// Creates (truncates) `path` and maps at least `initial_bytes`
    /// (rounded up to one page minimum — an empty file cannot be mapped).
    pub fn create<P: AsRef<Path>>(path: P, initial_bytes: u64) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        let len = initial_bytes.max(PAGE_SIZE).div_ceil(PAGE_SIZE) * PAGE_SIZE;
        file.set_len(len)?;
        // SAFETY: file-backed mapping of a file we hold open; length set above.
        let mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };
        Ok(Self {
            inner: RefCell::new(Inner { file, mmap }),
            path,
        })
    }

    /// Backing file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flushes mapped pages to the backing file (checked persistence).
    /// [`Memory::fence`] calls this best-effort (the trait has no `Result`).
    pub fn flush(&self) -> std::io::Result<()> {
        self.inner.borrow().mmap.flush()
    }

    fn ensure_len(inner: &mut Inner, needed: u64) {
        if needed <= inner.mmap.len() as u64 {
            return;
        }
        let new_len = needed.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        inner.file.set_len(new_len).expect("mmap backing file grow");
        // SAFETY: same file, still open; length extended above.
        inner.mmap = unsafe { memmap2::MmapMut::map_mut(&inner.file).expect("mmap remap") };
    }
}

impl Memory for MmapMemory {
    fn read(&self, offset: u64, buf: &mut [u8]) {
        let inner = self.inner.borrow();
        let start = (offset as usize).min(inner.mmap.len());
        let end = (start + buf.len()).min(inner.mmap.len());
        let n = end - start;
        buf[..n].copy_from_slice(&inner.mmap[start..end]);
        for b in buf[n..].iter_mut() {
            *b = 0;
        }
    }

    fn read_mapped(&self, offset: u64, buf: &mut [u8]) {
        let inner = self.inner.borrow();
        let start = (offset as usize).min(inner.mmap.len());
        let end = (start + buf.len()).min(inner.mmap.len());
        let n = end - start;
        debug_assert_eq!(n, buf.len(), "read_mapped range must be mapped");
        buf[..n].copy_from_slice(&inner.mmap[start..end]);
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        let mut inner = self.inner.borrow_mut();
        let end = offset + buf.len() as u64;
        Self::ensure_len(&mut inner, end);
        let start = offset as usize;
        inner.mmap[start..start + buf.len()].copy_from_slice(buf);
    }

    fn size(&self) -> u64 {
        self.inner.borrow().mmap.len().div_ceil(PAGE_SIZE as usize) as u64
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        let mut inner = self.inner.borrow_mut();
        let target = (inner.mmap.len() as u64 + pages * PAGE_SIZE).div_ceil(PAGE_SIZE) * PAGE_SIZE;
        Self::ensure_len(&mut inner, target);
        Ok(inner.mmap.len().div_ceil(PAGE_SIZE as usize) as u64)
    }

    fn fence(&self) {
        let _ = self.inner.borrow().mmap.flush();
    }
}
