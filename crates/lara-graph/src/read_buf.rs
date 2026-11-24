//! A buffer for reads from the page store: small reads land on the stack, larger ones in one
//! heap buffer.

use crate::memory::Memory;
use crate::page_store::Pages;
use crate::units::ByteOffset;
use alloc::vec::Vec;

/// Bytes of the stack buffer a read lands in first: a run of up to 32 slots, the first read of a
/// short window, or the values of a few edges. A visit of a small bucket then allocates nothing:
/// on the IC a heap buffer for each visit cost about 250 instructions, a ninth of a node visit
/// (`visit_run_1024`).
const SMALL_READ: usize = 128;
const _: () = assert!(SMALL_READ >= 32 * 4);

/// Where reads land: a small stack buffer, zeroed when first used, else one heap buffer sized by
/// the first read that needs it and never zeroed again.
pub(crate) struct ReadBuf {
    small: Option<[u8; SMALL_READ]>,
    heap: Vec<u8>,
}

impl ReadBuf {
    pub(crate) const fn new() -> Self {
        Self {
            small: None,
            heap: Vec::new(),
        }
    }

    /// `len` bytes to read into, holding whatever an earlier read left there.
    pub(crate) fn take(&mut self, len: usize) -> &mut [u8] {
        if len <= SMALL_READ {
            let small = self.small.get_or_insert([0; SMALL_READ]);
            small.get_mut(..len).unwrap_or(&mut [])
        } else {
            if self.heap.len() < len {
                self.heap.resize(len, 0);
            }
            self.heap.get_mut(..len).unwrap_or(&mut [])
        }
    }

    /// Reads `len` bytes at `at`.
    pub(crate) fn fill<M: Memory>(
        &mut self,
        view: &Pages<'_, M>,
        at: ByteOffset,
        len: usize,
    ) -> &[u8] {
        let bytes = self.take(len);
        view.read_into(at, bytes);
        bytes
    }
}
