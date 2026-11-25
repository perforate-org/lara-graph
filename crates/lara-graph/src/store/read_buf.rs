//! A buffer for reads from the page store: small reads land on the stack, larger ones in one
//! heap buffer.

use crate::store::memory::{Memory, zeroed};
use crate::store::page_store::Pages;
use crate::store::units::ByteOffset;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

/// Bytes of the stack buffer a read lands in first: a run of up to 32 slots, the first read of a
/// short window, or the values of a few edges. A visit of a small bucket then allocates nothing:
/// on the IC a heap buffer for each visit cost about 250 instructions, a ninth of a node visit
/// (`visit_run_1024`). It is never zeroed: a read fills it in place (see
/// [`Memory::read_uninit`]), where zeroing it first cost about 250 instructions a read.
const SMALL_READ: usize = 128;
const _: () = assert!(SMALL_READ >= 32 * 4);

/// Where reads land: a small stack buffer that reads fill without zeroing it, else one heap
/// buffer sized by the first read that needs it and never zeroed again.
pub(crate) struct ReadBuf {
    small: [MaybeUninit<u8>; SMALL_READ],
    heap: Vec<u8>,
}

impl ReadBuf {
    pub(crate) const fn new() -> Self {
        Self {
            small: [MaybeUninit::uninit(); SMALL_READ],
            heap: Vec::new(),
        }
    }

    /// `len` bytes to fill: zero on the stack, else holding whatever an earlier read left there.
    pub(crate) fn take(&mut self, len: usize) -> &mut [u8] {
        if len <= SMALL_READ {
            return zeroed(self.small.get_mut(..len).unwrap_or(&mut []));
        }
        self.heap_of(len)
    }

    /// Reads `len` bytes at `at`.
    pub(crate) fn fill<M: Memory>(
        &mut self,
        view: &Pages<'_, M>,
        at: ByteOffset,
        len: usize,
    ) -> &[u8] {
        if len <= SMALL_READ {
            return view.read_uninit(at, self.small.get_mut(..len).unwrap_or(&mut []));
        }
        let bytes = self.heap_of(len);
        view.read_into(at, bytes);
        bytes
    }

    /// The heap buffer's first `len` bytes, holding whatever an earlier read left there.
    fn heap_of(&mut self, len: usize) -> &mut [u8] {
        if self.heap.len() < len {
            self.heap.resize(len, 0);
        }
        self.heap.get_mut(..len).unwrap_or(&mut [])
    }
}
