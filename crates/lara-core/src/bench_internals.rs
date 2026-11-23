//! Unstable access to internals for measurement crates (`bench-internals` feature).
//!
//! Not part of the API: it is excluded from the public API snapshot, may change without notice,
//! and exists so experiments run the real allocator instead of a copy of it.

use crate::graph::Graph;
use crate::id::{Direction, Label, NodeId};
use crate::label_bucket_list::LabelBucketKey;
use crate::memory::{GrowFailed, Memory, VecMemory};
use crate::span_alloc::{SpanTag, TagAllocator};
use crate::units::SpanLen;
use core::cell::Cell;

/// Pages of the target index of `v`'s `(label, direction)` bucket: 0 when it has none.
#[must_use]
pub fn index_pages<M: Memory>(
    g: &Graph<M>,
    v: NodeId,
    label: Label,
    direction: Direction,
) -> usize {
    g.index_pages(v, LabelBucketKey { label, direction })
}

/// The slab span allocator over a heap memory that counts its accesses.
pub struct SpanAllocator {
    inner: TagAllocator<CountingMemory>,
}

/// How often the allocator read and wrote its memory: a proxy for stable-memory calls on the IC.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryOps {
    reads: u64,
    writes: u64,
}

impl MemoryOps {
    /// Read calls.
    #[must_use]
    pub const fn reads(&self) -> u64 {
        self.reads
    }

    /// Write calls.
    #[must_use]
    pub const fn writes(&self) -> u64 {
        self.writes
    }
}

/// A [`VecMemory`] that counts read and write calls.
#[derive(Debug, Default)]
struct CountingMemory {
    inner: VecMemory,
    reads: Cell<u64>,
    writes: Cell<u64>,
}

impl Memory for CountingMemory {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        self.inner.grow(pages)
    }

    fn read(&self, offset: u64, buf: &mut [u8]) {
        self.reads.set(self.reads.get() + 1);
        self.inner.read(offset, buf);
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        self.writes.set(self.writes.get() + 1);
        self.inner.write(offset, buf);
    }
}

/// One allocated span; hand it back to [`SpanAllocator::release`].
#[derive(Debug)]
pub struct Span {
    tag: SpanTag,
}

impl Span {
    /// First slot.
    #[must_use]
    pub fn start(&self) -> u64 {
        self.tag.range().start().get()
    }

    /// Length in slots.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.tag.range().len().get()
    }

    /// Never true: spans are non-empty. Present for the `len` convention.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tag.range().is_empty()
    }
}

impl Default for SpanAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl SpanAllocator {
    /// An allocator over a fresh heap memory.
    ///
    /// # Panics
    ///
    /// If the heap memory cannot be formatted.
    #[must_use]
    pub fn new() -> Self {
        match TagAllocator::create(CountingMemory::default()) {
            Ok(inner) => Self { inner },
            Err(error) => panic!("formatting a fresh memory failed: {error:?}"),
        }
    }

    /// Allocates `len` slots; `None` for zero or when memory is exhausted.
    pub fn allocate(&mut self, len: u32) -> Option<Span> {
        let tag = self.inner.allocate(SpanLen::new(len)?).ok()?;
        Some(Span { tag })
    }

    /// Returns a span.
    pub fn release(&mut self, span: Span) {
        let Span { tag } = span;
        self.inner.release(tag);
    }

    /// Grows `span` in place to `len` slots; hands it back unchanged when the space after it is
    /// taken (or `len` is zero).
    ///
    /// # Errors
    ///
    /// The unchanged span, when it cannot grow in place.
    pub fn try_extend(&mut self, span: Span, len: u32) -> Result<Span, Span> {
        let Some(len) = SpanLen::new(len) else {
            return Err(span);
        };
        match self.inner.try_extend(span.tag, len) {
            Ok(tag) => Ok(Span { tag }),
            Err(blocked) => Err(Span {
                tag: blocked.into_tag(),
            }),
        }
    }

    /// Memory calls the allocator has made so far.
    #[must_use]
    pub fn memory_ops(&self) -> MemoryOps {
        let memory = self.inner.memory();
        MemoryOps {
            reads: memory.reads.get(),
            writes: memory.writes.get(),
        }
    }

    /// Bytes the allocator spends on tag records (one per segment, live, free or recycled).
    #[must_use]
    pub fn tag_bytes(&self) -> u64 {
        self.inner.tag_bytes()
    }

    /// One past the highest slot ever handed out.
    #[must_use]
    pub fn frontier(&self) -> u64 {
        self.inner.frontier().get()
    }
}
