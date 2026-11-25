//! The edge store: every label bucket's slots, header, values and target index in the edge
//! pages, and the label width table that says how wide each label's values are. [`EdgeStore`]
//! makes a bucket's next state, which the graph publishes in the bucket's list entry;
//! [`EdgeView`] reads. Over the bucket layer, beside the node layer; each bucket's own formats
//! and operations are in `crate::bucket`.

mod build;
mod compact;
mod halves;
mod insert;
mod target_index;
mod values;
mod widths;

pub(crate) use build::BuiltHalf;
pub(crate) use compact::Compacted;
pub(crate) use insert::HoleCursor;
pub(crate) use target_index::{Batch, Between};
pub(crate) use values::visit_joined;
pub(crate) use widths::Widths;

use crate::bucket::LabelBucket;
use crate::bucket::pages_per_half;
use crate::bucket::reserved;
use crate::bucket::stored::HEADER_CLASS;
use crate::bucket::values::PAGES_PER_WRITE;
use crate::id::Width;
use crate::store::memory::Memory;
use crate::store::memory::WriteError;
use crate::store::page_store::{PageStore, Pages};

/// The edge store: every bucket's slots, header, index and values in the edge pages.
pub(crate) struct EdgeStore<M: Memory> {
    store: PageStore<M>,
}

/// What reads see of the edge store: lookups by position and by target, values, and where an
/// insert would go. Nothing through it writes.
pub(crate) struct EdgeView<'a, M: Memory> {
    edges: &'a EdgeStore<M>,
    /// How wide each label's values are.
    widths: &'a Widths,
}

impl<'a, M: Memory> EdgeView<'a, M> {
    pub(crate) const fn new(edges: &'a EdgeStore<M>, widths: &'a Widths) -> Self {
        Self { edges, widths }
    }

    /// The edge pages, for reads of a bucket's slots alone (see `crate::bucket::slots`).
    #[inline]
    pub(crate) fn pages(&self) -> Pages<'a, M> {
        self.edges.store.pages()
    }
}

/// The pages a write may take, reserved before it changes anything.
#[derive(Clone, Copy)]
pub(crate) enum Writes {
    /// `count` halves of a label of `width`, and block batches of `batches` pages.
    Halves {
        width: Option<Width>,
        count: u32,
        batches: u32,
    },
    /// `count` values written into existing buckets.
    Values { count: u32 },
}

impl<M: Memory> EdgeStore<M> {
    pub(crate) const fn new(store: PageStore<M>) -> Self {
        Self { store }
    }

    /// The edge pages' store (tests).
    #[cfg(test)]
    pub(crate) const fn store(&self) -> &PageStore<M> {
        &self.store
    }

    /// The edge pages' store, to allocate from (tests).
    #[cfg(test)]
    pub(crate) const fn store_mut(&mut self) -> &mut PageStore<M> {
        &mut self.store
    }

    pub(crate) fn into_memories(self) -> (M, M) {
        self.store.into_memories()
    }

    /// Reserves the pages `writes` can take.
    pub(crate) fn reserve(&mut self, writes: Writes) -> Result<(), WriteError> {
        let pages = match writes {
            Writes::Halves {
                width,
                count,
                batches,
            } => pages_per_half(width) * count + batches,
            // A header, a value plane for a bucket that had none, and the value's write.
            Writes::Values { count } => (2 + PAGES_PER_WRITE) * count,
        };
        self.store.reserve(pages)
    }

    /// Stores `bucket`'s header: gives a bucket with values or an index one, releases one nothing
    /// needs any more, and writes it. Returns the bucket as its list entry is to stand. Within
    /// reserved memory.
    pub(crate) fn seal(&mut self, bucket: LabelBucket) -> LabelBucket {
        let bucket = self.settle(bucket);
        bucket.write_header(&self.store);
        bucket
    }

    fn settle(&mut self, bucket: LabelBucket) -> LabelBucket {
        match (bucket.extended(), bucket.header) {
            (true, None) => LabelBucket {
                header: Some(reserved(self.store.allocate(HEADER_CLASS))),
                ..bucket
            },
            (false, Some(header)) => {
                self.store.release(header);
                LabelBucket {
                    header: None,
                    ..bucket
                }
            }
            (true, Some(_)) | (false, None) => bucket,
        }
    }
}
