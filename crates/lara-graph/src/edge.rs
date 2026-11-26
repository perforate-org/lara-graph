//! The edge store: every label bucket's slots, header, values and target index in the edge
//! pages, and the label width table that says how wide each label's values are. [`EdgeStore`]
//! makes a bucket's next state, which the graph publishes in the bucket's list entry;
//! [`EdgeView`] reads. Over the bucket layer, beside the node layer; each bucket's own formats
//! and operations are in `crate::bucket`.

mod build;
mod compact;
mod halves;
mod insert;
mod migrate;
mod target_index;
mod values;
mod widths;

pub(crate) use build::BuiltHalf;
pub(crate) use compact::Compacted;
pub(crate) use insert::HoleCursor;
#[cfg(test)]
pub(crate) use migrate::extents as migration_extents;
pub(crate) use target_index::{Batch, Between};
pub(crate) use values::visit_joined;
pub(crate) use widths::Widths;

use crate::bucket::LabelBucket;
use crate::bucket::pages_per_half;
use crate::bucket::reserved;
use crate::bucket::stored::{HEADER_CLASS, set_migration_record};
use crate::bucket::values::PAGES_PER_WRITE;
use crate::id::Width;
use crate::store::memory::Memory;
use crate::store::memory::WriteError;
use crate::store::page_store::ListPool;
use crate::store::page_store::{Extent, PageStore, Pages};

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
    /// `count` halves of a label of `width`, block batches of `batches` pages, and `lists`
    /// pages for the label bucket lists the halves may grow, which share the store.
    Halves {
        width: Option<Width>,
        count: u32,
        batches: u32,
        lists: u32,
    },
    /// `count` values written into existing buckets.
    Values { count: u32 },
}

impl<M: Memory> EdgeStore<M> {
    pub(crate) const fn new(store: PageStore<M>) -> Self {
        Self { store }
    }

    /// The page store, which the label bucket lists share.
    pub(crate) const fn store(&self) -> &PageStore<M> {
        &self.store
    }

    /// The page store, to allocate edge extents from (tests).
    #[cfg(test)]
    pub(crate) const fn store_mut(&mut self) -> &mut PageStore<M> {
        &mut self.store
    }

    /// The page store, lent to label bucket lists and node label data: their chunk extents come
    /// from pages of their own.
    pub(crate) fn lists_mut(&mut self) -> ListPool<'_, M> {
        self.store.lists()
    }

    pub(crate) fn into_memory(self) -> M {
        self.store.into_memory()
    }

    /// Reserves the pages `writes` can take.
    pub(crate) fn reserve(&mut self, writes: Writes) -> Result<(), WriteError> {
        let pages = match writes {
            Writes::Halves {
                width,
                count,
                batches,
                lists,
            } => pages_per_half(width) * count + batches + lists,
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

    /// A new header, with no migration record. Kept out of line: a bucket takes one once, and
    /// inlined into the insert path it cost each insert about 40 instructions.
    #[cold]
    #[inline(never)]
    fn new_header(&mut self) -> Extent {
        let header = reserved(self.store.allocate(HEADER_CLASS));
        set_migration_record(&self.store, header, None);
        header
    }

    fn settle(&mut self, bucket: LabelBucket) -> LabelBucket {
        match (bucket.extended(), bucket.header) {
            (true, None) => LabelBucket {
                header: Some(self.new_header()),
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
