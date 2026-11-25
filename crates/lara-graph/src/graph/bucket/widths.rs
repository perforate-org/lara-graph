//! The label width table: which labels' edges carry a value, and how wide it is.
//!
//! The table is one extent in the label bucket list store, its entries sorted by label, referenced
//! from the nodes header (see `graph`). The graph keeps a copy in the heap: it owns its memories.
//!
//! # Table
//!
//! ```text
//! ---------------------------------------- <- Extent start
//! Entry count = E                 ↕ 2 bytes
//! ----------------------------------------
//! Entry 0: label                  ↕ 2 bytes
//! ----------------------------------------
//! Entry 0: width                  ↕ 2 bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Entry E - 1                     ↕ 4 bytes
//! ----------------------------------------
//! Unused space
//! ```

use crate::graph::{Error, Graph};
use crate::id::{Label, Width};
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, Region};
use crate::store::page_store::{ByteClass, EXTENT_BYTES, Extent, PageStore};
use crate::store::units::ByteOffset;
use alloc::vec::Vec;

/// Where the table's reference sits in the nodes header.
pub(in crate::graph) const AT_WIDTHS: u64 = 8;
const ENTRY_BYTES: usize = 4;
/// Labels with a width: one page of entries.
const MAX_WIDTHS: usize = (4096 - 2) / ENTRY_BYTES;

/// The table and where it is stored: its entries in label order, as they are stored.
#[derive(Default)]
pub(in crate::graph) struct Widths {
    sorted: Vec<(Label, Width)>,
    at: Option<Extent>,
}

impl Widths {
    /// Reads the table referenced from the nodes header.
    pub(in crate::graph) fn load<M: Memory>(nodes: &M, label_bucket_lists: &PageStore<M>) -> Self {
        let [present, e @ ..]: [u8; 1 + EXTENT_BYTES] =
            Region::new(nodes).read_array(ByteOffset::new(AT_WIDTHS));
        if present == 0 {
            return Self::default();
        }
        let at = Extent::decode(e);
        let view = label_bucket_lists.pages();
        let count = usize::from(u16::from_le_bytes(view.read_array(at.offset())));
        let bytes = view.read(ByteOffset::new(at.offset().get() + 2), count * ENTRY_BYTES);
        let sorted = bytes
            .as_chunks::<ENTRY_BYTES>()
            .0
            .iter()
            .map(|&[l0, l1, w0, w1]| {
                let width = Width::new(u16::from_le_bytes([w0, w1]))
                    .unwrap_or_else(|| corrupt("label width"));
                (Label::new(u16::from_le_bytes([l0, l1])), width)
            })
            .collect();
        Self {
            sorted,
            at: Some(at),
        }
    }

    pub(in crate::graph) fn get(&self, label: Label) -> Option<Width> {
        let i = self.sorted.binary_search_by_key(&label, |&(l, _)| l).ok()?;
        self.sorted.get(i).map(|&(_, width)| width)
    }
}

impl<M: Memory> Graph<M> {
    /// Declares that `label`'s edges carry a value of `width` bytes, on every side. Constant
    /// time: existing edges read as zeros until a value is written to them. Declaring the same
    /// width again changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::WidthConflict`] if `label` already has another width, [`Error::TooManyWidths`]
    /// past 1023 labels, [`Error::OutOfMemory`]. On error nothing changes.
    pub fn declare_width(&mut self, label: Label, width: Width) -> Result<(), Error> {
        match self.widths.get(label) {
            Some(declared) if declared == width => return Ok(()),
            Some(declared) => return Err(Error::WidthConflict { label, declared }),
            None if self.widths.sorted.len() >= MAX_WIDTHS => return Err(Error::TooManyWidths),
            None => {}
        }
        let mut sorted = self.widths.sorted.clone();
        let place = sorted.partition_point(|&(l, _)| l < label);
        sorted.insert(place, (label, width));
        let count = u16::try_from(sorted.len()).map_err(|_| Error::TooManyWidths)?;
        let bytes: Vec<u8> = count
            .to_le_bytes()
            .into_iter()
            .chain(sorted.iter().flat_map(|(l, w)| {
                let [l0, l1] = l.get().to_le_bytes();
                let [w0, w1] = w.get().to_le_bytes();
                [l0, l1, w0, w1]
            }))
            .collect();
        let class = ByteClass::for_bytes(u32::try_from(bytes.len()).unwrap_or(u32::MAX))
            .unwrap_or_else(|| corrupt("width table beyond a page"));
        let at = self.label_bucket_lists.allocate(class)?;
        self.label_bucket_lists.write(at.offset(), &bytes);
        let reference: Vec<u8> = [1].into_iter().chain(at.encode()).collect();
        self.region()
            .overwrite(ByteOffset::new(AT_WIDTHS), &reference);
        if let Some(old) = self.widths.at {
            self.label_bucket_lists.release(old);
        }
        self.widths = Widths {
            sorted,
            at: Some(at),
        };
        Ok(())
    }

    /// The width of `label`'s values, if it has been declared.
    #[must_use]
    pub fn width(&self, label: Label) -> Option<Width> {
        self.widths.get(label)
    }

    /// The width table's extent (test invariant checks).
    #[cfg(test)]
    pub(in crate::graph) const fn widths_extent(&self) -> Option<Extent> {
        self.widths.at
    }
}
