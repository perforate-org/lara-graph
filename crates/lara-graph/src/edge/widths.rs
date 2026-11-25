//! The label width table: which labels' edges carry a value, and how wide it is.
//!
//! The table is one extent in the label bucket list store, its entries sorted by label, referenced
//! from the nodes header (see `node::store`). The graph keeps a copy in the heap: it owns its memories.
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

use crate::error::Error;
use crate::id::{Label, Width};
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use crate::store::page_store::{ByteClass, Extent, PageStore};
use crate::store::units::ByteOffset;
use alloc::vec::Vec;

const ENTRY_BYTES: usize = 4;
/// Labels with a width: one page of entries.
const MAX_WIDTHS: usize = (4096 - 2) / ENTRY_BYTES;

/// The table and where it is stored: its entries in label order, as they are stored.
#[derive(Default)]
pub(crate) struct Widths {
    sorted: Vec<(Label, Width)>,
    at: Option<Extent>,
}

impl Widths {
    /// The table's extent (test invariant checks).
    #[cfg(test)]
    pub(crate) const fn extent(&self) -> Option<Extent> {
        self.at
    }

    /// `label`'s width, checking that `value` matches it.
    pub(crate) fn check(&self, label: Label, value: &[u8]) -> Result<Option<Width>, Error> {
        let width = self.get(label);
        let expected = width.map_or(0, Width::get);
        if usize::from(expected) == value.len() {
            Ok(width)
        } else {
            Err(Error::WrongValueLength {
                label,
                expected,
                actual: value.len(),
            })
        }
    }

    /// Reads the table at `at`, which the nodes header names; an empty one for `None`.
    pub(crate) fn load<M: Memory>(at: Option<Extent>, label_bucket_lists: &PageStore<M>) -> Self {
        let Some(at) = at else {
            return Self::default();
        };
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

    pub(crate) fn get(&self, label: Label) -> Option<Width> {
        let i = self.sorted.binary_search_by_key(&label, |&(l, _)| l).ok()?;
        self.sorted.get(i).map(|&(_, width)| width)
    }
}

impl Widths {
    /// Gives `label` values of `width` bytes: writes a new table to `lists`, releases the old
    /// one, and returns where the new table is, for the nodes header; `None` when `label` already has
    /// `width`. On error nothing changes.
    pub(crate) fn declare<M: Memory>(
        &mut self,
        lists: &mut PageStore<M>,
        label: Label,
        width: Width,
    ) -> Result<Option<Extent>, Error> {
        match self.get(label) {
            Some(declared) if declared == width => return Ok(None),
            Some(declared) => return Err(Error::WidthConflict { label, declared }),
            None if self.sorted.len() >= MAX_WIDTHS => return Err(Error::TooManyWidths),
            None => {}
        }
        let mut sorted = self.sorted.clone();
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
        let at = lists.allocate(class)?;
        lists.write(at.offset(), &bytes);
        if let Some(old) = self.at {
            lists.release(old);
        }
        *self = Self {
            sorted,
            at: Some(at),
        };
        Ok(Some(at))
    }
}
