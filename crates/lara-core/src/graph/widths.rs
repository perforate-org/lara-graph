//! The label width table: which labels' edges carry a value, and how wide it is.
//!
//! The table is one extent in the list store, `count u16` then `(label u16, width u16)` entries
//! sorted by label, referenced from bytes 8..16 of the vertices header (a present flag and the
//! extent). The graph keeps a copy in the heap: it owns its memories.

use super::{Error, Graph};
use crate::corrupt::corrupt;
use crate::id::{Label, Width};
use crate::memory::{Memory, Region};
use crate::page_store::{ByteClass, EXTENT_BYTES, Extent, PageStore};
use crate::units::ByteOffset;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// Where the table's reference sits in the vertices header.
const AT_WIDTHS: u64 = 8;
const ENTRY_BYTES: usize = 4;
/// Labels with a width: one page of entries.
const MAX_WIDTHS: usize = (4096 - 2) / ENTRY_BYTES;

/// The table and where it is stored.
#[derive(Default)]
pub(super) struct Widths {
    map: BTreeMap<Label, Width>,
    at: Option<Extent>,
}

impl Widths {
    /// Reads the table referenced from the vertices header.
    pub(super) fn load<M: Memory>(vertices: &M, lists: &PageStore<M>) -> Self {
        let [present, e @ ..]: [u8; 1 + EXTENT_BYTES] =
            Region::new(vertices).read_array(ByteOffset::new(AT_WIDTHS));
        if present == 0 {
            return Self::default();
        }
        let at = Extent::decode(e);
        let view = lists.pages();
        let count = usize::from(u16::from_le_bytes(view.read_array(at.offset())));
        let bytes = view.read(ByteOffset::new(at.offset().get() + 2), count * ENTRY_BYTES);
        let map = bytes
            .as_chunks::<ENTRY_BYTES>()
            .0
            .iter()
            .map(|&[l0, l1, w0, w1]| {
                let width = Width::new(u16::from_le_bytes([w0, w1]))
                    .unwrap_or_else(|| corrupt("label width"));
                (Label::new(u16::from_le_bytes([l0, l1])), width)
            })
            .collect();
        Self { map, at: Some(at) }
    }

    pub(super) fn get(&self, label: Label) -> Option<Width> {
        self.map.get(&label).copied()
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
            None if self.widths.map.len() >= MAX_WIDTHS => return Err(Error::TooManyWidths),
            None => {}
        }
        let mut map = self.widths.map.clone();
        map.insert(label, width);
        let count = u16::try_from(map.len()).map_err(|_| Error::TooManyWidths)?;
        let bytes: Vec<u8> = count
            .to_le_bytes()
            .into_iter()
            .chain(map.iter().flat_map(|(l, w)| {
                let [l0, l1] = l.get().to_le_bytes();
                let [w0, w1] = w.get().to_le_bytes();
                [l0, l1, w0, w1]
            }))
            .collect();
        let class = ByteClass::for_bytes(u32::try_from(bytes.len()).unwrap_or(u32::MAX))
            .unwrap_or_else(|| corrupt("width table beyond a page"));
        let at = self.lists.allocate(class)?;
        self.lists.write(at.offset(), &bytes);
        let reference: Vec<u8> = [1].into_iter().chain(at.encode()).collect();
        self.region()
            .overwrite(ByteOffset::new(AT_WIDTHS), &reference);
        if let Some(old) = self.widths.at {
            self.lists.release(old);
        }
        self.widths = Widths { map, at: Some(at) };
        Ok(())
    }

    /// The width of `label`'s values, if it has been declared.
    #[must_use]
    pub fn width(&self, label: Label) -> Option<Width> {
        self.widths.get(label)
    }

    /// The width table's extent (test invariant checks).
    #[cfg(test)]
    pub(super) const fn widths_extent(&self) -> Option<Extent> {
        self.widths.at
    }
}
