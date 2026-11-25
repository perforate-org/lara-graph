//! The label width table: which labels' edges carry a value, and how wide it is.
//!
//! The table is a value plane in the label bucket list store with a two-byte cell per label: the
//! label's width, or zero for none. Its root is in the nodes header (see `node::store`). Any of
//! the 65,536 labels can have a width, and a plane takes pages only where labels have one. The
//! graph keeps the labels that have a width in the heap, sorted: it owns its memories.

use crate::bucket::reserved;
use crate::bucket::values::{self, PAGES_PER_WRITE, Plane};
use crate::error::Error;
use crate::id::{Label, Width};
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use crate::store::page_store::PageStore;
use alloc::vec::Vec;
use core::ops::ControlFlow;

/// A cell: the label's width, zero for none.
const CELL: Width = match Width::new(2) {
    Some(width) => width,
    None => panic!("a cell is two bytes"),
};

/// The labels with a width, in label order, and the plane that stores them.
pub(crate) struct Widths {
    sorted: Vec<(Label, Width)>,
    plane: Plane,
}

impl Default for Widths {
    fn default() -> Self {
        Self {
            sorted: Vec::new(),
            plane: Plane::Empty,
        }
    }
}

impl Widths {
    /// The table's plane (test invariant checks).
    #[cfg(test)]
    pub(crate) const fn plane(&self) -> Plane {
        self.plane
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

    /// Reads the table whose root the nodes header holds.
    pub(crate) fn load<M: Memory>(plane: Plane, label_bucket_lists: &PageStore<M>) -> Self {
        let mut sorted = Vec::new();
        values::visit(
            &label_bucket_lists.pages(),
            plane,
            CELL,
            (0, 1 << 16, false),
            |at, bytes| {
                for (label, &[w0, w1]) in (at..).zip(bytes.as_chunks::<2>().0) {
                    if let Some(width) = Width::new(u16::from_le_bytes([w0, w1])) {
                        let label = u16::try_from(label).unwrap_or_else(|_| corrupt("label"));
                        sorted.push((Label::new(label), width));
                    }
                }
                ControlFlow::<()>::Continue(())
            },
        );
        Self { sorted, plane }
    }

    pub(crate) fn get(&self, label: Label) -> Option<Width> {
        let i = self.sorted.binary_search_by_key(&label, |&(l, _)| l).ok()?;
        self.sorted.get(i).map(|&(_, width)| width)
    }

    /// Gives `label` values of `width` bytes and returns the table's new root, for the nodes
    /// header; `None` when `label` already has `width`. On error nothing changes.
    pub(crate) fn declare<M: Memory>(
        &mut self,
        lists: &mut PageStore<M>,
        label: Label,
        width: Width,
    ) -> Result<Option<Plane>, Error> {
        let place = match self.sorted.binary_search_by_key(&label, |&(l, _)| l) {
            Ok(i) => {
                return match self.sorted.get(i) {
                    Some(&(_, declared)) if declared == width => Ok(None),
                    Some(&(_, declared)) => Err(Error::WidthConflict { label, declared }),
                    None => corrupt("a found width"),
                };
            }
            Err(place) => place,
        };
        lists.reserve(PAGES_PER_WRITE)?;
        self.plane = reserved(values::write(
            lists,
            self.plane,
            CELL,
            u32::from(label.get()),
            &width.get().to_le_bytes(),
        ));
        self.sorted.insert(place, (label, width));
        Ok(Some(self.plane))
    }
}
