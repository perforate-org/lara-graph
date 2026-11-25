//! The label width table: which labels' edges carry a value, and how wide it is.
//!
//! The table is a value plane in the label bucket list store with a four-byte cell per label: the
//! label's width (two bytes, zero for none), its current value version (one byte, see
//! [`ValueVersion`]) and a reserved byte. Its root is in the nodes header (see `node::store`). Any of
//! the 65,536 labels can have a width, and a plane takes pages only where labels have one. The
//! graph keeps the labels that have a width in the heap, sorted: it owns its memories.

use crate::bucket::values::{self, PAGES_PER_WRITE, Plane};
use crate::bucket::{Schema, reserved};
use crate::error::Error;
use crate::id::{Label, ValueVersion, Width};
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use crate::store::page_store::PageStore;
use alloc::vec::Vec;
use core::ops::ControlFlow;

/// A cell: the label's width (zero for none), its value version and a reserved byte.
const CELL: Width = match Width::new(4) {
    Some(width) => width,
    None => panic!("a cell is four bytes"),
};

/// `schema` as its cell holds it.
fn encode(schema: Schema) -> [u8; 4] {
    let [w0, w1] = schema.width.get().to_le_bytes();
    [w0, w1, schema.version.get(), 0]
}

/// The labels with a width, in label order, and the plane that stores them.
pub(crate) struct Widths {
    sorted: Vec<(Label, Schema)>,
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

    /// `label`'s width and version, checking that `value` matches the width.
    pub(crate) fn check(&self, label: Label, value: &[u8]) -> Result<Option<Schema>, Error> {
        let schema = self.schema(label);
        let expected = schema.map_or(0, |s| s.width.get());
        if usize::from(expected) == value.len() {
            Ok(schema)
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
                for (label, &[w0, w1, version, _]) in (at..).zip(bytes.as_chunks::<4>().0) {
                    if let Some(width) = Width::new(u16::from_le_bytes([w0, w1])) {
                        let label = u16::try_from(label).unwrap_or_else(|_| corrupt("label"));
                        let version = ValueVersion::new(version);
                        sorted.push((Label::new(label), Schema { width, version }));
                    }
                }
                ControlFlow::<()>::Continue(())
            },
        );
        Self { sorted, plane }
    }

    pub(crate) fn get(&self, label: Label) -> Option<Width> {
        self.schema(label).map(|schema| schema.width)
    }

    /// `label`'s width and value version, if it has a width.
    pub(crate) fn schema(&self, label: Label) -> Option<Schema> {
        let i = self.sorted.binary_search_by_key(&label, |&(l, _)| l).ok()?;
        self.sorted.get(i).map(|&(_, schema)| schema)
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
                    Some(&(_, declared)) if declared.width == width => Ok(None),
                    Some(&(_, declared)) => Err(Error::WidthConflict {
                        label,
                        declared: declared.width,
                    }),
                    None => corrupt("a found width"),
                };
            }
            Err(place) => place,
        };
        let schema = Schema {
            width,
            version: ValueVersion::FIRST,
        };
        self.write(lists, label, schema)?;
        self.sorted.insert(place, (label, schema));
        Ok(Some(self.plane))
    }

    /// Gives `label` values of `width` bytes in a new layout: its next value version, or the
    /// first when it had no width. Returns the version and the table's new root. On error
    /// nothing changes.
    pub(crate) fn change<M: Memory>(
        &mut self,
        lists: &mut PageStore<M>,
        label: Label,
        width: Width,
    ) -> Result<(ValueVersion, Plane), Error> {
        let found = self.sorted.binary_search_by_key(&label, |&(l, _)| l);
        let version = match found.ok().and_then(|i| self.sorted.get(i)) {
            None => ValueVersion::FIRST,
            Some(&(_, old)) => old.version.next().ok_or(Error::AddressSpaceExhausted)?,
        };
        let schema = Schema { width, version };
        self.write(lists, label, schema)?;
        match found {
            Ok(i) => {
                if let Some(entry) = self.sorted.get_mut(i) {
                    entry.1 = schema;
                }
            }
            Err(place) => self.sorted.insert(place, (label, schema)),
        }
        Ok((version, self.plane))
    }

    /// Writes `label`'s cell. On error nothing changes.
    fn write<M: Memory>(
        &mut self,
        lists: &mut PageStore<M>,
        label: Label,
        schema: Schema,
    ) -> Result<(), Error> {
        lists.reserve(PAGES_PER_WRITE)?;
        self.plane = reserved(values::write(
            lists,
            self.plane,
            CELL,
            u32::from(label.get()),
            &encode(schema),
        ));
        Ok(())
    }
}
