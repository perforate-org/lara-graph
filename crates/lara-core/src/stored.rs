//! How a bucket is stored in its list entry: a plain descriptor, or a header for a bucket with
//! more to it.
//!
//! The 16-byte list entry holds a [`Descriptor`] for a plain bucket. A bucket whose edges carry
//! values (never tiny: three inline slots leave no room for them), or a paged bucket with its
//! target index (or one dropped while its vertex detaches), holds `Extended`: the extent of a 32-byte header in the edge store with the slot
//! descriptor, the value plane's root and width, and the target index's root. Plain buckets pay
//! nothing; an extended one costs one more 32-byte read.

use crate::corrupt::corrupt;
use crate::descriptor::{DESCRIPTOR_BYTES, Descriptor};
use crate::id::Width;
use crate::memory::Memory;
use crate::page_store::{ByteClass, EXTENT_BYTES, Extent, PageId, PageStore, Pages};
use crate::targets::Root;
use crate::values::{Plane, ROOT_BYTES};

/// The descriptor kind byte of a header reference; descriptor kinds are 1 to 3.
const KIND_EXTENDED: u8 = 4;
/// Bytes of a header: descriptor, plane root, width (0 for none), index root page (`u32::MAX`
/// for none) and depth.
const HEADER_BYTES: usize = DESCRIPTOR_BYTES + ROOT_BYTES + 2 + 4 + 1;
const NO_INDEX: u32 = u32::MAX;
const DROPPED_INDEX: u32 = u32::MAX - 1;
/// The class headers are allocated in.
pub(crate) const HEADER_CLASS: ByteClass = ByteClass::SMALLEST;
const _: () = assert!(HEADER_BYTES <= 32);

/// A list entry's 16 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stored {
    Plain(Descriptor),
    Extended(Extent),
}

impl Stored {
    pub(crate) fn encode(&self) -> [u8; DESCRIPTOR_BYTES] {
        match self {
            Self::Plain(d) => d.encode(),
            Self::Extended(header) => {
                let mut out = [0u8; DESCRIPTOR_BYTES];
                for (dst, src) in out
                    .iter_mut()
                    .zip([KIND_EXTENDED].into_iter().chain(header.encode()))
                {
                    *dst = src;
                }
                out
            }
        }
    }

    pub(crate) fn decode(bytes: [u8; DESCRIPTOR_BYTES]) -> Self {
        let [kind, e0, e1, e2, e3, e4, e5, e6, ..] = bytes;
        if kind == KIND_EXTENDED {
            Self::Extended(Extent::decode([e0, e1, e2, e3, e4, e5, e6]))
        } else {
            Self::Plain(Descriptor::decode(bytes))
        }
    }
}

/// A bucket's values: the plane and the width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Values {
    pub(crate) plane: Plane,
    pub(crate) width: Width,
}

/// A bucket's target index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Index {
    /// No postings: a bucket too small for an index, or a paged one whose complete blocks hold
    /// no live edges.
    Empty,
    /// The tree of postings.
    Tree(Root),
    /// Released while its vertex detaches, as all its edges are going: lookups scan the bucket.
    Dropped,
}

impl Index {
    /// The tree, if there is one.
    pub(crate) const fn tree(self) -> Option<Root> {
        match self {
            Self::Tree(root) => Some(root),
            Self::Empty | Self::Dropped => None,
        }
    }

    /// The index holding `root`.
    pub(crate) const fn of(root: Option<Root>) -> Self {
        match root {
            Some(root) => Self::Tree(root),
            None => Self::Empty,
        }
    }
}

/// A bucket resolved from its list entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bucket {
    pub(crate) slots: Descriptor,
    /// The header, while the bucket has values or an index.
    pub(crate) header: Option<Extent>,
    pub(crate) values: Option<Values>,
    pub(crate) index: Index,
}

impl Bucket {
    /// A bucket with only slots.
    pub(crate) const fn plain(slots: Descriptor) -> Self {
        Self {
            slots,
            header: None,
            values: None,
            index: Index::Empty,
        }
    }

    /// Reads an extended bucket's header (one 32-byte read); a plain one needs nothing.
    pub(crate) fn resolve<M: Memory>(view: &Pages<'_, M>, stored: Stored) -> Self {
        let header = match stored {
            Stored::Plain(slots) => return Self::plain(slots),
            Stored::Extended(header) => header,
        };
        let bytes: [u8; HEADER_BYTES] = view.read_array(header.offset());
        let (d, rest) = bytes.split_at(DESCRIPTOR_BYTES);
        let (root, rest) = rest.split_at(ROOT_BYTES);
        let [w0, w1, i0, i1, i2, i3, depth] =
            <[u8; 7]>::try_from(rest).unwrap_or_else(|_| corrupt("header"));
        let slots = Descriptor::decode(d.try_into().unwrap_or_else(|_| corrupt("header")));
        let plane = Plane::decode(root.try_into().unwrap_or_else(|_| corrupt("header")));
        let values = Width::new(u16::from_le_bytes([w0, w1])).map(|width| Values { plane, width });
        let index = match u32::from_le_bytes([i0, i1, i2, i3]) {
            NO_INDEX => Index::Empty,
            DROPPED_INDEX => Index::Dropped,
            page => Index::Tree(Root {
                page: PageId::new(page),
                depth,
            }),
        };
        Self {
            slots,
            header: Some(header),
            values,
            index,
        }
    }

    /// Whether the bucket needs a header: it has values or an index.
    pub(crate) const fn extended(&self) -> bool {
        self.values.is_some() || !matches!(self.index, Index::Empty)
    }

    /// What the list entry holds for this bucket.
    pub(crate) const fn stored(&self) -> Stored {
        match self.header {
            None => Stored::Plain(self.slots),
            Some(header) => Stored::Extended(header),
        }
    }

    /// Writes the header, if the bucket has one.
    pub(crate) fn write_header<M: Memory>(&self, store: &PageStore<M>) {
        let Some(header) = self.header else { return };
        let (plane, width) = self
            .values
            .map_or((Plane::Empty, 0), |v| (v.plane, v.width.get()));
        let (page, depth) = match self.index {
            Index::Empty => (NO_INDEX, 0),
            Index::Dropped => (DROPPED_INDEX, 0),
            Index::Tree(r) => (r.page.get(), r.depth),
        };
        let bytes: alloc::vec::Vec<u8> = self
            .slots
            .encode()
            .into_iter()
            .chain(plane.encode())
            .chain(width.to_le_bytes())
            .chain(page.to_le_bytes())
            .chain([depth])
            .collect();
        store.write(header.offset(), &bytes);
    }
}

const _: () = assert!(EXTENT_BYTES < DESCRIPTOR_BYTES);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::Tiny;

    #[test]
    fn entries_round_trip_plain_and_valued() {
        let plain = Stored::Plain(Descriptor::Tiny(Tiny::EMPTY));
        let valued = Stored::Extended(Extent::whole(PageId::new(7)));
        for stored in [plain, valued] {
            assert_eq!(Stored::decode(stored.encode()), stored);
        }
    }
}
