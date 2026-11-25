//! How a [`LabelBucket`] is stored in its label bucket list entry: a plain descriptor, or a
//! header for a bucket with more to it.
//!
//! A list entry's 9 bytes after its key hold a listed [`Descriptor`] for a plain bucket: a tiny one,
//! or a run, including one that keeps its values beside its slots (see `descriptor`). A paged
//! bucket, and one whose values are in a value plane (a run too large for its values beside it,
//! or one of a label with wide values; never tiny: two inline slots leave no room for them), holds
//! `Extended`: the extent of a 32-byte header in the edge store with the full slot descriptor, the
//! value plane's root and width, and the target index's root (or a mark that it was dropped while
//! the node detaches). Plain buckets pay nothing; an extended one costs one more 32-byte read,
//! which a paged bucket's scan of at least a page of slots does not notice.
//!
//! # Extended entry
//!
//! ```text
//! ---------------------------------------- <- Offset 0
//! Reserved space                  ↕ 4 bytes
//! ----------------------------------------
//! Header extent, kind 3 in its    ↕ 5 bytes
//!   spare bits
//! ----------------------------------------
//! ```
//!
//! # Header
//!
//! A 32-byte extent in the edge store.
//!
//! ```text
//! ---------------------------------------- <- Offset 0
//! Full slot descriptor            ↕ 13 bytes
//! ---------------------------------------- <- Offset 13
//! Value plane root                ↕ 6 bytes
//! ---------------------------------------- <- Offset 19
//! Value width; 0 for none         ↕ 2 bytes
//! ----------------------------------------
//! Target index root page          ↕ 4 bytes       (u32::MAX none, u32::MAX - 1 dropped)
//! ----------------------------------------
//! Target index depth              ↕ 1 byte
//! ----------------------------------------
//! Reserved space                  ↕ 6 bytes
//! ----------------------------------------
//! ```

use crate::bucket::descriptor::{
    DESCRIPTOR_BYTES, Descriptor, KIND_HEADER, KIND_SHIFT, LISTED_BYTES, listed_kind,
};
use crate::bucket::target_index::Root;
use crate::bucket::values::{Plane, ROOT_BYTES};
use crate::bucket::{Index, LabelBucket, Values};
use crate::id::Width;
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use crate::store::page_store::{ByteClass, EXTENT_BYTES, Extent, PageId, PageStore, Pages};

/// Bytes of a header: descriptor, plane root, width (0 for none), index root page (`u32::MAX`
/// for none) and depth.
const HEADER_BYTES: usize = DESCRIPTOR_BYTES + ROOT_BYTES + 2 + 4 + 1;
const NO_INDEX: u32 = u32::MAX;
const DROPPED_INDEX: u32 = u32::MAX - 1;
/// The class headers are allocated in.
pub(crate) const HEADER_CLASS: ByteClass = ByteClass::B32;
const _: () = assert!(HEADER_BYTES <= 32);

/// A list entry's 9 bytes after its key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stored {
    Plain(Descriptor),
    Extended(Extent),
}

impl Stored {
    pub(crate) fn encode(&self) -> [u8; LISTED_BYTES] {
        match self {
            Self::Plain(d) => d.encode_listed(),
            Self::Extended(header) => {
                let [e0, e1, e2, e3, e4] = header.encode();
                let last = e4 | (KIND_HEADER << KIND_SHIFT);
                [0, 0, 0, 0, e0, e1, e2, e3, last]
            }
        }
    }

    /// The descriptor the entry stands for: in the entry, or in the header it points to.
    #[inline]
    pub(crate) fn slots<M: Memory>(self, edges: &Pages<'_, M>) -> Descriptor {
        match self {
            Self::Plain(slots) => slots,
            Self::Extended(header) => Descriptor::decode(edges.read_array(header.offset())),
        }
    }

    pub(crate) fn decode(bytes: [u8; LISTED_BYTES]) -> Self {
        if listed_kind(&bytes) == KIND_HEADER {
            let [.., e0, e1, e2, e3, e4] = bytes;
            Self::Extended(Extent::decode([e0, e1, e2, e3, e4]))
        } else {
            Self::Plain(Descriptor::decode_listed(bytes))
        }
    }
}

/// The bucket as its list entry and header store it.
impl LabelBucket {
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
        let mut bytes = [0u8; HEADER_BYTES];
        let (d, rest) = bytes.split_at_mut(DESCRIPTOR_BYTES);
        let (root, rest) = rest.split_at_mut(ROOT_BYTES);
        d.copy_from_slice(&self.slots.encode());
        root.copy_from_slice(&plane.encode());
        let [w0, w1] = width.to_le_bytes();
        let [i0, i1, i2, i3] = page.to_le_bytes();
        rest.copy_from_slice(&[w0, w1, i0, i1, i2, i3, depth]);
        store.write(header.offset(), &bytes);
    }
}

const _: () = assert!(EXTENT_BYTES < LISTED_BYTES);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::descriptor::Tiny;

    #[test]
    fn entries_round_trip_plain_and_valued() {
        let plain = Stored::Plain(Descriptor::Tiny(Tiny::EMPTY));
        let valued = Stored::Extended(Extent::whole(PageId::new(7)));
        for stored in [plain, valued] {
            assert_eq!(Stored::decode(stored.encode()), stored);
        }
    }
}

/// Proofs over every input, checked by Kani (`cargo kani -p lara-graph`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Any nine bytes but a paged kind's decode to a list entry that encodes back to itself.
    #[kani::proof]
    fn list_entries_round_trip() {
        let bytes: [u8; LISTED_BYTES] = kani::any();
        kani::assume(listed_kind(&bytes) != 2);
        let stored = Stored::decode(bytes);
        assert_eq!(Stored::decode(stored.encode()), stored);
    }
}
