//! How a [`LabelBucket`] is stored in its label bucket list entry: a plain descriptor, or a
//! header for a bucket with more to it.
//!
//! A list entry's 9 bytes after its key hold a listed [`Descriptor`] for a plain bucket: a tiny one,
//! or a run, including one that keeps its values beside its slots (see `descriptor`). A paged
//! bucket, and one whose values are in a value plane (a run too large for its values beside it,
//! or one of a label with wide values; never tiny: two inline slots leave no room for them), holds
//! `Extended`: the extent of a 32-byte header in the edge store with the full slot descriptor, the
//! value plane's root and width, and the target index's root (or a mark that it was dropped while
//! the node is deleted). Plain buckets pay nothing; an extended one costs one more 32-byte read,
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
//! Value version                   ↕ 1 byte        (see `crate::ValueVersion`)
//! ----------------------------------------
//! Migration record extent         ↕ 5 bytes       (top bit set while a `migrate_values` runs;
//!                                                  zeros for none)
//! ----------------------------------------
//! ```

use crate::bucket::descriptor::{
    DESCRIPTOR_BYTES, Descriptor, KIND_HEADER, KIND_SHIFT, LISTED_BYTES, listed_kind,
};
use crate::bucket::target_index::Root;
use crate::bucket::values::{Plane, ROOT_BYTES};
use crate::bucket::{Index, LabelBucket, Migration, Values};
use crate::id::{ValueVersion, Width};
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use crate::store::page_store::{ByteClass, EXTENT_BYTES, Extent, PageId, PageStore, Pages};
use crate::store::units::ByteOffset;

/// Bytes of a header: descriptor, plane root, width (0 for none), index root page (`u32::MAX`
/// for none) and depth.
const HEADER_BYTES: usize = DESCRIPTOR_BYTES + ROOT_BYTES + 2 + 4 + 1 + 1 + EXTENT_BYTES;
/// The bit of a migration record's last byte that marks it present.
const MIGRATING: u8 = 0b1000_0000;
/// Where in a header the migration record's extent is.
const AT_MIGRATION: usize = HEADER_BYTES - EXTENT_BYTES;

/// The record of the migration under way in the bucket `header` heads, if any.
pub(crate) fn migration_record<M: Memory>(view: &Pages<'_, M>, header: Extent) -> Option<Extent> {
    let bytes: [u8; EXTENT_BYTES] = view.read_array(migration_at(header));
    let [.., last] = bytes;
    (last & MIGRATING != 0).then(|| Extent::decode(bytes))
}

/// Points `header` at `record`, or at none: what a new header starts with.
pub(crate) fn set_migration_record<M: Memory, D: Memory>(
    store: &PageStore<M, D>,
    header: Extent,
    record: Option<Extent>,
) {
    let bytes = record.map_or([0; EXTENT_BYTES], |r| {
        let [r0, r1, r2, r3, r4] = r.encode();
        [r0, r1, r2, r3, r4 | MIGRATING]
    });
    store.write(migration_at(header), &bytes);
}

fn migration_at(header: Extent) -> ByteOffset {
    let within = u64::try_from(AT_MIGRATION).unwrap_or(0);
    ByteOffset::new(header.offset().get() + within)
}
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
        let [w0, w1, i0, i1, i2, i3, depth, version, m0, m1, m2, m3, m4] =
            <[u8; 13]>::try_from(rest).unwrap_or_else(|_| corrupt("header"));
        let slots = Descriptor::decode(d.try_into().unwrap_or_else(|_| corrupt("header")));
        let plane = Plane::decode(root.try_into().unwrap_or_else(|_| corrupt("header")));
        // The record's extent stays in the header (see `migration_record`).
        let _ = (m0, m1, m2, m3);
        let migration = if m4 & MIGRATING == 0 {
            Migration::Idle
        } else {
            Migration::UnderWay
        };
        let values = Width::new(u16::from_le_bytes([w0, w1])).map(|width| Values {
            plane,
            width,
            version: ValueVersion::new(version),
            migration,
        });
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
    pub(crate) fn write_header<M: Memory, D: Memory>(&self, store: &PageStore<M, D>) {
        let Some(header) = self.header else { return };
        let (plane, width, version) = self.values.map_or((Plane::Empty, 0, 0), |v| {
            (v.plane, v.width.get(), v.version.get())
        });
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
        let (fields, _) = rest.split_at_mut(8);
        fields.copy_from_slice(&[w0, w1, i0, i1, i2, i3, depth, version]);
        // The migration record's place is written apart (see `set_migration_record`).
        store.write(header.offset(), bytes.get(..AT_MIGRATION).unwrap_or(&[]));
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
