//! How a bucket is stored in its list entry: a plain descriptor, or a header for a bucket whose
//! edges carry values.
//!
//! The 16-byte list entry holds a [`Descriptor`] for a bucket without values. A bucket with values
//! (never tiny: three inline slots leave no room for them) holds `Valued`, the extent of a
//! 32-byte header in the edge store: the slot descriptor, the value plane's root and the width.
//! Buckets without values pay nothing; a bucket with values costs one more 32-byte read.

use crate::corrupt::corrupt;
use crate::descriptor::{DESCRIPTOR_BYTES, Descriptor};
use crate::id::Width;
use crate::memory::Memory;
use crate::page_store::{ByteClass, EXTENT_BYTES, Extent, PageStore, Pages};
use crate::values::{Plane, ROOT_BYTES};

/// The descriptor kind byte of a header reference; descriptor kinds are 1 to 3.
const KIND_VALUED: u8 = 4;
/// Bytes of a header: descriptor, plane root, width.
const HEADER_BYTES: usize = DESCRIPTOR_BYTES + ROOT_BYTES + 2;
/// The class headers are allocated in.
pub(crate) const HEADER_CLASS: ByteClass = ByteClass::SMALLEST;
const _: () = assert!(HEADER_BYTES <= 32);

/// A list entry's 16 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stored {
    Plain(Descriptor),
    Valued(Extent),
}

impl Stored {
    pub(crate) fn encode(&self) -> [u8; DESCRIPTOR_BYTES] {
        match self {
            Self::Plain(d) => d.encode(),
            Self::Valued(header) => {
                let mut out = [0u8; DESCRIPTOR_BYTES];
                for (dst, src) in out
                    .iter_mut()
                    .zip([KIND_VALUED].into_iter().chain(header.encode()))
                {
                    *dst = src;
                }
                out
            }
        }
    }

    pub(crate) fn decode(bytes: [u8; DESCRIPTOR_BYTES]) -> Self {
        let [kind, e0, e1, e2, e3, e4, e5, e6, ..] = bytes;
        if kind == KIND_VALUED {
            Self::Valued(Extent::decode([e0, e1, e2, e3, e4, e5, e6]))
        } else {
            Self::Plain(Descriptor::decode(bytes))
        }
    }
}

/// A bucket's values: where its header is, the plane and the width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Values {
    pub(crate) header: Extent,
    pub(crate) plane: Plane,
    pub(crate) width: Width,
}

/// A bucket resolved from its list entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bucket {
    pub(crate) slots: Descriptor,
    pub(crate) values: Option<Values>,
}

impl Bucket {
    /// Reads a valued bucket's header (one 32-byte read); a plain one needs nothing.
    pub(crate) fn resolve<M: Memory>(view: &Pages<'_, M>, stored: Stored) -> Self {
        match stored {
            Stored::Plain(slots) => Self {
                slots,
                values: None,
            },
            Stored::Valued(header) => {
                let bytes: [u8; HEADER_BYTES] = view.read_array(header.offset());
                let (d, rest) = bytes.split_at(DESCRIPTOR_BYTES);
                let (root, width) = rest.split_at(ROOT_BYTES);
                let slots = Descriptor::decode(d.try_into().unwrap_or_else(|_| corrupt("header")));
                let plane = Plane::decode(root.try_into().unwrap_or_else(|_| corrupt("header")));
                let width = width
                    .try_into()
                    .ok()
                    .and_then(|w| Width::new(u16::from_le_bytes(w)))
                    .unwrap_or_else(|| corrupt("header width"));
                Self {
                    slots,
                    values: Some(Values {
                        header,
                        plane,
                        width,
                    }),
                }
            }
        }
    }

    /// What the list entry holds for this bucket.
    pub(crate) const fn stored(&self) -> Stored {
        match self.values {
            None => Stored::Plain(self.slots),
            Some(values) => Stored::Valued(values.header),
        }
    }

    /// Writes a valued bucket's header. A plain bucket's state is its list entry.
    pub(crate) fn write_header<M: Memory>(&self, store: &PageStore<M>) {
        if let Some(values) = self.values {
            let bytes: alloc::vec::Vec<u8> = self
                .slots
                .encode()
                .into_iter()
                .chain(values.plane.encode())
                .chain(values.width.get().to_le_bytes())
                .collect();
            store.write(values.header.offset(), &bytes);
        }
    }
}

const _: () = assert!(EXTENT_BYTES < DESCRIPTOR_BYTES);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::Tiny;
    use crate::page_store::PageId;

    #[test]
    fn entries_round_trip_plain_and_valued() {
        let plain = Stored::Plain(Descriptor::Tiny(Tiny::EMPTY));
        let valued = Stored::Valued(Extent::whole(PageId::new(7)));
        for stored in [plain, valued] {
            assert_eq!(Stored::decode(stored.encode()), stored);
        }
    }
}
