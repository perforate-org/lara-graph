//! Directory records: one per page. The only place that knows their wire layout; sentinel
//! values exist only here.
//!
//! # Record layout
//!
//! ```text
//! ---------------------------------------- <- Offset 0
//! Kind: 1 free, 2 chunk, 3 whole  ↕ 1 byte
//! ----------------------------------------
//! Class id (chunk)                ↕ 1 byte
//! ----------------------------------------
//! Extents in use (chunk)          ↕ 2 bytes
//! ----------------------------------------
//! Previous page (chunk)           ↕ 4 bytes
//! ----------------------------------------
//! Next page (free, chunk)         ↕ 4 bytes
//! ----------------------------------------
//! Slots (chunk)                   ↕ 16 bytes      (a bitmap; for the 16 B class a slab's free
//!                                                  list head, then its bump index)
//! ----------------------------------------
//! Reserved space                  ↕ 4 bytes
//! ----------------------------------------
//! ```

use super::{ByteClass, PageId};
use crate::store::corrupt::corrupt;

/// Bytes per directory record.
pub(super) const RECORD_BYTES: u64 = 32;

/// What a page is used for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PageRecord {
    /// On the free list.
    Free { next: Option<PageId> },
    /// Split into extents of one class; `bitmap` marks the allocated ones. While it has both
    /// used and free slots it is on its class's partial list (`prev`, `next`).
    Chunk {
        class: ByteClass,
        used: u16,
        prev: Option<PageId>,
        next: Option<PageId>,
        bitmap: u128,
    },
    /// One whole-page extent.
    Whole,
}

const NONE: u32 = u32::MAX;
const KIND_FREE: u8 = 1;
const KIND_CHUNK: u8 = 2;
const KIND_WHOLE: u8 = 3;

pub(super) fn encode_link(link: Option<PageId>) -> [u8; 4] {
    link.map_or(NONE, PageId::get).to_le_bytes()
}

pub(super) fn decode_link(bytes: [u8; 4]) -> Option<PageId> {
    let raw = u32::from_le_bytes(bytes);
    (raw != NONE).then_some(PageId::new(raw))
}

impl PageRecord {
    // As the module's Record layout diagram shows.
    pub(super) fn encode(self) -> [u8; 32] {
        let (kind, class, used, prev, next, bitmap) = match self {
            Self::Free { next } => (KIND_FREE, 0, 0, None, next, 0),
            Self::Chunk {
                class,
                used,
                prev,
                next,
                bitmap,
            } => (KIND_CHUNK, class.id(), used, prev, next, bitmap),
            Self::Whole => (KIND_WHOLE, 0, 0, None, None, 0),
        };
        // Laid out by hand: chaining the parts as iterators cost about 1,500 instructions a record
        // on the IC, a fifth of an insert into a hub (`hub_insert_1024`).
        let [u0, u1] = used.to_le_bytes();
        let [p0, p1, p2, p3] = encode_link(prev);
        let [n0, n1, n2, n3] = encode_link(next);
        let [
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
        ] = bitmap.to_le_bytes();
        [
            kind, class, u0, u1, p0, p1, p2, p3, n0, n1, n2, n3, b0, b1, b2, b3, b4, b5, b6, b7,
            b8, b9, b10, b11, b12, b13, b14, b15, 0, 0, 0, 0,
        ]
    }

    pub(super) fn decode(bytes: [u8; 32]) -> Self {
        let [
            kind,
            class,
            u0,
            u1,
            p0,
            p1,
            p2,
            p3,
            n0,
            n1,
            n2,
            n3,
            bitmap @ ..,
            _,
            _,
            _,
            _,
        ] = bytes;
        let next = decode_link([n0, n1, n2, n3]);
        match kind {
            KIND_FREE => Self::Free { next },
            KIND_WHOLE => Self::Whole,
            KIND_CHUNK => Self::Chunk {
                class: ByteClass::from_id(class)
                    .filter(|c| !c.is_page())
                    .unwrap_or_else(|| corrupt("chunk class")),
                used: u16::from_le_bytes([u0, u1]),
                prev: decode_link([p0, p1, p2, p3]),
                next,
                bitmap: u128::from_le_bytes(bitmap),
            },
            _ => corrupt("page record kind"),
        }
    }
}

/// Proofs over every input, checked by Kani (`cargo kani -p lara-graph`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Any page id an extent can name survives a link's encoding.
    #[kani::proof]
    fn links_round_trip() {
        let raw: u32 = kani::any();
        kani::assume(raw < crate::store::page_store::MAX_PAGES);
        let link = Some(PageId::new(raw));
        assert_eq!(decode_link(encode_link(link)), link);
        assert_eq!(decode_link(encode_link(None)), None);
    }
}
