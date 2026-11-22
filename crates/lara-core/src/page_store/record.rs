//! Directory records: one per page. The only place that knows their wire layout; sentinel
//! values exist only here.

use super::{ByteClass, PageId};
use crate::corrupt::corrupt;

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
    // Layout: kind u8 | class u8 | used u16 | prev u32 | next u32 | bitmap u128 | reserved u32.
    pub(super) fn encode(self) -> [u8; 32] {
        let (kind, class, used, prev, next, bitmap) = match self {
            Self::Free { next } => (KIND_FREE, 0, 0, None, next, 0),
            Self::Chunk {
                class,
                used,
                prev,
                next,
                bitmap,
            } => (KIND_CHUNK, class.log2(), used, prev, next, bitmap),
            Self::Whole => (KIND_WHOLE, 0, 0, None, None, 0),
        };
        let bytes = [kind, class]
            .into_iter()
            .chain(used.to_le_bytes())
            .chain(encode_link(prev))
            .chain(encode_link(next))
            .chain(bitmap.to_le_bytes());
        let mut out = [0u8; 32];
        for (dst, src) in out.iter_mut().zip(bytes) {
            *dst = src;
        }
        out
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
                class: ByteClass::from_log2(class)
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
