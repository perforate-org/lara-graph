//! Byte classes and the slot bitmaps of the chunk pages they are carved from.

use super::PAGE_BYTES;

/// An allocation size from 16 bytes to a whole page, stored as its id: 16, the powers of two from
/// 32 to 4096 and three quarters of each from 64 to 1024, ascending. Between 32 and 1024 a class
/// is at most 1.5 times the one below it, so an extent wastes at most a third of itself. Above, a
/// 1536-byte class would fit two to a page like 2048 and save nothing.
///
/// The 16-byte class holds one-entry label bucket lists and the runs of three or four slots that
/// small buckets grow into first. Its 256 extents a page outnumber a directory record's 128-bit
/// bitmap, so its pages are slabs instead (see [`Slots`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ByteClass(pub(super) u8);

/// Bytes of each class, by id; ids past a page's read as a page.
pub(super) const CLASS_BYTES: [u16; 16] = [
    16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 4096, 4096,
];
/// Extents of each class a page holds, by id.
pub(super) const CLASS_PER_PAGE: [u16; 14] = [256, 128, 85, 64, 42, 32, 21, 16, 10, 8, 5, 4, 2, 1];
/// Bytes of the link a freed slab extent holds: the next free extent's index.
pub(super) const SLAB_LINK_BYTES: usize = 2;
/// A slab link to no extent.
const NO_SLAB_LINK: u16 = u16::MAX;

impl ByteClass {
    pub(super) const PAGE_ID: u8 = 13;
    /// Chunk classes, i.e. classes below a page.
    pub(super) const CHUNK_CLASSES: u64 = 13;
    pub(crate) const PAGE: Self = Self(Self::PAGE_ID);
    /// The smallest class: 16 bytes.
    pub(crate) const SMALLEST: Self = Self(0);
    /// 32 bytes.
    pub(crate) const B32: Self = Self(1);
    /// 48 bytes: bucket headers.
    pub(crate) const B48: Self = Self(2);

    /// The smallest class holding `bytes`; `None` past a page.
    pub(crate) fn for_bytes(bytes: u32) -> Option<Self> {
        if bytes > PAGE_BYTES {
            return None;
        }
        if bytes <= 16 {
            return Some(Self::SMALLEST);
        }
        if bytes <= 32 {
            return Some(Self::B32);
        }
        // `bytes` lies in (2^(k-1), 2^k] with 6 <= k <= 12.
        let k = u8::try_from(bytes.next_power_of_two().trailing_zeros()).ok()?;
        let id = if k <= 10 {
            let power = 2 * (k - 5) + 1;
            if bytes <= 3 << (k - 2) {
                power - 1
            } else {
                power
            }
        } else {
            k + 1
        };
        Some(Self(id))
    }

    pub(crate) fn from_id(id: u8) -> Option<Self> {
        (id <= Self::PAGE_ID).then_some(Self(id))
    }

    pub(crate) const fn id(self) -> u8 {
        self.0
    }

    pub(crate) fn bytes(self) -> u32 {
        // Masked to the table's length, so the lookup needs no bounds check.
        CLASS_BYTES
            .get(usize::from(self.0 & 15))
            .map_or(PAGE_BYTES, |&bytes| u32::from(bytes))
    }

    pub(crate) const fn is_page(self) -> bool {
        self.0 == Self::PAGE_ID
    }

    /// Extents of this class per chunk page, the in-page bitmap's included.
    pub(super) fn per_page(self) -> u16 {
        CLASS_PER_PAGE
            .get(usize::from(self.0))
            .copied()
            .unwrap_or(1)
    }

    /// Whether the class's pages are slabs rather than bitmaps.
    pub(super) const fn is_slab(self) -> bool {
        self.0 == Self::SMALLEST.0
    }

    /// Position of this chunk class among the partial-list heads.
    pub(super) fn lane(self) -> u64 {
        u64::from(self.0)
    }
}

/// What an extent's 4-bit class code stands for, so that decoding one is a single table load.
#[derive(Clone, Copy)]
pub(super) struct Code {
    pub(super) class: ByteClass,
    /// The index bits above the extent's 7: 128 for the 16-byte class's code 1.
    pub(super) high: u16,
    /// The class's last index.
    pub(super) last: u16,
    pub(super) bytes: u16,
}

/// Codes 0 and 1 are the 16-byte class's, 2 to 14 the other ids' plus one, 15 a page's (no
/// extent's: it clamps). A test checks the table against [`CLASS_BYTES`] and [`CLASS_PER_PAGE`].
pub(super) const CODES: [Code; 16] = [
    code(0, 0, 255, 16),
    code(0, 128, 255, 16),
    code(1, 0, 127, 32),
    code(2, 0, 84, 48),
    code(3, 0, 63, 64),
    code(4, 0, 41, 96),
    code(5, 0, 31, 128),
    code(6, 0, 20, 192),
    code(7, 0, 15, 256),
    code(8, 0, 9, 384),
    code(9, 0, 7, 512),
    code(10, 0, 4, 768),
    code(11, 0, 3, 1024),
    code(12, 0, 1, 2048),
    code(13, 0, 0, 4096),
    code(13, 0, 0, 4096),
];

const fn code_of_page() -> Code {
    code(ByteClass::PAGE_ID, 0, 0, 4096)
}

const fn code(id: u8, high: u16, last: u16, bytes: u16) -> Code {
    Code {
        class: ByteClass(id),
        high,
        last,
        bytes,
    }
}

impl Code {
    /// The code in the low four bits of `bits`.
    pub(super) fn of(bits: u64) -> Self {
        let code = usize::try_from(bits & 15).unwrap_or(15);
        CODES.get(code).copied().unwrap_or(code_of_page())
    }
}

/// Which extents of a chunk page are free, as its directory record's 16 slot bytes hold them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slots {
    /// A bit per extent, set while it is allocated: classes of at most 128 extents a page.
    Bitmap(u128),
    /// The 16-byte class's 256 extents: those from `bump` on were never handed out, and freed ones
    /// form a list through their first [`SLAB_LINK_BYTES`] bytes from `head`. Allocating takes
    /// `bump` without reading anything, or pops the list with one two-byte read; freeing writes
    /// one link into the freed extent.
    Slab { head: Option<u16>, bump: u16 },
}

impl Slots {
    /// A fresh page's: nothing allocated.
    pub(super) const fn fresh(class: ByteClass) -> Self {
        if class.is_slab() {
            Self::Slab {
                head: None,
                bump: 0,
            }
        } else {
            Self::Bitmap(0)
        }
    }

    /// Slots of `class` from a record's slot bytes.
    pub(super) fn decode(class: ByteClass, raw: u128) -> Self {
        if !class.is_slab() {
            return Self::Bitmap(raw);
        }
        let head = u16::try_from(raw & 0xFFFF).unwrap_or(NO_SLAB_LINK);
        // At most 256 were handed out.
        let bump = u16::try_from((raw >> 16) & 0x1FF).unwrap_or(0).min(256);
        Self::Slab {
            head: decode_slab_link(head),
            bump,
        }
    }

    pub(super) fn encode(self) -> u128 {
        match self {
            Self::Bitmap(bits) => bits,
            Self::Slab { head, bump } => {
                u128::from(encode_slab_link(head)) | (u128::from(bump) << 16)
            }
        }
    }
}

pub(super) const fn encode_slab_link(link: Option<u16>) -> u16 {
    match link {
        Some(index) => index,
        None => NO_SLAB_LINK,
    }
}

pub(super) fn decode_slab_link(raw: u16) -> Option<u16> {
    (raw != NO_SLAB_LINK).then_some(raw)
}
