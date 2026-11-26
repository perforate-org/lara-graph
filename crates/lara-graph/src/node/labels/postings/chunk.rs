//! A chunk's container: the low 16 bits of its members, as a sorted list or a bitmap.

use super::Change;
use crate::bucket::values::{self, Plane};
use crate::id::NodeId;
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::{ByteClass, EXTENT_BYTES, Extent, PageId, PageStore, Pages};
use crate::store::read_buf::ReadBuf;
use crate::store::txn::SpanTxn;
use crate::store::units::ByteOffset;
use alloc::vec::Vec;
use core::ops::ControlFlow;

pub(super) const CHUNK_CELL: usize = 11;
/// Members a list container holds: two bytes each, one page.
const MAX_LISTED: usize = 2048;
const EMPTY: u8 = 0;
const LISTED: u8 = 1;
const BITMAP: u8 = 2;
const _: () = assert!(EXTENT_BYTES <= 8);

/// A chunk's container.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Chunk {
    Empty,
    Listed { extent: Extent, members: usize },
    Bitmap { pages: [PageId; 2], members: usize },
}

impl Chunk {
    pub(super) fn read<M: Memory>(view: &Pages<'_, M>, chunks: Plane, chunk: u32) -> Self {
        let mut bytes = [0u8; CHUNK_CELL];
        values::read(view, chunks, super::width(CHUNK_CELL), chunk, &mut bytes);
        Self::decode(bytes)
    }

    pub(super) fn decode(bytes: [u8; CHUNK_CELL]) -> Self {
        let [m0, m1, kind, p0, p1, p2, p3, p4, p5, p6, p7] = bytes;
        let members = usize::from(u16::from_le_bytes([m0, m1])) + 1;
        match kind {
            EMPTY => Self::Empty,
            LISTED => Self::Listed {
                extent: Extent::decode([p0, p1, p2, p3, p4, p5]),
                members,
            },
            BITMAP => Self::Bitmap {
                pages: [
                    PageId::new(u32::from_le_bytes([p0, p1, p2, p3])),
                    PageId::new(u32::from_le_bytes([p4, p5, p6, p7])),
                ],
                members,
            },
            _ => corrupt("chunk kind"),
        }
    }

    pub(super) fn encode(self) -> [u8; CHUNK_CELL] {
        let less = |members: usize| u16::try_from(members - 1).unwrap_or(u16::MAX).to_le_bytes();
        match self {
            Self::Empty => [0; CHUNK_CELL],
            Self::Listed { extent, members } => {
                let [m0, m1] = less(members);
                let [e0, e1, e2, e3, e4, e5] = extent.encode();
                [m0, m1, LISTED, e0, e1, e2, e3, e4, e5, 0, 0]
            }
            Self::Bitmap { pages, members } => {
                let [m0, m1] = less(members);
                let [first, second] = pages;
                let [p0, p1, p2, p3] = first.get().to_le_bytes();
                let [q0, q1, q2, q3] = second.get().to_le_bytes();
                [m0, m1, BITMAP, p0, p1, p2, p3, q0, q1, q2, q3]
            }
        }
    }

    /// The container with `low` added, within [`super::PAGES_PER_ADD`] reserved pages.
    pub(super) fn added<M: Memory>(
        self,
        store: &mut PageStore<M>,
        low: u16,
    ) -> Result<Self, WriteError> {
        Ok(match self {
            Self::Empty => {
                let extent = store.allocate(ByteClass::SMALLEST)?;
                store.write(extent.offset(), &low.to_le_bytes());
                Self::Listed { extent, members: 1 }
            }
            Self::Listed { extent, members } => listed_add(store, extent, members, low)?,
            Self::Bitmap { pages, members } => {
                flip(store, pages, low, Change::Joins);
                Self::Bitmap {
                    pages,
                    members: members + 1,
                }
            }
        })
    }

    /// The container with `low` removed, in place. Never allocates.
    pub(super) fn removed<M: Memory>(self, store: &mut PageStore<M>, low: u16) -> Self {
        match self {
            Self::Empty => corrupt("a posting removed from an empty chunk"),
            Self::Listed { extent, members } => listed_remove(store, extent, members, low),
            Self::Bitmap { pages, members: 1 } => {
                for page in pages {
                    store.release(Extent::whole(page));
                }
                Self::Empty
            }
            Self::Bitmap { pages, members } => {
                flip(store, pages, low, Change::Leaves);
                Self::Bitmap {
                    pages,
                    members: members - 1,
                }
            }
        }
    }

    /// Calls `f` with each member at or past id `from`, ascending; `base` is the chunk's first id.
    pub(super) fn visit<M: Memory, B>(
        self,
        view: &Pages<'_, M>,
        (base, from): (u32, u32),
        f: &mut impl FnMut(NodeId) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        let mut emit = |low: u32| {
            let id = base | low;
            if id < from {
                return ControlFlow::Continue(());
            }
            f(NodeId::new(id))
        };
        match self {
            Self::Empty => ControlFlow::Continue(()),
            Self::Listed { extent, members } => {
                let mut buf = ReadBuf::new();
                buf.fill(view, extent.offset(), members * 2)
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .try_for_each(|&low| emit(u32::from(u16::from_le_bytes(low))))
            }
            Self::Bitmap { pages, .. } => (0u32..)
                .zip(pages)
                .try_for_each(|(half, page)| bitmap_page(view, page, half << 15, &mut emit)),
        }
    }

    /// The extents the container owns (test invariant checks).
    #[cfg(test)]
    pub(super) fn extents(self) -> Vec<Extent> {
        match self {
            Self::Empty => Vec::new(),
            Self::Listed { extent, .. } => alloc::vec![extent],
            Self::Bitmap { pages, .. } => pages.map(Extent::whole).to_vec(),
        }
    }
}

/// Calls `emit` with each set bit of one bitmap page, lowest first, offset by `base`.
fn bitmap_page<M: Memory, B>(
    view: &Pages<'_, M>,
    page: PageId,
    base: u32,
    emit: &mut impl FnMut(u32) -> ControlFlow<B>,
) -> ControlFlow<B> {
    let bytes = view.read(page.offset(), 4096);
    for (byte_at, &byte) in (0u32..).zip(&bytes) {
        let mut rest = byte;
        while rest != 0 {
            emit(base | (byte_at << 3) | rest.trailing_zeros())?;
            rest &= rest - 1;
        }
    }
    ControlFlow::Continue(())
}

/// Sets `low`'s bit in a bitmap as it joins, clears it as it leaves.
fn flip<M: Memory>(store: &PageStore<M>, pages: [PageId; 2], low: u16, change: Change) {
    let low = u32::from(low);
    let [first, second] = pages;
    let page = if low >> 15 == 0 { first } else { second };
    let at = ByteOffset::new(page.offset().get() + u64::from((low & 0x7FFF) >> 3));
    let mask = 1u8 << (low & 7);
    let [byte] = store.pages().read_array(at);
    let byte = match change {
        Change::Joins => byte | mask,
        Change::Leaves => byte & !mask,
    };
    store.write(at, &[byte]);
}

/// Where member `i` of a list container is.
fn lows_at(extent: Extent, i: usize) -> ByteOffset {
    ByteOffset::new(extent.offset().get() + u64::try_from(i * 2).unwrap_or(0))
}

/// Where `low` is, or belongs, among a list's members.
fn position(members: &[u8], low: u16) -> Result<usize, usize> {
    members
        .as_chunks::<2>()
        .0
        .binary_search_by(|&l| u16::from_le_bytes(l).cmp(&low))
}

/// A list container with `low` added: at the end in place when it is the largest and there is
/// room (one 2-byte read and write), else into a larger extent or the middle, or a bitmap once
/// it would pass [`MAX_LISTED`].
fn listed_add<M: Memory>(
    store: &mut PageStore<M>,
    extent: Extent,
    members: usize,
    low: u16,
) -> Result<Chunk, WriteError> {
    let has_room = (members + 1) * 2 <= usize::try_from(extent.len()).unwrap_or(0);
    let last = u16::from_le_bytes(store.pages().read_array(lows_at(extent, members - 1)));
    if has_room && last < low {
        store.write(lows_at(extent, members), &low.to_le_bytes());
        return Ok(Chunk::Listed {
            extent,
            members: members + 1,
        });
    }
    if members + 1 > MAX_LISTED {
        return to_bitmap(store, extent, members, low);
    }
    let mut buf = ReadBuf::new();
    let old = buf.fill(&store.pages(), extent.offset(), members * 2);
    let at = position(old, low).unwrap_or_else(|at| at);
    // The new member, then the ones after it: all that moves.
    let tail: Vec<u8> = low
        .to_le_bytes()
        .into_iter()
        .chain(old.get(at * 2..).unwrap_or(&[]).iter().copied())
        .collect();
    let target = if has_room {
        extent
    } else {
        let len = u32::try_from((members + 1) * 2).unwrap_or(u32::MAX);
        let grown = store.allocate(ByteClass::for_bytes(len).unwrap_or(ByteClass::PAGE))?;
        store.write(grown.offset(), old.get(..at * 2).unwrap_or(&[]));
        grown
    };
    store.write(lows_at(target, at), &tail);
    if target != extent {
        store.release(extent);
    }
    Ok(Chunk::Listed {
        extent: target,
        members: members + 1,
    })
}

/// A list container with `low` removed: the members after it move down one.
fn listed_remove<M: Memory>(
    store: &mut PageStore<M>,
    extent: Extent,
    members: usize,
    low: u16,
) -> Chunk {
    if members == 1 {
        store.release(extent);
        return Chunk::Empty;
    }
    let mut buf = ReadBuf::new();
    let old = buf.fill(&store.pages(), extent.offset(), members * 2);
    let at = position(old, low).unwrap_or_else(|_| corrupt("a posting removed that is not there"));
    let tail = old.get((at + 1) * 2..).unwrap_or(&[]).to_vec();
    store.write(lows_at(extent, at), &tail);
    Chunk::Listed {
        extent,
        members: members - 1,
    }
}

/// A full list container, with `low` added, as a bitmap of its chunk.
fn to_bitmap<M: Memory>(
    store: &mut PageStore<M>,
    extent: Extent,
    members: usize,
    low: u16,
) -> Result<Chunk, WriteError> {
    let mut txn = SpanTxn::begin(store);
    let pages = [
        txn.allocate(ByteClass::PAGE)?.page(),
        txn.allocate(ByteClass::PAGE)?.page(),
    ];
    txn.commit();
    let mut bits = [[0u8; 4096]; 2];
    let mut set = |l: u16| {
        let l = usize::from(l);
        if let Some(b) = bits
            .get_mut(l >> 15)
            .and_then(|p| p.get_mut((l & 0x7FFF) >> 3))
        {
            *b |= 1 << (l & 7);
        }
    };
    let old = store.pages().read(extent.offset(), members * 2);
    for &l in old.as_chunks::<2>().0 {
        set(u16::from_le_bytes(l));
    }
    set(low);
    for (page, half) in pages.iter().zip(&bits) {
        store.write(page.offset(), half);
    }
    store.release(extent);
    Ok(Chunk::Bitmap {
        pages,
        members: members + 1,
    })
}

/// Proofs over every input, checked by Kani (`cargo kani -p lara-graph`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Any cell of a known kind decodes to a container that encodes back to itself.
    #[kani::proof]
    fn chunk_cells_round_trip() {
        let bytes: [u8; CHUNK_CELL] = kani::any();
        kani::assume(bytes[2] == EMPTY || bytes[2] == LISTED || bytes[2] == BITMAP);
        let chunk = Chunk::decode(bytes);
        assert!(Chunk::decode(chunk.encode()) == chunk);
    }
}
