//! Internal nodes' layout: the child count, then the children's first keys, pages and counts in
//! three arrays, so that a lookup reads only the keys it probes and the counts it sums.

use super::leaf::LEAF_BYTES;
use crate::corrupt::corrupt;
use crate::memory::Memory;
use crate::page_store::{PageId, PageStore, Pages};
use crate::units::ByteOffset;
use alloc::vec::Vec;

const NODE_HEADER: usize = 4;
/// Bytes a child takes across the three arrays.
const CHILD_BYTES: usize = 16;
/// Children per internal node: a page of them; tests use a few so that trees grow deep early.
pub(super) const FANOUT: usize =
    crate::scale::Scale::pick(8, (LEAF_BYTES - NODE_HEADER) / CHILD_BYTES);
const KEYS: usize = NODE_HEADER;
const PAGES: usize = KEYS + 8 * FANOUT;
const COUNTS: usize = PAGES + 4 * FANOUT;

/// A child as its parent lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Child {
    pub(super) first: (u32, u32),
    pub(super) page: PageId,
    pub(super) count: u32,
}

pub(super) fn at(page: PageId, offset: usize) -> ByteOffset {
    page.offset()
        .advance(u64::try_from(offset).unwrap_or(u64::MAX))
        .unwrap_or_else(|_| corrupt("index page offset"))
}

/// Fills `buf` from `offset` into `page`.
pub(super) fn read_into<M: Memory>(
    view: &Pages<'_, M>,
    page: PageId,
    offset: usize,
    buf: &mut [u8],
) {
    view.read_into(at(page, offset), buf);
}

fn u32_at<M: Memory>(view: &Pages<'_, M>, page: PageId, offset: usize) -> u32 {
    u32::from_le_bytes(view.read_array(at(page, offset)))
}

/// How many children an internal node lists.
pub(super) fn node_len<M: Memory>(view: &Pages<'_, M>, page: PageId) -> usize {
    usize::from(u16::from_le_bytes(view.read_array(page.offset())))
}

pub(super) fn key_at<M: Memory>(view: &Pages<'_, M>, page: PageId, i: usize) -> (u32, u32) {
    let [t0, t1, t2, t3, b0, b1, b2, b3] = view.read_array(at(page, KEYS + 8 * i));
    (
        u32::from_le_bytes([t0, t1, t2, t3]),
        u32::from_le_bytes([b0, b1, b2, b3]),
    )
}

pub(super) fn page_at<M: Memory>(view: &Pages<'_, M>, page: PageId, i: usize) -> PageId {
    PageId::new(u32_at(view, page, PAGES + 4 * i))
}

pub(super) fn count_at<M: Memory>(view: &Pages<'_, M>, page: PageId, i: usize) -> u32 {
    u32_at(view, page, COUNTS + 4 * i)
}

/// The counts of the first `n` children, with one read.
pub(super) fn counts<M: Memory>(view: &Pages<'_, M>, page: PageId, n: usize) -> Vec<u32> {
    let mut bytes = alloc::vec![0u8; 4 * n];
    read_into(view, page, COUNTS, &mut bytes);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| u32::from_le_bytes(c))
        .collect()
}

/// The child a key falls in: the last whose first key is at or below it (or the first).
pub(super) fn child_of<M: Memory>(
    view: &Pages<'_, M>,
    page: PageId,
    n: usize,
    key: (u32, u32),
) -> usize {
    let (mut low, mut high) = (0, n);
    while low < high {
        let mid = low + (high - low) / 2;
        if key_at(view, page, mid) <= key {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low.saturating_sub(1)
}

/// Every child of an internal node.
pub(super) fn children<M: Memory>(view: &Pages<'_, M>, page: PageId) -> Vec<Child> {
    let n = node_len(view, page);
    let mut keys = alloc::vec![0u8; 8 * n];
    let mut pages = alloc::vec![0u8; 4 * n];
    read_into(view, page, KEYS, &mut keys);
    read_into(view, page, PAGES, &mut pages);
    keys.as_chunks::<8>()
        .0
        .iter()
        .zip(pages.as_chunks::<4>().0)
        .zip(counts(view, page, n))
        .map(|(([t0, t1, t2, t3, b0, b1, b2, b3], p), count)| Child {
            first: (
                u32::from_le_bytes([*t0, *t1, *t2, *t3]),
                u32::from_le_bytes([*b0, *b1, *b2, *b3]),
            ),
            page: PageId::new(u32::from_le_bytes(*p)),
            count,
        })
        .collect()
}

/// Writes an internal node listing `children`.
pub(super) fn write_node<M: Memory>(store: &PageStore<M>, page: PageId, children: &[Child]) {
    let n = u16::try_from(children.len()).unwrap_or_else(|_| corrupt("index node size"));
    let [n0, n1] = n.to_le_bytes();
    store.write(page.offset(), &[n0, n1, 0, 0]);
    let keys: Vec<u8> = children
        .iter()
        .flat_map(|c| {
            c.first
                .0
                .to_le_bytes()
                .into_iter()
                .chain(c.first.1.to_le_bytes())
        })
        .collect();
    let pages: Vec<u8> = children
        .iter()
        .flat_map(|c| c.page.get().to_le_bytes())
        .collect();
    let counts: Vec<u8> = children
        .iter()
        .flat_map(|c| c.count.to_le_bytes())
        .collect();
    store.write(at(page, KEYS), &keys);
    store.write(at(page, PAGES), &pages);
    store.write(at(page, COUNTS), &counts);
}

/// Writes child `i`'s count.
pub(super) fn write_child<M: Memory>(store: &PageStore<M>, page: PageId, i: usize, count: u32) {
    store.write(at(page, COUNTS + 4 * i), &count.to_le_bytes());
}
