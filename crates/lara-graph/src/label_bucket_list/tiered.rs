//! Tiered bucket lists: a top extent of leaf references over pages of sorted entries.
//!
//! A lookup reads the top and one leaf. Adding to a full leaf splits it into two half-full pages;
//! removing a leaf's last entry releases the page, and a list down to one leaf turns flat again.
//!
//! # Top
//!
//! ```text
//! ---------------------------------------- <- Extent start
//! Leaf count = L                  ↕ 2 bytes
//! ---------------------------------------- <- Offset 2
//! Leaf 0: first label             ↕ 2 bytes
//! ----------------------------------------
//! Leaf 0: first direction         ↕ 1 byte
//! ----------------------------------------
//! Leaf 0: page id                 ↕ 4 bytes
//! ----------------------------------------
//! Leaf 0: entry count             ↕ 1 byte
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Leaf L - 1                      ↕ 8 bytes
//! ----------------------------------------
//! Unused space
//! ```
//!
//! # Leaf page
//!
//! ```text
//! ---------------------------------------- <- Page start
//! Entry 0                         ↕ 19 bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Entry count - 1 (at most 214)   ↕ 19 bytes
//! ----------------------------------------
//! Unused space
//! ```

use super::{
    EntryAt, LEAF_ENTRIES, LabelBucketKey, LabelBucketList, Listed, Lookup, Shape, class_len,
    encode_entry, flat, read_entries, search, shift_in, shift_out,
};
use crate::corrupt::corrupt;
use crate::memory::{Memory, WriteError};
use crate::page_store::{ByteClass, Extent, PageId, PageStore, Pages};
use crate::stored::Stored;
use crate::txn::SpanTxn;
use crate::units::ByteOffset;
use alloc::vec::Vec;
use core::ops::ControlFlow;

const TOP_HEADER_BYTES: usize = 2;
const TOP_ENTRY_BYTES: usize = 8;
/// Leaves one list can have: one page of top entries. Tests use a handful.
const TOP_ENTRIES: usize =
    crate::scale::Scale::pick(6, (4096 - TOP_HEADER_BYTES) / TOP_ENTRY_BYTES);

/// One leaf page as the top lists it.
#[derive(Clone, Copy)]
struct Leaf {
    first: LabelBucketKey,
    page: PageId,
    count: u8,
}

impl Leaf {
    fn encode(self) -> [u8; TOP_ENTRY_BYTES] {
        let [k0, k1, k2] = self.first.encode();
        let [p0, p1, p2, p3] = self.page.get().to_le_bytes();
        [k0, k1, k2, p0, p1, p2, p3, self.count]
    }

    fn decode([k0, k1, k2, p0, p1, p2, p3, count]: [u8; TOP_ENTRY_BYTES]) -> Self {
        Self {
            first: LabelBucketKey::decode([k0, k1, k2]),
            page: PageId::new(u32::from_le_bytes([p0, p1, p2, p3])),
            count,
        }
    }

    fn entries(self) -> usize {
        usize::from(self.count)
    }
}

const fn tiered(top: Extent) -> LabelBucketList {
    LabelBucketList(Shape::Tiered { top })
}

fn read_top<M: Memory>(pages: &Pages<'_, M>, top: Extent) -> Vec<Leaf> {
    let leaves = usize::from(u16::from_le_bytes(pages.read_array(top.offset())));
    let bytes = pages.read(leaf_at(top, 0), leaves * TOP_ENTRY_BYTES);
    bytes
        .as_chunks::<TOP_ENTRY_BYTES>()
        .0
        .iter()
        .map(|&raw| Leaf::decode(raw))
        .collect()
}

fn leaf_at(top: Extent, i: usize) -> ByteOffset {
    let within = TOP_HEADER_BYTES + i * TOP_ENTRY_BYTES;
    ByteOffset::new(top.offset().get() + u64::try_from(within).unwrap_or(u64::MAX))
}

fn leaf(leaves: &[Leaf], i: usize) -> Leaf {
    leaves
        .get(i)
        .copied()
        .unwrap_or_else(|| corrupt("bucket list leaf"))
}

fn top_class(leaves: usize) -> ByteClass {
    let bytes = TOP_HEADER_BYTES + leaves * TOP_ENTRY_BYTES;
    ByteClass::for_bytes(u32::try_from(bytes).unwrap_or(u32::MAX))
        .unwrap_or_else(|| corrupt("bucket list top beyond a page"))
}

/// Rewrites the whole top. It must fit `top`'s class.
fn write_top<M: Memory>(store: &PageStore<M>, top: Extent, leaves: &[Leaf]) {
    let count = u16::try_from(leaves.len()).unwrap_or_else(|_| corrupt("bucket list leaves"));
    let bytes: Vec<u8> = count
        .to_le_bytes()
        .into_iter()
        .chain(leaves.iter().flat_map(|l| l.encode()))
        .collect();
    store.write(top.offset(), &bytes);
}

fn write_entries<M: Memory>(
    store: &PageStore<M>,
    page: PageId,
    entries: &[(LabelBucketKey, Stored)],
) {
    let mut bytes = Vec::with_capacity(entries.len() * super::ENTRY_BYTES);
    for (k, d) in entries {
        bytes.extend_from_slice(&encode_entry(*k, d));
    }
    store.write(page.offset(), &bytes);
}

/// The leaf for `entries`, a non-empty sorted run written to `page`.
fn leaf_of(page: PageId, entries: &[(LabelBucketKey, Stored)]) -> Leaf {
    let first = entries
        .first()
        .map_or_else(|| corrupt("an empty leaf"), |(k, _)| *k);
    let count = u8::try_from(entries.len()).unwrap_or_else(|_| corrupt("leaf beyond a page"));
    Leaf { first, page, count }
}

pub(super) fn lookup<M: Memory>(pages: &Pages<'_, M>, top: Extent, key: LabelBucketKey) -> Lookup {
    let leaves = read_top(pages, top);
    // The last leaf starting at or before `key`; keys before every leaf belong to the first.
    let i = leaves.partition_point(|l| l.first <= key).saturating_sub(1);
    let found = leaf(&leaves, i);
    let index = u16::try_from(i).unwrap_or_else(|_| corrupt("bucket list leaves"));
    search(pages, found.page.offset(), found.entries(), key, index)
}

/// Where the first leaf's entries start.
pub(super) fn first_leaf<M: Memory>(pages: &Pages<'_, M>, top: Extent) -> ByteOffset {
    Leaf::decode(pages.read_array(leaf_at(top, 0)))
        .page
        .offset()
}

pub(super) fn has_room<M: Memory>(pages: &Pages<'_, M>, top: Extent, adding: usize) -> bool {
    let leaves = usize::from(u16::from_le_bytes(pages.read_array(top.offset())));
    leaves + adding <= TOP_ENTRIES
}

/// Every leaf's entries, in order, until `f` breaks.
pub(super) fn for_each<M: Memory, B>(
    pages: &Pages<'_, M>,
    top: Extent,
    f: &mut impl FnMut(LabelBucketKey, Listed) -> ControlFlow<B>,
) -> Option<B> {
    for leaf in read_top(pages, top) {
        if let Some(value) = super::for_each_in(pages, leaf.page.offset(), leaf.entries(), f) {
            return Some(value);
        }
    }
    None
}

#[cfg(test)]
pub(super) fn check_shape<M: Memory>(pages: &Pages<'_, M>, top: Extent) {
    let leaves = read_top(pages, top);
    assert!(
        (2..=TOP_ENTRIES).contains(&leaves.len()),
        "{} leaves",
        leaves.len()
    );
    for l in leaves {
        assert!(
            (1..=LEAF_ENTRIES).contains(&l.entries()),
            "leaf of {}",
            l.count
        );
    }
}

#[cfg(test)]
pub(super) fn extents<M: Memory>(pages: &Pages<'_, M>, top: Extent) -> Vec<Extent> {
    core::iter::once(top)
        .chain(
            read_top(pages, top)
                .into_iter()
                .map(|l| Extent::whole(l.page)),
        )
        .collect()
}

/// A tiered list of two half-full leaves holding `all`, sorted. On error nothing is allocated.
pub(super) fn from_entries<M: Memory>(
    store: &mut PageStore<M>,
    all: &[(LabelBucketKey, Stored)],
) -> Result<LabelBucketList, WriteError> {
    let mut txn = SpanTxn::begin(store);
    let left_page = txn.allocate(ByteClass::PAGE)?.page();
    let right_page = txn.allocate(ByteClass::PAGE)?.page();
    let top = txn.allocate(top_class(2))?;
    txn.commit();
    let (left, right) = all.split_at(all.len() / 2);
    write_entries(store, left_page, left);
    write_entries(store, right_page, right);
    write_top(
        store,
        top,
        &[leaf_of(left_page, left), leaf_of(right_page, right)],
    );
    Ok(tiered(top))
}

pub(super) fn add<M: Memory>(
    store: &mut PageStore<M>,
    top: Extent,
    gap: EntryAt,
    key: LabelBucketKey,
    d: &Stored,
) -> Result<LabelBucketList, WriteError> {
    let mut leaves = read_top(&store.pages(), top);
    let l = usize::from(gap.leaf);
    let target = leaf(&leaves, l);
    if target.entries() < LEAF_ENTRIES {
        shift_in(
            store,
            target.page.offset(),
            target.entries(),
            gap.index,
            (key, d),
        );
        let first = if gap.index == 0 { key } else { target.first };
        let next = Leaf {
            first,
            count: target.count + 1,
            ..target
        };
        store.write(leaf_at(top, l), &next.encode());
        return Ok(tiered(top));
    }
    // A full leaf splits; allocate first so that an error changes nothing.
    let grows = TOP_HEADER_BYTES + (leaves.len() + 1) * TOP_ENTRY_BYTES > class_len(top);
    let mut txn = SpanTxn::begin(store);
    let page = txn.allocate(ByteClass::PAGE)?.page();
    let new_top = if grows {
        Some(txn.allocate(top_class(leaves.len() + 1))?)
    } else {
        None
    };
    txn.commit();
    let mut all = read_entries(&store.pages(), target.page.offset(), target.entries());
    all.insert(gap.index.min(all.len()), (key, *d));
    let (left, right) = all.split_at(all.len() / 2);
    write_entries(store, target.page, left);
    write_entries(store, page, right);
    if let Some(slot) = leaves.get_mut(l) {
        *slot = leaf_of(target.page, left);
    }
    leaves.insert(l + 1, leaf_of(page, right));
    let Some(bigger) = new_top else {
        write_top(store, top, &leaves);
        return Ok(tiered(top));
    };
    write_top(store, bigger, &leaves);
    store.release(top);
    Ok(tiered(bigger))
}

/// Removes `entry`. A leaf left empty is released; a list left with one leaf turns flat.
pub(super) fn remove<M: Memory>(
    store: &mut PageStore<M>,
    top: Extent,
    entry: EntryAt,
) -> LabelBucketList {
    let mut leaves = read_top(&store.pages(), top);
    let l = usize::from(entry.leaf);
    let target = leaf(&leaves, l);
    if target.count > 1 {
        shift_out(store, target.page.offset(), target.entries(), entry.index);
        let first = if entry.index == 0 {
            let (key, _) = read_entries(&store.pages(), target.page.offset(), 1)
                .first()
                .copied()
                .unwrap_or_else(|| corrupt("an empty leaf"));
            key
        } else {
            target.first
        };
        let next = Leaf {
            first,
            count: target.count - 1,
            ..target
        };
        store.write(leaf_at(top, l), &next.encode());
        return tiered(top);
    }
    store.release(Extent::whole(target.page));
    leaves.remove(l);
    if let [only] = leaves.as_slice() {
        store.release(top);
        return flat(Extent::whole(only.page), only.count);
    }
    write_top(store, top, &leaves);
    tiered(top)
}
