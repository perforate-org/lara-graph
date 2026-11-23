//! Per-vertex bucket lists: which `(label, side)` buckets a vertex has, and their descriptors.
//!
//! Entries are 19 bytes (label, side, 16-byte descriptor), sorted by key. A list starts **flat**:
//! one extent in its own page store, read whole and searched in memory; adding a bucket shifts
//! entries in place or moves the list to the next class. A flat list that would outgrow a page
//! becomes **tiered** (`tiered`): a small top extent lists leaf pages with their first key and
//! count, and each leaf holds a page of entries. Every update copies at most one page. Lists are
//! independent: changing one never touches another vertex.

mod tiered;

use crate::corrupt::corrupt;
use crate::descriptor::DESCRIPTOR_BYTES;
use crate::id::{Label, Side};
use crate::memory::{Memory, WriteError};
use crate::page_store::{ByteClass, EXTENT_BYTES, Extent, PageStore, Pages};
use crate::stored::Stored;
use crate::units::ByteOffset;
use alloc::vec::Vec;

/// Bytes per list entry.
const ENTRY_BYTES: usize = 3 + DESCRIPTOR_BYTES;
/// Entries a flat list or a leaf holds: one page. Tests use a handful so that lists tier early.
const LEAF_ENTRIES: usize = if cfg!(test) { 8 } else { 4096 / ENTRY_BYTES };
const _: () = assert!(crate::page_store::PAGE_BYTES == 4096);
/// Pages one [`add`] can take from the store: tiering takes two leaves and a top.
pub(crate) const PAGES_PER_ADD: u32 = 3;
/// Bytes of a vertex's list reference.
pub(crate) const LIST_REF_BYTES: usize = EXTENT_BYTES + 1;
/// The count byte that marks a tiered list; flat counts stay below it.
const TIERED: u8 = u8::MAX;
// Flat counts never reach [`TIERED`] (0xFF).
const _: () = assert!(LEAF_ENTRIES < 0xFF);

/// Which bucket of a vertex: a label on one side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BucketKey {
    pub(crate) label: Label,
    pub(crate) side: Side,
}

const SIDE_OUT: u8 = 0;
const SIDE_IN: u8 = 1;
const SIDE_UNDIRECTED: u8 = 2;

impl BucketKey {
    fn encode(self) -> [u8; 3] {
        let [l0, l1] = self.label.get().to_le_bytes();
        let side = match self.side {
            Side::Out => SIDE_OUT,
            Side::In => SIDE_IN,
            Side::Undirected => SIDE_UNDIRECTED,
        };
        [l0, l1, side]
    }

    fn decode([l0, l1, side]: [u8; 3]) -> Self {
        let side = match side {
            SIDE_OUT => Side::Out,
            SIDE_IN => Side::In,
            SIDE_UNDIRECTED => Side::Undirected,
            _ => corrupt("bucket side"),
        };
        Self {
            label: Label::new(u16::from_le_bytes([l0, l1])),
            side,
        }
    }
}

/// Where a vertex's bucket list is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BucketList(Shape);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// One extent of `count` entries.
    Flat { extent: Extent, count: u8 },
    /// A top extent listing leaf pages.
    Tiered { top: Extent },
}

/// Where an entry is, or where a new one belongs: leaf `leaf` (0 for a flat list), entry `index`
/// in it, at byte offset `at`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EntryAt {
    leaf: u16,
    index: usize,
    at: ByteOffset,
}

/// The result of looking a key up in a list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lookup {
    /// The bucket exists.
    Present { entry: EntryAt, stored: Stored },
    /// No bucket; a new entry belongs at `gap`.
    Absent { gap: EntryAt },
}

/// Layout: extent (7) | count u8, [`TIERED`] for a tiered list. An empty list is all zeros.
pub(crate) fn encode_ref(list: Option<BucketList>) -> [u8; LIST_REF_BYTES] {
    let mut out = [0u8; LIST_REF_BYTES];
    let encoded = list.map(|BucketList(shape)| match shape {
        Shape::Flat { extent, count } => (extent, count),
        Shape::Tiered { top } => (top, TIERED),
    });
    if let Some((extent, count)) = encoded {
        for (dst, src) in out
            .iter_mut()
            .zip(extent.encode().into_iter().chain([count]))
        {
            *dst = src;
        }
    }
    out
}

pub(crate) fn decode_ref(bytes: [u8; LIST_REF_BYTES]) -> Option<BucketList> {
    let [e0, e1, e2, e3, e4, e5, e6, count] = bytes;
    (count != 0).then(|| {
        let extent = Extent::decode([e0, e1, e2, e3, e4, e5, e6]);
        BucketList(match count {
            TIERED => Shape::Tiered { top: extent },
            count => Shape::Flat { extent, count },
        })
    })
}

/// Finds a bucket with one read of a flat list, or a top and a leaf read of a tiered one.
pub(crate) fn lookup<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<BucketList>,
    key: BucketKey,
) -> Lookup {
    match list.map(|BucketList(shape)| shape) {
        None => Lookup::Absent {
            gap: EntryAt {
                leaf: 0,
                index: 0,
                at: ByteOffset::new(0),
            },
        },
        Some(Shape::Flat { extent, count }) => {
            search(pages, extent.offset(), usize::from(count), key, 0)
        }
        Some(Shape::Tiered { top }) => tiered::lookup(pages, top, key),
    }
}

/// Searches the `count` sorted entries at `base`, leaf `leaf` of their list.
fn search<M: Memory>(
    pages: &Pages<'_, M>,
    base: ByteOffset,
    count: usize,
    key: BucketKey,
    leaf: u16,
) -> Lookup {
    let place = |index| EntryAt {
        leaf,
        index,
        at: entry_offset(base, index),
    };
    pages.with_read(base, count * ENTRY_BYTES, |bytes| {
        let (entries, _) = bytes.as_chunks::<ENTRY_BYTES>();
        let found =
            entries.binary_search_by(|&[k0, k1, k2, ..]| BucketKey::decode([k0, k1, k2]).cmp(&key));
        match found {
            Ok(index) => Lookup::Present {
                entry: place(index),
                stored: entries
                    .get(index)
                    .map_or_else(|| corrupt("list entry"), |&e| decode_entry(e).1),
            },
            Err(index) => Lookup::Absent { gap: place(index) },
        }
    })
}

/// A vertex's first bucket in key order, with one small read (two for a tiered list).
pub(crate) fn first<M: Memory>(
    pages: &Pages<'_, M>,
    list: BucketList,
) -> (EntryAt, BucketKey, Stored) {
    let (leaf_page, leaf) = match list.0 {
        Shape::Flat { extent, .. } => (extent.offset(), 0),
        Shape::Tiered { top } => (tiered::first_leaf(pages, top), 0),
    };
    let (key, descriptor) = decode_entry(pages.read_array(leaf_page));
    let entry = EntryAt {
        leaf,
        index: 0,
        at: leaf_page,
    };
    (entry, key, descriptor)
}

/// Whether `adding` more buckets fit. Flat lists always make room by tiering; a tiered list
/// can split a leaf per bucket while its top has entries to spare.
pub(crate) fn has_room<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<BucketList>,
    adding: usize,
) -> bool {
    match list.map(|BucketList(shape)| shape) {
        None | Some(Shape::Flat { .. }) => true,
        Some(Shape::Tiered { top }) => tiered::has_room(pages, top, adding),
    }
}

/// The keys of a vertex's buckets, in order.
pub(crate) fn keys<M: Memory>(pages: &Pages<'_, M>, list: Option<BucketList>) -> Vec<BucketKey> {
    entries(pages, list).into_iter().map(|(k, _)| k).collect()
}

/// Every entry of a list, in order.
pub(crate) fn entries<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<BucketList>,
) -> Vec<(BucketKey, Stored)> {
    match list.map(|BucketList(shape)| shape) {
        None => Vec::new(),
        Some(Shape::Flat { extent, count }) => {
            read_entries(pages, extent.offset(), usize::from(count))
        }
        Some(Shape::Tiered { top }) => tiered::entries(pages, top),
    }
}

fn read_entries<M: Memory>(
    pages: &Pages<'_, M>,
    base: ByteOffset,
    count: usize,
) -> Vec<(BucketKey, Stored)> {
    let bytes = pages.read(base, count * ENTRY_BYTES);
    bytes
        .as_chunks::<ENTRY_BYTES>()
        .0
        .iter()
        .map(|&entry| decode_entry(entry))
        .collect()
}

/// Every bucket of a vertex (test invariant checks).
#[cfg(test)]
pub(crate) fn buckets<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<BucketList>,
) -> Vec<(BucketKey, Stored)> {
    entries(pages, list)
}

/// Panics unless the list's shape is within its limits: a flat list or each leaf holds 1 to
/// [`LEAF_ENTRIES`] entries, and a tiered list has 2 to `TOP_ENTRIES` leaves (test checks).
#[cfg(test)]
pub(crate) fn check_shape<M: Memory>(pages: &Pages<'_, M>, list: BucketList) {
    match list.0 {
        Shape::Flat { count, .. } => {
            assert!(
                (1..=LEAF_ENTRIES).contains(&usize::from(count)),
                "flat count {count}"
            );
        }
        Shape::Tiered { top } => tiered::check_shape(pages, top),
    }
}

/// Every extent a list owns (test invariant checks).
#[cfg(test)]
pub(crate) fn extents<M: Memory>(pages: &Pages<'_, M>, list: BucketList) -> Vec<Extent> {
    match list.0 {
        Shape::Flat { extent, .. } => alloc::vec![extent],
        Shape::Tiered { top } => tiered::extents(pages, top),
    }
}

fn decode_entry(entry: [u8; ENTRY_BYTES]) -> (BucketKey, Stored) {
    let [k0, k1, k2, rest @ ..] = entry;
    (BucketKey::decode([k0, k1, k2]), Stored::decode(rest))
}

fn encode_entry(key: BucketKey, d: &Stored) -> impl Iterator<Item = u8> {
    key.encode().into_iter().chain(d.encode())
}

/// Publishes a bucket's new descriptor in place. Cannot fail.
pub(crate) fn set<M: Memory>(store: &PageStore<M>, entry: EntryAt, d: &Stored) {
    store.write(ByteOffset::new(entry.at.get() + 3), &d.encode());
}

/// Adds a bucket for `key` at `gap`. Returns the list's new reference; on error the old list is
/// unchanged. The caller checks [`has_room`] first.
pub(crate) fn add<M: Memory>(
    store: &mut PageStore<M>,
    list: Option<BucketList>,
    gap: EntryAt,
    key: BucketKey,
    d: &Stored,
) -> Result<BucketList, WriteError> {
    match list.map(|BucketList(shape)| shape) {
        None => add_flat(store, None, gap.index, key, d),
        Some(Shape::Flat { extent, count }) if usize::from(count) < LEAF_ENTRIES => {
            add_flat(store, Some((extent, count)), gap.index, key, d)
        }
        Some(Shape::Flat { extent, count }) => {
            let mut all = read_entries(&store.pages(), extent.offset(), usize::from(count));
            all.insert(gap.index.min(all.len()), (key, *d));
            let list = tiered::from_entries(store, &all)?;
            store.release(extent);
            Ok(list)
        }
        Some(Shape::Tiered { top }) => tiered::add(store, top, gap, key, d),
    }
}

fn add_flat<M: Memory>(
    store: &mut PageStore<M>,
    list: Option<(Extent, u8)>,
    index: usize,
    key: BucketKey,
    d: &Stored,
) -> Result<BucketList, WriteError> {
    let count = list.map_or(0, |(_, c)| usize::from(c));
    let new_count = u8::try_from(count + 1).unwrap_or_else(|_| corrupt("flat list count"));
    let bytes = (count + 1) * ENTRY_BYTES;
    if let Some((extent, _)) = list.filter(|(e, _)| bytes <= class_len(*e)) {
        // Room in place: shift the tail up one entry and write the new one.
        shift_in(store, extent.offset(), count, index, (key, d));
        return Ok(flat(extent, new_count));
    }
    let class = ByteClass::for_bytes(u32::try_from(bytes).unwrap_or(u32::MAX))
        .unwrap_or_else(|| corrupt("list beyond a page"));
    let extent = store.allocate(class)?;
    if let Some((old, _)) = list {
        // Copy the old entries around the gap, then write the new one into it.
        let view = store.pages();
        let head = index * ENTRY_BYTES;
        view.with_read(old.offset(), count * ENTRY_BYTES, |all| {
            let (before, after) = all.split_at(head.min(all.len()));
            store.write(extent.offset(), before);
            store.write(entry_offset(extent.offset(), index + 1), after);
        });
        store.release(old);
    }
    write_entry(store, entry_offset(extent.offset(), index), key, d);
    Ok(flat(extent, new_count))
}

/// Removes `entry`. Returns the list's new reference, or `None` when it is now empty (its
/// extents are released). Never fails.
pub(crate) fn remove<M: Memory>(
    store: &mut PageStore<M>,
    list: BucketList,
    entry: EntryAt,
) -> Option<BucketList> {
    match list.0 {
        Shape::Flat { extent, count } => {
            if count <= 1 {
                store.release(extent);
                return None;
            }
            shift_out(store, extent.offset(), usize::from(count), entry.index);
            Some(flat(extent, count - 1))
        }
        Shape::Tiered { top } => Some(tiered::remove(store, top, entry)),
    }
}

/// Closes the gap entry `index` leaves among `count` entries at `base`.
fn shift_out<M: Memory>(store: &PageStore<M>, base: ByteOffset, count: usize, index: usize) {
    let view = store.pages();
    let len = (count - index - 1) * ENTRY_BYTES;
    view.with_read(entry_offset(base, index + 1), len, |tail| {
        store.write(entry_offset(base, index), tail);
    });
}

/// Opens a gap at entry `index` among `count` entries at `base` and writes the new entry there.
fn shift_in<M: Memory>(
    store: &PageStore<M>,
    base: ByteOffset,
    count: usize,
    index: usize,
    (key, d): (BucketKey, &Stored),
) {
    let view = store.pages();
    let len = (count - index) * ENTRY_BYTES;
    view.with_read(entry_offset(base, index), len, |tail| {
        store.write(entry_offset(base, index + 1), tail);
    });
    write_entry(store, entry_offset(base, index), key, d);
}

/// Writes one entry at `at`.
fn write_entry<M: Memory>(store: &PageStore<M>, at: ByteOffset, key: BucketKey, d: &Stored) {
    let mut bytes = [0u8; ENTRY_BYTES];
    for (dst, src) in bytes.iter_mut().zip(encode_entry(key, d)) {
        *dst = src;
    }
    store.write(at, &bytes);
}

const fn flat(extent: Extent, count: u8) -> BucketList {
    BucketList(Shape::Flat { extent, count })
}

fn class_len(extent: Extent) -> usize {
    usize::try_from(extent.len()).unwrap_or(0)
}

fn entry_offset(base: ByteOffset, index: usize) -> ByteOffset {
    ByteOffset::new(base.get() + u64::try_from(index * ENTRY_BYTES).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests;
