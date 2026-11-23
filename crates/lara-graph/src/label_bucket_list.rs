//! Per-node label bucket lists: which `(label, direction)` label buckets a node has, and their
//! descriptors.
//!
//! Entries are 19 bytes (label, direction, 16-byte descriptor), sorted by key. A list starts
//! **flat**: one extent in its own page store, read whole and searched in memory; adding a bucket
//! shifts entries in place or moves the list to the next class. A flat list that would outgrow a
//! page becomes **tiered** (`tiered`): a small top extent lists leaf pages with their first key and
//! count, and each leaf holds a page of entries. Every update copies at most one page. Lists are
//! independent: changing one never touches another node.
//!
//! # Reference
//!
//! A node's record holds it ([`LABEL_BUCKET_LIST_REF_BYTES`]); an empty list is all zeros.
//!
//! ```text
//! ---------------------------------------- <- Offset 0
//! Extent: the flat list or the top↕ 7 bytes
//! ----------------------------------------
//! Entry count; 0xFF if tiered     ↕ 1 byte
//! ----------------------------------------
//! ```
//!
//! # Flat list
//!
//! ```text
//! ---------------------------------------- <- Extent start
//! Entry 0                         ↕ 19 bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Entry count - 1                 ↕ 19 bytes
//! ----------------------------------------
//! Unused space
//! ```
//!
//! # Entry
//!
//! Entries are sorted by label, then direction.
//!
//! ```text
//! ---------------------------------------- <- Offset 0
//! Label                           ↕ 2 bytes
//! ----------------------------------------
//! Direction: 0 out, 1 in, 2 undir.↕ 1 byte
//! ----------------------------------------
//! Stored: descriptor or header ref↕ 16 bytes
//! ----------------------------------------
//! ```

mod tiered;

use crate::corrupt::corrupt;
use crate::descriptor::DESCRIPTOR_BYTES;
use crate::id::{Direction, Label};
use crate::memory::{Memory, WriteError};
use crate::page_store::{ByteClass, EXTENT_BYTES, Extent, PageStore, Pages};
use crate::stored::Stored;
use crate::units::ByteOffset;
use alloc::vec::Vec;

/// Bytes per list entry.
const ENTRY_BYTES: usize = 3 + DESCRIPTOR_BYTES;
/// Entries a flat list or a leaf holds: one page. Tests use a handful so that lists tier early.
const LEAF_ENTRIES: usize = crate::scale::Scale::pick(8, 4096 / ENTRY_BYTES);
const _: () = assert!(crate::page_store::PAGE_BYTES == 4096);
/// Pages one [`add`] can take from the store: tiering takes two leaves and a top.
pub(crate) const PAGES_PER_ADD: u32 = 3;
/// Bytes of a node's label bucket list reference.
pub(crate) const LABEL_BUCKET_LIST_REF_BYTES: usize = EXTENT_BYTES + 1;
/// The count byte that marks a tiered list; flat counts stay below it.
const TIERED: u8 = u8::MAX;
// Flat counts never reach [`TIERED`] (0xFF).
const _: () = assert!(LEAF_ENTRIES < 0xFF);

/// Which label bucket of a node: a label in one direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct LabelBucketKey {
    pub(crate) label: Label,
    pub(crate) direction: Direction,
}

const DIRECTION_OUTGOING: u8 = 0;
const DIRECTION_INCOMING: u8 = 1;
const DIRECTION_UNDIRECTED: u8 = 2;

impl LabelBucketKey {
    /// The key's place in key order: the label, then the direction.
    fn rank(self) -> u32 {
        rank_of(self.encode())
    }

    fn encode(self) -> [u8; 3] {
        let [l0, l1] = self.label.get().to_le_bytes();
        let direction = match self.direction {
            Direction::Outgoing => DIRECTION_OUTGOING,
            Direction::Incoming => DIRECTION_INCOMING,
            Direction::Undirected => DIRECTION_UNDIRECTED,
        };
        [l0, l1, direction]
    }

    fn decode([l0, l1, direction]: [u8; 3]) -> Self {
        let direction = match direction {
            DIRECTION_OUTGOING => Direction::Outgoing,
            DIRECTION_INCOMING => Direction::Incoming,
            DIRECTION_UNDIRECTED => Direction::Undirected,
            _ => corrupt("bucket direction"),
        };
        Self {
            label: Label::new(u16::from_le_bytes([l0, l1])),
            direction,
        }
    }
}

/// Where a node's label bucket list is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LabelBucketList(Shape);

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

/// As the module's Reference diagram shows.
pub(crate) fn encode_ref(list: Option<LabelBucketList>) -> [u8; LABEL_BUCKET_LIST_REF_BYTES] {
    let encoded = list.map(|LabelBucketList(shape)| match shape {
        Shape::Flat { extent, count } => (extent, count),
        Shape::Tiered { top } => (top, TIERED),
    });
    encoded.map_or([0; LABEL_BUCKET_LIST_REF_BYTES], |(extent, count)| {
        let [e0, e1, e2, e3, e4, e5, e6] = extent.encode();
        [e0, e1, e2, e3, e4, e5, e6, count]
    })
}

pub(crate) fn decode_ref(bytes: [u8; LABEL_BUCKET_LIST_REF_BYTES]) -> Option<LabelBucketList> {
    let [e0, e1, e2, e3, e4, e5, e6, count] = bytes;
    (count != 0).then(|| {
        let extent = Extent::decode([e0, e1, e2, e3, e4, e5, e6]);
        LabelBucketList(match count {
            TIERED => Shape::Tiered { top: extent },
            count => Shape::Flat { extent, count },
        })
    })
}

/// Finds a bucket with one read of a flat list, or a top and a leaf read of a tiered one.
pub(crate) fn lookup<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<LabelBucketList>,
    key: LabelBucketKey,
) -> Lookup {
    match list.map(|LabelBucketList(shape)| shape) {
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

impl LabelBucketList {
    /// A flat list's extent and entry count; a tiered list needs its top and a leaf.
    pub(crate) fn flat(self) -> Option<(Extent, usize)> {
        match self.0 {
            Shape::Flat { extent, count } => Some((extent, usize::from(count))),
            Shape::Tiered { .. } => None,
        }
    }
}

/// Finds a bucket's stored form, without the entry's position: a read that only scans slots
/// does not build an [`EntryAt`] it will discard.
pub(crate) fn lookup_stored<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<LabelBucketList>,
    key: LabelBucketKey,
) -> Option<Stored> {
    match list.map(|LabelBucketList(shape)| shape) {
        None => None,
        Some(Shape::Flat { extent, count }) => {
            search_stored(pages, extent.offset(), usize::from(count), key)
        }
        Some(Shape::Tiered { top }) => match tiered::lookup(pages, top, key) {
            Lookup::Present { stored, .. } => Some(stored),
            Lookup::Absent { .. } => None,
        },
    }
}

/// [`search`], returning only what was found.
fn search_stored<M: Memory>(
    pages: &Pages<'_, M>,
    base: ByteOffset,
    count: usize,
    key: LabelBucketKey,
) -> Option<Stored> {
    let want = key.rank();
    pages.with_read(base, count * ENTRY_BYTES, |bytes| {
        let (entries, _) = bytes.as_chunks::<ENTRY_BYTES>();
        entries
            .binary_search_by(|&[k0, k1, k2, ..]| rank_of([k0, k1, k2]).cmp(&want))
            .ok()
            .and_then(|index| entries.get(index).map(|&entry| decode_entry(entry).1))
    })
}

/// [`lookup_stored`] over entries already copied into `bytes`, at `offset`.
pub(crate) fn lookup_stored_in(
    bytes: &[u8],
    offset: usize,
    count: usize,
    key: LabelBucketKey,
) -> Option<Stored> {
    let want = key.rank();
    let rest = bytes.get(offset..)?;
    let entries = rest.get(..count * ENTRY_BYTES)?;
    let (entries, _) = entries.as_chunks::<ENTRY_BYTES>();
    entries
        .binary_search_by(|&[k0, k1, k2, ..]| rank_of([k0, k1, k2]).cmp(&want))
        .ok()
        .and_then(|index| entries.get(index).map(|&entry| decode_entry(entry).1))
}

/// Searches the `count` sorted entries at `base`, leaf `leaf` of their list.
fn search<M: Memory>(
    pages: &Pages<'_, M>,
    base: ByteOffset,
    count: usize,
    key: LabelBucketKey,
    leaf: u16,
) -> Lookup {
    let place = |index| EntryAt {
        leaf,
        index,
        at: entry_offset(base, index),
    };
    // Entries are compared as encoded, and only the one found is decoded: decoding each key
    // the search probes cost about a seventh of a node visit on the IC (`degree_1024`).
    let want = key.rank();
    pages.with_read(base, count * ENTRY_BYTES, |bytes| {
        let (entries, _) = bytes.as_chunks::<ENTRY_BYTES>();
        let found = entries.binary_search_by(|&[k0, k1, k2, ..]| rank_of([k0, k1, k2]).cmp(&want));
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

/// A node's first bucket in key order, with one small read (two for a tiered list).
pub(crate) fn first<M: Memory>(
    pages: &Pages<'_, M>,
    list: LabelBucketList,
) -> (EntryAt, LabelBucketKey, Stored) {
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
    list: Option<LabelBucketList>,
    adding: usize,
) -> bool {
    match list.map(|LabelBucketList(shape)| shape) {
        None | Some(Shape::Flat { .. }) => true,
        Some(Shape::Tiered { top }) => tiered::has_room(pages, top, adding),
    }
}

/// The keys of a node's buckets, in order.
pub(crate) fn keys<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<LabelBucketList>,
) -> Vec<LabelBucketKey> {
    entries(pages, list).into_iter().map(|(k, _)| k).collect()
}

/// Every entry of a list, in order.
pub(crate) fn entries<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<LabelBucketList>,
) -> Vec<(LabelBucketKey, Stored)> {
    match list.map(|LabelBucketList(shape)| shape) {
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
) -> Vec<(LabelBucketKey, Stored)> {
    let bytes = pages.read(base, count * ENTRY_BYTES);
    bytes
        .as_chunks::<ENTRY_BYTES>()
        .0
        .iter()
        .map(|&entry| decode_entry(entry))
        .collect()
}

/// Every bucket of a node (test invariant checks).
#[cfg(test)]
pub(crate) fn buckets<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<LabelBucketList>,
) -> Vec<(LabelBucketKey, Stored)> {
    entries(pages, list)
}

/// Panics unless the list's shape is within its limits: a flat list or each leaf holds 1 to
/// [`LEAF_ENTRIES`] entries, and a tiered list has 2 to `TOP_ENTRIES` leaves (test checks).
#[cfg(test)]
pub(crate) fn check_shape<M: Memory>(pages: &Pages<'_, M>, list: LabelBucketList) {
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
pub(crate) fn extents<M: Memory>(pages: &Pages<'_, M>, list: LabelBucketList) -> Vec<Extent> {
    match list.0 {
        Shape::Flat { extent, .. } => alloc::vec![extent],
        Shape::Tiered { top } => tiered::extents(pages, top),
    }
}

/// An encoded key's place in key order, without decoding it: the label, then the direction
/// byte, whose values follow [`Direction`]'s order (checked by a test).
fn rank_of([l0, l1, direction]: [u8; 3]) -> u32 {
    (u32::from(u16::from_le_bytes([l0, l1])) << 8) | u32::from(direction)
}

fn decode_entry(entry: [u8; ENTRY_BYTES]) -> (LabelBucketKey, Stored) {
    let [k0, k1, k2, rest @ ..] = entry;
    (LabelBucketKey::decode([k0, k1, k2]), Stored::decode(rest))
}

/// Laid out by hand, as are the other hot encoders: chaining parts as iterators compiles to
/// slow code on the IC (a 32-byte directory record cost about 1,500 instructions that way).
fn encode_entry(key: LabelBucketKey, d: &Stored) -> [u8; ENTRY_BYTES] {
    let [k0, k1, k2] = key.encode();
    let [
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        s8,
        s9,
        s10,
        s11,
        s12,
        s13,
        s14,
        s15,
    ] = d.encode();
    [
        k0, k1, k2, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
    ]
}

/// Publishes a bucket's new descriptor in place. Cannot fail.
pub(crate) fn set<M: Memory>(store: &PageStore<M>, entry: EntryAt, d: &Stored) {
    store.write(ByteOffset::new(entry.at.get() + 3), &d.encode());
}

/// Adds a bucket for `key` at `gap`. Returns the list's new reference; on error the old list is
/// unchanged. The caller checks [`has_room`] first.
pub(crate) fn add<M: Memory>(
    store: &mut PageStore<M>,
    list: Option<LabelBucketList>,
    gap: EntryAt,
    key: LabelBucketKey,
    d: &Stored,
) -> Result<LabelBucketList, WriteError> {
    match list.map(|LabelBucketList(shape)| shape) {
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
    key: LabelBucketKey,
    d: &Stored,
) -> Result<LabelBucketList, WriteError> {
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
    list: LabelBucketList,
    entry: EntryAt,
) -> Option<LabelBucketList> {
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
    (key, d): (LabelBucketKey, &Stored),
) {
    let view = store.pages();
    let len = (count - index) * ENTRY_BYTES;
    view.with_read(entry_offset(base, index), len, |tail| {
        store.write(entry_offset(base, index + 1), tail);
    });
    write_entry(store, entry_offset(base, index), key, d);
}

/// Writes one entry at `at`.
fn write_entry<M: Memory>(store: &PageStore<M>, at: ByteOffset, key: LabelBucketKey, d: &Stored) {
    store.write(at, &encode_entry(key, d));
}

const fn flat(extent: Extent, count: u8) -> LabelBucketList {
    LabelBucketList(Shape::Flat { extent, count })
}

fn class_len(extent: Extent) -> usize {
    usize::try_from(extent.len()).unwrap_or(0)
}

fn entry_offset(base: ByteOffset, index: usize) -> ByteOffset {
    ByteOffset::new(base.get() + u64::try_from(index * ENTRY_BYTES).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests;
