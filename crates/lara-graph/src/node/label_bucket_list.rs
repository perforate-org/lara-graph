//! Per-node label bucket lists: which `(label, direction)` label buckets a node has, and their
//! descriptors.
//!
//! Entries are 12 bytes (label and a 10-byte listed descriptor, the direction in its free bits),
//! sorted by key. A list starts
//! **flat**: one extent in the page store, read whole and searched in memory; adding a bucket
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
//! Extent: the flat list or the top↕ 6 bytes
//! ----------------------------------------
//! Entry count; 0xFF if tiered     ↕ 1 byte
//! ----------------------------------------
//! ```
//!
//! # Flat list
//!
//! ```text
//! ---------------------------------------- <- Extent start
//! Entry 0                         ↕ 12 bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Entry count - 1                 ↕ 12 bytes
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
//! Stored: listed descriptor or    ↕ 10 bytes      (bits 1-2 of the last byte the direction:
//!   header ref                                     0 out, 1 in, 2 undirected)
//! ----------------------------------------
//! ```

mod tiered;

use crate::bucket::LabelBucketKey;
use crate::bucket::descriptor::{LISTED_BYTES, LISTED_FREE_BITS};
use crate::bucket::stored::Stored;
use crate::id::{Direction, EdgeLabel};
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::{ByteClass, EXTENT_BYTES, Extent, PageStore, Pages};
use crate::store::units::ByteOffset;
use alloc::vec::Vec;
use core::ops::ControlFlow;

/// Bytes per list entry.
const ENTRY_BYTES: usize = 2 + LISTED_BYTES;
/// Entries a flat list or a leaf holds: as many as a count byte can take but for 255 ([`TIERED`]),
/// 254 of the 341 a page has room for. Tests use a handful so that lists tier early.
const LEAF_ENTRIES: usize = crate::store::scale::Scale::pick(8, 254);
const _: () = assert!(LEAF_ENTRIES * ENTRY_BYTES <= 4096);
const _: () = assert!(crate::store::page_store::PAGE_BYTES == 4096);
/// Pages one [`add`] can take from the store: tiering takes two leaves and a top.
pub(crate) const PAGES_PER_ADD: u32 = 3;
/// Bytes of a node's label bucket list reference.
pub(crate) const LABEL_BUCKET_LIST_REF_BYTES: usize = EXTENT_BYTES + 1;
/// The count byte that marks a tiered list; flat counts stay below it.
const TIERED: u8 = u8::MAX;
// Flat counts never reach [`TIERED`] (0xFF).
const _: () = assert!(LEAF_ENTRIES < 0xFF);

const DIRECTION_OUTGOING: u8 = 0;
const DIRECTION_INCOMING: u8 = 1;
const DIRECTION_UNDIRECTED: u8 = 2;
/// Where an entry keeps its direction: the listed descriptor's free bits.
const DIRECTION_SHIFT: u32 = LISTED_FREE_BITS.trailing_zeros();
const _: () = assert!(DIRECTION_UNDIRECTED << DIRECTION_SHIFT & !LISTED_FREE_BITS == 0);

/// A key as the list stores and orders it.
impl LabelBucketKey {
    /// The key's place in key order: the label, then the direction.
    fn rank(self) -> u32 {
        (u32::from(self.label.get()) << 8) | u32::from(self.direction_bits())
    }

    /// The direction as an entry's last byte holds it.
    fn direction_bits(self) -> u8 {
        let [_, _, direction] = self.encode();
        direction << DIRECTION_SHIFT
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

    #[inline]
    fn decode([l0, l1, direction]: [u8; 3]) -> Self {
        let direction = match direction {
            DIRECTION_OUTGOING => Direction::Outgoing,
            DIRECTION_INCOMING => Direction::Incoming,
            DIRECTION_UNDIRECTED => Direction::Undirected,
            _ => corrupt("bucket direction"),
        };
        Self {
            label: EdgeLabel::new(u16::from_le_bytes([l0, l1])),
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
    /// The leaf, and in the top two bits the key's direction, which [`set`] keeps beside the
    /// descriptor it writes: a field of its own made every lookup's copies of this larger.
    leaf_and_direction: u16,
    index: usize,
    at: ByteOffset,
}

/// Where [`EntryAt`] keeps the direction: above any leaf's index.
const LEAF_DIRECTION_SHIFT: u32 = 14;
const _: () = assert!(tiered::TOP_ENTRIES < 1 << LEAF_DIRECTION_SHIFT);

impl EntryAt {
    fn new(leaf: u16, index: usize, at: ByteOffset, key: LabelBucketKey) -> Self {
        let [_, _, direction] = key.encode();
        Self {
            leaf_and_direction: leaf | (u16::from(direction) << LEAF_DIRECTION_SHIFT),
            index,
            at,
        }
    }

    const fn leaf(self) -> u16 {
        self.leaf_and_direction & ((1 << LEAF_DIRECTION_SHIFT) - 1)
    }

    /// The direction as the entry's last byte holds it.
    fn direction_bits(self) -> u8 {
        let direction = u8::try_from(self.leaf_and_direction >> LEAF_DIRECTION_SHIFT).unwrap_or(0);
        direction << DIRECTION_SHIFT
    }
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
        let [e0, e1, e2, e3, e4, e5] = extent.encode();
        [e0, e1, e2, e3, e4, e5, count]
    })
}

pub(crate) fn decode_ref(bytes: [u8; LABEL_BUCKET_LIST_REF_BYTES]) -> Option<LabelBucketList> {
    let [e0, e1, e2, e3, e4, e5, count] = bytes;
    (count != 0).then(|| {
        let extent = Extent::decode([e0, e1, e2, e3, e4, e5]);
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
            gap: EntryAt::new(0, 0, ByteOffset::new(0), key),
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
    pages.with_read(base, count * ENTRY_BYTES, |bytes| {
        find_stored(bytes.as_chunks::<ENTRY_BYTES>().0, key)
    })
}

/// Entries up to which a list is searched in order rather than by halves.
const SCANNED_ENTRIES: usize = 8;

/// The bucket `key` names among sorted `entries`: a short list compared entry by entry on the
/// key's encoded bytes, a longer one searched by rank.
///
/// Inlined into every lookup: out of line, batched degrees and visits cost 2-5 % more
/// (`degree_1024_batched`).
#[expect(clippy::inline_always, reason = "measured: see the doc")]
#[inline(always)]
fn find_stored(entries: &[[u8; ENTRY_BYTES]], key: LabelBucketKey) -> Option<Stored> {
    let entry = if entries.len() <= SCANNED_ENTRIES {
        let label = key.label.get().to_le_bytes();
        let direction = key.direction_bits();
        entries
            .iter()
            .find(|&&[l0, l1, .., last]| [l0, l1] == label && last & LISTED_FREE_BITS == direction)
    } else {
        let want = key.rank();
        let found = entries.binary_search_by(|e| rank_of(e).cmp(&want));
        found.ok().and_then(|index| entries.get(index))
    };
    entry.map(|e| Stored::decode(listed_of(e)))
}

/// [`lookup_stored`] over entries already copied into `bytes`, at `offset`.
///
/// Inlined into the read batch that calls it: left to the codegen units, it kept a stack frame
/// and batched degrees and visits cost 3-5 % more (`degree_1024_batched`).
#[inline]
pub(crate) fn lookup_stored_in(
    bytes: &[u8],
    offset: usize,
    count: usize,
    key: LabelBucketKey,
) -> Option<Stored> {
    let rest = bytes.get(offset..)?;
    let entries = rest.get(..count * ENTRY_BYTES)?;
    find_stored(entries.as_chunks::<ENTRY_BYTES>().0, key)
}

/// Searches the `count` sorted entries at `base`, leaf `leaf` of their list.
fn search<M: Memory>(
    pages: &Pages<'_, M>,
    base: ByteOffset,
    count: usize,
    key: LabelBucketKey,
    leaf: u16,
) -> Lookup {
    let place = |index| EntryAt::new(leaf, index, entry_offset(base, index), key);
    // Entries are compared as encoded, and only the one found is decoded: decoding each key
    // the search probes cost about a seventh of a node visit on the IC (`degree_1024`).
    let want = key.rank();
    pages.with_read(base, count * ENTRY_BYTES, |bytes| {
        let (entries, _) = bytes.as_chunks::<ENTRY_BYTES>();
        let found = entries.binary_search_by(|e| rank_of(e).cmp(&want));
        match found {
            Ok(index) => Lookup::Present {
                entry: place(index),
                stored: entries
                    .get(index)
                    .map_or_else(|| corrupt("list entry"), |e| Stored::decode(listed_of(e))),
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
    let entry = EntryAt::new(leaf, 0, leaf_page, key);
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

/// A listed bucket's stored form, decoded only when asked for: a walk that wants only the keys
/// does not decode descriptors.
#[derive(Clone, Copy)]
pub(crate) struct Listed([u8; LISTED_BYTES]);

impl Listed {
    #[inline]
    pub(crate) fn stored(self) -> Stored {
        Stored::decode(self.0)
    }
}

/// Every entry of a list, in order, until `f` breaks.
#[inline]
pub(crate) fn for_each<M: Memory, B>(
    pages: &Pages<'_, M>,
    list: Option<LabelBucketList>,
    mut f: impl FnMut(LabelBucketKey, Listed) -> ControlFlow<B>,
) -> Option<B> {
    match list.map(|LabelBucketList(shape)| shape) {
        None => None,
        Some(Shape::Flat { extent, count }) => {
            for_each_in(pages, extent.offset(), usize::from(count), &mut f)
        }
        Some(Shape::Tiered { top }) => tiered::for_each(pages, top, &mut f),
    }
}

/// [`for_each`] over `count` entries at `base`.
pub(crate) fn for_each_in<M: Memory, B>(
    pages: &Pages<'_, M>,
    base: ByteOffset,
    count: usize,
    f: &mut impl FnMut(LabelBucketKey, Listed) -> ControlFlow<B>,
) -> Option<B> {
    pages.with_read(base, count * ENTRY_BYTES, |bytes| {
        for_each_bytes(bytes, 0, count, f)
    })
}

/// [`for_each`] over `count` encoded entries already in `bytes`, starting at `at`.
pub(crate) fn for_each_bytes<B>(
    bytes: &[u8],
    at: usize,
    count: usize,
    f: &mut impl FnMut(LabelBucketKey, Listed) -> ControlFlow<B>,
) -> Option<B> {
    let entries = bytes.get(at..)?.get(..count * ENTRY_BYTES)?;
    let (entries, _) = entries.as_chunks::<ENTRY_BYTES>();
    for entry in entries {
        if let ControlFlow::Break(value) = f(key_of(entry), Listed(listed_of(entry))) {
            return Some(value);
        }
    }
    None
}

/// Every entry of a list, in order (test invariant checks).
#[cfg(test)]
pub(crate) fn entries<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<LabelBucketList>,
) -> Vec<(LabelBucketKey, Stored)> {
    let mut all = Vec::new();
    for_each(pages, list, |key, listed| {
        all.push((key, listed.stored()));
        ControlFlow::<()>::Continue(())
    });
    all
}

/// Writes one-entry flat lists for `entries`, allocating their extents in one batch.
#[expect(
    clippy::indexing_slicing,
    reason = "entry offsets stay inside the page"
)]
pub(crate) fn one_batch<M: Memory>(
    store: &mut PageStore<M>,
    entries: &[(LabelBucketKey, Stored)],
) -> Result<Vec<LabelBucketList>, WriteError> {
    let class = ByteClass::for_bytes(u32::try_from(ENTRY_BYTES).unwrap_or(u32::MAX))
        .unwrap_or_else(|| corrupt("bucket list entry class"));
    let extents = store.allocate_many(class, u32::try_from(entries.len()).unwrap_or(u32::MAX))?;
    let mut out = Vec::with_capacity(entries.len());
    let mut page: Option<crate::store::page_store::PageId> = None;
    let mut buffer = [0u8; 4096];
    for (&extent, &(key, stored)) in extents.iter().zip(entries) {
        if page != Some(extent.page()) {
            if let Some(previous) = page {
                store.write(previous.offset(), &buffer);
            }
            page = Some(extent.page());
            store.pages().read_into(extent.page().offset(), &mut buffer);
        }
        let within = usize::try_from(extent.offset().get() - extent.page().offset().get())
            .unwrap_or_else(|_| corrupt("bucket list offset"));
        buffer[within..within + ENTRY_BYTES].copy_from_slice(&encode_entry(key, &stored));
        out.push(LabelBucketList(Shape::Flat { extent, count: 1 }));
    }
    if let Some(page) = page {
        store.write(page.offset(), &buffer);
    }
    Ok(out)
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

/// An entry's place in key order, without decoding its key: the label, then the direction bits,
/// whose values follow [`Direction`]'s order (checked by a test).
#[inline]
fn rank_of(entry: &[u8; ENTRY_BYTES]) -> u32 {
    let &[l0, l1, .., last] = entry;
    (u32::from(u16::from_le_bytes([l0, l1])) << 8) | u32::from(last & LISTED_FREE_BITS)
}

#[inline]
fn key_of(entry: &[u8; ENTRY_BYTES]) -> LabelBucketKey {
    let &[l0, l1, .., last] = entry;
    LabelBucketKey::decode([l0, l1, (last & LISTED_FREE_BITS) >> DIRECTION_SHIFT])
}

/// An entry's listed descriptor, its direction bits left in: decoding ignores them.
#[inline]
const fn listed_of(entry: &[u8; ENTRY_BYTES]) -> [u8; LISTED_BYTES] {
    let &[_, _, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] = entry;
    [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]
}

#[inline]
fn decode_entry(entry: [u8; ENTRY_BYTES]) -> (LabelBucketKey, Stored) {
    (key_of(&entry), Stored::decode(listed_of(&entry)))
}

/// The listed descriptor with `direction` bits in its free ones.
#[inline]
fn with_direction(d: &Stored, direction: u8) -> [u8; LISTED_BYTES] {
    let [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] = d.encode();
    [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9 | direction]
}

/// Laid out by hand, as are the other hot encoders: chaining parts as iterators compiles to
/// slow code on the IC (a 32-byte directory record cost about 1,500 instructions that way).
fn encode_entry(key: LabelBucketKey, d: &Stored) -> [u8; ENTRY_BYTES] {
    let [l0, l1] = key.label.get().to_le_bytes();
    let [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] = with_direction(d, key.direction_bits());
    [l0, l1, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]
}

/// Publishes a bucket's new descriptor in place, keeping the entry's direction. Cannot fail.
pub(crate) fn set<M: Memory>(store: &PageStore<M>, entry: EntryAt, d: &Stored) {
    store.write(
        ByteOffset::new(entry.at.get() + 2),
        &with_direction(d, entry.direction_bits()),
    );
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

/// Proofs over every input, checked by Kani (`cargo kani -p lara-graph`).
#[cfg(kani)]
mod proofs {
    use super::*;

    fn any_key() -> LabelBucketKey {
        let direction: u8 = kani::any();
        kani::assume(direction <= DIRECTION_UNDIRECTED);
        let [l0, l1] = kani::any::<u16>().to_le_bytes();
        LabelBucketKey::decode([l0, l1, direction])
    }

    /// Keys rank in the order they compare: a list sorted by rank is sorted by key.
    #[kani::proof]
    fn keys_rank_in_key_order() {
        let (a, b) = (any_key(), any_key());
        assert_eq!(a.rank().cmp(&b.rank()), a.cmp(&b));
    }

    /// An entry holds its key and its listed bytes apart: its rank is its key's, and the
    /// direction bits leave what the listed bytes decode to alone.
    #[kani::proof]
    fn entries_hold_key_and_descriptor_apart() {
        let key = any_key();
        let listed: [u8; LISTED_BYTES] = kani::any();
        kani::assume(crate::bucket::descriptor::listed_kind(&listed) != 2);
        let stored = Stored::decode(listed);
        let entry = encode_entry(key, &stored);
        assert_eq!(key_of(&entry), key);
        assert_eq!(rank_of(&entry), key.rank());
        assert_eq!(Stored::decode(listed_of(&entry)), stored);
    }

    /// Any seven bytes decode to a list reference that encodes back to itself.
    #[kani::proof]
    fn references_round_trip() {
        let list = decode_ref(kani::any());
        assert_eq!(decode_ref(encode_ref(list)), list);
    }
}
