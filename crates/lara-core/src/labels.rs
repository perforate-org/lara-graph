//! Per-vertex bucket lists: which `(label, side)` buckets a vertex has, and their descriptors.
//!
//! A vertex's list is one extent in its own page store, entries sorted by key. Each entry is
//! 19 bytes: the label, the side and the bucket's 16-byte descriptor. A lookup reads the whole
//! list once and searches it in memory; adding a bucket shifts entries in place or moves the list
//! to the next class (at most one page). Lists are independent: adding a bucket never touches
//! another vertex.

use crate::corrupt::corrupt;
use crate::descriptor::{DESCRIPTOR_BYTES, Descriptor};
use crate::id::{Label, Side};
use crate::memory::{Memory, WriteError};
use crate::page_store::{ByteClass, EXTENT_BYTES, Extent, PageStore, Pages};
use crate::units::ByteOffset;
use alloc::vec::Vec;

/// Bytes per list entry.
const ENTRY_BYTES: usize = 3 + DESCRIPTOR_BYTES;
/// Buckets one vertex can have: one page (4096 bytes) of entries.
pub(crate) const MAX_BUCKETS: usize = 4096 / ENTRY_BYTES;
/// Pages one [`add`] can take from the store: its new list extent.
pub(crate) const PAGES_PER_ADD: u32 = 1;
const _: () = assert!(crate::page_store::PAGE_BYTES == 4096);
/// Bytes of a vertex's list reference.
pub(crate) const LIST_REF_BYTES: usize = EXTENT_BYTES + 1;

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

/// Where a vertex's bucket list is and how many entries it has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BucketList {
    extent: Extent,
    count: u8,
}

/// The result of looking a label up in a list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lookup {
    /// The bucket exists at entry `index`.
    Present {
        index: usize,
        descriptor: Descriptor,
    },
    /// No bucket; a new entry belongs at `index`.
    Absent { index: usize },
}

/// Layout: extent (7) | count u8. An empty list is all zeros.
pub(crate) fn encode_ref(list: Option<BucketList>) -> [u8; LIST_REF_BYTES] {
    let mut out = [0u8; LIST_REF_BYTES];
    if let Some(list) = list {
        for (dst, src) in out
            .iter_mut()
            .zip(list.extent.encode().into_iter().chain([list.count]))
        {
            *dst = src;
        }
    }
    out
}

pub(crate) fn decode_ref(bytes: [u8; LIST_REF_BYTES]) -> Option<BucketList> {
    let [e0, e1, e2, e3, e4, e5, e6, count] = bytes;
    (count != 0).then(|| BucketList {
        extent: Extent::decode([e0, e1, e2, e3, e4, e5, e6]),
        count,
    })
}

fn entries<M: Memory>(pages: &Pages<'_, M>, list: BucketList) -> Vec<(BucketKey, Descriptor)> {
    let bytes = pages.read(list.extent.offset(), usize::from(list.count) * ENTRY_BYTES);
    bytes
        .as_chunks::<ENTRY_BYTES>()
        .0
        .iter()
        .map(|&entry| decode_entry(entry))
        .collect()
}

fn decode_entry(entry: [u8; ENTRY_BYTES]) -> (BucketKey, Descriptor) {
    let [k0, k1, k2, rest @ ..] = entry;
    (BucketKey::decode([k0, k1, k2]), Descriptor::decode(rest))
}

fn encode_entry(key: BucketKey, d: &Descriptor) -> impl Iterator<Item = u8> {
    key.encode().into_iter().chain(d.encode())
}

/// Finds a bucket in a vertex's list with one read.
pub(crate) fn lookup<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<BucketList>,
    key: BucketKey,
) -> Lookup {
    let Some(list) = list else {
        return Lookup::Absent { index: 0 };
    };
    let entries = entries(pages, list);
    match entries.binary_search_by_key(&key, |(k, _)| *k) {
        Ok(index) => {
            let (_, descriptor) = entries
                .get(index)
                .copied()
                .unwrap_or_else(|| corrupt("list entry"));
            Lookup::Present { index, descriptor }
        }
        Err(index) => Lookup::Absent { index },
    }
}

/// A vertex's first bucket in key order, with one small read.
pub(crate) fn first<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<BucketList>,
) -> Option<(BucketKey, Descriptor)> {
    let list = list?;
    Some(decode_entry(pages.read_array(list.extent.offset())))
}

/// How many buckets a list has.
pub(crate) fn len(list: Option<BucketList>) -> usize {
    list.map_or(0, |l| usize::from(l.count))
}

/// The keys of a vertex's buckets, in order.
pub(crate) fn keys<M: Memory>(pages: &Pages<'_, M>, list: Option<BucketList>) -> Vec<BucketKey> {
    list.map_or_else(Vec::new, |list| {
        entries(pages, list).into_iter().map(|(k, _)| k).collect()
    })
}

/// Every bucket of a vertex (test invariant checks).
#[cfg(test)]
pub(crate) fn buckets<M: Memory>(
    pages: &Pages<'_, M>,
    list: Option<BucketList>,
) -> Vec<(BucketKey, Descriptor)> {
    list.map_or_else(Vec::new, |list| entries(pages, list))
}

/// The list's extent (test invariant checks).
#[cfg(test)]
pub(crate) const fn extent(list: BucketList) -> Extent {
    list.extent
}

/// Publishes a bucket's new descriptor in place. Cannot fail.
pub(crate) fn set<M: Memory>(store: &PageStore<M>, list: BucketList, index: usize, d: &Descriptor) {
    let at = entry_offset(list.extent, index) + 3;
    store.write(ByteOffset::new(at), &d.encode());
}

/// Adds a bucket for `key` at entry `index`. Returns the list's new reference; on error the
/// old list is unchanged. The caller keeps lists within [`MAX_BUCKETS`].
pub(crate) fn add<M: Memory>(
    store: &mut PageStore<M>,
    list: Option<BucketList>,
    index: usize,
    key: BucketKey,
    d: &Descriptor,
) -> Result<BucketList, WriteError> {
    let count = len(list);
    let new_count = u8::try_from(count + 1)
        .ok()
        .filter(|_| count < MAX_BUCKETS)
        .unwrap_or_else(|| corrupt("bucket list beyond a page"));
    let bytes = (count + 1) * ENTRY_BYTES;
    let mut all: Vec<u8> = list.map_or_else(Vec::new, |l| {
        store.pages().read(l.extent.offset(), count * ENTRY_BYTES)
    });
    let split = index * ENTRY_BYTES;
    let tail = all.split_off(split.min(all.len()));
    all.extend(encode_entry(key, d));
    all.extend(tail);
    if let Some(old) = list.filter(|l| bytes <= class_len(l.extent)) {
        // Room in place: rewrite from the new entry on.
        let at = entry_offset(old.extent, index);
        store.write(ByteOffset::new(at), all.get(split..).unwrap_or(&[]));
        return Ok(BucketList {
            extent: old.extent,
            count: new_count,
        });
    }
    let class = ByteClass::for_bytes(u32::try_from(bytes).unwrap_or(u32::MAX))
        .unwrap_or_else(|| corrupt("list beyond a page"));
    let extent = store.allocate(class)?;
    store.write(extent.offset(), &all);
    if let Some(old) = list {
        store.release(old.extent);
    }
    Ok(BucketList {
        extent,
        count: new_count,
    })
}

/// Removes entry `index`. Returns the list's new reference, or `None` when it is now empty (its
/// extent is released). Never fails.
pub(crate) fn remove<M: Memory>(
    store: &mut PageStore<M>,
    list: BucketList,
    index: usize,
) -> Option<BucketList> {
    let count = usize::from(list.count);
    if count <= 1 {
        store.release(list.extent);
        return None;
    }
    let tail_start = entry_offset(list.extent, index + 1);
    let tail = store.pages().read(
        ByteOffset::new(tail_start),
        (count - index - 1) * ENTRY_BYTES,
    );
    store.write(ByteOffset::new(entry_offset(list.extent, index)), &tail);
    Some(BucketList {
        extent: list.extent,
        count: list.count - 1,
    })
}

fn class_len(extent: Extent) -> usize {
    usize::try_from(extent.len()).unwrap_or(0)
}

fn entry_offset(extent: Extent, index: usize) -> u64 {
    extent.offset().get() + u64::try_from(index * ENTRY_BYTES).unwrap_or(u64::MAX)
}
