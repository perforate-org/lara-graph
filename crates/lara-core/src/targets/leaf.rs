//! Index leaves: a page of postings in independently coded segments, laid out as a slotted page
//! so that a change reads and writes only the segments it touches.
//!
//! Layout: segment count u16 and the low end of the slots u16, then a table entry per segment in
//! key order ([`ENTRY_BYTES`]: its first posting's target u32 and block u32, the total count of
//! its postings u32, then packed in eight bytes its slot's offset and capacity, the length of
//! its code, its Rice parameter `k`, block width `w` and posting count). Slots fill the leaf from
//! its end down; the space between the table and the lowest slot is free.
//!
//! A segment (`segment`) codes a run of sorted postings: each target gap after the first in
//! Rice(`k`) (0 for the same target), each count (one bit for one, else Elias gamma), each block
//! in `w` bits. A rewritten segment that still fits its slot stays there; one that outgrows it
//! moves to a new slot with some slack, leaving the old one as garbage, which rewriting the whole
//! leaf reclaims when space runs out (`write`). A lookup reads the table entries it probes and
//! one slot; a change also writes that slot and its table entry.

mod segment;
mod write;

use super::Posting;
use super::tree::Change;
use crate::corrupt::corrupt;
use alloc::vec::Vec;
use core::ops::Range;
use segment::{Entry, Patched, merge, rebuild};
use write::{Split, Touched, gather, in_place, pack};

pub(super) const LEAF_BYTES: usize = 4096;
/// Bytes a leaf may use: a page; tests use less so that trees grow deep early.
const CAPACITY: usize = if cfg!(test) { 1024 } else { LEAF_BYTES };
const HEAD_BYTES: usize = 4;
const ENTRY_BYTES: usize = 20;

/// How a leaf's page is read: fills the buffer from the offset.
pub(super) type Read<'a> = &'a dyn Fn(usize, &mut [u8]);

/// A leaf's header; its table entries and slots are read as they are needed.
struct Leaf<'r> {
    read: Read<'r>,
    n: usize,
    low: usize,
}

impl<'r> Leaf<'r> {
    fn open(read: Read<'r>) -> Self {
        let mut head = [0u8; HEAD_BYTES];
        read(0, &mut head);
        let [n0, n1, l0, l1] = head;
        Self {
            read,
            n: usize::from(u16::from_le_bytes([n0, n1])),
            low: usize::from(u16::from_le_bytes([l0, l1])),
        }
    }

    const fn len(&self) -> usize {
        self.n
    }

    /// The raw table entries `range`, with one read.
    fn raw(&self, range: Range<usize>) -> Vec<u8> {
        let range = range.start.min(self.n)..range.end.min(self.n);
        let mut bytes = alloc::vec![0u8; range.len() * ENTRY_BYTES];
        (self.read)(table_end(range.start), &mut bytes);
        bytes
    }

    fn entry(&self, i: usize) -> Option<Entry> {
        if i >= self.n {
            return None;
        }
        let mut bytes = [0u8; ENTRY_BYTES];
        (self.read)(table_end(i), &mut bytes);
        Some(Entry::from_bytes(bytes))
    }

    /// The first key of segment `i`, reading only it.
    fn first(&self, i: usize) -> Option<(u32, u32)> {
        if i >= self.n {
            return None;
        }
        let mut bytes = [0u8; 8];
        (self.read)(table_end(i), &mut bytes);
        let [t0, t1, t2, t3, b0, b1, b2, b3] = bytes;
        Some((
            u32::from_le_bytes([t0, t1, t2, t3]),
            u32::from_le_bytes([b0, b1, b2, b3]),
        ))
    }

    /// The totals of segments `range`, in order.
    fn totals(&self, range: Range<usize>) -> Vec<u32> {
        self.raw(range)
            .as_chunks::<ENTRY_BYTES>()
            .0
            .iter()
            .map(|&[_, _, _, _, _, _, _, _, c0, c1, c2, c3, ..]| {
                u32::from_le_bytes([c0, c1, c2, c3])
            })
            .collect()
    }

    /// The segment a key falls in: the last that starts at or below it (or the first).
    fn segment_of(&self, key: (u32, u32)) -> usize {
        let (mut low, mut high) = (0, self.n);
        while low < high {
            let mid = low + (high - low) / 2;
            if self.first(mid).is_some_and(|f| f <= key) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        low.saturating_sub(1)
    }

    /// The total count of the segments before `s`, summed from whichever end is nearer when the
    /// leaf's total is known.
    fn before(&self, s: usize, total: Option<u32>) -> u32 {
        match total {
            Some(total) if 2 * s > self.n => total - self.totals(s..self.n).iter().sum::<u32>(),
            Some(_) | None => self.totals(0..s).iter().sum(),
        }
    }

    /// The segment holding the `rank`-th unit of count and the rank within it; counted from the
    /// end when the rank is in the second half of a known `total`.
    fn locate(&self, rank: u32, total: Option<u32>) -> Option<(usize, u32)> {
        let totals = self.totals(0..self.n);
        let find = |order: &mut dyn Iterator<Item = usize>, mut left: u32| {
            order
                .map(|i| (i, totals.get(i).copied().unwrap_or(0)))
                .find_map(|(i, t)| {
                    let here = left < t;
                    if !here {
                        left -= t;
                    }
                    here.then_some((i, t, left))
                })
        };
        match total {
            Some(total) if rank < total && 2 * rank >= total => {
                let (s, t, back) = find(&mut (0..self.n).rev(), total - 1 - rank)?;
                Some((s, t - 1 - back))
            }
            Some(_) | None => find(&mut (0..self.n), rank).map(|(s, _, left)| (s, left)),
        }
    }
}

const fn table_end(segments: usize) -> usize {
    HEAD_BYTES + segments * ENTRY_BYTES
}

fn header(segments: usize, low: usize) -> [u8; HEAD_BYTES] {
    let narrow = |v: usize| u16::try_from(v).unwrap_or_else(|_| corrupt("index leaf size"));
    let [n0, n1] = narrow(segments).to_le_bytes();
    let [l0, l1] = narrow(low).to_le_bytes();
    [n0, n1, l0, l1]
}

/// Every posting of a leaf.
#[cfg(test)]
pub(super) fn decode(read: Read<'_>) -> Vec<Posting> {
    let leaf = Leaf::open(read);
    (0..leaf.len())
        .filter_map(|i| leaf.entry(i))
        .flat_map(|e| e.decode(read))
        .collect()
}

/// The total count of the postings before `key`, decoding part of one segment. `total` is the
/// leaf's total when its parent knows it.
pub(super) fn count_before(read: Read<'_>, key: (u32, u32), total: Option<u32>) -> u32 {
    let leaf = Leaf::open(read);
    let s = leaf.segment_of(key);
    let mut within = 0;
    if let Some(e) = leaf.entry(s) {
        e.walk(read, |p| {
            if p.key() >= key {
                return true;
            }
            within += p.count;
            false
        });
    }
    leaf.before(s, total) + within
}

/// The posting holding the `rank`-th unit of count (0-based) and the rank within it, decoding
/// part of one segment; `None` past the leaf's total. `total` is the leaf's total when its parent
/// knows it.
pub(super) fn select(read: Read<'_>, rank: u32, total: Option<u32>) -> Option<(Posting, u32)> {
    let leaf = Leaf::open(read);
    let (s, mut left) = leaf.locate(rank, total)?;
    let mut found = None;
    leaf.entry(s)?.walk(read, |p| {
        if left < p.count {
            found = Some((p, left));
            return true;
        }
        left -= p.count;
        false
    });
    found
}

/// Calls `f` with the leaf's postings from `key` on, in order, until it returns true; whether it
/// did.
pub(super) fn walk_from(
    read: Read<'_>,
    key: (u32, u32),
    mut f: impl FnMut(Posting) -> bool,
) -> bool {
    let leaf = Leaf::open(read);
    let mut stopped = false;
    for e in (leaf.segment_of(key)..leaf.len()).filter_map(|i| leaf.entry(i)) {
        e.walk(read, |p| {
            stopped = p.key() >= key && f(p);
            stopped
        });
        if stopped {
            break;
        }
    }
    stopped
}

/// A whole leaf as its parent lists it: first key, total count and bytes.
pub(super) struct Image {
    pub(super) first: (u32, u32),
    pub(super) count: u32,
    pub(super) bytes: Vec<u8>,
}

/// What a batch of changes does to a leaf.
pub(super) enum Update {
    /// The leaf stays on its page with these `(offset, bytes)` writes, and the change in its
    /// total. Its parent's key for it stays a valid lower bound whatever the changes did to its
    /// first key, so that key is left as it is.
    InPlace {
        writes: Vec<(usize, Vec<u8>)>,
        delta: i64,
    },
    /// Whole leaves replace it: none when it empties, more than one when it outgrows a page.
    Replace(Vec<Image>),
}

/// Applies sorted, combined changes to a leaf (none for a new tree), re-coding only the segments
/// they touch.
pub(super) fn update(old: Option<Read<'_>>, changes: &[Change]) -> Update {
    let Some(read) = old else {
        let fresh: Vec<_> = rebuild(&merge(&[], changes))
            .into_iter()
            .map(|c| (c, true))
            .collect();
        return Update::Replace(pack(&fresh, Split::Fill));
    };
    let leaf = Leaf::open(read);
    let mut groups: Vec<(usize, &[Change])> = Vec::new();
    let mut rest = changes;
    while let Some(c) = rest.first() {
        // The segment the next change falls in takes the changes below the next one's start.
        let index = leaf.segment_of(c.key);
        let bound = leaf.first(index + 1);
        let split = bound.map_or(rest.len(), |b| rest.partition_point(|c| c.key < b));
        let (mine, later) = rest.split_at(split.max(1));
        rest = later;
        groups.push((index, mine));
    }
    // Removals only lower counts of postings that exist: they rewrite counts in place.
    if changes.iter().all(|c| c.delta < 0)
        && let Some(writes) = patched(&leaf, &groups)
    {
        let delta = changes.iter().map(|c| c.delta).sum();
        return Update::InPlace { writes, delta };
    }
    let touched: Vec<Touched> = groups
        .iter()
        .map(|&(index, mine)| touch(&leaf, index, mine))
        .collect();
    // Changes only to the last segment are appends.
    let split = match touched.as_slice() {
        [only] if only.index + 1 == leaf.len() => Split::Fill,
        _ => Split::Even,
    };
    in_place(&leaf, &touched)
        .unwrap_or_else(|| Update::Replace(pack(&gather(&leaf, touched), split)))
}

/// A change written over one segment in its own slot.
pub(super) struct Patch {
    index: usize,
    patched: Patched,
}

impl Patch {
    /// Calls `write` with each `(offset, bytes)` the patch writes: the changed code bytes, and
    /// the segment's table entry.
    pub(super) fn write(&self, mut write: impl FnMut(usize, &[u8])) {
        let (offset, bytes) = self.patched.changed();
        write(offset, bytes);
        write(table_end(self.index), &self.patched.entry.to_bytes());
    }
}

/// The patch that lowers the count of the posting at `key` by one in place, or `None` when the
/// leaf lacks it.
pub(super) fn lowered(read: Read<'_>, key: (u32, u32)) -> Option<Patch> {
    let leaf = Leaf::open(read);
    patch(&leaf, leaf.segment_of(key), &[Change { key, delta: -1 }])
}

/// The patch that raises the count of the posting at `key` by one in its segment's slot (adding
/// the posting), or `None` when it does not fit there.
pub(super) fn raised(read: Read<'_>, key: (u32, u32)) -> Option<Patch> {
    let leaf = Leaf::open(read);
    let index = leaf.segment_of(key);
    let patched = leaf.entry(index)?.raise(read, key)?;
    Some(Patch { index, patched })
}

#[inline]
fn patch(leaf: &Leaf<'_>, index: usize, changes: &[Change]) -> Option<Patch> {
    let patched = leaf.entry(index)?.lower(leaf.read, changes)?;
    Some(Patch { index, patched })
}

/// The writes that apply every group's decrements by lowering counts in place, or `None` when one
/// of them names a posting its segment lacks.
fn patched(leaf: &Leaf<'_>, groups: &[(usize, &[Change])]) -> Option<Vec<(usize, Vec<u8>)>> {
    let mut writes = Vec::with_capacity(2 * groups.len());
    for &(index, changes) in groups {
        patch(leaf, index, changes)?.write(|at, bytes| writes.push((at, bytes.to_vec())));
    }
    Some(writes)
}

/// Segment `index` re-coded with `changes`; a single replacement that still fits keeps its slot.
fn touch(leaf: &Leaf<'_>, index: usize, changes: &[Change]) -> Touched {
    let old = leaf.entry(index).unwrap_or_else(|| corrupt("index leaf"));
    let by = rebuild(&merge(&old.decode(leaf.read), changes));
    let keep = match by.as_slice() {
        [one] => Some(old.slot()).filter(|&(_, cap)| one.bytes.len() <= cap),
        _ => None,
    };
    Touched {
        index,
        old,
        by,
        keep,
    }
}
