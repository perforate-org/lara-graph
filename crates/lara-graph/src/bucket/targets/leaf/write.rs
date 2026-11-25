//! Writing changed segments: in place, into the leaf's free space, or by rewriting the leaf
//! (splitting it when it outgrows a page).

use super::segment::{Coded, Entry};
use super::{CAPACITY, ENTRY_BYTES, HEAD_BYTES, Image, Leaf, Update, header, table_end};
use crate::store::corrupt::corrupt;
use alloc::vec::Vec;

/// A segment the changes touched: its index and entry, what replaces it, and the slot a single
/// replacement may stay in.
pub(super) struct Touched {
    pub(super) index: usize,
    pub(super) old: Entry,
    pub(super) by: Vec<Coded>,
    pub(super) keep: Option<(usize, usize)>,
}

/// A write of `bytes` at an offset into the leaf.
type Write = (usize, Vec<u8>);

/// Replacement segments placed in the leaf: each replaced index with the new entries, the slot
/// writes, and the slots' new low end.
struct Placed {
    entries: Vec<(usize, Vec<Entry>)>,
    writes: Vec<Write>,
    low: usize,
}

/// The writes that apply `touched` to `leaf` on its own page, or `None` when they do not fit in
/// its free space (or the leaf empties).
pub(super) fn in_place(leaf: &Leaf<'_>, touched: &[Touched]) -> Option<Update> {
    let segments = leaf.len() + touched.iter().map(|t| t.by.len()).sum::<usize>() - touched.len();
    if segments == 0 {
        return None;
    }
    let Placed {
        entries: placed,
        mut writes,
        low,
    } = place(leaf.low, touched)?;
    if low < table_end(segments) {
        return None;
    }
    if low != leaf.low || segments != leaf.len() {
        writes.push((0, header(segments, low).to_vec()));
    }
    writes.extend(table_writes(leaf, &placed, segments));
    let delta = placed
        .iter()
        .flat_map(|p| &p.1)
        .map(|e| i64::from(e.total))
        .sum::<i64>()
        - touched.iter().map(|t| i64::from(t.old.total)).sum::<i64>();
    Some(Update::InPlace { writes, delta })
}

/// Each replacement's table entries after placing its slots (its own slot when it may keep it,
/// else new ones taken from the free space below `low`), the slot writes, and the new low end.
fn place(mut low: usize, touched: &[Touched]) -> Option<Placed> {
    let mut writes = Vec::with_capacity(touched.len() * 2 + 1);
    let mut placed = Vec::with_capacity(touched.len());
    for t in touched {
        let mut entries = Vec::with_capacity(t.by.len());
        for c in &t.by {
            let slot = if let Some(slot) = t.keep {
                slot
            } else {
                let cap = with_slack(c.bytes.len());
                low = low.checked_sub(cap)?;
                (low, cap)
            };
            writes.push((slot.0, c.bytes.clone()));
            entries.push(Entry::placed(c, slot));
        }
        placed.push((t.index, entries));
    }
    Some(Placed {
        entries: placed,
        writes,
        low,
    })
}

/// The table writes: just the replaced entries when each segment was replaced by one, else the
/// table from the first replaced entry on, as the later ones shift.
fn table_writes(leaf: &Leaf<'_>, placed: &[(usize, Vec<Entry>)], segments: usize) -> Vec<Write> {
    if segments == leaf.len() {
        return placed
            .iter()
            .flat_map(|(index, entries)| {
                entries
                    .iter()
                    .map(move |e| (table_end(*index), e.to_bytes().to_vec()))
            })
            .collect();
    }
    let from = placed.first().map_or(0, |p| p.0);
    let old = leaf.raw(from..leaf.len());
    let raw = |i: usize| {
        old.get((i - from) * ENTRY_BYTES..(i - from + 1) * ENTRY_BYTES)
            .unwrap_or_default()
    };
    let mut table = Vec::with_capacity((segments - from) * ENTRY_BYTES);
    let mut next = from;
    for (index, entries) in placed {
        table.extend((next..*index).flat_map(raw));
        table.extend(entries.iter().flat_map(|e| e.to_bytes()));
        next = index + 1;
    }
    table.extend((next..leaf.len()).flat_map(raw));
    alloc::vec![(table_end(from), table)]
}

/// A slot's capacity for `len` coded bytes: room for a few more postings.
const fn with_slack(len: usize) -> usize {
    len + len / 4 + 8
}

/// Every segment of the leaf as coded bytes, reading untouched ones from their slots, with
/// whether it was just re-coded (and so gets slack).
pub(super) fn gather(leaf: &Leaf<'_>, touched: Vec<Touched>) -> Vec<(Coded, bool)> {
    let mut out = Vec::with_capacity(leaf.len() + touched.len());
    let mut next = 0;
    let kept = |i: usize| {
        let e = leaf.entry(i).unwrap_or_else(|| corrupt("index leaf"));
        (e.kept(leaf.read), false)
    };
    for t in touched {
        out.extend((next..t.index).map(kept));
        out.extend(t.by.into_iter().map(|c| (c, true)));
        next = t.index + 1;
    }
    out.extend((next..leaf.len()).map(kept));
    out
}

/// The slot capacity a segment gets in a freshly written leaf: at least a byte, so that the slot
/// starts inside the leaf even for an empty code (one posting without count or block bits).
fn cap_of((coded, fresh): &(Coded, bool)) -> usize {
    if *fresh {
        with_slack(coded.bytes.len())
    } else {
        coded.bytes.len().max(1)
    }
}

fn size(segments: &[(Coded, bool)]) -> usize {
    HEAD_BYTES
        + segments
            .iter()
            .map(|s| ENTRY_BYTES + cap_of(s))
            .sum::<usize>()
}

/// A leaf holding `segments`, slots packed down from its end.
fn image(segments: &[(Coded, bool)]) -> Image {
    let mut low = CAPACITY;
    let mut bytes = alloc::vec![0u8; CAPACITY];
    let mut table = Vec::with_capacity(segments.len() * ENTRY_BYTES);
    for s in segments {
        let cap = cap_of(s);
        low -= cap;
        if let Some(slot) = bytes.get_mut(low..low + s.0.bytes.len()) {
            slot.copy_from_slice(&s.0.bytes);
        }
        table.extend(Entry::placed(&s.0, (low, cap)).to_bytes());
    }
    let head = header(segments.len(), low);
    if let Some(dst) = bytes.get_mut(..head.len()) {
        dst.copy_from_slice(&head);
    }
    if let Some(dst) = bytes.get_mut(head.len()..head.len() + table.len()) {
        dst.copy_from_slice(&table);
    }
    Image {
        first: segments.first().map_or((0, 0), |s| s.0.first),
        count: segments.iter().map(|s| s.0.total).sum(),
        bytes,
    }
}

/// How a rewritten leaf's segments are split into leaves.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Split {
    /// Changes landed at the end: earlier leaves are filled, as appends will not reach them.
    Fill,
    /// Changes landed anywhere: the leaves share the segments evenly.
    Even,
}

/// Fill a rewritten leaf may keep before it splits, leaving room for later changes in place.
const FULL: usize = CAPACITY * 7 / 8;

/// Splits segments into leaves.
pub(super) fn pack(segments: &[(Coded, bool)], split: Split) -> Vec<Image> {
    if segments.is_empty() {
        return Vec::new();
    }
    let total = size(segments);
    if total <= FULL {
        return alloc::vec![image(segments)];
    }
    let chunks = match split {
        Split::Fill => filled(segments),
        Split::Even => even(segments, total),
    };
    chunks.into_iter().map(image).collect()
}

/// Leaves filled up to [`FULL`] in order.
fn filled(segments: &[(Coded, bool)]) -> Vec<&[(Coded, bool)]> {
    let mut chunks = Vec::new();
    let mut rest = segments;
    while !rest.is_empty() {
        let mut used = HEAD_BYTES;
        let take = rest
            .iter()
            .take_while(|s| {
                used += ENTRY_BYTES + cap_of(s);
                used <= FULL
            })
            .count()
            .max(1);
        let (chunk, later) = rest.split_at(take.min(rest.len()));
        chunks.push(chunk);
        rest = later;
    }
    chunks
}

/// At least two leaves sharing the segments evenly, each within a page.
fn even(segments: &[(Coded, bool)], total: usize) -> Vec<&[(Coded, bool)]> {
    let mut parts = total.div_ceil(FULL).max(2);
    loop {
        let per = segments.len().div_ceil(parts);
        let chunks: Vec<&[(Coded, bool)]> = segments.chunks(per).collect();
        if chunks.iter().all(|c| size(c) <= CAPACITY) {
            return chunks;
        }
        parts += 1;
        if parts > segments.len() {
            corrupt("an index segment larger than a leaf");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::targets::Posting;
    use crate::bucket::targets::leaf::segment::rebuild;

    /// A posting alone, with a count of one, codes to no bits at all. Kept as it is in a rewritten
    /// leaf, its slot still starts inside the leaf, where its table entry can name it.
    #[test]
    fn an_empty_code_gets_a_slot_inside_the_leaf() {
        let lone = rebuild(&[Posting {
            target: 7,
            block: 3,
            count: 1,
        }]);
        let [coded] = lone.as_slice() else {
            panic!("one segment")
        };
        assert!(coded.bytes.is_empty(), "no count or block bits");
        let image = image(&[(coded.clone(), false)]);
        let read = |offset: usize, buf: &mut [u8]| {
            buf.copy_from_slice(&image.bytes[offset..offset + buf.len()]);
        };
        let entry = Leaf::open(&read).entry(0).unwrap();
        assert!(entry.offset < CAPACITY, "slot at {}", entry.offset);
    }
}
