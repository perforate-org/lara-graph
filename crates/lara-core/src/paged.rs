//! The page index of a paged bucket.
//!
//! Data pages hold 1024 slots each; position `p` is slot `p % 1024` of data page `p / 1024`.
//! The index lists data pages in order. Each entry is 8 bytes: a page id, a count and a slot.
//! - At the bottom level, the count is the data page's tombstones, so a scan skips dead pages
//!   without reading them, and the slot is the page's first hole, so an insert that reuses holes
//!   finds one without reading the page.
//! - Above it, the count is how many of the node's children hold a hole an insert can fill, so a
//!   search for one goes straight to it.
//!
//! The index is a tree of fixed fanout [`FANOUT`] (512 entries fill a page). Its depth follows
//! from the number of data pages and is never stored. The top level is an extent sized to its
//! entries; every lower node is a whole page. Growth is bounded: the top grows by copying at most
//! one page of entries, and a full one-page top becomes a child of a new top without copying.

use crate::corrupt::corrupt;
use crate::descriptor::{PAGE_SLOTS, Paged};
use crate::memory::{Memory, WriteError};
use crate::page_store::{ByteClass, Extent, PageId, PageStore, Pages};
use crate::txn::SpanTxn;
use crate::units::ByteOffset;
use alloc::vec::Vec;
use core::ops::ControlFlow;

mod growth;

pub(crate) use growth::append_page;

/// Bytes per index entry.
const ENTRY: usize = 8;
const ENTRY_BYTES: u32 = 8;
/// Entries per index page.
pub(crate) const FANOUT: u32 = 4096 / ENTRY_BYTES;
/// The first-hole slot of a page without holes.
pub(crate) const NO_HOLE: u16 = 1024;

/// A data page, its tombstone count and the slot of its first hole ([`NO_HOLE`] for none). A
/// hollow entry (`page: None`) stands for a page whose positions are all dead: its page was
/// released without renumbering anything after it.
///
/// An entry above the bottom level is read as one too: its page is the child node, and its count
/// is how many of the child's entries have a fillable hole below them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Leaf {
    pub(crate) page: Option<PageId>,
    pub(crate) tombstones: u16,
    pub(crate) first_hole: u16,
}

/// Whether an entry at `level` has a hole an insert can fill below it: a present data page with a
/// tombstone, or a node with such a child. A hollow page's holes are not filled: its page is gone.
const fn fillable(level: u32, entry: Leaf) -> bool {
    entry.tombstones > 0 && (level > 1 || entry.page.is_some())
}

impl Leaf {
    /// A data page with `tombstones` holes, the first at slot `first_hole`.
    pub(crate) const fn data(page: PageId, tombstones: u16, first_hole: u16) -> Self {
        Self {
            page: Some(page),
            tombstones,
            first_hole,
        }
    }

    /// A node entry: its child, and how many of the child's entries have a fillable hole.
    const fn node(page: PageId, holed: u16) -> Self {
        Self {
            page: Some(page),
            tombstones: holed,
            first_hole: NO_HOLE,
        }
    }

    /// The page, where one must exist (index nodes, the last data page).
    pub(crate) fn present(self) -> PageId {
        self.page
            .unwrap_or_else(|| corrupt("hollow entry where a page must exist"))
    }
}

/// Encoded page id of a hollow entry.
const HOLLOW: u32 = u32::MAX;

/// Data pages a bucket of `stored` positions uses.
pub(crate) fn data_pages(stored: u32) -> u32 {
    stored.div_ceil(PAGE_SLOTS)
}

/// Index levels needed for `pages` data pages.
fn depth(pages: u32) -> u32 {
    let mut depth = 1;
    let mut reach = u64::from(FANOUT);
    while u64::from(pages) > reach {
        depth += 1;
        reach *= u64::from(FANOUT);
    }
    depth
}

/// Data pages one entry at `level` covers (level 1 is the bottom).
fn span(level: u32) -> u64 {
    u64::from(FANOUT).pow(level - 1)
}

fn entry_at(node: ByteOffset, index: u64) -> ByteOffset {
    ByteOffset::new(node.get() + index * u64::from(ENTRY_BYTES))
}

fn encode(leaf: Leaf) -> [u8; ENTRY] {
    let [p0, p1, p2, p3] = leaf.page.map_or(HOLLOW, PageId::get).to_le_bytes();
    let [t0, t1] = leaf.tombstones.to_le_bytes();
    let [h0, h1] = leaf.first_hole.to_le_bytes();
    [p0, p1, p2, p3, t0, t1, h0, h1]
}

fn decode(bytes: [u8; ENTRY]) -> Leaf {
    let [p0, p1, p2, p3, t0, t1, h0, h1] = bytes;
    let raw = u32::from_le_bytes([p0, p1, p2, p3]);
    Leaf {
        page: (raw != HOLLOW).then_some(PageId::new(raw)),
        tombstones: u16::from_le_bytes([t0, t1]),
        first_hole: u16::from_le_bytes([h0, h1]),
    }
}

fn index_class(entries: u32) -> ByteClass {
    ByteClass::for_bytes(entries * ENTRY_BYTES)
        .unwrap_or_else(|| corrupt("index level beyond a page"))
}

/// One index node during a walk: where it is, its level, and the first data page it covers.
#[derive(Clone, Copy)]
struct Node {
    offset: ByteOffset,
    level: u32,
    first: u64,
}

impl Node {
    fn top(paged: &Paged) -> Self {
        Self {
            offset: paged.index.offset(),
            level: depth(data_pages(paged.stored)),
            first: 0,
        }
    }
}

/// Whether a walk goes below a node entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Descend {
    Into,
    Past,
}

/// Entries read at a time: a walk that stops early reads little of a large node.
const WALK_ENTRIES: u64 = 64;

/// Visits the entries under `node` whose subtrees hold data pages numbered `from` or later, in
/// order, parents before their children, as `(level, first data page covered, entry)`, until
/// `on_entry` breaks; it says for each node entry whether to go below it. Level-1 entries are
/// data pages. Reads each node from its first such entry, a window at a time.
fn walk<M: Memory, B>(
    view: &Pages<'_, M>,
    node: Node,
    total: u32,
    from: u64,
    on_entry: &mut impl FnMut(u32, u64, Leaf) -> ControlFlow<B, Descend>,
) -> ControlFlow<B> {
    let covered = span(node.level);
    let entries = (u64::from(total) - node.first)
        .div_ceil(covered)
        .min(u64::from(FANOUT));
    let mut start = from.saturating_sub(node.first) / covered;
    while start < entries {
        let end = (start + WALK_ENTRIES).min(entries);
        let len = usize::try_from((end - start) * u64::from(ENTRY_BYTES))
            .unwrap_or_else(|_| corrupt("index"));
        let bytes = view.read(entry_at(node.offset, start), len);
        for (i, &raw) in (start..).zip(bytes.as_chunks::<ENTRY>().0) {
            let entry = decode(raw);
            let first = node.first + i * covered;
            let descend = on_entry(node.level, first, entry)?;
            if node.level > 1 && descend == Descend::Into {
                let child = Node {
                    offset: entry.present().offset(),
                    level: node.level - 1,
                    first,
                };
                walk(view, child, total, from, on_entry)?;
            }
        }
        start = end;
    }
    ControlFlow::Continue(())
}

/// Visits every index entry, as `(level, entry)`.
fn every_entry<M: Memory>(view: &Pages<'_, M>, paged: &Paged, mut f: impl FnMut(u32, Leaf)) {
    let _: ControlFlow<()> = walk(
        view,
        Node::top(paged),
        data_pages(paged.stored),
        0,
        &mut |level, _, entry| {
            f(level, entry);
            ControlFlow::Continue(Descend::Into)
        },
    );
}

/// Calls `f` with each data page numbered `from` or later, as `(number, leaf)`, until it
/// breaks. Reads only the index entries from there on.
pub(crate) fn visit_leaves<M: Memory, B>(
    view: &Pages<'_, M>,
    paged: &Paged,
    from: u32,
    mut f: impl FnMut(u32, Leaf) -> ControlFlow<B>,
) -> ControlFlow<B> {
    walk(
        view,
        Node::top(paged),
        data_pages(paged.stored),
        u64::from(from),
        &mut |level, first, entry| {
            if level == 1 {
                f(page_number(first), entry)?;
            }
            ControlFlow::Continue(Descend::Into)
        },
    )
}

/// Every data page in order, with its tombstone count.
pub(crate) fn leaves<M: Memory>(view: &Pages<'_, M>, paged: &Paged) -> Vec<Leaf> {
    let mut out = Vec::new();
    every_entry(view, paged, |level, entry| {
        if level == 1 {
            out.push(entry);
        }
    });
    out
}

/// Levels the index of any bucket can have: three cover every `u32` position.
const MAX_DEPTH: usize = 3;

/// The entries on the way to data page `number`, top first, ending at its bottom-level entry.
fn path<M: Memory>(view: &Pages<'_, M>, paged: &Paged, number: u32) -> Path {
    let mut entries = [ByteOffset::new(0); MAX_DEPTH];
    let mut node = paged.index.offset();
    let depth = depth(data_pages(paged.stored));
    for (slot, level) in entries.iter_mut().zip((1..=depth).rev()) {
        let at = entry_at(node, (u64::from(number) / span(level)) % u64::from(FANOUT));
        *slot = at;
        if level > 1 {
            node = decode(view.read_array(at)).present().offset();
        }
    }
    let len = usize::try_from(depth).unwrap_or(MAX_DEPTH);
    if len > MAX_DEPTH {
        corrupt("index depth");
    }
    Path { entries, len }
}

/// The index entries above and at a data page's, top first.
struct Path {
    entries: [ByteOffset; MAX_DEPTH],
    len: usize,
}

impl Path {
    fn entries(&self) -> &[ByteOffset] {
        self.entries.get(..self.len).unwrap_or(&[])
    }

    /// The data page's own entry.
    fn bottom(&self) -> ByteOffset {
        self.entries()
            .last()
            .copied()
            .unwrap_or_else(|| corrupt("index depth"))
    }
}

/// Where the bottom-level entry of data page `number` is.
fn leaf_entry<M: Memory>(view: &Pages<'_, M>, paged: &Paged, number: u32) -> ByteOffset {
    path(view, paged, number).bottom()
}

/// Rewrites data page `number`'s entry with `make` applied to it, and keeps the counts above it
/// in step when whether it has a fillable hole changes. Returns the old entry.
fn rewrite_leaf<M: Memory>(
    store: &PageStore<M>,
    paged: &Paged,
    number: u32,
    make: impl FnOnce(Leaf) -> Leaf,
) -> Leaf {
    let view = store.pages();
    let path = path(&view, paged, number);
    let at = path.bottom();
    let old = decode(view.read_array(at));
    let new = make(old);
    store.write(at, &encode(new));
    let mut gained = match (fillable(1, old), fillable(1, new)) {
        (false, true) => Some(true),
        (true, false) => Some(false),
        (true, true) | (false, false) => None,
    };
    for &above in path.entries().iter().rev().skip(1) {
        let Some(gain) = gained else { break };
        let entry = decode(view.read_array(above));
        let count = if gain {
            entry.tombstones.checked_add(1)
        } else {
            entry.tombstones.checked_sub(1)
        }
        .unwrap_or_else(|| corrupt("index hole count"));
        store.write(above, &encode(Leaf::node(entry.present(), count)));
        // The node above sees a change only when this one starts or stops having holes.
        gained = ((entry.tombstones == 0) != (count == 0)).then_some(gain);
    }
    old
}

/// The first data page numbered `from` or later with a hole an insert can fill, and its entry.
/// Goes below a node entry only when its count says a child has one; a child that holds `from`
/// may still have none at `from` or later, so the walk goes on past it.
pub(crate) fn first_fillable<M: Memory>(
    view: &Pages<'_, M>,
    paged: &Paged,
    from: u32,
) -> Option<(u32, Leaf)> {
    let found = walk(
        view,
        Node::top(paged),
        data_pages(paged.stored),
        u64::from(from),
        &mut |level, first, entry| match (fillable(level, entry), level) {
            (false, _) => ControlFlow::Continue(Descend::Past),
            (true, 1) => ControlFlow::Break((page_number(first), entry)),
            (true, _) => ControlFlow::Continue(Descend::Into),
        },
    );
    match found {
        ControlFlow::Break(found) => Some(found),
        ControlFlow::Continue(()) => None,
    }
}

fn page_number(first: u64) -> u32 {
    u32::try_from(first).unwrap_or_else(|_| corrupt("page number"))
}

/// Data page `number` of a paged bucket.
pub(crate) fn data_page<M: Memory>(view: &Pages<'_, M>, paged: &Paged, number: u32) -> Leaf {
    decode(view.read_array(leaf_entry(view, paged, number)))
}

/// Records that the edge at `slot` of data page `number` was removed.
pub(crate) fn removed<M: Memory>(store: &PageStore<M>, paged: &Paged, number: u32, slot: u16) {
    rewrite_leaf(store, paged, number, |leaf| Leaf {
        tombstones: leaf
            .tombstones
            .checked_add(1)
            .unwrap_or_else(|| corrupt("tombstone count")),
        first_hole: leaf.first_hole.min(slot),
        ..leaf
    });
}

/// Records that the hole at `slot` of data page `number` was filled. When it was the page's
/// first, `next_hole` finds the next one: the first after `slot`.
pub(crate) fn filled<M: Memory>(
    store: &PageStore<M>,
    paged: &Paged,
    number: u32,
    slot: u16,
    next_hole: impl FnOnce() -> u16,
) {
    rewrite_leaf(store, paged, number, |leaf| {
        let tombstones = leaf
            .tombstones
            .checked_sub(1)
            .unwrap_or_else(|| corrupt("tombstone count"));
        let first_hole = match (tombstones, leaf.first_hole == slot) {
            (0, _) => NO_HOLE,
            (_, true) => next_hole(),
            (_, false) => leaf.first_hole,
        };
        Leaf {
            tombstones,
            first_hole,
            ..leaf
        }
    });
}

/// Releases data page `number` (all of its positions are dead) and leaves a hollow entry, so
/// no later position moves.
pub(crate) fn hollow<M: Memory>(store: &mut PageStore<M>, paged: &Paged, number: u32) {
    let leaf = rewrite_leaf(store, paged, number, |_| Leaf {
        page: None,
        tombstones: narrow(PAGE_SLOTS),
        first_hole: 0,
    });
    store.release(Extent::whole(leaf.present()));
}

/// Overwrites data page `number`'s tombstone count and first hole.
pub(crate) fn set_holes<M: Memory>(
    store: &PageStore<M>,
    paged: &Paged,
    number: u32,
    (tombstones, first_hole): (u16, u16),
) {
    rewrite_leaf(store, paged, number, |leaf| Leaf {
        tombstones,
        first_hole,
        ..leaf
    });
}

/// Every index node below the top (whole pages), for releasing a demoted bucket's index.
pub(crate) fn index_nodes<M: Memory>(view: &Pages<'_, M>, paged: &Paged) -> Vec<PageId> {
    let mut out = Vec::new();
    every_entry(view, paged, |level, entry| {
        if level > 1 {
            out.push(entry.present());
        }
    });
    out
}

/// Bytes of `entries` index entries.
fn byte_len_of(entries: u32) -> usize {
    usize::try_from(entries * ENTRY_BYTES).unwrap_or_else(|_| corrupt("index"))
}

fn stored_count(count: usize) -> u32 {
    u32::try_from(count).unwrap_or_else(|_| corrupt("index entry count"))
}

fn narrow(count: u32) -> u16 {
    u16::try_from(count).unwrap_or_else(|_| corrupt("page slot count"))
}

/// The index of a new paged bucket: the full run's page first (with its tombstones), then a new
/// empty page. Returns the index and the new page.
pub(crate) fn start<M: Memory>(
    store: &mut PageStore<M>,
    first: Leaf,
) -> Result<(Extent, PageId), WriteError> {
    let mut txn = SpanTxn::begin(store);
    let page = txn.allocate(ByteClass::PAGE)?;
    let index = txn.allocate(index_class(2))?;
    txn.commit();
    store.write(entry_at(index.offset(), 0), &encode(first));
    store.write(
        entry_at(index.offset(), 1),
        &encode(Leaf::data(page.page(), 0, NO_HOLE)),
    );
    Ok((index, page.page()))
}

/// Checks that every count above the bottom level is the number of its child's entries with a
/// fillable hole (invariant checks).
#[cfg(test)]
pub(crate) fn check_hole_counts<M: Memory>(view: &Pages<'_, M>, paged: &Paged) {
    fn count<M: Memory>(view: &Pages<'_, M>, node: Node, total: u32) -> u16 {
        let mut holed = 0u16;
        let _: ControlFlow<()> = walk(view, node, total, node.first, &mut |level, first, entry| {
            if level == node.level {
                if level > 1 {
                    let child = Node {
                        offset: entry.present().offset(),
                        level: level - 1,
                        first,
                    };
                    assert_eq!(count(view, child, total), entry.tombstones, "level {level}");
                }
                holed += u16::from(fillable(level, entry));
            }
            ControlFlow::Continue(Descend::Past)
        });
        holed
    }
    count(view, Node::top(paged), data_pages(paged.stored));
}

/// Every extent the index itself uses, for invariant checks: the top and every lower node.
#[cfg(test)]
pub(crate) fn index_extents<M: Memory>(view: &Pages<'_, M>, paged: &Paged) -> Vec<Extent> {
    let mut out = alloc::vec![paged.index];
    every_entry(view, paged, |level, entry| {
        if level > 1 {
            out.push(Extent::whole(entry.present()));
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_agree() {
        assert_eq!(u32::from(NO_HOLE), PAGE_SLOTS);
        assert_eq!(u32::try_from(ENTRY).unwrap(), ENTRY_BYTES);
    }
}
