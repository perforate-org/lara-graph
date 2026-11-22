//! The page index of a paged bucket.
//!
//! Data pages hold 1024 slots each; position `p` is slot `p % 1024` of data page `p / 1024`.
//! The index lists data pages in order. Each entry is 6 bytes: a page id and, at the bottom
//! level, that data page's tombstone count (so a scan skips dead pages without reading them).
//!
//! The index is a tree of fixed fanout [`FANOUT`] (682 entries fill a page). Its depth follows
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

/// Bytes per index entry.
const ENTRY_BYTES: u32 = 6;
/// Entries per index page.
pub(crate) const FANOUT: u32 = 682;

/// A data page and its tombstone count. A hollow entry (`page: None`) stands for a page whose
/// positions are all dead: its page was released without renumbering anything after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Leaf {
    pub(crate) page: Option<PageId>,
    pub(crate) tombstones: u16,
}

impl Leaf {
    pub(crate) const fn live(page: PageId, tombstones: u16) -> Self {
        Self {
            page: Some(page),
            tombstones,
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

fn encode(leaf: Leaf) -> [u8; 6] {
    let [p0, p1, p2, p3] = leaf.page.map_or(HOLLOW, PageId::get).to_le_bytes();
    let [t0, t1] = leaf.tombstones.to_le_bytes();
    [p0, p1, p2, p3, t0, t1]
}

fn decode(bytes: [u8; 6]) -> Leaf {
    let [p0, p1, p2, p3, t0, t1] = bytes;
    let raw = u32::from_le_bytes([p0, p1, p2, p3]);
    Leaf {
        page: (raw != HOLLOW).then_some(PageId::new(raw)),
        tombstones: u16::from_le_bytes([t0, t1]),
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

/// Visits every entry under `node` in order, parents before their children, as
/// `(level, entry)`; level-1 entries are data pages.
fn walk<M: Memory>(
    view: &Pages<'_, M>,
    node: Node,
    total: u32,
    on_entry: &mut impl FnMut(u32, Leaf),
) {
    let covered = span(node.level);
    let entries = (u64::from(total) - node.first)
        .div_ceil(covered)
        .min(u64::from(FANOUT));
    let len =
        usize::try_from(entries * u64::from(ENTRY_BYTES)).unwrap_or_else(|_| corrupt("index"));
    let bytes = view.read(node.offset, len);
    for (i, &raw) in (0u64..).zip(bytes.as_chunks::<6>().0) {
        let entry = decode(raw);
        on_entry(node.level, entry);
        if node.level > 1 {
            let child = Node {
                offset: entry.present().offset(),
                level: node.level - 1,
                first: node.first + i * covered,
            };
            walk(view, child, total, on_entry);
        }
    }
}

/// Every data page in order, with its tombstone count.
pub(crate) fn leaves<M: Memory>(view: &Pages<'_, M>, paged: &Paged) -> Vec<Leaf> {
    let mut out = Vec::new();
    walk(
        view,
        Node::top(paged),
        data_pages(paged.stored),
        &mut |level, entry| {
            if level == 1 {
                out.push(entry);
            }
        },
    );
    out
}

/// Where the bottom-level entry of data page `number` is.
fn leaf_entry<M: Memory>(view: &Pages<'_, M>, paged: &Paged, number: u32) -> ByteOffset {
    let mut node = paged.index.offset();
    for level in (1..=depth(data_pages(paged.stored))).rev() {
        let at = entry_at(node, (u64::from(number) / span(level)) % u64::from(FANOUT));
        if level == 1 {
            return at;
        }
        node = decode(view.read_array(at)).present().offset();
    }
    corrupt("index depth")
}

/// Data page `number` of a paged bucket.
pub(crate) fn data_page<M: Memory>(view: &Pages<'_, M>, paged: &Paged, number: u32) -> Leaf {
    decode(view.read_array(leaf_entry(view, paged, number)))
}

/// Whether a data page gains or loses a tombstone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TombstoneChange {
    /// An edge was removed.
    Add,
    /// A hole was filled.
    Remove,
}

/// Updates the tombstone count of data page `number`.
pub(crate) fn change_tombstones<M: Memory>(
    store: &PageStore<M>,
    paged: &Paged,
    number: u32,
    change: TombstoneChange,
) {
    let at = leaf_entry(&store.pages(), paged, number);
    let leaf = decode(store.pages().read_array(at));
    let tombstones = match change {
        TombstoneChange::Add => leaf.tombstones.checked_add(1),
        TombstoneChange::Remove => leaf.tombstones.checked_sub(1),
    };
    let tombstones = tombstones.unwrap_or_else(|| corrupt("tombstone count"));
    store.write(at, &encode(Leaf { tombstones, ..leaf }));
}

/// Releases data page `number` (all of its positions are dead) and leaves a hollow entry, so
/// no later position moves.
pub(crate) fn hollow<M: Memory>(store: &mut PageStore<M>, paged: &Paged, number: u32) {
    let at = leaf_entry(&store.pages(), paged, number);
    let leaf = decode(store.pages().read_array(at));
    store.write(
        at,
        &encode(Leaf {
            page: None,
            tombstones: narrow(PAGE_SLOTS),
        }),
    );
    store.release(Extent::whole(leaf.present()));
}

/// Overwrites data page `number`'s tombstone count.
pub(crate) fn set_tombstones<M: Memory>(
    store: &PageStore<M>,
    paged: &Paged,
    number: u32,
    tombstones: u16,
) {
    let at = leaf_entry(&store.pages(), paged, number);
    let leaf = decode(store.pages().read_array(at));
    store.write(at, &encode(Leaf { tombstones, ..leaf }));
}

/// Every index node below the top (whole pages), for releasing a demoted bucket's index.
pub(crate) fn index_nodes<M: Memory>(view: &Pages<'_, M>, paged: &Paged) -> Vec<PageId> {
    let mut out = Vec::new();
    walk(
        view,
        Node::top(paged),
        data_pages(paged.stored),
        &mut |level, entry| {
            if level > 1 {
                out.push(entry.present());
            }
        },
    );
    out
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
        &encode(Leaf::live(page.page(), 0)),
    );
    Ok((index, page.page()))
}

/// How adding a data page changes the index's top level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopChange {
    /// The top has room.
    Keep,
    /// The top moves to an extent for this many entries (copying at most one page).
    Grow { entries: u32 },
    /// The full one-page top becomes the first child of a new two-entry top.
    Deepen,
}

/// What adding data page `number` needs.
struct Growth {
    number: u32,
    depth: u32,
    top: TopChange,
    /// Levels below the top that need a new node, bottom first.
    fresh_levels: Vec<u32>,
}

/// The allocations a growth made.
struct Grown {
    data: PageId,
    top: Extent,
    fresh: Vec<(u32, PageId)>,
}

fn plan(paged: &Paged) -> Growth {
    let number = data_pages(paged.stored);
    let (old_depth, depth) = (depth(number), depth(number + 1));
    let fresh_levels = (1..depth.min(old_depth + 1))
        .filter(|&level| u64::from(number) % span(level + 1) == 0)
        .collect();
    let top = if depth > old_depth {
        TopChange::Deepen
    } else {
        let entries =
            u32::try_from(u64::from(number + 1).div_ceil(span(old_depth))).unwrap_or(FANOUT);
        if entries * ENTRY_BYTES > paged.index.len() {
            TopChange::Grow { entries }
        } else {
            TopChange::Keep
        }
    };
    Growth {
        number,
        depth,
        top,
        fresh_levels,
    }
}

/// Adds a data page after the last (full) one. Returns the index's new top and the new page.
pub(crate) fn append_page<M: Memory>(
    store: &mut PageStore<M>,
    paged: &Paged,
) -> Result<(Extent, PageId), WriteError> {
    let growth = plan(paged);
    let grown = allocate(store, &growth, paged.index)?;
    // Everything below writes into existing or newly allocated space: nothing fails.
    match growth.top {
        TopChange::Keep => {}
        TopChange::Deepen => {
            let first = Leaf::live(paged.index.page(), 0);
            store.write(entry_at(grown.top.offset(), 0), &encode(first));
        }
        TopChange::Grow { .. } => {
            let entries = u64::from(growth.number).div_ceil(span(growth.depth));
            let len = usize::try_from(entries * u64::from(ENTRY_BYTES))
                .unwrap_or_else(|_| corrupt("index"));
            let bytes = store.pages().read(paged.index.offset(), len);
            store.write(grown.top.offset(), &bytes);
            store.release(paged.index);
        }
    }
    link_path(store, &growth, &grown);
    Ok((grown.top, grown.data))
}

/// Allocates the data page, fresh nodes and new top together: all or nothing.
fn allocate<M: Memory>(
    store: &mut PageStore<M>,
    growth: &Growth,
    old_top: Extent,
) -> Result<Grown, WriteError> {
    let mut txn = SpanTxn::begin(store);
    let data = txn.allocate(ByteClass::PAGE)?.page();
    let mut fresh = Vec::new();
    for &level in &growth.fresh_levels {
        fresh.push((level, txn.allocate(ByteClass::PAGE)?.page()));
    }
    let top = match growth.top {
        TopChange::Keep => old_top,
        TopChange::Grow { entries } => txn.allocate(index_class(entries))?,
        TopChange::Deepen => txn.allocate(index_class(2))?,
    };
    txn.commit();
    Ok(Grown { data, top, fresh })
}

/// Walks the new page's path from the top, linking fresh nodes and writing its entry.
fn link_path<M: Memory>(store: &PageStore<M>, growth: &Growth, grown: &Grown) {
    let number = u64::from(growth.number);
    let mut node = grown.top.offset();
    for level in (2..=growth.depth).rev() {
        let at = entry_at(node, (number / span(level)) % u64::from(FANOUT));
        let child = match grown.fresh.iter().find(|(l, _)| *l == level - 1) {
            Some(&(_, page)) => {
                store.write(at, &encode(Leaf::live(page, 0)));
                page
            }
            None => decode(store.pages().read_array(at)).present(),
        };
        node = child.offset();
    }
    let at = entry_at(node, number % u64::from(FANOUT));
    store.write(at, &encode(Leaf::live(grown.data, 0)));
}

/// Every extent the index itself uses, for invariant checks: the top and every lower node.
#[cfg(test)]
pub(crate) fn index_extents<M: Memory>(view: &Pages<'_, M>, paged: &Paged) -> Vec<Extent> {
    let mut out = alloc::vec![paged.index];
    walk(
        view,
        Node::top(paged),
        data_pages(paged.stored),
        &mut |level, entry| {
            if level > 1 {
                out.push(Extent::whole(entry.present()));
            }
        },
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::PAGES_PER_INSERT;

    /// Pages appending data page `number` (after full pages `0..number`) takes.
    fn pages_taken(number: u32) -> u32 {
        // A one-page top: `Grow` never applies, `Keep` and `Deepen` are decided by depth.
        let paged = Paged {
            index: Extent::whole(PageId::new(0)),
            stored: number * PAGE_SLOTS,
            live: number * PAGE_SLOTS,
        };
        let growth = plan(&paged);
        let top = match growth.top {
            TopChange::Keep => 0,
            TopChange::Grow { .. } | TopChange::Deepen => 1,
        };
        1 + top + u32::try_from(growth.fresh_levels.len()).unwrap()
    }

    #[test]
    fn deepening_to_three_levels_takes_the_most_pages_an_insert_reserves() {
        let two_levels = FANOUT;
        let three_levels = FANOUT * FANOUT;
        assert_eq!(pages_taken(two_levels), 3, "data, top, one fresh node");
        assert_eq!(
            pages_taken(three_levels),
            PAGES_PER_INSERT,
            "data, top, two fresh nodes"
        );
        let last = u32::MAX / PAGE_SLOTS;
        assert_eq!(depth(last + 1), 3, "three levels cover every u32 position");
        for number in [1, two_levels - 1, two_levels + 1, three_levels + 1, last] {
            assert!(pages_taken(number) <= PAGES_PER_INSERT, "page {number}");
        }
    }
}
