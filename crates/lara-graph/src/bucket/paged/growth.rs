//! Adding a data page to a paged bucket's index: the top grows by copying at most one page of
//! entries, a full one-page top becomes the first child of a new top without copying, and the
//! nodes on the new page's path are allocated with it, all or nothing.

use super::{
    ENTRY, ENTRY_BYTES, FANOUT, Leaf, NO_HOLE, Node, byte_len_of, data_pages, decode, depth,
    encode, entry_at, fillable, index_class, narrow, span, stored_count,
};
use crate::bucket::descriptor::Paged;
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::{ByteClass, Extent, PageId, PageStore};
use crate::store::txn::SpanTxn;
use alloc::vec::Vec;

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
            // The old top becomes the new top's first child: count its entries with holes.
            let old = Node::top(paged);
            let bytes = store
                .pages()
                .read(paged.index.offset(), byte_len_of(FANOUT));
            let holed = bytes
                .as_chunks::<ENTRY>()
                .0
                .iter()
                .filter(|&&raw| fillable(old.level, decode(raw)))
                .count();
            let first = Leaf::node(paged.index.page(), narrow(stored_count(holed)));
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
                store.write(at, &encode(Leaf::node(page, 0)));
                page
            }
            None => decode(store.pages().read_array(at)).present(),
        };
        node = child.offset();
    }
    let at = entry_at(node, number % u64::from(FANOUT));
    store.write(at, &encode(Leaf::data(grown.data, 0, NO_HOLE)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::descriptor::PAGE_SLOTS;
    use crate::bucket::slots::PAGES_PER_INSERT;

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
