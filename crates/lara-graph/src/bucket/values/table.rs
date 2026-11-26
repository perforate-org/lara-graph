//! The table form of a value plane: a radix tree of node pages over value pages.
//!
//! A node page holds 1024 page ids, `u32::MAX` for a missing child; missing nodes and pages read
//! as zeros. Writing anywhere creates at most the path to it.

use super::{Plane, page_len};
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::{ByteClass, Extent, PAGE_BYTES, PageId, PageStore, Pages};
use crate::store::txn::SpanTxn;
use crate::store::units::ByteOffset;
use alloc::vec;
use alloc::vec::Vec;

/// Data pages a table of `depth` levels covers.
pub(super) fn reach(depth: u8) -> u64 {
    u64::from(NODE_ENTRIES).pow(u32::from(depth))
}

pub(super) fn entry_at(node: PageId, index: u64) -> ByteOffset {
    ByteOffset::new(node.offset().get() + index * 4)
}

/// The node entry for data page `number` at `level` (1 is the bottom).
pub(super) fn index_at(number: u64, level: u8) -> u64 {
    (number / reach(level - 1)) % u64::from(NODE_ENTRIES)
}

/// Value page `number` of a table, if it exists.
pub(super) fn data_page<M: Memory>(
    view: &Pages<'_, M>,
    root: PageId,
    depth: u8,
    number: u64,
) -> Option<PageId> {
    if number >= reach(depth) {
        return None;
    }
    let mut node = root;
    for level in (1..=depth).rev() {
        let raw = u32::from_le_bytes(view.read_array(entry_at(node, index_at(number, level))));
        if raw == NONE {
            return None;
        }
        node = PageId::new(raw);
    }
    Some(node)
}

/// A new table holding `first` (a run's cells, as value page 0) and value page `number`.
pub(super) fn table_with<M: Memory>(
    store: &mut PageStore<M>,
    first: Option<Extent>,
    number: u64,
) -> Result<Plane, WriteError> {
    let root = store.allocate(ByteClass::PAGE)?.page();
    write_node(store, root);
    if let Some(run) = first {
        let page0 = store.allocate(ByteClass::PAGE)?.page();
        let mut content = vec![0u8; page_len()];
        let len = usize::try_from(run.len()).unwrap_or(0);
        if let Some(head) = content.get_mut(..len) {
            store.pages().read_into(run.offset(), head);
        }
        store.write(page0.offset(), &content);
        link(store, (root, 1), 0, Some(page0))?;
    }
    grow_table(store, (root, 1), number)
}

/// Levels needed to reach data page `number`.
pub(super) fn depth_for(number: u64) -> u8 {
    (1..=MAX_DEPTH)
        .find(|&d| number < reach(d))
        .unwrap_or_else(|| corrupt("a value page beyond every position"))
}

/// The table, deepened as needed, with value page `number` present (zeroed if new).
pub(super) fn grow_table<M: Memory>(
    store: &mut PageStore<M>,
    (root, depth): (PageId, u8),
    number: u64,
) -> Result<Plane, WriteError> {
    let mut root = root;
    let mut depth = depth;
    if number >= reach(depth) {
        // Deepen: each new root's first entry is the old root. Allocate them all first.
        let target = depth_for(number);
        let mut txn = SpanTxn::begin(store);
        let mut roots = Vec::new();
        for _ in depth..target {
            roots.push(txn.allocate(ByteClass::PAGE)?.page());
        }
        txn.commit();
        for new_root in roots {
            write_node(store, new_root);
            store.write(entry_at(new_root, 0), &root.get().to_le_bytes());
            root = new_root;
            depth += 1;
        }
    }
    link(store, (root, depth), number, None)
}

/// Makes value page `number` present under `(root, depth)`: `page` if given, else a new zeroed
/// page. Missing nodes on the way are created.
pub(super) fn link<M: Memory>(
    store: &mut PageStore<M>,
    (root, depth): (PageId, u8),
    number: u64,
    page: Option<PageId>,
) -> Result<Plane, WriteError> {
    let table = Plane::Table { root, depth };
    // Walk down to the first missing entry.
    let mut node = root;
    let mut level = depth;
    while level > 0 {
        let at = entry_at(node, index_at(number, level));
        let raw = u32::from_le_bytes(store.pages().read_array(at));
        if raw == NONE {
            break;
        }
        node = PageId::new(raw);
        level -= 1;
    }
    if level == 0 {
        return Ok(table);
    }
    // Below `node`, `level - 1` nodes and the value page are missing.
    let mut txn = SpanTxn::begin(store);
    let mut chain = Vec::new();
    for _ in 1..level {
        chain.push(txn.allocate(ByteClass::PAGE)?.page());
    }
    let data = match page {
        Some(page) => page,
        None => txn.allocate(ByteClass::PAGE)?.page(),
    };
    txn.commit();
    if page.is_none() {
        store.write(data.offset(), &vec![0u8; page_len()]);
    }
    for &fresh in &chain {
        write_node(store, fresh);
    }
    chain.push(data);
    let mut parent = node;
    for (l, &child) in (1..=level).rev().zip(&chain) {
        store.write(
            entry_at(parent, index_at(number, l)),
            &child.get().to_le_bytes(),
        );
        parent = child;
    }
    Ok(table)
}

/// Fills a node page with "no page" entries.
pub(super) fn write_node<M: Memory>(store: &PageStore<M>, node: PageId) {
    store.write(node.offset(), &vec![0xFFu8; page_len()]);
}

pub(super) fn release_node<M: Memory>(store: &mut PageStore<M>, node: PageId, level: u8) {
    if level > 0 {
        let bytes = store.pages().read(node.offset(), page_len());
        let children: Vec<PageId> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&raw| u32::from_le_bytes(raw))
            .filter(|&raw| raw != NONE)
            .map(PageId::new)
            .collect();
        for child in children {
            release_node(store, child, level - 1);
        }
    }
    store.release(Extent::whole(node));
}

/// Releases the value pages numbered `first..end` under `node` (at `level`, its subtree starting
/// at page `base`), and every node they leave empty. Returns whether `node` is now empty; the
/// caller releases it then.
pub(super) fn release_pages<M: Memory>(
    store: &mut PageStore<M>,
    (node, level, base): (PageId, u8, u64),
    range: (u64, u64),
) -> bool {
    let span = reach(level - 1);
    let bytes = store.pages().read(node.offset(), page_len());
    let mut empty = true;
    for (i, &raw) in (0u64..).zip(bytes.as_chunks::<4>().0) {
        let raw = u32::from_le_bytes(raw);
        if raw == NONE {
            continue;
        }
        let lo = base + i * span;
        let child = (PageId::new(raw), level - 1, lo);
        if release_child(store, child, (lo + span, range)) {
            store.write(entry_at(node, i), &NONE.to_le_bytes());
        } else {
            empty = false;
        }
    }
    empty
}

/// [`release_pages`] for one child covering pages `lo..hi`; returns whether it is gone.
fn release_child<M: Memory>(
    store: &mut PageStore<M>,
    (child, level, lo): (PageId, u8, u64),
    (hi, (first, end)): (u64, (u64, u64)),
) -> bool {
    if hi <= first || lo >= end {
        return false;
    }
    if first <= lo && hi <= end {
        release_node(store, child, level);
        return true;
    }
    // Partly covered: only a node (level > 0) can be, since pages are whole units.
    let emptied = release_pages(store, (child, level, lo), (first, end));
    if emptied {
        store.release(Extent::whole(child));
    }
    emptied
}

/// Page ids per node page.
pub(super) const NODE_ENTRIES: u32 = PAGE_BYTES / 4;

/// The node entry of a missing page.
pub(super) const NONE: u32 = u32::MAX;

/// Levels a table can have: enough for every `u32` position at one cell per page.
pub(super) const MAX_DEPTH: u8 = 4;
