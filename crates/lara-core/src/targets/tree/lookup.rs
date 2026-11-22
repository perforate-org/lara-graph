//! Reading the tree: rank and selection by count, and walking it whole.

use super::Root;
use super::node::{child_of, children, counts, node_len, page_at, read_into};
use crate::memory::Memory;
use crate::page_store::{Extent, PageStore, Pages};
use crate::targets::{Posting, leaf};
#[cfg(any(test, feature = "bench-internals"))]
use alloc::vec::Vec;

/// The total count of the postings before `key`.
pub(crate) fn count_before<M: Memory>(view: &Pages<'_, M>, root: Root, key: (u32, u32)) -> u32 {
    let mut before = 0;
    let mut page = root.page;
    let mut total = None;
    for _ in 0..root.depth {
        let n = node_len(view, page);
        let i = child_of(view, page, n, key);
        let sums = counts(view, page, 0..i + 1);
        before += sums.iter().take(i).sum::<u32>();
        total = sums.last().copied();
        page = page_at(view, page, i);
    }
    let read = |offset, buf: &mut [u8]| read_into(view, page, offset, buf);
    before + leaf::count_before(&read, key, total)
}

/// The posting holding the `rank`-th unit of count overall (0-based), with the rank within it.
pub(crate) fn select<M: Memory>(
    view: &Pages<'_, M>,
    root: Root,
    rank: u32,
) -> Option<(Posting, u32)> {
    let mut left = rank;
    let mut page = root.page;
    let mut total = None;
    for _ in 0..root.depth {
        let n = node_len(view, page);
        let sums = counts(view, page, 0..n);
        let mut next = None;
        for (i, &count) in sums.iter().enumerate() {
            if left < count {
                next = Some(i);
                total = Some(count);
                break;
            }
            left -= count;
        }
        page = page_at(view, page, next?);
    }
    let read = |offset, buf: &mut [u8]| read_into(view, page, offset, buf);
    leaf::select(&read, left, total)
}

/// Releases every page of the tree.
pub(crate) fn release<M: Memory>(store: &mut PageStore<M>, root: Root) {
    if root.depth > 0 {
        for child in children(&store.pages(), root.page) {
            release(
                store,
                Root {
                    page: child.page,
                    depth: root.depth - 1,
                },
            );
        }
    }
    store.release(Extent::whole(root.page));
}

/// Every posting with a count in key order (test checks); postings whose count reached zero
/// wait in their segments until it is rewritten.
#[cfg(test)]
pub(crate) fn postings<M: Memory>(view: &Pages<'_, M>, root: Option<Root>) -> Vec<Posting> {
    let Some(root) = root else {
        return Vec::new();
    };
    if root.depth == 0 {
        let read = |offset, buf: &mut [u8]| read_into(view, root.page, offset, buf);
        return leaf::decode(&read)
            .into_iter()
            .filter(|p| p.count > 0)
            .collect();
    }
    children(view, root.page)
        .into_iter()
        .flat_map(|c| {
            let below = Root {
                page: c.page,
                depth: root.depth - 1,
            };
            postings(view, Some(below))
        })
        .collect()
}

/// Every page of the tree (test checks and measurements).
#[cfg(any(test, feature = "bench-internals"))]
pub(crate) fn extents<M: Memory>(view: &Pages<'_, M>, root: Option<Root>) -> Vec<Extent> {
    let Some(root) = root else {
        return Vec::new();
    };
    let mut out = alloc::vec![Extent::whole(root.page)];
    if root.depth > 0 {
        for c in children(view, root.page) {
            let below = Root {
                page: c.page,
                depth: root.depth - 1,
            };
            out.extend(extents(view, Some(below)));
        }
    }
    out
}
