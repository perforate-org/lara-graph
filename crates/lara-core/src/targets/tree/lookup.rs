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
        let sums = counts(view, page, i + 1);
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
        let sums = counts(view, page, n);
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

/// Where the `k`-th (0-based) unit of count among a target's postings is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Nth {
    /// In this posting, at this rank within it.
    At(Posting, u32),
    /// Past them: the target's postings hold this many units in all.
    Past(u32),
}

/// The `k`-th unit of count among `target`'s postings. Usually one descent: the walk of the leaf
/// where they start; a target whose postings run on into the next leaf is counted from the root.
pub(crate) fn nth<M: Memory>(view: &Pages<'_, M>, root: Root, target: u32, k: u32) -> Nth {
    let mut left = k;
    let mut found = None;
    let complete = walk_target(view, root, target, |p| {
        found = if p.target != target {
            Some(Nth::Past(k - left))
        } else if left < p.count {
            Some(Nth::At(p, left))
        } else {
            left -= p.count;
            None
        };
        found.is_some()
    });
    match (found, complete) {
        (Some(nth), _) => nth,
        (None, true) => Nth::Past(k - left),
        (None, false) => {
            let first = count_before(view, root, (target, 0));
            match select(view, root, first + k) {
                Some((p, within)) if p.target == target => Nth::At(p, within),
                Some(_) | None => Nth::Past(count_before(view, root, (target, u32::MAX)) - first),
            }
        }
    }
}

/// The units of count among `target`'s postings in blocks before `block`. Usually one descent, as
/// for [`nth`].
pub(crate) fn rank<M: Memory>(view: &Pages<'_, M>, root: Root, target: u32, block: u32) -> u32 {
    let mut sum = 0;
    let complete = walk_target(view, root, target, |p| {
        let done = (p.target, p.block) >= (target, block);
        if !done {
            sum += p.count;
        }
        done
    });
    if complete {
        sum
    } else {
        count_before(view, root, (target, block)) - count_before(view, root, (target, 0))
    }
}

/// The last block where `target` has a unit of count, if any. Usually one descent, as for
/// [`nth`].
pub(crate) fn last_block<M: Memory>(view: &Pages<'_, M>, root: Root, target: u32) -> Option<u32> {
    let mut last = None;
    let complete = walk_target(view, root, target, |p| {
        if p.target != target {
            return true;
        }
        if p.count > 0 {
            last = Some(p.block);
        }
        false
    });
    if complete {
        return last;
    }
    let end = count_before(view, root, (target, u32::MAX));
    if end == count_before(view, root, (target, 0)) {
        return None;
    }
    select(view, root, end - 1).map(|(p, _)| p.block)
}

/// Calls `f` with the postings from `target`'s first on, in the leaf where they start, until it
/// returns true. Whether that walk saw all it needed: `f` stopped it, or the leaf is the last.
fn walk_target<M: Memory>(
    view: &Pages<'_, M>,
    root: Root,
    target: u32,
    f: impl FnMut(Posting) -> bool,
) -> bool {
    let key = (target, 0);
    let mut page = root.page;
    let mut last = true;
    for _ in 0..root.depth {
        let n = node_len(view, page);
        let i = child_of(view, page, n, key);
        last &= i + 1 == n;
        page = page_at(view, page, i);
    }
    let read = |offset, buf: &mut [u8]| read_into(view, page, offset, buf);
    leaf::walk_from(&read, key, f) || last
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

/// Every posting with a count in key order (test checks); postings at zero (a leaf's last, or
/// left by a batch of decrements) are left out.
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

/// How many postings of each leaf have a count of zero, leaf by leaf (test checks).
#[cfg(test)]
pub(crate) fn zeros_per_leaf<M: Memory>(view: &Pages<'_, M>, root: Option<Root>) -> Vec<usize> {
    let Some(root) = root else {
        return Vec::new();
    };
    if root.depth == 0 {
        let read = |offset, buf: &mut [u8]| read_into(view, root.page, offset, buf);
        return alloc::vec![leaf::decode(&read).iter().filter(|p| p.count == 0).count()];
    }
    children(view, root.page)
        .into_iter()
        .flat_map(|c| {
            let below = Root {
                page: c.page,
                depth: root.depth - 1,
            };
            zeros_per_leaf(view, Some(below))
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
