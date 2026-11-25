//! Reading the tree: rank and selection by count, and walking it whole.

use super::Root;
use super::node::{child_of, children, count_at, counts, node_len, page_at, read_into};
use crate::bucket::target_index::{Posting, leaf};
use crate::store::memory::Memory;
use crate::store::page_store::{Extent, PageId, PageStore, Pages};
#[cfg(test)]
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
/// where they start, or, for a target with many postings there, the leaf's segment totals; a
/// target whose postings run on into the next leaf is counted from the root.
pub(crate) fn nth<M: Memory>(view: &Pages<'_, M>, root: Root, target: u32, k: u32) -> Nth {
    let mut left = k;
    let mut found = None;
    let walked = walk_target(view, root, target, |p| {
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
    let local = match (found, walked) {
        (Some(nth), _) => Some(nth),
        (None, Walked::Whole) => Some(Nth::Past(k - left)),
        (None, Walked::RanOff) => None,
        (None, Walked::Long(leaf)) => {
            let read = |offset, buf: &mut [u8]| read_into(view, leaf.page, offset, buf);
            let first = leaf::count_before(&read, (target, 0), leaf.total);
            let past =
                || Nth::Past(leaf::count_before(&read, (target, u32::MAX), leaf.total) - first);
            match leaf::select(&read, first + k, leaf.total) {
                Some((p, within)) if p.target == target => Some(Nth::At(p, within)),
                Some(_) => Some(past()),
                None => (leaf.after == After::End).then(past),
            }
        }
    };
    local.unwrap_or_else(|| {
        let first = count_before(view, root, (target, 0));
        match select(view, root, first + k) {
            Some((p, within)) if p.target == target => Nth::At(p, within),
            Some(_) | None => Nth::Past(count_before(view, root, (target, u32::MAX)) - first),
        }
    })
}

/// The units of count among `target`'s postings in blocks before `block`. Usually one descent, as
/// for [`nth`].
pub(crate) fn rank<M: Memory>(view: &Pages<'_, M>, root: Root, target: u32, block: u32) -> u32 {
    let mut sum = 0;
    let walked = walk_target(view, root, target, |p| {
        let done = (p.target, p.block) >= (target, block);
        if !done {
            sum += p.count;
        }
        done
    });
    let local = match walked {
        Walked::Whole => Some(sum),
        Walked::RanOff => None,
        Walked::Long(leaf) => {
            let read = |offset, buf: &mut [u8]| read_into(view, leaf.page, offset, buf);
            let before = leaf::count_before(&read, (target, block), leaf.total);
            // All of a leaf before the key, unless it is the last: the key may lie further on.
            (leaf.after == After::End || leaf.total != Some(before))
                .then(|| before - leaf::count_before(&read, (target, 0), leaf.total))
        }
    };
    local.unwrap_or_else(|| {
        count_before(view, root, (target, block)) - count_before(view, root, (target, 0))
    })
}

/// The last block where `target` has a unit of count, if any. Usually one descent, as for
/// [`nth`].
pub(crate) fn last_block<M: Memory>(view: &Pages<'_, M>, root: Root, target: u32) -> Option<u32> {
    let mut last = None;
    let walked = walk_target(view, root, target, |p| {
        if p.target != target {
            return true;
        }
        if p.count > 0 {
            last = Some(p.block);
        }
        false
    });
    let local = match walked {
        Walked::Whole => Some(last),
        Walked::RanOff => None,
        Walked::Long(leaf) => {
            let read = |offset, buf: &mut [u8]| read_into(view, leaf.page, offset, buf);
            let end = leaf::count_before(&read, (target, u32::MAX), leaf.total);
            // All of a leaf before the target's end, unless it is the last: it may go on.
            (leaf.after == After::End || leaf.total != Some(end)).then(|| {
                (end > leaf::count_before(&read, (target, 0), leaf.total))
                    .then(|| leaf::select(&read, end - 1, leaf.total).map(|(p, _)| p.block))
                    .flatten()
            })
        }
    };
    local.unwrap_or_else(|| {
        let end = count_before(view, root, (target, u32::MAX));
        if end == count_before(view, root, (target, 0)) {
            return None;
        }
        select(view, root, end - 1).map(|(p, _)| p.block)
    })
}

/// Postings a walk decodes before it turns to the leaf's segment totals: a full segment's worth.
/// A target with many postings in a leaf is then found in a few segment reads rather than posting
/// by posting, which cost about 250K instructions to pair an edge of a hub with 64 targets in turn
/// (`counterpart_parallel_*`).
const WALK_POSTINGS: u32 = 48;

/// The leaf where a target's postings start.
struct InLeaf {
    page: PageId,
    /// Its total, when its parent lists it (not for a root leaf).
    total: Option<u32>,
    after: After,
}

/// What follows a leaf in the tree.
#[derive(Clone, Copy, PartialEq, Eq)]
enum After {
    /// Nothing: it is the last leaf, so a target's postings end in it.
    End,
    /// More leaves, where a target's postings may go on.
    MoreLeaves,
}

/// How far a walk of a target's postings got.
enum Walked {
    /// `f` stopped it, or the leaf ended and is the last: it saw all it needed.
    Whole,
    /// The leaf ended and is not the last: the postings may go on in the next.
    RanOff,
    /// It passed [`WALK_POSTINGS`] postings first: the leaf's totals answer faster.
    Long(InLeaf),
}

/// Calls `f` with the postings from `target`'s first on, in the leaf where they start, until it
/// returns true or [`WALK_POSTINGS`] have gone by.
fn walk_target<M: Memory>(
    view: &Pages<'_, M>,
    root: Root,
    target: u32,
    mut f: impl FnMut(Posting) -> bool,
) -> Walked {
    let key = (target, 0);
    let mut page = root.page;
    let mut last = true;
    // The leaf's place in its parent, to read its total only if the walk turns to the totals.
    let mut parent = None;
    for _ in 0..root.depth {
        let n = node_len(view, page);
        let i = child_of(view, page, n, key);
        last &= i + 1 == n;
        parent = Some((page, i));
        page = page_at(view, page, i);
    }
    let read = |offset, buf: &mut [u8]| read_into(view, page, offset, buf);
    let mut seen = 0;
    let mut long = false;
    let stopped = leaf::walk_from(&read, key, |p| {
        seen += 1;
        long = seen > WALK_POSTINGS;
        long || f(p)
    });
    match (long, stopped || last) {
        (true, _) => Walked::Long(InLeaf {
            page,
            total: parent.map(|(node, i)| count_at(view, node, i)),
            after: if last { After::End } else { After::MoreLeaves },
        }),
        (false, true) => Walked::Whole,
        (false, false) => Walked::RanOff,
    }
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

/// Every posting with a count in key order (test checks); postings at zero (a segment's first, a
/// leaf's last, or left by a batch of decrements) are left out.
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

/// How many postings of each segment have a count of zero, segment by segment (test checks).
#[cfg(test)]
pub(crate) fn zeros_per_segment<M: Memory>(view: &Pages<'_, M>, root: Option<Root>) -> Vec<usize> {
    let Some(root) = root else {
        return Vec::new();
    };
    if root.depth == 0 {
        let read = |offset, buf: &mut [u8]| read_into(view, root.page, offset, buf);
        return leaf::zeros_per_segment(&read);
    }
    children(view, root.page)
        .into_iter()
        .flat_map(|c| {
            let below = Root {
                page: c.page,
                depth: root.depth - 1,
            };
            zeros_per_segment(view, Some(below))
        })
        .collect()
}

/// Every page of the tree (test checks and measurements).
#[cfg(test)]
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
