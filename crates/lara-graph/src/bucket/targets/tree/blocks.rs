//! The blocks where a target has entries, listed in one walk of its postings.

use super::Root;
use super::node::{child_of, key_at, node_len, page_at, read_into};
use crate::bucket::targets::leaf;
use crate::store::memory::Memory;
use crate::store::page_store::{PageId, Pages};

/// Calls `f` with each block where `target` has a unit of count, ascending. One descent per leaf
/// its postings touch, each read once: listing a target's blocks with [`nth`] instead would
/// descend and walk again for every block.
pub(crate) fn visit_blocks<M: Memory>(
    view: &Pages<'_, M>,
    root: Root,
    target: u32,
    mut f: impl FnMut(u32),
) {
    let mut key = (target, 0);
    loop {
        let (page, next) = leaf_and_next(view, root, key);
        let read = |offset, buf: &mut [u8]| read_into(view, page, offset, buf);
        let ended = leaf::walk_from(&read, key, |p| {
            if p.target != target {
                return true;
            }
            if p.count > 0 {
                f(p.block);
            }
            false
        });
        match next {
            Some(first) if !ended && first.0 == target => key = first,
            Some(_) | None => return,
        }
    }
}

/// The leaf where `key` belongs, and the first key of the leaf after it, if any.
fn leaf_and_next<M: Memory>(
    view: &Pages<'_, M>,
    root: Root,
    key: (u32, u32),
) -> (PageId, Option<(u32, u32)>) {
    let mut page = root.page;
    let mut next = None;
    for _ in 0..root.depth {
        let n = node_len(view, page);
        let i = child_of(view, page, n, key);
        if i + 1 < n {
            next = Some(key_at(view, page, i + 1));
        }
        page = page_at(view, page, i);
    }
    (page, next)
}
