//! The counted B+ tree over postings.
//!
//! An internal node is a page: its child count u16 and two spare bytes, then three arrays of
//! [`FANOUT`] slots each: the children's first keys (target u32, block u32), their pages (u32),
//! and their total counts (u32). Lookups read the keys they probe and the counts they sum, not
//! whole nodes; a change that leaves every node on its page writes only the entries it alters.
//! Leaves are `leaf` pages. A batch of changes is planned in memory first, which tells exactly
//! how many pages it needs, then committed (`commit`).

mod commit;
mod lookup;
mod node;

use super::leaf;
use crate::corrupt::corrupt;
use crate::memory::{Memory, WriteError};
use crate::page_store::{PageId, PageStore, Pages};
use alloc::vec::Vec;
use commit::{Body, Fresh, Kept, Kid, Top, Work, shifted};
pub(crate) use commit::{Plan, commit};
#[cfg(any(test, feature = "bench-internals"))]
pub(crate) use lookup::extents;
#[cfg(test)]
pub(crate) use lookup::postings;
pub(crate) use lookup::{Nth, last_block, nth, rank, release};
#[cfg(test)]
pub(crate) use lookup::{count_before, select};
use node::{
    Child, FANOUT, at, child_of, children, count_at, key_at, node_len, page_at, read_into,
    write_child,
};

/// Where a tree is: its root page and how many internal levels lie above the leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Root {
    pub(crate) page: PageId,
    pub(crate) depth: u8,
}

/// A change to the count of the posting at `key`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Change {
    pub(crate) key: (u32, u32),
    pub(crate) delta: i64,
}

/// Sorts changes and sums those to one key.
fn combine(changes: &[Change]) -> Vec<Change> {
    let mut sorted = changes.to_vec();
    sorted.sort_unstable_by_key(|c| c.key);
    let mut out: Vec<Change> = Vec::with_capacity(sorted.len());
    for c in sorted {
        match out.last_mut() {
            Some(last) if last.key == c.key => last.delta += c.delta,
            Some(_) | None => out.push(c),
        }
    }
    out.retain(|c| c.delta != 0);
    out
}

/// Applies `changes` to the tree (counts that reach zero drop their postings) within memory the
/// caller reserved: [`super::PAGES_PER_CHANGE`] per edge changed, or the plan's pages. An
/// allocation failure means the reservation was too small.
pub(crate) fn apply_reserved<M: Memory>(
    store: &mut PageStore<M>,
    root: Option<Root>,
    changes: &[Change],
) -> Result<Option<Root>, WriteError> {
    commit(store, plan(&store.pages(), root, changes))
}

/// Lowers the count of the posting at `key` by one, in place: no page moves and none is
/// allocated. A count that reaches zero stays until its segment is next rewritten.
pub(crate) fn lower<M: Memory>(store: &PageStore<M>, root: Root, key: (u32, u32)) {
    lower_under(store, (root.page, root.depth), key);
}

fn lower_under<M: Memory>(store: &PageStore<M>, (page, level): (PageId, u8), key: (u32, u32)) {
    let view = store.pages();
    if level == 0 {
        let read = |offset, buf: &mut [u8]| read_into(&view, page, offset, buf);
        leaf::lowered(&read, key)
            .unwrap_or_else(|| corrupt("an index decrement without its posting"))
            .write(|offset, bytes| store.write(at(page, offset), bytes));
        return;
    }
    let i = child_of(&view, page, node_len(&view, page), key);
    lower_under(store, (page_at(&view, page, i), level - 1), key);
    write_child(store, page, i, shifted(count_at(&view, page, i), -1));
}

/// What `changes` would do to the tree, read but not written: [`Plan::pages`] tells what to
/// reserve, and [`commit`] writes it.
pub(crate) fn plan<M: Memory>(view: &Pages<'_, M>, root: Option<Root>, changes: &[Change]) -> Plan {
    let changes = combine(changes);
    let depth = root.map_or(0, |r| r.depth);
    if changes.is_empty() {
        return Plan {
            top: Top::Unchanged(root),
            depth,
        };
    }
    let outcome = match root {
        Some(root) => node(view, (root.page, root.depth), &changes),
        None => match leaf::update(None, &changes) {
            leaf::Update::Replace(images) => Outcome::Replaced(fresh_leaves(None, images)),
            leaf::Update::InPlace { .. } => corrupt("a new index leaf in place"),
        },
    };
    let (top, depth) = match (outcome, root) {
        (Outcome::Same(kept), Some(root)) => (Top::Kept(root, kept), depth),
        (Outcome::Same(_), None) => corrupt("a new index kept in place"),
        (Outcome::Replaced(nodes), _) => grow(nodes, depth),
    };
    Plan { top, depth }
}

/// A new top for `nodes` at `depth`: none when empty; levels above them while there are several.
fn grow(mut nodes: Vec<Fresh>, mut depth: u8) -> (Top, u8) {
    while nodes.len() > 1 {
        let kids: Vec<Kid> = nodes.into_iter().map(Kid::Fresh).collect();
        nodes = pack(None, kids);
        depth += 1;
    }
    match nodes.pop() {
        Some(top) => (Top::Fresh(top), depth),
        None => corrupt("an index rewritten to nothing"),
    }
}

/// What changes did to a node.
enum Outcome {
    /// It stays on its page.
    Same(Kept),
    /// These nodes replace it: none when it emptied, several when it split.
    Replaced(Vec<Fresh>),
}

/// Plans `changes` (sorted) under `page` at `level`.
fn node<M: Memory>(
    view: &Pages<'_, M>,
    (page, level): (PageId, u8),
    changes: &[Change],
) -> Outcome {
    if level == 0 {
        return leaf_node(view, page, changes);
    }
    let results = descend(view, (page, level), changes);
    if results.iter().all(|(_, r)| matches!(r, Outcome::Same(_))) {
        Outcome::Same(kept_node(page, results))
    } else {
        Outcome::Replaced(relist(view, page, results))
    }
}

/// Plans `changes` in the leaf at `page`.
fn leaf_node<M: Memory>(view: &Pages<'_, M>, page: PageId, changes: &[Change]) -> Outcome {
    let read = |offset, buf: &mut [u8]| read_into(view, page, offset, buf);
    match leaf::update(Some(&read), changes) {
        leaf::Update::InPlace { writes, delta } => Outcome::Same(Kept {
            page,
            delta,
            work: Work::Leaf(writes),
        }),
        leaf::Update::Replace(images) => Outcome::Replaced(fresh_leaves(Some(page), images)),
    }
}

/// An internal node whose changed children all stay on their pages.
fn kept_node(page: PageId, results: Vec<(usize, Outcome)>) -> Kept {
    let kids: Vec<(usize, Kept)> = results
        .into_iter()
        .filter_map(|(i, r)| match r {
            Outcome::Same(k) => Some((i, k)),
            Outcome::Replaced(_) => None,
        })
        .collect();
    Kept {
        page,
        delta: kids.iter().map(|(_, k)| k.delta).sum(),
        work: Work::Node(kids),
    }
}

/// An internal node some of whose children split or emptied: its children relisted and repacked.
fn relist<M: Memory>(
    view: &Pages<'_, M>,
    page: PageId,
    results: Vec<(usize, Outcome)>,
) -> Vec<Fresh> {
    let mut next: Vec<Kid> = children(view, page).into_iter().map(Kid::Old).collect();
    for (i, result) in results.into_iter().rev() {
        let Some(Kid::Old(old)) = next.get(i) else {
            corrupt("an index child changed twice");
        };
        let old = *old;
        match result {
            Outcome::Same(kept) => {
                let child = Child {
                    count: shifted(old.count, kept.delta),
                    ..old
                };
                if let Some(slot) = next.get_mut(i) {
                    *slot = Kid::Kept(child, kept);
                }
            }
            Outcome::Replaced(by) => {
                next.splice(i..=i, by.into_iter().map(Kid::Fresh));
            }
        }
    }
    pack(Some(page), next)
}

/// Hands each child of the internal node at `page` the changes that fall in it; returns what
/// becomes of each child they change.
fn descend<M: Memory>(
    view: &Pages<'_, M>,
    (page, level): (PageId, u8),
    changes: &[Change],
) -> Vec<(usize, Outcome)> {
    let n = node_len(view, page);
    let mut results = Vec::new();
    let mut rest = changes;
    while let Some(first) = rest.first() {
        // The child the next change falls in takes the changes below the next child's start.
        let i = child_of(view, page, n, first.key);
        let bound = (i + 1 < n).then(|| key_at(view, page, i + 1));
        let split = bound.map_or(rest.len(), |b| rest.partition_point(|c| c.key < b));
        let (mine, later) = rest.split_at(split.max(1));
        rest = later;
        results.push((i, node(view, (page_at(view, page, i), level - 1), mine)));
    }
    results
}

/// Fresh leaves for `images`, the first on `page`. A rewrite never empties a leaf: a leaf is
/// rewritten only for a change that is not a decrement, and that leaves its posting.
fn fresh_leaves(page: Option<PageId>, images: Vec<leaf::Image>) -> Vec<Fresh> {
    if images.is_empty() {
        corrupt("an index leaf emptied by a rewrite");
    }
    let mut reuse = page;
    images
        .into_iter()
        .map(|image| Fresh {
            page: reuse.take(),
            first: image.first,
            count: image.count,
            body: Body::Leaf(image.bytes),
        })
        .collect()
}

/// Internal nodes for `kids` (some), packed evenly, the first on `page`.
fn pack(page: Option<PageId>, kids: Vec<Kid>) -> Vec<Fresh> {
    let parts = kids.len().div_ceil(FANOUT);
    let per = kids.len().div_ceil(parts);
    let mut reuse = page;
    let mut out = Vec::with_capacity(parts);
    let mut kids = kids.into_iter().peekable();
    while kids.peek().is_some() {
        let part: Vec<Kid> = kids.by_ref().take(per).collect();
        out.push(Fresh {
            page: reuse.take(),
            first: part.first().map_or((0, 0), Kid::first),
            count: part.iter().map(Kid::count).sum(),
            body: Body::Node(part),
        });
    }
    out
}
