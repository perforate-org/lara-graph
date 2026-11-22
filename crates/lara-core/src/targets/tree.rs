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
pub(crate) use lookup::{count_before, release, select};
use node::{Child, FANOUT, child_of, children, key_at, node_len, page_at, read_into};

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
    sorted.sort_by_key(|c| c.key);
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

/// What `changes` would do to the tree, read but not written: [`Plan::pages`] tells what to
/// reserve, and [`commit`] writes it.
pub(crate) fn plan<M: Memory>(view: &Pages<'_, M>, root: Option<Root>, changes: &[Change]) -> Plan {
    let changes = combine(changes);
    let mut freed = Vec::new();
    let depth = root.map_or(0, |r| r.depth);
    if changes.is_empty() {
        return Plan {
            top: Top::Unchanged(root),
            freed,
            depth,
        };
    }
    let outcome = match root {
        Some(root) => node(view, (root.page, root.depth), &changes, &mut freed),
        None => match leaf::update(None, &changes) {
            leaf::Update::Replace(images) => {
                Outcome::Replaced(fresh_leaves(None, images, &mut freed))
            }
            leaf::Update::InPlace { .. } => corrupt("a new index leaf in place"),
        },
    };
    let (top, depth) = match (outcome, root) {
        (Outcome::Same(kept), Some(root)) => (Top::Kept(root, kept), depth),
        (Outcome::Same(_), None) => corrupt("a new index kept in place"),
        (Outcome::Replaced(nodes), _) => grow(nodes, depth),
    };
    Plan { top, freed, depth }
}

/// A new top for `nodes` at `depth`: none when empty; levels above them while there are several.
fn grow(mut nodes: Vec<Fresh>, mut depth: u8) -> (Top, u8) {
    while nodes.len() > 1 {
        let kids: Vec<Kid> = nodes.into_iter().map(Kid::Fresh).collect();
        nodes = pack(None, kids, &mut Vec::new());
        depth += 1;
    }
    match nodes.pop() {
        Some(top) => (Top::Fresh(top), depth),
        None => (Top::Empty, 0),
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
    freed: &mut Vec<PageId>,
) -> Outcome {
    if level == 0 {
        return leaf_node(view, page, changes, freed);
    }
    let results = descend(view, (page, level), changes, freed);
    if results.iter().all(|(_, r)| matches!(r, Outcome::Same(_))) {
        Outcome::Same(kept_node(page, results))
    } else {
        Outcome::Replaced(relist(view, page, results, freed))
    }
}

/// Plans `changes` in the leaf at `page`.
fn leaf_node<M: Memory>(
    view: &Pages<'_, M>,
    page: PageId,
    changes: &[Change],
    freed: &mut Vec<PageId>,
) -> Outcome {
    let read = |offset, buf: &mut [u8]| read_into(view, page, offset, buf);
    match leaf::update(Some(&read), changes) {
        leaf::Update::InPlace {
            writes,
            first,
            delta,
        } => Outcome::Same(Kept {
            page,
            first,
            delta,
            work: Work::Leaf(writes),
        }),
        leaf::Update::Replace(images) => Outcome::Replaced(fresh_leaves(Some(page), images, freed)),
    }
}

/// An internal node whose changed children all stay on their pages.
fn kept_node(page: PageId, results: Vec<(usize, Outcome)>) -> Kept {
    let mut first = None;
    let mut delta = 0;
    let kids: Vec<(usize, Kept)> = results
        .into_iter()
        .filter_map(|(i, r)| match r {
            Outcome::Same(k) => Some((i, k)),
            Outcome::Replaced(_) => None,
        })
        .inspect(|(i, k)| {
            delta += k.delta;
            if *i == 0 {
                first = k.first;
            }
        })
        .collect();
    Kept {
        page,
        first,
        delta,
        work: Work::Node(kids),
    }
}

/// An internal node some of whose children split or emptied: its children relisted and repacked.
fn relist<M: Memory>(
    view: &Pages<'_, M>,
    page: PageId,
    results: Vec<(usize, Outcome)>,
    freed: &mut Vec<PageId>,
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
                    first: kept.first.unwrap_or(old.first),
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
    pack(Some(page), next, freed)
}

/// Hands each child of the internal node at `page` the changes that fall in it; returns what
/// becomes of each child they change.
fn descend<M: Memory>(
    view: &Pages<'_, M>,
    (page, level): (PageId, u8),
    changes: &[Change],
    freed: &mut Vec<PageId>,
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
        results.push((
            i,
            node(view, (page_at(view, page, i), level - 1), mine, freed),
        ));
    }
    results
}

/// Fresh leaves for `images`, the first on `page`; `page` is freed when there are none.
fn fresh_leaves(
    page: Option<PageId>,
    images: Vec<leaf::Image>,
    freed: &mut Vec<PageId>,
) -> Vec<Fresh> {
    if images.is_empty() {
        freed.extend(page);
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

/// Internal nodes for `kids`, packed evenly, the first on `page`; `page` is freed when there are
/// none.
fn pack(page: Option<PageId>, kids: Vec<Kid>, freed: &mut Vec<PageId>) -> Vec<Fresh> {
    if kids.is_empty() {
        freed.extend(page);
        return Vec::new();
    }
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
