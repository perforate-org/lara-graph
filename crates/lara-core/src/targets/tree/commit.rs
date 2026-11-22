//! A planned change to the tree, and writing it.
//!
//! Planning (`tree`) reads the tree and decides everything a batch of changes does: the bytes each
//! leaf and node gets, and which nodes need new pages. Committing writes that plan, taking exactly
//! the pages it counted, so a batch is reserved for and applied in one pass of the work.

use super::Root;
use super::node::{Child, at, count_at, node_len, page_at, write_child, write_node};
use crate::corrupt::corrupt;
use crate::memory::{Memory, WriteError};
use crate::page_store::{ByteClass, Extent, PageId, PageStore};
use alloc::vec::Vec;

type Key = (u32, u32);

/// A node that stays on its page: what changes in it, its new first key if that changed, and the
/// change in its total.
pub(super) struct Kept {
    pub(super) page: PageId,
    pub(super) first: Option<Key>,
    pub(super) delta: i64,
    pub(super) work: Work,
}

/// What changes in a kept node.
pub(super) enum Work {
    /// A leaf's writes, `(offset, bytes)`.
    Leaf(Vec<(usize, Vec<u8>)>),
    /// An internal node's changed children, by index, each kept on its own page.
    Node(Vec<(usize, Kept)>),
}

/// A node to write in full: on `page` (one it replaces) or a new page.
pub(super) struct Fresh {
    pub(super) page: Option<PageId>,
    pub(super) first: Key,
    pub(super) count: u32,
    pub(super) body: Body,
}

pub(super) enum Body {
    Leaf(Vec<u8>),
    Node(Vec<Kid>),
}

/// A child of a fresh internal node.
pub(super) enum Kid {
    /// Unchanged.
    Old(Child),
    /// On its page, changed as planned; listed with its new first key and total.
    Kept(Child, Kept),
    /// Written in full.
    Fresh(Fresh),
}

impl Kid {
    pub(super) const fn first(&self) -> Key {
        match self {
            Self::Old(c) | Self::Kept(c, _) => c.first,
            Self::Fresh(f) => f.first,
        }
    }

    pub(super) const fn count(&self) -> u32 {
        match self {
            Self::Old(c) | Self::Kept(c, _) => c.count,
            Self::Fresh(f) => f.count,
        }
    }
}

/// What a batch of changes does to the tree.
pub(crate) struct Plan {
    pub(super) top: Top,
    /// Pages whose nodes emptied.
    pub(super) freed: Vec<PageId>,
    /// The depth of the tree the plan writes.
    pub(super) depth: u8,
}

pub(super) enum Top {
    /// Nothing changes.
    Unchanged(Option<Root>),
    /// The root stays on its page.
    Kept(Root, Kept),
    /// A new root, written in full (with any new levels above the old one).
    Fresh(Fresh),
    /// Every posting went.
    Empty,
}

impl Plan {
    /// The pages committing the plan allocates.
    pub(crate) fn pages(&self) -> u32 {
        match &self.top {
            Top::Unchanged(_) | Top::Empty => 0,
            Top::Kept(_, kept) => kept_pages(kept),
            Top::Fresh(fresh) => fresh_pages(fresh),
        }
    }
}

fn kept_pages(kept: &Kept) -> u32 {
    match &kept.work {
        Work::Leaf(_) => 0,
        Work::Node(kids) => kids.iter().map(|(_, k)| kept_pages(k)).sum(),
    }
}

fn fresh_pages(fresh: &Fresh) -> u32 {
    let own = u32::from(fresh.page.is_none());
    own + match &fresh.body {
        Body::Leaf(_) => 0,
        Body::Node(kids) => kids
            .iter()
            .map(|k| match k {
                Kid::Old(_) => 0,
                Kid::Kept(_, kept) => kept_pages(kept),
                Kid::Fresh(f) => fresh_pages(f),
            })
            .sum(),
    }
}

/// Writes `plan` within memory the caller reserved ([`Plan::pages`]); an allocation failure
/// means the reservation was too small.
pub(crate) fn commit<M: Memory>(
    store: &mut PageStore<M>,
    plan: Plan,
) -> Result<Option<Root>, WriteError> {
    let root = match plan.top {
        Top::Unchanged(root) => return Ok(root),
        Top::Kept(root, kept) => {
            write_kept(store, &kept);
            Some(root)
        }
        Top::Fresh(fresh) => Some(Root {
            page: write_fresh(store, fresh)?.page,
            depth: plan.depth,
        }),
        Top::Empty => None,
    };
    for page in plan.freed {
        store.release(Extent::whole(page));
    }
    Ok(root.map(|r| shrink(store, r)))
}

fn write_kept<M: Memory>(store: &PageStore<M>, kept: &Kept) {
    match &kept.work {
        Work::Leaf(writes) => {
            for (offset, bytes) in writes {
                store.write(at(kept.page, *offset), bytes);
            }
        }
        Work::Node(kids) => {
            for (i, kid) in kids {
                write_kept(store, kid);
                let count = shifted(count_at(&store.pages(), kept.page, *i), kid.delta);
                write_child(store, kept.page, *i, count, kid.first);
            }
        }
    }
}

fn write_fresh<M: Memory>(store: &mut PageStore<M>, fresh: Fresh) -> Result<Child, WriteError> {
    let page = match fresh.page {
        Some(page) => page,
        None => store.allocate(ByteClass::PAGE)?.page(),
    };
    match fresh.body {
        Body::Leaf(bytes) => store.write(page.offset(), &bytes),
        Body::Node(kids) => {
            let mut children = Vec::with_capacity(kids.len());
            for kid in kids {
                children.push(match kid {
                    Kid::Old(c) => c,
                    Kid::Kept(c, kept) => {
                        write_kept(store, &kept);
                        c
                    }
                    Kid::Fresh(f) => write_fresh(store, f)?,
                });
            }
            write_node(store, page, &children);
        }
    }
    Ok(Child {
        first: fresh.first,
        page,
        count: fresh.count,
    })
}

/// A root with one child gives way to it.
fn shrink<M: Memory>(store: &mut PageStore<M>, mut top: Root) -> Root {
    while top.depth > 0 {
        let view = store.pages();
        if node_len(&view, top.page) != 1 {
            break;
        }
        let only = page_at(&view, top.page, 0);
        store.release(Extent::whole(top.page));
        top = Root {
            page: only,
            depth: top.depth - 1,
        };
    }
    top
}

/// `count` changed by `delta`.
pub(super) fn shifted(count: u32, delta: i64) -> u32 {
    u32::try_from(i64::from(count) + delta).unwrap_or_else(|_| corrupt("an index count overflow"))
}
