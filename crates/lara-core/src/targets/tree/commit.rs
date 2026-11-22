//! A planned change to the tree, and writing it.
//!
//! Planning (`tree`) reads the tree and decides everything a batch of changes does: the bytes each
//! leaf and node gets, and which nodes need new pages. Committing writes that plan, taking exactly
//! the pages it counted, so a batch is reserved for and applied in one pass of the work.

use super::Root;
use super::node::{Child, at, count_at, write_child, write_node};
use crate::corrupt::corrupt;
use crate::memory::{Memory, WriteError};
use crate::page_store::{ByteClass, PageId, PageStore};
use alloc::vec::Vec;

type Key = (u32, u32);

/// A node that stays on its page: what changes in it, and the change in its total. Its parent's
/// key for it stays: routing only ever sends a node keys at or above that key, so whatever the
/// changes do to its first key, the parent's stays a valid lower bound.
pub(super) struct Kept {
    pub(super) page: PageId,
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
}

impl Plan {
    /// The pages committing the plan allocates.
    pub(crate) fn pages(&self) -> u32 {
        match &self.top {
            // A kept node's changed children are all kept too: none takes a page.
            Top::Unchanged(_) | Top::Kept(..) => 0,
            Top::Fresh(fresh) => fresh_pages(fresh),
        }
    }
}

fn fresh_pages(fresh: &Fresh) -> u32 {
    let own = u32::from(fresh.page.is_none());
    own + match &fresh.body {
        Body::Leaf(_) => 0,
        Body::Node(kids) => kids
            .iter()
            .map(|k| match k {
                Kid::Old(_) | Kid::Kept(..) => 0,
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
    };
    Ok(root)
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
                write_child(store, kept.page, *i, count);
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

/// `count` changed by `delta`.
pub(super) fn shifted(count: u32, delta: i64) -> u32 {
    u32::try_from(i64::from(count) + delta).unwrap_or_else(|_| corrupt("an index count overflow"))
}
