//! Compaction: reclaims a bucket's tombstones in one bounded step.
//!
//! Every move keeps live edges in their relative order (they are packed toward the front), so
//! insertion order and the pairing of parallel edges by rank (ADR 0048) survive. Each moved edge
//! is reported. One step moves at most one page of edges:
//!
//! - Tiny and run buckets are packed in one step; a run shrinks to the smallest class that holds
//!   its edges, to a tiny bucket at three or fewer, and an empty bucket goes away.
//! - A paged bucket with at most [`DEMOTE_AT`] live edges is packed into a run (or tiny bucket).
//! - Otherwise one page-level action per step: a dead page (not the last) is released and its
//!   index entry left hollow, or a page with holes absorbs the next page with content when both
//!   fit in one page. No position after them moves. The last page is left to inserts, which
//!   reuse its holes.

use crate::bucket::{self, byte_len, narrow, run_class, stored_u32};
use crate::descriptor::{Descriptor, PAGE_SLOTS, Paged, Run, Slot, TINY_SLOTS, Tiny};
use crate::id::{Edge, Move, Position, Progress};
use crate::memory::{Memory, WriteError};
use crate::page_store::{Extent, PageId, PageStore};
use crate::paged::{self, Leaf};
use alloc::vec::Vec;

/// Paged buckets with this many live edges or fewer are packed back into a run.
pub(crate) const DEMOTE_AT: u32 = PAGE_SLOTS / 2;

/// What a step leaves.
pub(crate) enum Outcome {
    /// The bucket's next descriptor (possibly unchanged).
    Bucket(Descriptor),
    /// The bucket has no positions left; its label entry goes.
    Empty,
}

pub(crate) struct Step {
    pub(crate) outcome: Outcome,
    pub(crate) moves: Vec<Move>,
    pub(crate) progress: Progress,
    /// A data page the step released, its index entry left hollow: its positions hold no edges.
    pub(crate) hollowed: Option<u32>,
}

impl Step {
    fn unchanged(d: Descriptor) -> Self {
        Self {
            outcome: Outcome::Bucket(d),
            moves: Vec::new(),
            progress: Progress::Done,
            hollowed: None,
        }
    }
}

/// Whether a compacted bucket may become tiny. A bucket with values may not: its three inline
/// slots leave no room for them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Inline {
    Allowed,
    Forbidden,
}

/// One bounded compaction step. On error nothing changed.
pub(crate) fn step<M: Memory>(
    d: Descriptor,
    store: &mut PageStore<M>,
    inline: Inline,
) -> Result<Step, WriteError> {
    if d.live() == d.stored() {
        return Ok(Step::unchanged(d));
    }
    match d {
        Descriptor::Tiny(_) | Descriptor::Run(_) => pack_small(d, store, inline),
        Descriptor::Paged(paged) if paged.live <= DEMOTE_AT => demote(paged, store, inline),
        Descriptor::Paged(paged) => Ok(page_step(paged, store)),
    }
}

/// Packs a tiny or run bucket into the smallest form that holds its live edges.
fn pack_small<M: Memory>(
    d: Descriptor,
    store: &mut PageStore<M>,
    inline: Inline,
) -> Result<Step, WriteError> {
    let edges = live_edges(&d, store);
    let old_extent = match d {
        Descriptor::Run(run) => Some(run.extent),
        Descriptor::Tiny(_) | Descriptor::Paged(_) => None,
    };
    // Reuse the run's extent when the packed edges stay a run of the same class. A tiny or empty
    // result uses no extent, so the old one is released below.
    let stays_run = edges.len() > TINY_SLOTS || (inline == Inline::Forbidden && !edges.is_empty());
    let reuse = old_extent.filter(|e| stays_run && e.class() == run_class(stored_u32(edges.len())));
    let outcome = rebuild(&edges, reuse, store, inline)?;
    if let Some(old) = old_extent.filter(|old| Some(*old) != reuse) {
        store.release(old);
    }
    Ok(Step {
        outcome,
        moves: moves_to_front(&edges, 0),
        progress: Progress::Done,
        hollowed: None,
    })
}

/// Packs a paged bucket with few live edges into a run or tiny bucket, releasing its pages.
fn demote<M: Memory>(
    paged: Paged,
    store: &mut PageStore<M>,
    inline: Inline,
) -> Result<Step, WriteError> {
    let edges = live_edges(&Descriptor::Paged(paged), store);
    // Allocate first: if that fails, the paged bucket is untouched.
    let outcome = rebuild(&edges, None, store, inline)?;
    let view = store.pages();
    let data: Vec<PageId> = paged::leaves(&view, &paged)
        .into_iter()
        .filter_map(|l| l.page)
        .collect();
    let nodes = paged::index_nodes(&view, &paged);
    for page in data.into_iter().chain(nodes) {
        store.release(Extent::whole(page));
    }
    store.release(paged.index);
    Ok(Step {
        outcome,
        moves: moves_to_front(&edges, 0),
        progress: Progress::Done,
        hollowed: None,
    })
}

/// What one page-level step of a paged bucket does.
enum PageAction {
    /// Release this dead page (not the last) and leave its entry hollow.
    Hollow(u32),
    /// Pack the second page's live edges into the first.
    Merge((u32, Leaf), (u32, Leaf)),
    Nothing,
}

/// The page-level action for `paged`, read-only.
fn choose<M: Memory>(paged: &Paged, store: &PageStore<M>) -> PageAction {
    let leaves = paged::leaves(&store.pages(), paged);
    let last = stored_u32(leaves.len()) - 1;
    let full = narrow(PAGE_SLOTS);
    let candidates = (0u32..).zip(&leaves).take_while(|(n, _)| *n < last);
    for (number, leaf) in candidates.filter(|(_, l)| l.page.is_some() && l.tombstones > 0) {
        if leaf.tombstones == full {
            return PageAction::Hollow(number);
        }
        let next = (number + 1..last).find(|&n| leaf_at(&leaves, n).page.is_some());
        if let Some(next) = next {
            let (first, second) = (leaf_at(&leaves, number), leaf_at(&leaves, next));
            if u32::from(first.tombstones) + u32::from(second.tombstones) >= PAGE_SLOTS {
                return PageAction::Merge((number, first), (next, second));
            }
        }
    }
    PageAction::Nothing
}

/// The moves the next [`step`] of a large paged bucket will report, without changing anything;
/// empty for every other bucket. Lets the caller plan what the moves cost before the step.
pub(crate) fn preview<M: Memory>(d: Descriptor, store: &PageStore<M>) -> Vec<Move> {
    let Descriptor::Paged(paged) = d else {
        return Vec::new();
    };
    if d.live() == d.stored() || paged.live <= DEMOTE_AT {
        return Vec::new();
    }
    match choose(&paged, store) {
        PageAction::Merge(first, second) => {
            let edges = merged_edges(store, first, second);
            moves_to_front(&edges, first.0 * PAGE_SLOTS)
        }
        PageAction::Hollow(_) | PageAction::Nothing => Vec::new(),
    }
}

/// One page-level action, or none when nothing can be reclaimed this way.
fn page_step<M: Memory>(paged: Paged, store: &mut PageStore<M>) -> Step {
    let step = |moves, hollowed| Step {
        outcome: Outcome::Bucket(Descriptor::Paged(paged)),
        moves,
        progress: Progress::Remaining,
        hollowed: Some(hollowed),
    };
    match choose(&paged, store) {
        PageAction::Hollow(number) => {
            paged::hollow(store, &paged, number);
            step(Vec::new(), number)
        }
        PageAction::Merge(first, second) => {
            let moves = merge(store, &paged, first, second);
            step(moves, second.0)
        }
        PageAction::Nothing => Step::unchanged(Descriptor::Paged(paged)),
    }
}

/// The live edges of two pages, in position order.
fn merged_edges<M: Memory>(
    store: &PageStore<M>,
    first: (u32, Leaf),
    second: (u32, Leaf),
) -> Vec<Edge> {
    [first, second]
        .into_iter()
        .flat_map(|(number, leaf)| page_edges(store, number, leaf.present()))
        .collect()
}

/// Packs data page `second` into data page `first` (their live edges fit one page), then
/// releases `second` and leaves its entry hollow. Pages between them are hollow, so order holds.
fn merge<M: Memory>(
    store: &mut PageStore<M>,
    paged: &Paged,
    first: (u32, Leaf),
    second: (u32, Leaf),
) -> Vec<Move> {
    let edges = merged_edges(store, first, second);
    let base = first.0 * PAGE_SLOTS;
    let mut bytes: Vec<u8> = edges
        .iter()
        .flat_map(|e| Slot::Live(e.target()).encode())
        .collect();
    bytes.resize(byte_len(PAGE_SLOTS), 0xFF);
    store.write(first.1.present().offset(), &bytes);
    paged::set_tombstones(
        store,
        paged,
        first.0,
        narrow(PAGE_SLOTS - stored_u32(edges.len())),
    );
    paged::hollow(store, paged, second.0);
    moves_to_front(&edges, base)
}

/// The live edges of data page `number`, with their bucket positions.
fn page_edges<M: Memory>(store: &PageStore<M>, number: u32, page: PageId) -> Vec<Edge> {
    let bytes = store.pages().read(page.offset(), byte_len(PAGE_SLOTS));
    let base = number * PAGE_SLOTS;
    (base..)
        .zip(Slot::decode_all(&bytes))
        .filter_map(|(at, slot)| match slot {
            Slot::Live(target) => Some(Edge::new(Position::new(at), target)),
            Slot::Tombstone => None,
        })
        .collect()
}

/// A tiny bucket for up to three edges where allowed, a run for more (in `reuse` if given, else
/// a new extent), or nothing for none.
fn rebuild<M: Memory>(
    edges: &[Edge],
    reuse: Option<Extent>,
    store: &mut PageStore<M>,
    inline: Inline,
) -> Result<Outcome, WriteError> {
    let count = stored_u32(edges.len());
    if edges.is_empty() {
        return Ok(Outcome::Empty);
    }
    if edges.len() <= TINY_SLOTS && inline == Inline::Allowed {
        let tiny = edges
            .iter()
            .fold(Tiny::EMPTY, |t, e| t.push(Slot::Live(e.target())));
        return Ok(Outcome::Bucket(Descriptor::Tiny(tiny)));
    }
    let extent = match reuse {
        Some(extent) => extent,
        None => store.allocate(run_class(count))?,
    };
    let bytes: Vec<u8> = edges
        .iter()
        .flat_map(|e| Slot::Live(e.target()).encode())
        .collect();
    store.write(extent.offset(), &bytes);
    let packed = narrow(count);
    Ok(Outcome::Bucket(Descriptor::Run(Run {
        extent,
        stored: packed,
        live: packed,
    })))
}

fn live_edges<M: Memory>(d: &Descriptor, store: &PageStore<M>) -> Vec<Edge> {
    let mut edges = Vec::new();
    bucket::read(d, &store.pages(), &mut edges);
    edges
}

/// The moves of packing `edges` (in position order) to consecutive positions from `base`.
fn moves_to_front(edges: &[Edge], base: u32) -> Vec<Move> {
    (base..)
        .zip(edges)
        .filter(|(to, e)| e.position().index() != *to)
        .map(|(to, e)| Move::new(e.position(), Position::new(to), e.target()))
        .collect()
}

fn leaf_at(leaves: &[Leaf], number: u32) -> Leaf {
    usize::try_from(number)
        .ok()
        .and_then(|i| leaves.get(i).copied())
        .unwrap_or_else(|| crate::corrupt::corrupt("index entry"))
}
