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

use crate::bucket::descriptor::{
    Beside, Descriptor, PAGE_SLOTS, Paged, Run, Slot, TINY_SLOTS, Tiny, run_class,
};
use crate::bucket::paged::{self, Leaf};
use crate::bucket::slots::{self, byte_len, narrow, stored_u32};
use crate::id::Width;
use crate::id::{Neighbor, Position};
use crate::outcome::{Move, Progress};
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::{Extent, PageId, PageStore};
use alloc::vec::Vec;

/// Paged buckets with this many live edges or fewer are packed back into a run. Half a page below
/// the point where a run becomes paged, so a bucket near the line does not change form back and
/// forth.
pub(crate) const DEMOTE_AT: u32 = PAGE_SLOTS / 2;

/// What a step leaves.
pub(crate) enum Outcome {
    /// The bucket's next descriptor (possibly unchanged).
    LabelBucket(Descriptor),
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
            outcome: Outcome::LabelBucket(d),
            moves: Vec::new(),
            progress: Progress::Done,
            hollowed: None,
        }
    }
}

/// Whether a compacted bucket may become tiny. A bucket with values may not: its two inline
/// slots leave no room for them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Inline {
    Allowed,
    Forbidden,
}

/// What the next step of a bucket's compaction will do, read but not written: [`Planned::moves`]
/// tells what moves, and [`apply`] does it.
pub(crate) struct Planned {
    action: Action,
    moves: Vec<Move>,
}

/// One bounded step's action, with the live edges it packs.
enum Action {
    Unchanged(Descriptor),
    /// A tiny or run bucket, packed into the smallest form that holds its edges.
    Small(Descriptor, Vec<Neighbor>),
    /// A paged bucket with few live edges, packed into a run or tiny bucket.
    Demote(Paged, Vec<Neighbor>),
    /// A paged bucket's page-level action.
    Page(Paged, PageAction, Vec<Neighbor>),
}

impl Planned {
    /// The moves the step will make, in position order.
    pub(crate) fn moves(&self) -> &[Move] {
        &self.moves
    }

    /// Whether the step does nothing: no holes, or none a step can reclaim.
    pub(crate) const fn is_idle(&self) -> bool {
        matches!(
            self.action,
            Action::Unchanged(_) | Action::Page(_, PageAction::Nothing, _)
        )
    }

    /// Whether the step leaves the bucket paged.
    pub(crate) const fn stays_paged(&self) -> bool {
        matches!(self.action, Action::Page(..))
    }
}

/// Plans the next bounded compaction step of `d`.
pub(crate) fn plan<M: Memory>(d: Descriptor, store: &PageStore<M>) -> Planned {
    let (action, base) = if d.live() == d.stored() {
        (Action::Unchanged(d), 0)
    } else {
        match d {
            Descriptor::Tiny(_) | Descriptor::Run(_) => {
                (Action::Small(d, live_edges(&d, store)), 0)
            }
            Descriptor::Paged(paged) if paged.live <= DEMOTE_AT => {
                (Action::Demote(paged, live_edges(&d, store)), 0)
            }
            Descriptor::Paged(paged) => match choose(&paged, store) {
                PageAction::Merge(first, second) => {
                    let edges = merged_edges(store, first, second);
                    let action = PageAction::Merge(first, second);
                    (Action::Page(paged, action, edges), first.0 * PAGE_SLOTS)
                }
                action @ (PageAction::Hollow(_) | PageAction::Nothing) => {
                    (Action::Page(paged, action, Vec::new()), 0)
                }
            },
        }
    };
    let edges = match &action {
        Action::Unchanged(_) => &[][..],
        Action::Small(_, edges) | Action::Demote(_, edges) | Action::Page(_, _, edges) => edges,
    };
    let moves = moves_to_front(edges, base);
    Planned { action, moves }
}

/// Applies a planned step. On error nothing changed.
pub(crate) fn apply<M: Memory>(
    planned: Planned,
    store: &mut PageStore<M>,
    inline: Inline,
) -> Result<Step, WriteError> {
    let Planned { action, moves } = planned;
    let (outcome, progress, hollowed) = match action {
        Action::Unchanged(d) => return Ok(Step::unchanged(d)),
        Action::Small(d, edges) => (pack_small(d, &edges, store, inline)?, Progress::Done, None),
        Action::Demote(paged, edges) => {
            (demote(paged, &edges, store, inline)?, Progress::Done, None)
        }
        Action::Page(paged, action, edges) => match page_step(paged, action, &edges, store) {
            Some(hollowed) => (
                Outcome::LabelBucket(Descriptor::Paged(paged)),
                Progress::Remaining,
                Some(hollowed),
            ),
            None => return Ok(Step::unchanged(Descriptor::Paged(paged))),
        },
    };
    Ok(Step {
        outcome,
        moves,
        progress,
        hollowed,
    })
}

/// Packs a tiny or run bucket's live `edges` into the smallest form that holds them. Values
/// beside a run's slots are packed with them.
fn pack_small<M: Memory>(
    d: Descriptor,
    edges: &[Neighbor],
    store: &mut PageStore<M>,
    inline: Inline,
) -> Result<Outcome, WriteError> {
    let Descriptor::Run(run) = d else {
        return rebuild(edges, (None, None), store, inline);
    };
    // The packed edges' cells, read before the run is rewritten (perhaps in place).
    let cells = run
        .values
        .map(|beside| (beside, packed_cells(&run, edges, store, beside.width)));
    // Reuse the run's extent when the packed edges stay a run of the same class. A tiny or empty
    // result uses no extent, so the old one is released below.
    let stays_run = edges.len() > TINY_SLOTS || (inline == Inline::Forbidden && !edges.is_empty());
    let count = stored_u32(edges.len());
    let reuse =
        Some(run.extent).filter(|e| stays_run && e.class() == run_class(count, run.width()));
    let outcome = rebuild(edges, (reuse, run.values), store, inline)?;
    if let (Outcome::LabelBucket(Descriptor::Run(packed)), Some((_, cells))) = (&outcome, &cells) {
        store.write(packed.cells_offset(), cells);
    }
    if Some(run.extent) != reuse {
        store.release(run.extent);
    }
    Ok(outcome)
}

/// The cells of `edges` (live edges of `run`, in position order), packed in that order.
fn packed_cells<M: Memory>(
    run: &Run,
    edges: &[Neighbor],
    store: &PageStore<M>,
    width: Width,
) -> Vec<u8> {
    let w = usize::from(width.get());
    let all = store
        .pages()
        .read(run.cells_offset(), usize::from(run.stored) * w);
    edges
        .iter()
        .flat_map(|e| {
            let at = usize::try_from(e.position().index()).unwrap_or(usize::MAX) * w;
            all.get(at..at + w).unwrap_or(&[]).iter().copied()
        })
        .collect()
}

/// Packs a paged bucket's few live `edges` into a run or tiny bucket, releasing its pages.
fn demote<M: Memory>(
    paged: Paged,
    edges: &[Neighbor],
    store: &mut PageStore<M>,
    inline: Inline,
) -> Result<Outcome, WriteError> {
    // Allocate first: if that fails, the paged bucket is untouched. Its values stay in their
    // plane.
    let outcome = rebuild(edges, (None, None), store, inline)?;
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
    Ok(outcome)
}

/// What one page-level step of a paged bucket does.
#[derive(Clone, Copy)]
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

/// Applies a page-level action; returns the data page it released (its entry left hollow), or
/// `None` when there was nothing to do.
fn page_step<M: Memory>(
    paged: Paged,
    action: PageAction,
    edges: &[Neighbor],
    store: &mut PageStore<M>,
) -> Option<u32> {
    match action {
        PageAction::Hollow(number) => {
            paged::hollow(store, &paged, number);
            Some(number)
        }
        PageAction::Merge(first, second) => {
            merge(store, &paged, (first, second), edges);
            Some(second.0)
        }
        PageAction::Nothing => None,
    }
}

/// The live edges of two pages, in position order.
fn merged_edges<M: Memory>(
    store: &PageStore<M>,
    first: (u32, Leaf),
    second: (u32, Leaf),
) -> Vec<Neighbor> {
    [first, second]
        .into_iter()
        .flat_map(|(number, leaf)| page_edges(store, number, leaf.present()))
        .collect()
}

/// Packs data page `second` into data page `first` (their live `edges` fit one page), then
/// releases `second` and leaves its entry hollow. Pages between them are hollow, so order holds.
fn merge<M: Memory>(
    store: &mut PageStore<M>,
    paged: &Paged,
    (first, second): ((u32, Leaf), (u32, Leaf)),
    edges: &[Neighbor],
) {
    let mut bytes: Vec<u8> = edges
        .iter()
        .flat_map(|e| Slot::Live(e.target()).encode())
        .collect();
    bytes.resize(byte_len(PAGE_SLOTS), 0);
    store.write(first.1.present().offset(), &bytes);
    // The live edges are packed at the front: the first hole follows them.
    let live = narrow(stored_u32(edges.len()));
    let holes = (narrow(PAGE_SLOTS) - live, live.min(paged::NO_HOLE));
    paged::set_holes(store, paged, first.0, holes);
    paged::hollow(store, paged, second.0);
}

/// The live edges of data page `number`, with their bucket positions.
fn page_edges<M: Memory>(store: &PageStore<M>, number: u32, page: PageId) -> Vec<Neighbor> {
    let bytes = store.pages().read(page.offset(), byte_len(PAGE_SLOTS));
    let base = number * PAGE_SLOTS;
    (base..)
        .zip(Slot::decode_all(&bytes))
        .filter_map(|(at, slot)| match slot {
            Slot::Live(target) => Some(Neighbor::new(Position::new(at), target)),
            Slot::Tombstone => None,
        })
        .collect()
}

/// A tiny bucket for up to three edges where allowed, a run for more (in `reuse` if given, else
/// a new extent; with room for values of `values` beside its slots if set), or nothing for none.
fn rebuild<M: Memory>(
    edges: &[Neighbor],
    (reuse, values): (Option<Extent>, Option<Beside>),
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
        return Ok(Outcome::LabelBucket(Descriptor::Tiny(tiny)));
    }
    let extent = match reuse {
        Some(extent) => extent,
        None => store.allocate(run_class(count, values.map(|b| b.width)))?,
    };
    let bytes: Vec<u8> = edges
        .iter()
        .flat_map(|e| Slot::Live(e.target()).encode())
        .collect();
    store.write(extent.offset(), &bytes);
    let packed = narrow(count);
    Ok(Outcome::LabelBucket(Descriptor::Run(Run {
        extent,
        stored: packed,
        live: packed,
        values,
    })))
}

fn live_edges<M: Memory>(d: &Descriptor, store: &PageStore<M>) -> Vec<Neighbor> {
    let mut edges = Vec::new();
    slots::read(d, &store.pages(), &mut edges);
    edges
}

/// The moves of packing `edges` (in position order) to consecutive positions from `base`.
fn moves_to_front(edges: &[Neighbor], base: u32) -> Vec<Move> {
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
        .unwrap_or_else(|| crate::store::corrupt::corrupt("index entry"))
}
