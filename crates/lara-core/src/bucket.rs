//! Bucket operations: insert, remove and read one bucket's edges by position.
//!
//! Every operation takes a descriptor and returns the next one; the caller publishes it.
//!
//! - **Positions are stable.** Removal writes a tombstone in place; growth copies tombstones
//!   along. Only compaction (not here) renumbers positions.
//! - **Equal-target edges stay in insertion order.** An insert reuses a hole only if no live edge
//!   with the same target lies after it. Forward and reverse buckets therefore agree on the
//!   order of parallel edges without coordinating.
//! - **Cost is bounded.** An insert copies at most one run below a page, or one page of index
//!   entries; a full one-page run becomes the first data page without copying.

use crate::corrupt::corrupt;
use crate::descriptor::{Descriptor, PAGE_SLOTS, Paged, Run, SLOT_BYTES, Slot, TINY_SLOTS, Tiny};
use crate::id::{Edge, Position, VertexId};
use crate::memory::{Memory, WriteError};
use crate::page_store::{ByteClass, PageStore, Pages};
use crate::paged::{self, Leaf, TombstoneChange};
use crate::units::{ByteOffset, Overflow};
use alloc::vec::Vec;
use core::ops::ControlFlow;

/// Pages one [`insert`] can take from the store. The worst case is a paged bucket whose index
/// deepens: a data page, a new top and the fresh nodes on the path below it (two at most, as
/// three levels cover every `u32` position). Promotion takes two (a page and an index), growth
/// one.
pub(crate) const PAGES_PER_INSERT: u32 = 4;

/// Adds an edge to `target`. Returns the bucket's next descriptor and the edge's position.
pub(crate) fn insert<M: Memory>(
    d: Descriptor,
    target: VertexId,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Position), WriteError> {
    let edge = Slot::Live(target);
    match d {
        Descriptor::Tiny(tiny) => {
            if let Some(hole) = reusable_hole(tiny.slots(), target) {
                return Ok((Descriptor::Tiny(tiny.set(hole, edge)), position(hole)));
            }
            if !tiny.is_full() {
                let at = tiny.slots().len();
                return Ok((Descriptor::Tiny(tiny.push(edge)), position(at)));
            }
            tiny_to_run(tiny, edge, store)
        }
        Descriptor::Run(run) => run_insert(run, target, store),
        Descriptor::Paged(paged) => paged_insert(paged, target, store),
    }
}

/// Removes the edge at `at`. Returns the next descriptor and the removed edge's target, or
/// `None` when no live edge is there. Never fails: it writes in place.
pub(crate) fn remove<M: Memory>(
    d: Descriptor,
    at: Position,
    store: &PageStore<M>,
) -> Option<(Descriptor, VertexId)> {
    if at.index() >= d.stored() {
        return None;
    }
    match d {
        Descriptor::Tiny(tiny) => {
            let i = usize::try_from(at.index()).ok()?;
            let Slot::Live(target) = *tiny.slots().get(i)? else {
                return None;
            };
            Some((Descriptor::Tiny(tiny.set(i, Slot::Tombstone)), target))
        }
        Descriptor::Run(run) => {
            let slot_at = slot_offset(run.extent.offset(), at.index());
            let Slot::Live(target) = Slot::decode(store.pages().read_array(slot_at)) else {
                return None;
            };
            store.write(slot_at, &Slot::Tombstone.encode());
            Some((
                Descriptor::Run(Run {
                    live: run.live - 1,
                    ..run
                }),
                target,
            ))
        }
        Descriptor::Paged(paged) => {
            let number = at.index() / PAGE_SLOTS;
            let page = paged::data_page(&store.pages(), &paged, number).page?;
            let slot_at = slot_offset(page.offset(), at.index() % PAGE_SLOTS);
            let Slot::Live(target) = Slot::decode(store.pages().read_array(slot_at)) else {
                return None;
            };
            store.write(slot_at, &Slot::Tombstone.encode());
            paged::change_tombstones(store, &paged, number, TombstoneChange::Add);
            Some((
                Descriptor::Paged(Paged {
                    live: paged.live - 1,
                    ..paged
                }),
                target,
            ))
        }
    }
}

/// The live edge at `at`, if any. Reads the descriptor and page data only.
pub(crate) fn get<M: Memory>(
    d: &Descriptor,
    at: Position,
    view: &Pages<'_, M>,
) -> Option<VertexId> {
    if at.index() >= d.stored() {
        return None;
    }
    let slot = match d {
        Descriptor::Tiny(tiny) => *tiny.slots().get(usize::try_from(at.index()).ok()?)?,
        Descriptor::Run(run) => {
            Slot::decode(view.read_array(slot_offset(run.extent.offset(), at.index())))
        }
        Descriptor::Paged(paged) => {
            let page = paged::data_page(view, paged, at.index() / PAGE_SLOTS).page?;
            Slot::decode(view.read_array(slot_offset(page.offset(), at.index() % PAGE_SLOTS)))
        }
    };
    match slot {
        Slot::Live(target) => Some(target),
        Slot::Tombstone => None,
    }
}

/// Appends the bucket's live edges to `out`, in position order. Reads the descriptor and page
/// data only: one read for a run; for a paged bucket, the index and each page that is not all
/// tombstones.
pub(crate) fn read<M: Memory>(d: &Descriptor, view: &Pages<'_, M>, out: &mut Vec<Edge>) {
    scan(d, view, 0, |first, bytes| {
        push_live(Slot::decode_all(bytes), first, out);
        ControlFlow::<()>::Continue(())
    });
}

/// The first live edge at `from` or later that `wanted` accepts; `wanted` sees each live edge
/// in position order until then. Reads only up to it, skipping pages that are all tombstones.
pub(crate) fn find<M: Memory>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    from: Position,
    mut wanted: impl FnMut(Edge) -> bool,
) -> Option<Edge> {
    let from = from.index();
    scan(d, view, from, |first, bytes| {
        let hit = (first..)
            .zip(Slot::decode_all(bytes))
            .skip_while(|(at, _)| *at < from)
            .find_map(|(at, slot)| match slot {
                Slot::Live(target) => {
                    Some(Edge::new(Position::new(at), target)).filter(|e| wanted(*e))
                }
                Slot::Tombstone => None,
            });
        hit.map_or(ControlFlow::Continue(()), ControlFlow::Break)
    })
}

/// Feeds the bucket's stored slots, as bytes with the position of the first, to `visit` until it
/// breaks: the tiny slots, the run, or each data page that is not all tombstones and does not end
/// before `from`.
fn scan<M: Memory, T>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    from: u32,
    mut visit: impl FnMut(u32, &[u8]) -> ControlFlow<T>,
) -> Option<T> {
    let flow = match d {
        Descriptor::Tiny(tiny) => {
            let bytes: Vec<u8> = tiny.slots().iter().flat_map(|s| s.encode()).collect();
            visit(0, &bytes)
        }
        Descriptor::Run(run) => visit(
            0,
            &view.read(run.extent.offset(), byte_len(u32::from(run.stored))),
        ),
        Descriptor::Paged(paged) => {
            (0u32..)
                .zip(paged::leaves(view, paged))
                .try_for_each(|(number, leaf)| {
                    let first = number * PAGE_SLOTS;
                    let count = (paged.stored - first).min(PAGE_SLOTS);
                    let live_page = leaf.page.filter(|_| u32::from(leaf.tombstones) < count);
                    match live_page {
                        Some(page) if first + count > from => {
                            visit(first, &view.read(page.offset(), byte_len(count)))
                        }
                        Some(_) | None => ControlFlow::Continue(()),
                    }
                })
        }
    };
    match flow {
        ControlFlow::Break(found) => Some(found),
        ControlFlow::Continue(()) => None,
    }
}

// --- runs --------------------------------------------------------------------------------------

fn tiny_to_run<M: Memory>(
    tiny: Tiny,
    edge: Slot,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Position), WriteError> {
    let stored = TINY_SLOTS + 1;
    let extent = store.allocate(run_class(stored_u32(stored)))?;
    let bytes: Vec<u8> = tiny
        .slots()
        .iter()
        .chain([&edge])
        .flat_map(|s| s.encode())
        .collect();
    store.write(extent.offset(), &bytes);
    let live = Descriptor::Tiny(tiny).live() + 1;
    let run = Run {
        extent,
        stored: narrow(stored_u32(stored)),
        live: narrow(live),
    };
    Ok((Descriptor::Run(run), position(TINY_SLOTS)))
}

fn run_insert<M: Memory>(
    run: Run,
    target: VertexId,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Position), WriteError> {
    let edge = Slot::Live(target);
    let stored = u32::from(run.stored);
    let live = run.live + 1;
    if run.live < run.stored {
        let bytes = store.pages().read(run.extent.offset(), byte_len(stored));
        let slots: Vec<Slot> = Slot::decode_all(&bytes).collect();
        if let Some(hole) = reusable_hole(&slots, target) {
            store.write(
                slot_offset(run.extent.offset(), stored_u32(hole)),
                &edge.encode(),
            );
            return Ok((Descriptor::Run(Run { live, ..run }), position(hole)));
        }
    }
    let next = Run {
        stored: run.stored + 1,
        live,
        ..run
    };
    if stored < run.capacity() {
        store.write(slot_offset(run.extent.offset(), stored), &edge.encode());
        return Ok((Descriptor::Run(next), Position::new(stored)));
    }
    if run.capacity() < PAGE_SLOTS {
        // Move to the next class. One allocation: a failure leaves nothing behind.
        let extent = store.allocate(run_class(stored + 1))?;
        let bytes = store.pages().read(run.extent.offset(), byte_len(stored));
        store.write(extent.offset(), &bytes);
        store.write(slot_offset(extent.offset(), stored), &edge.encode());
        store.release(run.extent);
        return Ok((
            Descriptor::Run(Run { extent, ..next }),
            Position::new(stored),
        ));
    }
    // A full page: it becomes data page 0 as it is, and the new edge opens page 1.
    let tombstones = narrow(stored - u32::from(run.live));
    let (index, page) = paged::start(store, Leaf::live(run.extent.page(), tombstones))?;
    store.write(page.offset(), &edge.encode());
    let paged = Paged {
        index,
        stored: stored + 1,
        live: u32::from(live),
    };
    Ok((Descriptor::Paged(paged), Position::new(stored)))
}

// --- pages -------------------------------------------------------------------------------------

fn paged_insert<M: Memory>(
    paged: Paged,
    target: VertexId,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Position), WriteError> {
    let edge = Slot::Live(target);
    let live = paged
        .live
        .checked_add(1)
        .ok_or(WriteError::Overflow(Overflow))?;
    let last = paged::data_pages(paged.stored) - 1;
    let first = last * PAGE_SLOTS;
    let in_last = paged.stored - first;
    let leaf = paged::data_page(&store.pages(), &paged, last);
    // The last data page is never hollow: compaction keeps it.
    let last_page = leaf.present();
    // Holes are reused in the last page only: earlier holes precede every edge of that page.
    if leaf.tombstones > 0 {
        let bytes = store.pages().read(last_page.offset(), byte_len(in_last));
        let slots: Vec<Slot> = Slot::decode_all(&bytes).collect();
        if let Some(hole) = reusable_hole(&slots, target) {
            store.write(
                slot_offset(last_page.offset(), stored_u32(hole)),
                &edge.encode(),
            );
            paged::change_tombstones(store, &paged, last, TombstoneChange::Remove);
            let at = Position::new(first + stored_u32(hole));
            return Ok((Descriptor::Paged(Paged { live, ..paged }), at));
        }
    }
    let stored = paged
        .stored
        .checked_add(1)
        .ok_or(WriteError::Overflow(Overflow))?;
    let at = Position::new(paged.stored);
    if in_last < PAGE_SLOTS {
        store.write(slot_offset(last_page.offset(), in_last), &edge.encode());
        return Ok((
            Descriptor::Paged(Paged {
                stored,
                live,
                ..paged
            }),
            at,
        ));
    }
    let (index, page) = paged::append_page(store, &paged)?;
    store.write(page.offset(), &edge.encode());
    Ok((
        Descriptor::Paged(Paged {
            index,
            stored,
            live,
        }),
        at,
    ))
}

// --- helpers -----------------------------------------------------------------------------------

/// The first hole after the last live edge to `target`: filling it keeps equal-target edges in
/// insertion order. `None` when there is no such hole.
fn reusable_hole(slots: &[Slot], target: VertexId) -> Option<usize> {
    let mut hole = None;
    for (i, slot) in slots.iter().enumerate().rev() {
        match slot {
            Slot::Live(t) if *t == target => break,
            Slot::Tombstone => hole = Some(i),
            Slot::Live(_) => {}
        }
    }
    hole
}

fn push_live(slots: impl Iterator<Item = Slot>, first: u32, out: &mut Vec<Edge>) {
    for (i, slot) in (first..).zip(slots) {
        if let Slot::Live(target) = slot {
            out.push(Edge::new(Position::new(i), target));
        }
    }
}

/// The class of a run holding `slots` slots.
pub(crate) fn run_class(slots: u32) -> ByteClass {
    ByteClass::for_bytes(slots * SLOT_BYTES).unwrap_or_else(|| corrupt("run beyond a page"))
}

fn slot_offset(start: ByteOffset, slot: u32) -> ByteOffset {
    ByteOffset::new(start.get() + u64::from(slot) * u64::from(SLOT_BYTES))
}

pub(crate) fn byte_len(slots: u32) -> usize {
    usize::try_from(slots * SLOT_BYTES).unwrap_or_else(|_| corrupt("read length"))
}

fn position(i: usize) -> Position {
    Position::new(stored_u32(i))
}

pub(crate) fn stored_u32(i: usize) -> u32 {
    u32::try_from(i).unwrap_or_else(|_| corrupt("bucket position"))
}

/// A count that fits a run's 16-bit fields (runs hold at most a page of slots).
pub(crate) fn narrow(count: u32) -> u16 {
    u16::try_from(count).unwrap_or_else(|_| corrupt("run count"))
}

/// Every extent a bucket owns (test invariant checks).
#[cfg(test)]
pub(crate) fn owned_extents<M: Memory>(
    d: &Descriptor,
    view: &Pages<'_, M>,
) -> Vec<crate::page_store::Extent> {
    match d {
        Descriptor::Tiny(_) => Vec::new(),
        Descriptor::Run(run) => alloc::vec![run.extent],
        Descriptor::Paged(paged) => {
            let whole = crate::page_store::Extent::whole;
            let mut out: Vec<_> = paged::leaves(view, paged)
                .iter()
                .filter_map(|leaf| leaf.page.map(whole))
                .collect();
            out.extend(paged::index_extents(view, paged));
            out
        }
    }
}
