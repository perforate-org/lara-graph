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
use crate::id::{Placement, Position, VertexId};
use crate::memory::{Memory, WriteError};
use crate::page_store::{ByteClass, PageStore, Pages};
use crate::paged::{self, Leaf, TombstoneChange};
use crate::units::{ByteOffset, Overflow};
use alloc::vec::Vec;

mod positions;
mod scan;

pub(crate) use positions::read_positions;
pub(crate) use scan::{Span, Window, find, read, visit};

/// Pages one [`insert`] can take from the store. The worst case is a paged bucket whose index
/// deepens: a data page, a new top and the fresh nodes on the path below it (two at most, as
/// three levels cover every `u32` position). Promotion takes two (a page and an index), growth
/// one.
pub(crate) const PAGES_PER_INSERT: u32 = 4;

/// Adds an edge to `target` where `placement` allows. Returns the bucket's next descriptor and
/// the edge's position. [`Placement::Append`] never reads the bucket's slots.
pub(crate) fn insert<M: Memory>(
    d: Descriptor,
    target: VertexId,
    placement: Placement,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Position), WriteError> {
    let edge = Slot::Live(target);
    let reuse = placement == Placement::ReuseHoles;
    match d {
        Descriptor::Tiny(tiny) => {
            let hole = reuse.then(|| reusable_hole(tiny.slots(), target)).flatten();
            if let Some(hole) = hole {
                return Ok((Descriptor::Tiny(tiny.set(hole, edge)), position(hole)));
            }
            if !tiny.is_full() {
                let at = tiny.slots().len();
                return Ok((Descriptor::Tiny(tiny.push(edge)), position(at)));
            }
            tiny_to_run(tiny, edge, store)
        }
        Descriptor::Run(run) => run_insert(run, target, placement, store),
        Descriptor::Paged(paged) => paged_insert(paged, target, placement, store),
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
    placement: Placement,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Position), WriteError> {
    let edge = Slot::Live(target);
    let stored = u32::from(run.stored);
    let live = run.live + 1;
    if placement == Placement::ReuseHoles && run.live < run.stored {
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
    placement: Placement,
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
    if placement == Placement::ReuseHoles && leaf.tombstones > 0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{Edge, Order};
    use crate::memory::VecMemory;
    use core::ops::ControlFlow;

    /// Short windows read a span in either order, across pages and with holes.
    #[test]
    fn short_windows_read_spans_both_ways() {
        let mut store = PageStore::create(VecMemory::new(), VecMemory::new()).unwrap();
        let mut d = Descriptor::EMPTY;
        let total = PAGE_SLOTS + 300;
        for i in 0..total {
            d = insert(d, VertexId::new(i), Placement::Append, &mut store)
                .unwrap()
                .0;
        }
        for at in (0..total).step_by(7) {
            d = remove(d, Position::new(at), &store).unwrap().0;
        }
        let (low, high) = (900, PAGE_SLOTS + 200);
        let live: Vec<u32> = (low..high).filter(|at| at % 7 != 0).collect();
        for order in [Order::Ascending, Order::Descending] {
            let mut seen = Vec::new();
            visit(
                &d,
                &store.pages(),
                Span::new(low, high, order),
                Window::Short,
                |e| {
                    assert_eq!(e.target().index(), e.position().index());
                    seen.push(e.position().index());
                    ControlFlow::<()>::Continue(())
                },
            );
            let mut want = live.clone();
            if order == Order::Descending {
                want.reverse();
            }
            assert_eq!(seen, want, "{order:?}");
        }
    }

    /// `find` and `visit` start at `from` in every bucket form, including past a page boundary.
    #[test]
    fn searches_start_at_their_position() {
        let mut store = PageStore::create(VecMemory::new(), VecMemory::new()).unwrap();
        for total in [3, 10, PAGE_SLOTS * 2 + 7] {
            let mut d = Descriptor::EMPTY;
            for _ in 0..total {
                d = insert(d, VertexId::new(1), Placement::ReuseHoles, &mut store)
                    .unwrap()
                    .0;
            }
            let view = store.pages();
            for from in [0, 1, total / 2, PAGE_SLOTS + 1, total - 1, total] {
                let found = find(&d, &view, Position::new(from), |_| true);
                let expected =
                    (from < total).then(|| Edge::new(Position::new(from), VertexId::new(1)));
                assert_eq!(found, expected, "{total} edges, from {from}");
            }
        }
    }
}
