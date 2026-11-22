//! Bucket operations: insert, remove and read one bucket's edges by position.
//!
//! Every operation takes a descriptor and returns the next one; the caller publishes it.
//!
//! - **Positions are stable.** Removal writes a tombstone in place; growth copies tombstones
//!   along. Only compaction (not here) renumbers positions.
//! - **Equal-target edges stay in insertion order.** An insert reuses a hole only if no live edge
//!   with the same target lies after it ([`hole`]). Forward and reverse buckets therefore agree
//!   on the order of parallel edges without coordinating.
//! - **Cost is bounded.** An insert copies at most one run below a page, or one page of index
//!   entries; a full one-page run becomes the first data page without copying.

use crate::corrupt::corrupt;
use crate::descriptor::{
    Descriptor, PAGE_SLOTS, Paged, Run, SLOT_BYTES, Slot, TINY_SLOTS, TOMBSTONE, Tiny,
};
use crate::id::{Position, VertexId};
use crate::memory::{Memory, WriteError};
use crate::page_store::{ByteClass, PageId, PageStore, Pages};
use crate::paged::{self, Leaf};
use crate::units::{ByteOffset, Overflow};
use alloc::vec::Vec;

mod positions;
mod scan;

pub(crate) use positions::read_positions;
pub(crate) use scan::{Span, Window, find, find_in, read, visit};

/// Pages one [`insert`] can take from the store. The worst case is a paged bucket whose index
/// deepens: a data page, a new top and the fresh nodes on the path below it (two at most, as
/// three levels cover every `u32` position). Promotion takes two (a page and an index), growth
/// one.
pub(crate) const PAGES_PER_INSERT: u32 = 4;

/// Where an insert puts its edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Spot {
    /// After every stored position. Never reads the bucket's slots.
    End,
    /// Into this hole, one [`hole`] found.
    Hole(Position),
}

/// Adds an edge to `target` at `spot`. Returns the bucket's next descriptor and the edge's
/// position. Filling a hole never fails: it writes in place.
pub(crate) fn insert<M: Memory>(
    d: Descriptor,
    target: VertexId,
    spot: Spot,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Position), WriteError> {
    let edge = Slot::Live(target);
    if let Spot::Hole(at) = spot {
        return Ok((fill(d, at, edge, store), at));
    }
    match d {
        Descriptor::Tiny(tiny) if !tiny.is_full() => {
            let at = tiny.slots().len();
            Ok((Descriptor::Tiny(tiny.push(edge)), position(at)))
        }
        Descriptor::Tiny(tiny) => tiny_to_run(tiny, edge, store),
        Descriptor::Run(run) => run_append(run, target, store),
        Descriptor::Paged(paged) => paged_append(paged, target, store),
    }
}

/// The hole an insert of an edge to `target` may fill: the first after every live edge to
/// `target`, so equal-target edges keep their order. `None` when there is no such hole.
///
/// A small bucket reads its slots. A paged one takes the position of its last live edge to
/// `target` from `last_paged` (its target index knows it), then goes to the first page after it
/// with a hole, by the page index's counts.
pub(crate) fn hole<M: Memory>(
    d: &Descriptor,
    target: VertexId,
    view: &Pages<'_, M>,
    last_paged: impl FnOnce() -> Option<Position>,
) -> Option<Position> {
    if d.live() == d.stored() {
        return None;
    }
    match d {
        Descriptor::Tiny(tiny) => reusable_hole(tiny.slots(), target).map(position),
        Descriptor::Run(run) => {
            let bytes = view.read(run.extent.offset(), byte_len(u32::from(run.stored)));
            let slots: Vec<Slot> = Slot::decode_all(&bytes).collect();
            reusable_hole(&slots, target).map(position)
        }
        Descriptor::Paged(paged) => paged_hole(view, paged, last_paged()),
    }
}

/// The first hole in a paged bucket after `after`.
fn paged_hole<M: Memory>(
    view: &Pages<'_, M>,
    paged: &Paged,
    after: Option<Position>,
) -> Option<Position> {
    let start = after.map_or(0, |p| p.index() + 1);
    let mut from = start / PAGE_SLOTS;
    loop {
        // Only the page holding `start` can have all its holes before it.
        let (number, leaf) = paged::first_fillable(view, paged, from)?;
        let base = number * PAGE_SLOTS;
        let low = start.max(base) - base;
        // The page's first hole is the answer unless `start` lies past it: then read on.
        let first = u32::from(leaf.first_hole);
        let slot = if first >= low {
            Some(first)
        } else {
            first_tombstone(view, leaf.present(), low..page_slots(paged, number))
        };
        if let Some(slot) = slot {
            return Some(Position::new(base + slot));
        }
        from = number + 1;
    }
}

/// The slots data page `number` stores: all of them but in the last page.
fn page_slots(paged: &Paged, number: u32) -> u32 {
    (paged.stored - number * PAGE_SLOTS).min(PAGE_SLOTS)
}

/// Slots read at a time while looking for a hole: a short first window, as the next hole is
/// usually near, then longer ones.
const HOLE_WINDOWS: (u32, u32) = (32, 256);

/// The first tombstone among `slots` of a data page, read a window at a time and compared as raw
/// slot bytes.
fn first_tombstone<M: Memory>(
    view: &Pages<'_, M>,
    page: PageId,
    slots: core::ops::Range<u32>,
) -> Option<u32> {
    let mut low = slots.start;
    let mut window = HOLE_WINDOWS.0;
    while low < slots.end {
        let high = (low + window).min(slots.end);
        window = HOLE_WINDOWS.1;
        let bytes = view.read(slot_offset(page.offset(), low), byte_len(high - low));
        let mut words = bytes.as_chunks::<4>().0.iter();
        if let Some(i) = words.position(|&s| u32::from_le_bytes(s) == TOMBSTONE) {
            return Some(low + stored_u32(i));
        }
        low = high;
    }
    None
}

/// Writes `edge` into the hole at `at`, which [`hole`] found. Never fails.
fn fill<M: Memory>(d: Descriptor, at: Position, edge: Slot, store: &PageStore<M>) -> Descriptor {
    match d {
        Descriptor::Tiny(tiny) => {
            let i = usize::try_from(at.index()).unwrap_or_else(|_| corrupt("tiny slot"));
            Descriptor::Tiny(tiny.set(i, edge))
        }
        Descriptor::Run(run) => {
            store.write(slot_offset(run.extent.offset(), at.index()), &edge.encode());
            Descriptor::Run(Run {
                live: run.live + 1,
                ..run
            })
        }
        Descriptor::Paged(paged) => {
            let (number, slot) = (at.index() / PAGE_SLOTS, at.index() % PAGE_SLOTS);
            let page = paged::data_page(&store.pages(), &paged, number).present();
            store.write(slot_offset(page.offset(), slot), &edge.encode());
            let next = || {
                let after = (slot + 1)..page_slots(&paged, number);
                first_tombstone(&store.pages(), page, after).map_or(paged::NO_HOLE, narrow)
            };
            paged::filled(store, &paged, number, narrow(slot), next);
            Descriptor::Paged(Paged {
                live: paged.live + 1,
                ..paged
            })
        }
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
            let (number, slot) = (at.index() / PAGE_SLOTS, at.index() % PAGE_SLOTS);
            let page = paged::data_page(&store.pages(), &paged, number).page?;
            let slot_at = slot_offset(page.offset(), slot);
            let Slot::Live(target) = Slot::decode(store.pages().read_array(slot_at)) else {
                return None;
            };
            store.write(slot_at, &Slot::Tombstone.encode());
            paged::removed(store, &paged, number, narrow(slot));
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

/// The same slots, never tiny: a tiny bucket becomes a run of the smallest class (a bucket with
/// values needs one); a run or paged bucket is unchanged.
pub(crate) fn without_tiny<M: Memory>(
    d: Descriptor,
    store: &mut PageStore<M>,
) -> Result<Descriptor, WriteError> {
    let Descriptor::Tiny(tiny) = d else {
        return Ok(d);
    };
    let stored = stored_u32(tiny.slots().len());
    let extent = store.allocate(run_class(stored.max(1)))?;
    let bytes: Vec<u8> = tiny.slots().iter().flat_map(|s| s.encode()).collect();
    store.write(extent.offset(), &bytes);
    Ok(Descriptor::Run(Run {
        extent,
        stored: narrow(stored),
        live: narrow(d.live()),
    }))
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

fn run_append<M: Memory>(
    run: Run,
    target: VertexId,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Position), WriteError> {
    let edge = Slot::Live(target);
    let stored = u32::from(run.stored);
    let live = run.live + 1;
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
    let first_hole = if tombstones == 0 {
        paged::NO_HOLE
    } else {
        let all = 0..PAGE_SLOTS;
        first_tombstone(&store.pages(), run.extent.page(), all).map_or(paged::NO_HOLE, narrow)
    };
    let first = Leaf::data(run.extent.page(), tombstones, first_hole);
    let (index, page) = paged::start(store, first)?;
    store.write(page.offset(), &edge.encode());
    let paged = Paged {
        index,
        stored: stored + 1,
        live: u32::from(live),
    };
    Ok((Descriptor::Paged(paged), Position::new(stored)))
}

// --- pages -------------------------------------------------------------------------------------

fn paged_append<M: Memory>(
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
            d = insert(d, VertexId::new(i), Spot::End, &mut store)
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

    /// Reusing inserts go to the first hole after the target's last edge, through a page index
    /// two levels deep, and the index's hole counts follow every fill and removal.
    #[test]
    fn holes_are_found_through_a_deep_page_index() {
        let mut store = PageStore::create(VecMemory::new(), VecMemory::new()).unwrap();
        let mut d = Descriptor::EMPTY;
        let page_count = paged::FANOUT + 3;
        let total = page_count * PAGE_SLOTS - 5;
        for i in 0..total {
            d = insert(d, VertexId::new(i), Spot::End, &mut store)
                .unwrap()
                .0;
            // A hole made before the index deepens is counted in the new top.
            if i == 2 * PAGE_SLOTS {
                d = remove(d, Position::new(3), &store).unwrap().0;
            }
        }
        let Descriptor::Paged(paged) = d else {
            panic!("a paged bucket")
        };
        assert!(
            paged::data_pages(paged.stored) > paged::FANOUT,
            "two index levels"
        );
        let holes = [
            3,
            PAGE_SLOTS * 300 + 7,
            PAGE_SLOTS * (paged::FANOUT + 1) + 1,
            total - 2,
        ];
        for &at in holes.iter().skip(1).rev() {
            d = remove(d, Position::new(at), &store).unwrap().0;
        }
        let check = |d: &Descriptor, store: &PageStore<VecMemory>| {
            let Descriptor::Paged(paged) = d else {
                panic!("a paged bucket")
            };
            paged::check_hole_counts(&store.pages(), paged);
        };
        check(&d, &store);
        let view = store.pages();
        let new = VertexId::new(u32::MAX - 1);
        // No edge to the target: the first hole. After its last edge: the first hole after it.
        assert_eq!(hole(&d, new, &view, || None), Some(Position::new(holes[0])));
        let after = |at: u32| move || Some(Position::new(at));
        assert_eq!(
            hole(&d, new, &view, after(holes[0])),
            Some(Position::new(holes[1]))
        );
        assert_eq!(
            hole(&d, new, &view, after(holes[1] + 5000)),
            Some(Position::new(holes[2]))
        );
        assert_eq!(hole(&d, new, &view, after(holes[3])), None);
        // Filling them in order empties every count, page by page.
        for &at in &holes {
            let found = hole(&d, new, &store.pages(), || None);
            assert_eq!(found, Some(Position::new(at)));
            d = insert(d, new, Spot::Hole(Position::new(at)), &mut store)
                .unwrap()
                .0;
            check(&d, &store);
        }
        assert_eq!(hole(&d, new, &store.pages(), || None), None);
        assert_eq!(get(&d, Position::new(holes[2]), &store.pages()), Some(new));
    }

    /// `find` and `visit` start at `from` in every bucket form, including past a page boundary.
    #[test]
    fn searches_start_at_their_position() {
        let mut store = PageStore::create(VecMemory::new(), VecMemory::new()).unwrap();
        for total in [3, 10, PAGE_SLOTS * 2 + 7] {
            let mut d = Descriptor::EMPTY;
            for _ in 0..total {
                d = insert(d, VertexId::new(1), Spot::End, &mut store)
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
