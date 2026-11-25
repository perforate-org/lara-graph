//! A bucket's slots: insert, remove and read one label bucket's edges by position.
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

use crate::bucket::descriptor::{
    Descriptor, PAGE_SLOTS, Paged, Run, SLOT_BYTES, Slot, TOMBSTONE, run_class,
};
use crate::bucket::paged;
use crate::bucket::values::{self, Plane};
use crate::id::{NodeId, Position, Width};
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::{PageId, PageStore, Pages};
use crate::store::units::ByteOffset;
use alloc::vec::Vec;

mod grow;
mod positions;
mod scan;

use grow::{cells_len, paged_append, run_append, tiny_to_run};
pub(crate) use positions::read_positions;
pub(crate) use scan::{Span, Window, find, find_in, nth_live, read, visit, visit_beside};

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
    target: NodeId,
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
    target: NodeId,
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
    paged_hole_from(view, paged, after.map_or(0, |p| p.index() + 1))
}

/// The first hole in a paged bucket at or after `start`.
pub(crate) fn hole_from<M: Memory>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    start: Position,
) -> Option<Position> {
    if d.live() == d.stored() {
        return None;
    }
    match d {
        Descriptor::Paged(paged) => paged_hole_from(view, paged, start.index()),
        Descriptor::Tiny(_) | Descriptor::Run(_) => None,
    }
}

fn paged_hole_from<M: Memory>(view: &Pages<'_, M>, paged: &Paged, start: u32) -> Option<Position> {
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
) -> Option<(Descriptor, NodeId)> {
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
pub(crate) fn get<M: Memory>(d: &Descriptor, at: Position, view: &Pages<'_, M>) -> Option<NodeId> {
    if at.index() >= d.stored() {
        return None;
    }
    match d {
        Descriptor::Tiny(tiny) => match *tiny.slots().get(usize::try_from(at.index()).ok()?)? {
            Slot::Live(target) => Some(target),
            Slot::Tombstone => None,
        },
        Descriptor::Run(run) => target_at(view, slot_offset(run.extent.offset(), at.index())),
        Descriptor::Paged(paged) => {
            let page = paged::data_page(view, paged, at.index() / PAGE_SLOTS).page?;
            target_at(view, slot_offset(page.offset(), at.index() % PAGE_SLOTS))
        }
    }
}

/// The live target in the stored slot at `at`, if any.
pub(crate) fn target_at<M: Memory>(view: &Pages<'_, M>, at: ByteOffset) -> Option<NodeId> {
    match Slot::decode(view.read_array(at)) {
        Slot::Live(target) => Some(target),
        Slot::Tombstone => None,
    }
}

/// The same slots, never tiny: a tiny bucket becomes a run of the smallest class (a bucket with
/// values in a plane needs one); a run or paged bucket is unchanged.
pub(crate) fn without_tiny<M: Memory>(
    d: Descriptor,
    store: &mut PageStore<M>,
) -> Result<Descriptor, WriteError> {
    let Descriptor::Tiny(tiny) = d else {
        return Ok(d);
    };
    let stored = stored_u32(tiny.slots().len());
    let extent = store.allocate(run_class(stored.max(1), None))?;
    store.write(extent.offset(), &tiny_bytes(tiny.slots()));
    Ok(Descriptor::Run(Run {
        extent,
        stored: narrow(stored),
        live: narrow(d.live()),
        values: None,
    }))
}

/// A tiny bucket or a run without values as a run with values of `width` beside its slots, all
/// zero, in the smallest class that holds one more slot. Within one page of copying.
pub(crate) fn with_values_beside<M: Memory>(
    d: Descriptor,
    width: Width,
    store: &mut PageStore<M>,
) -> Result<Descriptor, WriteError> {
    let (slots, old) = match d {
        Descriptor::Tiny(tiny) => (tiny_bytes(tiny.slots()), None),
        Descriptor::Run(run) if run.values.is_none() => (
            store
                .pages()
                .read(run.extent.offset(), byte_len(u32::from(run.stored))),
            Some(run.extent),
        ),
        Descriptor::Run(_) | Descriptor::Paged(_) => corrupt("values beside a run or pages"),
    };
    let stored = d.stored();
    let extent = store.allocate(run_class(stored + 1, Some(width)))?;
    let run = Run {
        extent,
        stored: narrow(stored),
        live: narrow(d.live()),
        values: Some(width),
    };
    store.write(extent.offset(), &slots);
    store.write(
        run.cells_offset(),
        &alloc::vec![0u8; cells_len(run.capacity(), width)],
    );
    if let Some(old) = old {
        store.release(old);
    }
    Ok(Descriptor::Run(run))
}

/// Whether appending to `d` needs its values moved out first: a page-sized run with values
/// beside its slots is full. Then [`values_out`] makes room.
pub(crate) fn full_beside(d: &Descriptor) -> bool {
    matches!(d, Descriptor::Run(run)
        if run.values.is_some() && u32::from(run.stored) >= run.capacity() && !run.can_grow())
}

/// A run with values beside its slots as a run of its slots alone, with room for one more, and
/// a value plane of its cells. Copies at most a page of each.
pub(crate) fn values_out<M: Memory>(
    run: Run,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Plane), WriteError> {
    let width = run
        .values
        .unwrap_or_else(|| corrupt("a run without values beside its slots"));
    let stored = u32::from(run.stored);
    let view = store.pages();
    let slots = view.read(run.extent.offset(), byte_len(stored));
    let cells = view.read(run.cells_offset(), cells_len(stored, width));
    let extent = store.allocate(run_class(stored + 1, None))?;
    let plane = match values::run_of(store, &cells) {
        Ok(plane) => plane,
        Err(error) => {
            store.release(extent);
            return Err(error);
        }
    };
    store.write(extent.offset(), &slots);
    store.release(run.extent);
    Ok((
        Descriptor::Run(Run {
            extent,
            values: None,
            ..run
        }),
        plane,
    ))
}

/// A tiny bucket's slots as stored bytes, slot by slot rather than through an iterator chain
/// over bytes, which compiles to slow code on the IC.
fn tiny_bytes(slots: &[Slot]) -> Vec<u8> {
    let mut bytes = alloc::vec![0u8; slots.len() * 4];
    for (dst, slot) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(slots) {
        *dst = slot.encode();
    }
    bytes
}

// --- helpers -----------------------------------------------------------------------------------

/// The first hole after the last live edge to `target`: filling it keeps equal-target edges in
/// insertion order. `None` when there is no such hole.
fn reusable_hole(slots: &[Slot], target: NodeId) -> Option<usize> {
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
) -> Vec<crate::store::page_store::Extent> {
    match d {
        Descriptor::Tiny(_) => Vec::new(),
        Descriptor::Run(run) => alloc::vec![run.extent],
        Descriptor::Paged(paged) => {
            let whole = crate::store::page_store::Extent::whole;
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
    use crate::id::Neighbor;
    use crate::input::Order;
    use crate::store::memory::VecMemory;
    use core::ops::ControlFlow;

    /// Short windows read a span in either order, across pages and with holes.
    #[test]
    fn short_windows_read_spans_both_ways() {
        let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
        let mut d = Descriptor::EMPTY;
        let total = PAGE_SLOTS + 300;
        for i in 0..total {
            d = insert(d, NodeId::new(i), Spot::End, &mut store).unwrap().0;
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
        let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
        let mut d = Descriptor::EMPTY;
        let page_count = paged::FANOUT + 3;
        let total = page_count * PAGE_SLOTS - 5;
        for i in 0..total {
            d = insert(d, NodeId::new(i), Spot::End, &mut store).unwrap().0;
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
        let new = NodeId::new(u32::MAX - 1);
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
        let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
        for total in [3, 10, PAGE_SLOTS * 2 + 7] {
            let mut d = Descriptor::EMPTY;
            for _ in 0..total {
                d = insert(d, NodeId::new(1), Spot::End, &mut store).unwrap().0;
            }
            let view = store.pages();
            for from in [0, 1, total / 2, PAGE_SLOTS + 1, total - 1, total] {
                let found = find(&d, &view, Position::new(from), |_| true);
                let expected =
                    (from < total).then(|| Neighbor::new(Position::new(from), NodeId::new(1)));
                assert_eq!(found, expected, "{total} edges, from {from}");
            }
        }
    }
}
