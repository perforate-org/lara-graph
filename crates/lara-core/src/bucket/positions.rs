//! Reading chosen positions of a bucket.

use super::scan::{Span, scan};
use super::stored_u32;
use crate::descriptor::{Descriptor, PAGE_SLOTS, Slot};
use crate::id::{Order, Position, VertexId};
use crate::memory::Memory;
use crate::page_store::Pages;
use core::ops::ControlFlow;

/// The targets at `positions` (ascending, distinct), `None` for a hole or a position past the
/// end, passed to `f` in that order. Reads each touched run or page once, from its first wanted
/// position to its last.
pub(crate) fn read_positions<M: Memory>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    positions: &[Position],
    mut f: impl FnMut(Position, Option<VertexId>),
) {
    let mut rest = positions;
    while let Some(&head) = rest.first() {
        // The positions sharing `head`'s page (a tiny bucket or a run is one page).
        let page = head.index() / PAGE_SLOTS;
        let same = rest
            .iter()
            .take_while(|p| p.index() / PAGE_SLOTS == page)
            .count();
        let (group, tail) = rest.split_at(same);
        rest = tail;
        read_group(d, view, group, &mut f);
    }
}

/// Wanted positions this sparse or sparser are read one slot at a time instead of as a range.
const SPARSE: u32 = 16;

/// [`read_positions`] for positions within one page: one read of the range they span, or a
/// point read each when they are sparse in it.
fn read_group<M: Memory>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    group: &[Position],
    f: &mut impl FnMut(Position, Option<VertexId>),
) {
    let low = group.first().map_or(0, |p| p.index());
    let high = group.last().map_or(low, |p| p.index() + 1).min(d.stored());
    if high.saturating_sub(low) > SPARSE * stored_u32(group.len()) {
        for &p in group {
            f(p, super::get(d, p, view));
        }
    } else {
        read_range(d, view, (group, Span::new(low, high, Order::Ascending)), f);
    }
}

/// Reports each of `group`'s positions from one read of `span`, which covers them.
fn read_range<M: Memory>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    (group, span): (&[Position], Span),
    f: &mut impl FnMut(Position, Option<VertexId>),
) {
    let mut wanted = group.iter().copied().peekable();
    // Reports every wanted position up to `at`: the slot's target at `at`, `None` before it.
    let mut reach = |at: Position, target: Option<VertexId>| {
        while let Some(p) = wanted.next_if(|p| *p <= at) {
            f(p, if p == at { target } else { None });
        }
    };
    scan(d, view, span, (PAGE_SLOTS, PAGE_SLOTS), |first, bytes| {
        for (at, slot) in (first..).zip(Slot::decode_all(bytes)) {
            let target = match slot {
                Slot::Live(target) => Some(target),
                Slot::Tombstone => None,
            };
            reach(Position::new(at), target);
        }
        ControlFlow::<()>::Continue(())
    });
    // Positions in a hollow or dead page, or past the end, were never reached.
    reach(Position::new(u32::MAX), None);
}
