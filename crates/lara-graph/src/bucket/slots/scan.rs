//! Reading a bucket: ranges of positions in either order, searches and chosen positions.

use super::{byte_len, slot_offset, stored_u32};
use crate::bucket::descriptor::{
    Descriptor, PAGE_SLOTS, Paged, Run, SLOT_BYTES, Slot, TINY_SLOTS, Tiny,
};
use crate::bucket::paged::{self, Leaf};
use crate::id::{Neighbor, Position, Width};
use crate::input::Order;
use crate::store::memory::Memory;
use crate::store::page_store::Pages;
use crate::store::read_buf::ReadBuf;
use crate::store::units::ByteOffset;
use alloc::vec::Vec;
use core::ops::ControlFlow;

/// How much one read of a bucket covers.
#[derive(Clone, Copy)]
pub(crate) enum Window {
    /// A whole run or data page: for scans that read everything.
    Page,
    /// 32 slots, then 128 at a time: for searches that usually stop early, often at once.
    Short,
}

impl Window {
    /// Slots of the first read, and of each one after it.
    const fn slots(self) -> (u32, u32) {
        match self {
            Self::Page => (PAGE_SLOTS, PAGE_SLOTS),
            Self::Short => (32, 128),
        }
    }
}

/// Positions `low..high` of a bucket, read in `order`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    low: u32,
    high: u32,
    order: Order,
}

impl Span {
    /// Positions `low..high` (empty when `high <= low`).
    pub(crate) const fn new(low: u32, high: u32, order: Order) -> Self {
        Self { low, high, order }
    }

    /// Every stored position of `d`.
    pub(crate) fn all(d: &Descriptor, order: Order) -> Self {
        Self::new(0, d.stored(), order)
    }
}

/// Calls `f` with each live edge in `span`, in its order, until it breaks. Reads the descriptor
/// and page data only, skipping pages that are all tombstones.
pub(crate) fn visit<M: Memory, B>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    span: Span,
    window: Window,
    mut f: impl FnMut(Neighbor) -> ControlFlow<B>,
) -> Option<B> {
    // A tiny bucket's slots go to `f` as they are, not through bytes as `scan` gives them: the
    // round trip cost about a fifth of visiting a one-edge bucket on the IC.
    if let Descriptor::Tiny(tiny) = d {
        return visit_tiny(tiny, span, &mut f).break_value();
    }
    scan(d, view, span, window.slots(), |first, bytes| {
        match span.order {
            Order::Ascending => emit_ascending(first, bytes, &mut f),
            Order::Descending => emit_descending(first, bytes, &mut f),
        }
    })
}

/// Calls `f` with each live edge of `run`, which keeps values of `width` beside its slots, and
/// its value, in `order`, until it breaks: one read covers the slots and the cells.
pub(crate) fn visit_beside<M: Memory, B>(
    run: &Run,
    width: Width,
    view: &Pages<'_, M>,
    order: Order,
    mut f: impl FnMut(Neighbor, &[u8]) -> ControlFlow<B>,
) -> Option<B> {
    let stored = usize::from(run.stored);
    let w = usize::from(width.get());
    let cells_at = usize::try_from(run.capacity() * SLOT_BYTES).unwrap_or(usize::MAX);
    let mut buf = ReadBuf::new();
    let bytes = buf.fill(view, run.extent.offset(), cells_at + stored * w);
    let (slots, rest) = bytes.split_at(cells_at.min(bytes.len()));
    let slots = slots.as_chunks::<4>().0.iter().take(stored);
    let mut emit = |(i, &raw): (usize, &[u8; 4])| match Slot::decode(raw) {
        Slot::Live(target) => {
            let at = i * w;
            f(
                Neighbor::new(Position::new(stored_u32(i)), target),
                rest.get(at..at + w).unwrap_or(&[]),
            )
        }
        Slot::Tombstone => ControlFlow::Continue(()),
    };
    match order {
        Order::Ascending => slots.enumerate().try_for_each(&mut emit),
        Order::Descending => slots.enumerate().rev().try_for_each(&mut emit),
    }
    .break_value()
}

/// Calls `f` with the live edges of a tiny bucket in `span`, in its order.
fn visit_tiny<B>(
    tiny: &Tiny,
    span: Span,
    f: &mut impl FnMut(Neighbor) -> ControlFlow<B>,
) -> ControlFlow<B> {
    let slots = tiny.slots();
    let high = usize::try_from(span.high).map_or(slots.len(), |h| h.min(slots.len()));
    let low = usize::try_from(span.low).map_or(high, |l| l.min(high));
    let mut edges = slots
        .get(low..high)
        .unwrap_or(&[])
        .iter()
        .enumerate()
        .filter_map(|(i, slot)| match *slot {
            Slot::Live(target) => Some(Neighbor::new(Position::new(stored_u32(low + i)), target)),
            Slot::Tombstone => None,
        });
    match span.order {
        Order::Ascending => edges.try_for_each(f),
        Order::Descending => edges.rev().try_for_each(f),
    }
}

/// Calls `f` with the live edges among `bytes`, slots from position `first` on, lowest first.
fn emit_ascending<B>(
    first: u32,
    bytes: &[u8],
    f: &mut impl FnMut(Neighbor) -> ControlFlow<B>,
) -> ControlFlow<B> {
    let slots = bytes.as_chunks::<4>().0;
    slots
        .iter()
        .enumerate()
        .try_for_each(|(i, &raw)| match Slot::decode(raw) {
            Slot::Live(target) => f(Neighbor::new(Position::new(first + stored_u32(i)), target)),
            Slot::Tombstone => ControlFlow::Continue(()),
        })
}

/// [`emit_ascending`], highest first.
fn emit_descending<B>(
    first: u32,
    bytes: &[u8],
    f: &mut impl FnMut(Neighbor) -> ControlFlow<B>,
) -> ControlFlow<B> {
    let slots = bytes.as_chunks::<4>().0;
    for (i, &raw) in slots.iter().enumerate().rev() {
        if let Slot::Live(target) = Slot::decode(raw) {
            f(Neighbor::new(Position::new(first + stored_u32(i)), target))?;
        }
    }
    ControlFlow::Continue(())
}

/// Appends the bucket's live edges to `out`, in position order: one read for a run; for a
/// paged bucket, the index and each page that is not all tombstones.
pub(crate) fn read<M: Memory>(d: &Descriptor, view: &Pages<'_, M>, out: &mut Vec<Neighbor>) {
    out.reserve(usize::try_from(d.live()).unwrap_or(0));
    // Its own loop, like `find`: it compiles to faster code here than going through `visit`.
    scan(
        d,
        view,
        Span::all(d, Order::Ascending),
        Window::Page.slots(),
        |first, bytes| {
            for (at, slot) in (first..).zip(Slot::decode_all(bytes)) {
                if let Slot::Live(target) = slot {
                    out.push(Neighbor::new(Position::new(at), target));
                }
            }
            ControlFlow::<()>::Continue(())
        },
    );
}

/// The first live edge at `from` or later that `wanted` accepts; `wanted` sees each live edge
/// in position order until then. Reads from `from` up to it, a short window at a time.
pub(crate) fn find<M: Memory>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    from: Position,
    wanted: impl FnMut(Neighbor) -> bool,
) -> Option<Neighbor> {
    find_in(d, view, (from.index(), d.stored()), wanted)
}

/// [`find`] among positions `low..high`.
pub(crate) fn find_in<M: Memory>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    (low, high): (u32, u32),
    mut wanted: impl FnMut(Neighbor) -> bool,
) -> Option<Neighbor> {
    // Its own loop rather than `visit`: the two compile to different code, and each is the
    // faster one for its use on the IC (`docs/ic-measurements-2026-11-22.md`).
    let span = Span::new(low, high.min(d.stored()), Order::Ascending);
    scan(d, view, span, Window::Short.slots(), |first, bytes| {
        for (at, slot) in (first..).zip(Slot::decode_all(bytes)) {
            if let Slot::Live(target) = slot {
                let e = Neighbor::new(Position::new(at), target);
                if wanted(e) {
                    return ControlFlow::Break(e);
                }
            }
        }
        ControlFlow::Continue(())
    })
}

/// Feeds the bucket's stored slots in `span`, as bytes with the position of the first, to
/// `visit` until it breaks: the tiny slots, or reads of the run and of each data page that is
/// not all tombstones, in the span's order, `window` slots at a time (the first read, then each
/// one after it).
pub(super) fn scan<M: Memory, T>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    span: Span,
    window: (u32, u32),
    mut visit: impl FnMut(u32, &[u8]) -> ControlFlow<T>,
) -> Option<T> {
    let reads = Reads {
        view,
        span,
        window: core::cell::Cell::new(window.0),
        then: window.1,
    };
    let flow = match d {
        Descriptor::Tiny(tiny) => scan_tiny(tiny, span, &mut visit),
        Descriptor::Run(run) => reads.span(
            &mut ReadBuf::new(),
            &mut visit,
            run.extent.offset(),
            0,
            u32::from(run.stored),
        ),
        Descriptor::Paged(paged) => reads.paged(paged, &mut ReadBuf::new(), &mut visit),
    };
    match flow {
        ControlFlow::Break(found) => Some(found),
        ControlFlow::Continue(()) => None,
    }
}

fn scan_tiny<T>(
    tiny: &Tiny,
    span: Span,
    visit: &mut impl FnMut(u32, &[u8]) -> ControlFlow<T>,
) -> ControlFlow<T> {
    // Slot by slot into fixed chunks, not through an iterator chain over bytes (slow on the IC).
    let mut bytes = [0u8; TINY_SLOTS * 4];
    let slots = tiny.slots();
    for (dst, slot) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(slots) {
        *dst = slot.encode();
    }
    let len = slots.len() * 4;
    let low = byte_len(span.low).min(len);
    let high = byte_len(span.high).clamp(low, len);
    visit(span.low, bytes.get(low..high).unwrap_or(&[]))
}

/// Where one [`scan`] reads from.
struct Reads<'v, 'm, M: Memory> {
    view: &'v Pages<'m, M>,
    span: Span,
    /// Slots of the next read.
    window: core::cell::Cell<u32>,
    /// Slots of each read after the first.
    then: u32,
}

impl<M: Memory> Reads<'_, '_, M> {
    /// The span's part of a paged bucket: its data pages in the span's order, each read only
    /// where the span covers it.
    fn paged<T>(
        &self,
        paged: &Paged,
        buf: &mut ReadBuf,
        visit: &mut impl FnMut(u32, &[u8]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        let page = |number: u32, leaf: Leaf| {
            let first = number * PAGE_SLOTS;
            let count = (paged.stored - first).min(PAGE_SLOTS);
            match leaf.page.filter(|_| u32::from(leaf.tombstones) < count) {
                Some(page) => self.span(buf, visit, page.offset(), first, count),
                None => ControlFlow::Continue(()),
            }
        };
        let pages_in = self.span.low / PAGE_SLOTS..self.span.high.div_ceil(PAGE_SLOTS);
        paged::visit_leaves(self.view, paged, (pages_in, self.span.order), page)
    }

    /// Feeds the part of the span among `count` slots at `at` (the first of them at position
    /// `first`), a window at a time in the span's order.
    fn span<T>(
        &self,
        buf: &mut ReadBuf,
        visit: &mut impl FnMut(u32, &[u8]) -> ControlFlow<T>,
        at: ByteOffset,
        first: u32,
        count: u32,
    ) -> ControlFlow<T> {
        let mut lo = self.span.low.saturating_sub(first).min(count);
        let mut hi = self.span.high.saturating_sub(first).min(count);
        while lo < hi {
            let n = (hi - lo).min(self.window.replace(self.then));
            let start = match self.span.order {
                Order::Ascending => {
                    lo += n;
                    lo - n
                }
                Order::Descending => {
                    hi -= n;
                    hi
                }
            };
            visit(
                first + start,
                buf.fill(self.view, slot_offset(at, start), byte_len(n)),
            )?;
        }
        ControlFlow::Continue(())
    }
}
