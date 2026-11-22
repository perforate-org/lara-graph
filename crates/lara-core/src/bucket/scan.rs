//! Reading a bucket: ranges of positions in either order, searches and chosen positions.

use super::{byte_len, slot_offset, stored_u32};
use crate::descriptor::{Descriptor, PAGE_SLOTS, Paged, Slot, TINY_SLOTS, Tiny};
use crate::id::{Edge, Order, Position};
use crate::memory::Memory;
use crate::page_store::Pages;
use crate::paged::{self, Leaf};
use crate::units::ByteOffset;
use alloc::vec::Vec;
use core::ops::ControlFlow;

/// How much one read of a bucket covers.
#[derive(Clone, Copy)]
pub(crate) enum Window {
    /// A whole run or data page: for scans that read everything.
    Page,
    /// 128 slots: for searches that usually stop early.
    Short,
}

impl Window {
    const fn slots(self) -> u32 {
        match self {
            Self::Page => PAGE_SLOTS,
            Self::Short => 128,
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
    mut f: impl FnMut(Edge) -> ControlFlow<B>,
) -> Option<B> {
    scan(d, view, span, window.slots(), |first, bytes| {
        match span.order {
            Order::Ascending => emit_ascending(first, bytes, &mut f),
            Order::Descending => emit_descending(first, bytes, &mut f),
        }
    })
}

/// Calls `f` with the live edges among `bytes`, slots from position `first` on, lowest first.
fn emit_ascending<B>(
    first: u32,
    bytes: &[u8],
    f: &mut impl FnMut(Edge) -> ControlFlow<B>,
) -> ControlFlow<B> {
    let slots = bytes.as_chunks::<4>().0;
    slots
        .iter()
        .enumerate()
        .try_for_each(|(i, &raw)| match Slot::decode(raw) {
            Slot::Live(target) => f(Edge::new(Position::new(first + stored_u32(i)), target)),
            Slot::Tombstone => ControlFlow::Continue(()),
        })
}

/// [`emit_ascending`], highest first.
fn emit_descending<B>(
    first: u32,
    bytes: &[u8],
    f: &mut impl FnMut(Edge) -> ControlFlow<B>,
) -> ControlFlow<B> {
    let slots = bytes.as_chunks::<4>().0;
    for (i, &raw) in slots.iter().enumerate().rev() {
        if let Slot::Live(target) = Slot::decode(raw) {
            f(Edge::new(Position::new(first + stored_u32(i)), target))?;
        }
    }
    ControlFlow::Continue(())
}

/// Appends the bucket's live edges to `out`, in position order: one read for a run; for a
/// paged bucket, the index and each page that is not all tombstones.
pub(crate) fn read<M: Memory>(d: &Descriptor, view: &Pages<'_, M>, out: &mut Vec<Edge>) {
    out.reserve(usize::try_from(d.live()).unwrap_or(0));
    // Its own loop, like `find`: it compiles to faster code here than going through `visit`.
    scan(
        d,
        view,
        Span::all(d, Order::Ascending),
        PAGE_SLOTS,
        |first, bytes| {
            for (at, slot) in (first..).zip(Slot::decode_all(bytes)) {
                if let Slot::Live(target) = slot {
                    out.push(Edge::new(Position::new(at), target));
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
    mut wanted: impl FnMut(Edge) -> bool,
) -> Option<Edge> {
    // Its own loop rather than `visit`: the two compile to different code, and each is the
    // faster one for its use on the IC (`docs/ic-measurements-2026-11-22.md`).
    let span = Span::new(from.index(), d.stored(), Order::Ascending);
    scan(d, view, span, Window::Short.slots(), |first, bytes| {
        for (at, slot) in (first..).zip(Slot::decode_all(bytes)) {
            if let Slot::Live(target) = slot {
                let e = Edge::new(Position::new(at), target);
                if wanted(e) {
                    return ControlFlow::Break(e);
                }
            }
        }
        ControlFlow::Continue(())
    })
}

/// Feeds the bucket's stored slots in `span`, as bytes with the position of the first, to
/// `visit` until it breaks: the tiny slots, or reads of at most `window` slots of the run and of
/// each data page that is not all tombstones, in the span's order.
pub(super) fn scan<M: Memory, T>(
    d: &Descriptor,
    view: &Pages<'_, M>,
    span: Span,
    window: u32,
    mut visit: impl FnMut(u32, &[u8]) -> ControlFlow<T>,
) -> Option<T> {
    // One buffer serves every read: sized by the first, never zeroed again.
    let mut buf = Vec::new();
    let reads = Reads { view, span, window };
    let flow = match d {
        Descriptor::Tiny(tiny) => scan_tiny(tiny, span, &mut visit),
        Descriptor::Run(run) => reads.span(
            &mut buf,
            &mut visit,
            run.extent.offset(),
            0,
            u32::from(run.stored),
        ),
        Descriptor::Paged(paged) => reads.paged(paged, &mut buf, &mut visit),
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
    let mut bytes = [0u8; TINY_SLOTS * 4];
    let encoded = tiny.slots().iter().flat_map(|s| s.encode());
    let len = bytes
        .iter_mut()
        .zip(encoded)
        .map(|(dst, src)| *dst = src)
        .count();
    let low = byte_len(span.low).min(len);
    let high = byte_len(span.high).clamp(low, len);
    visit(span.low, bytes.get(low..high).unwrap_or(&[]))
}

/// Where one [`scan`] reads from.
struct Reads<'v, 'm, M: Memory> {
    view: &'v Pages<'m, M>,
    span: Span,
    window: u32,
}

impl<M: Memory> Reads<'_, '_, M> {
    /// The span's part of a paged bucket: its data pages in the span's order, each read only
    /// where the span covers it.
    fn paged<T>(
        &self,
        paged: &Paged,
        buf: &mut Vec<u8>,
        visit: &mut impl FnMut(u32, &[u8]) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        let mut page = |number: u32, leaf: Leaf| {
            let first = number * PAGE_SLOTS;
            let count = (paged.stored - first).min(PAGE_SLOTS);
            match leaf.page.filter(|_| u32::from(leaf.tombstones) < count) {
                Some(page) => self.span(buf, visit, page.offset(), first, count),
                None => ControlFlow::Continue(()),
            }
        };
        let pages_in = self.span.low / PAGE_SLOTS..self.span.high.div_ceil(PAGE_SLOTS);
        match self.span.order {
            Order::Ascending => paged::visit_leaves(self.view, paged, pages_in.start, |n, leaf| {
                if pages_in.contains(&n) {
                    page(n, leaf)
                } else {
                    ControlFlow::Continue(())
                }
            }),
            Order::Descending => {
                let leaves = paged::leaves(self.view, paged);
                (0u32..)
                    .zip(leaves)
                    .filter(|(n, _)| pages_in.contains(n))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .try_for_each(|(n, leaf)| page(n, leaf))
            }
        }
    }

    /// Feeds the part of the span among `count` slots at `at` (the first of them at position
    /// `first`), a window at a time in the span's order.
    fn span<T>(
        &self,
        buf: &mut Vec<u8>,
        visit: &mut impl FnMut(u32, &[u8]) -> ControlFlow<T>,
        at: ByteOffset,
        first: u32,
        count: u32,
    ) -> ControlFlow<T> {
        let mut lo = self.span.low.saturating_sub(first).min(count);
        let mut hi = self.span.high.saturating_sub(first).min(count);
        while lo < hi {
            let n = (hi - lo).min(self.window);
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
                fill(self.view, buf, slot_offset(at, start), n),
            )?;
        }
        ControlFlow::Continue(())
    }
}

/// Reads `count` slots at `at` into `buf`, growing it only when it is too short.
fn fill<'b, M: Memory>(
    view: &Pages<'_, M>,
    buf: &'b mut Vec<u8>,
    at: ByteOffset,
    count: u32,
) -> &'b [u8] {
    let len = byte_len(count);
    if buf.len() < len {
        buf.resize(len, 0);
    }
    let bytes = buf.get_mut(..len).unwrap_or(&mut []);
    view.read_into(at, bytes);
    bytes
}
