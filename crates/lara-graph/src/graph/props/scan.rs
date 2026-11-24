//! Values-first scans and combined scans of edges with their values.

use super::super::{Error, Graph};
use crate::descriptor::{Descriptor, PAGE_SLOTS};
use crate::id::{Direction, Edge, Label, NodeId, Order, Position, Width};
use crate::label_bucket::{self, Span, Window, stored_u32};
use crate::label_bucket_list::LabelBucketKey;
use crate::memory::Memory;
use crate::page_store::Pages;
use crate::paged;
use crate::stored::LabelBucket;
use crate::values;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

impl<M: Memory> Graph<M> {
    /// Calls `f` with the value of every position of `v`'s `(label, direction)` bucket in `order`,
    /// until it breaks: the first phase of a values-first traversal. Reads the value plane only,
    /// skipping pages the slot index marks all dead; other removed positions come with their last
    /// value, so the second phase ([`Self::targets_at`]) drops them.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn visit_values<B>(
        &self,
        v: NodeId,
        (label, direction): (Label, Direction),
        order: Order,
        mut f: impl FnMut(Position, &[u8]) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let Some(found) = self.find_label_bucket(v, LabelBucketKey { label, direction })? else {
            return Ok(None);
        };
        Ok(self.visit_values_in(&found.bucket, label, order, &mut f))
    }

    /// Calls `f` with the value of every position of `bucket` in `order`, until it breaks.
    pub(in crate::graph) fn visit_values_in<B>(
        &self,
        bucket: &LabelBucket,
        label: Label,
        order: Order,
        f: &mut impl FnMut(Position, &[u8]) -> ControlFlow<B>,
    ) -> Option<B> {
        let width = self.widths.get(label)?;
        let view = self.edges.pages();
        let w = usize::from(width.get());
        let mut ranges = live_ranges(&view, &bucket.slots);
        if order == Order::Descending {
            ranges.reverse();
        }
        let plane = bucket.values.map_or(values::Plane::Empty, |c| c.plane);
        let descending = order == Order::Descending;
        for (low, high) in ranges {
            let flow = values::visit(
                &view,
                plane,
                width,
                (low, high, descending),
                |first, bytes| {
                    let cells = bytes.chunks_exact(w).enumerate();
                    let mut emit =
                        |(i, cell): (usize, &[u8])| f(Position::new(first + stored_u32(i)), cell);
                    if descending {
                        cells.rev().try_for_each(&mut emit)
                    } else {
                        cells.into_iter().try_for_each(&mut emit)
                    }
                },
            );
            if flow.is_some() {
                return flow;
            }
        }
        None
    }

    /// Calls `f` with each live edge of `v`'s `(label, direction)` bucket and its value, in
    /// `order`, until it breaks.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn visit_neighbors_with_values<B>(
        &self,
        v: NodeId,
        (label, direction): (Label, Direction),
        order: Order,
        mut f: impl FnMut(Edge, &[u8]) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let Some(found) = self.find_label_bucket(v, LabelBucketKey { label, direction })? else {
            return Ok(None);
        };
        let width = self.widths.get(label);
        Ok(visit_joined(
            &self.edges.pages(),
            &found.bucket,
            width,
            order,
            &mut f,
        ))
    }
}

/// Calls `f` with each live edge of `bucket` (a bucket of a label of `width`) and its value, in
/// `order`, until it breaks.
pub(in crate::graph) fn visit_joined<M: Memory, B>(
    view: &Pages<'_, M>,
    bucket: &LabelBucket,
    width: Option<Width>,
    order: Order,
    f: &mut impl FnMut(Edge, &[u8]) -> ControlFlow<B>,
) -> Option<B> {
    let slots = bucket.slots;
    let plane = bucket.values.map_or(values::Plane::Empty, |c| c.plane);
    // A slot page's worth of positions at a time: its live edges, then its values in one read.
    let mut windows: Vec<u32> = (0..slots.stored().div_ceil(PAGE_SLOTS)).collect();
    if order == Order::Descending {
        windows.reverse();
    }
    let mut join = Join {
        view,
        slots: &slots,
        plane,
        width,
        edges: Vec::new(),
        cells: Vec::new(),
    };
    for k in windows {
        let span = (k * PAGE_SLOTS, ((k + 1) * PAGE_SLOTS).min(slots.stored()));
        if let ControlFlow::Break(done) = join.window(span, order, f) {
            return Some(done);
        }
    }
    None
}

/// A combined scan's state: a window's live edges, then its values in one read.
struct Join<'a, 'v, M: Memory> {
    view: &'a Pages<'v, M>,
    slots: &'a Descriptor,
    plane: values::Plane,
    width: Option<Width>,
    edges: Vec<Edge>,
    cells: Vec<u8>,
}

impl<M: Memory> Join<'_, '_, M> {
    /// Calls `f` with each live edge of positions `low..high` and its value.
    fn window<B>(
        &mut self,
        (low, high): (u32, u32),
        order: Order,
        f: &mut impl FnMut(Edge, &[u8]) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        self.edges.clear();
        let span = Span::new(low, high, order);
        label_bucket::visit(self.slots, self.view, span, Window::Page, |e| {
            self.edges.push(e);
            ControlFlow::<()>::Continue(())
        });
        if self.edges.is_empty() {
            return ControlFlow::Continue(());
        }
        self.cells.clear();
        let w = self.width.map_or(0, |w| usize::from(w.get()));
        if let Some(width) = self.width {
            let cells = &mut self.cells;
            values::visit(
                self.view,
                self.plane,
                width,
                (low, high, false),
                |_, bytes| {
                    cells.extend_from_slice(bytes);
                    ControlFlow::<()>::Continue(())
                },
            );
        }
        for &e in &self.edges {
            let at = usize::try_from(e.position().index() - low).unwrap_or(0) * w;
            f(e, self.cells.get(at..at + w).unwrap_or(&[]))?;
        }
        ControlFlow::Continue(())
    }
}

/// The position ranges of a bucket's slot pages that are not all dead: the whole bucket unless
/// it is paged.
fn live_ranges<M: Memory>(view: &Pages<'_, M>, slots: &Descriptor) -> Vec<(u32, u32)> {
    let Descriptor::Paged(paged) = slots else {
        return vec![(0, slots.stored())];
    };
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for (number, leaf) in (0u32..).zip(paged::leaves(view, paged)) {
        let first = number * PAGE_SLOTS;
        let count = (paged.stored - first).min(PAGE_SLOTS);
        if leaf.page.is_some() && u32::from(leaf.tombstones) < count {
            match ranges.last_mut() {
                Some(last) if last.1 == first => last.1 = first + count,
                Some(_) | None => ranges.push((first, first + count)),
            }
        }
    }
    ranges
}
