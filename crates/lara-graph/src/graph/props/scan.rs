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
/// `order`, until it breaks. Reads the values a value page at a time, and the slots of those
/// positions while that page's bytes are at hand, so a value is never copied twice.
pub(in crate::graph) fn visit_joined<M: Memory, B>(
    view: &Pages<'_, M>,
    bucket: &LabelBucket,
    width: Option<Width>,
    order: Order,
    f: &mut impl FnMut(Edge, &[u8]) -> ControlFlow<B>,
) -> Option<B> {
    let slots = &bucket.slots;
    let Some(width) = width else {
        return label_bucket::visit(slots, view, Span::all(slots, order), Window::Page, |e| {
            f(e, &[])
        });
    };
    let plane = bucket.values.map_or(values::Plane::Empty, |c| c.plane);
    let w = usize::from(width.get());
    let descending = order == Order::Descending;
    let mut visit_range = |(low, high): (u32, u32)| {
        values::visit(
            view,
            plane,
            width,
            (low, high, descending),
            |first, bytes| {
                let span = Span::new(first, first + stored_u32(bytes.len() / w), order);
                match label_bucket::visit(slots, view, span, Window::Page, |e| {
                    let at = usize::try_from(e.position().index() - first).unwrap_or(0) * w;
                    f(e, bytes.get(at..at + w).unwrap_or(&[]))
                }) {
                    Some(done) => ControlFlow::Break(done),
                    None => ControlFlow::Continue(()),
                }
            },
        )
    };
    // A paged bucket skips its all-dead slot pages' values too.
    if !matches!(slots, Descriptor::Paged(_)) {
        return visit_range((0, slots.stored()));
    }
    let mut ranges = live_ranges(view, slots);
    if descending {
        ranges.reverse();
    }
    ranges.into_iter().find_map(visit_range)
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
