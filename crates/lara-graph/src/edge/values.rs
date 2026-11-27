//! A bucket's values in the edge store: the label's width, a bucket made ready for values, reading and
//! writing one, and scans that join values with edges.

use crate::bucket::descriptor::{self, Beside, Descriptor, PAGE_SLOTS};
use crate::bucket::paged;
use crate::bucket::reserved;
use crate::bucket::slots::{self, Span, Window, stored_u32};
use crate::bucket::values::{self, Plane};
use crate::bucket::{LabelBucket, Migration, Schema, Values};
use crate::edge::{EdgeStore, EdgeView};
use crate::id::{EdgeLabel, Neighbor, Position, Width};
use crate::input::Order;
use crate::store::memory::Memory;
use crate::store::page_store::Pages;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

impl<M: Memory> EdgeStore<M> {
    /// `bucket` with `value` at `position`, given a value plane if it has none. Within reserved
    /// memory.
    pub(crate) fn put_value(
        &mut self,
        bucket: LabelBucket,
        (schema, position): (Schema, Position),
        value: &[u8],
    ) -> LabelBucket {
        let width = schema.width;
        let mut bucket = if bucket.cells().is_none() {
            self.with_values(bucket, schema)
        } else {
            bucket
        };
        if let Some(cells) = bucket.cells() {
            let plane = reserved(values::write(
                &mut self.store,
                cells.plane,
                width,
                position.index(),
                value,
            ));
            bucket.set_plane(plane);
        }
        bucket
    }

    /// `bucket`, which has no values, ready for values of `width`, all zero: a small bucket as a
    /// run with them beside its slots, others (never tiny) with an empty value plane, which
    /// `Graph::publish` gives a header. Within reserved memory.
    pub(super) fn with_values(&mut self, bucket: LabelBucket, schema: Schema) -> LabelBucket {
        let width = schema.width;
        let small = match bucket.slots {
            Descriptor::Tiny(_) => true,
            Descriptor::Run(run) => u32::from(run.stored) < descriptor::max_beside(width),
            Descriptor::Paged(_) => false,
        };
        if small && descriptor::beside(width) && bucket.header.is_none() {
            let slots = reserved(slots::with_values_beside(
                bucket.slots,
                Beside::new(width, schema.version),
                &mut self.store,
            ));
            return LabelBucket { slots, ..bucket };
        }
        let slots = reserved(slots::without_tiny(bucket.slots, &mut self.store));
        LabelBucket {
            slots,
            values: Some(Values {
                plane: Plane::Empty,
                width,
                version: schema.version,
                migration: Migration::Idle,
            }),
            ..bucket
        }
    }
}

impl<M: Memory> EdgeView<'_, M> {
    /// The value at `position` of `bucket` (a bucket of `label`), whether or not an edge is live
    /// there: its cell, zeros before any write, empty for a label without a width.
    pub(crate) fn read_value(
        &self,
        bucket: &LabelBucket,
        label: EdgeLabel,
        position: Position,
    ) -> Vec<u8> {
        value_at(&self.pages(), self.widths.get(label), bucket, position)
    }

    /// Appends the value at each of `positions` of `bucket` (a bucket of `label`) to `out`, as
    /// [`Self::read_value`] gives each.
    pub(crate) fn read_values(
        &self,
        bucket: &LabelBucket,
        label: EdgeLabel,
        positions: &[Position],
        out: &mut Vec<u8>,
    ) {
        values_into(
            &self.pages(),
            self.widths.get(label),
            bucket,
            positions,
            out,
        );
    }

    /// Calls `f` with the value of every position of `bucket` in `order`, until it breaks.
    pub(crate) fn visit_values_in<B>(
        &self,
        bucket: &LabelBucket,
        label: EdgeLabel,
        order: Order,
        f: &mut impl FnMut(Position, &[u8]) -> ControlFlow<B>,
    ) -> Option<B> {
        // A bucket's stored values keep the width they were written with until migrated.
        let width = bucket
            .cells()
            .map(|c| c.width)
            .or_else(|| self.widths.get(label))?;
        let view = self.pages();
        let w = usize::from(width.get());
        let mut ranges = live_ranges(&view, &bucket.slots);
        if order == Order::Descending {
            ranges.reverse();
        }
        let plane = bucket.cells().map_or(Plane::Empty, |c| c.plane);
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
}

/// Calls `f` with each live edge of `bucket` (a bucket of a label of `width`) and its value, in
/// `order`, until it breaks. Reads the values a value page at a time, and the slots of those
/// positions while that page's bytes are at hand, so a value is never copied twice.
pub(crate) fn visit_joined<M: Memory, B>(
    view: &Pages<'_, M>,
    bucket: &LabelBucket,
    width: Option<Width>,
    order: Order,
    f: &mut impl FnMut(Neighbor, &[u8]) -> ControlFlow<B>,
) -> Option<B> {
    let slots = &bucket.slots;
    if let Descriptor::Run(run) = slots
        && let Some(beside) = run.values
    {
        return slots::visit_beside(run, beside.width, view, order, f);
    }
    // Values in a plane keep the width they were written with until migrated.
    let cells = bucket.values;
    let Some(width) = cells.map(|c| c.width).or(width) else {
        return slots::visit(slots, view, Span::all(slots, order), Window::Page, |e| {
            f(e, &[])
        });
    };
    let plane = cells.map_or(Plane::Empty, |c| c.plane);
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
                match slots::visit(slots, view, span, Window::Page, |e| {
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

/// Appends the value at each of `positions` of `bucket`, a bucket of a label of width `width`, to
/// `out`, as [`value_at`] gives each. A run of consecutive ascending positions is read a value
/// page at a time; one value at a time, with a buffer each, a batch of a thousand cost about
/// 1.1K instructions a value (Gleaph's `timer_maintenance_single_step_indexed_hub_2048`).
pub(super) fn values_into<M: Memory>(
    pages: &Pages<'_, M>,
    width: Option<Width>,
    bucket: &LabelBucket,
    positions: &[Position],
    out: &mut Vec<u8>,
) {
    let Some(cells) = bucket.cells() else {
        let len = positions.len() * width.map_or(0, |w| usize::from(w.get()));
        out.resize(out.len() + len, 0);
        return;
    };
    let w = usize::from(cells.width.get());
    let mut positions = positions.iter().map(|p| p.index()).peekable();
    while let Some(start) = positions.next() {
        let mut end = start.saturating_add(1);
        while positions.next_if_eq(&end).is_some() {
            end = end.saturating_add(1);
        }
        if end - start <= 1 {
            let at = out.len();
            out.resize(at + w, 0);
            if let Some(cell) = out.get_mut(at..) {
                values::read(pages, cells.plane, cells.width, start, cell);
            }
        } else {
            let _: Option<()> = values::visit(
                pages,
                cells.plane,
                cells.width,
                (start, end, false),
                |_, bytes| {
                    out.extend_from_slice(bytes);
                    ControlFlow::Continue(())
                },
            );
        }
    }
}

/// The value at `position` of `bucket`, a bucket of a label of width `width`: its cell, zeros
/// before any write, empty for a label without a width.
pub(super) fn value_at<M: Memory>(
    pages: &Pages<'_, M>,
    width: Option<Width>,
    bucket: &LabelBucket,
    position: Position,
) -> Vec<u8> {
    let Some(cells) = bucket.cells() else {
        return vec![0; width.map_or(0, |w| usize::from(w.get()))];
    };
    let mut out = vec![0; usize::from(cells.width.get())];
    values::read(pages, cells.plane, cells.width, position.index(), &mut out);
    out
}
