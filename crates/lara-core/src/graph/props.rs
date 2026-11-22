//! Edge values: writing, point and batch reads, values-first scans and combined scans.

use super::pair::{Found, key_of};
use super::{Error, Graph, reserved};
use crate::bucket::{self, Span, Window, stored_u32};
use crate::descriptor::{Descriptor, PAGE_SLOTS};
use crate::id::{Edge, EdgeRef, Label, Order, Position, Side, VertexId, Width};
use crate::labels::BucketKey;
use crate::memory::Memory;
use crate::page_store::Pages;
use crate::paged;
use crate::values;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

impl<M: Memory> Graph<M> {
    /// Replaces the value of the edge `half` belongs to, in both halves, and returns the value
    /// the canonical half held; `None` when `half` is not live. A bucket created before its
    /// label's width was declared gets its value plane here.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`], [`Error::WrongValueLength`], [`Error::OutOfMemory`]. On error
    /// nothing changes.
    pub fn set_value(&mut self, half: EdgeRef, value: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        let width = self.value_width(half.label(), value)?;
        let Some(pair) = self.pair(half)? else {
            return Ok(None);
        };
        let halves = super::pair::order(half, pair.other);
        let mut targets = vec![(half, pair.own)];
        if pair.other != half {
            targets.push((pair.other, pair.theirs));
        }
        let old = targets
            .iter()
            .find(|(h, _)| *h == halves.canonical())
            .map_or_else(Vec::new, |(h, found)| {
                self.read_value(found, h.label(), h.position())
            });
        let Some(width) = width else {
            return Ok(Some(old));
        };
        let pages = 2 + values::PAGES_PER_WRITE;
        self.edges
            .reserve(pages * u32::try_from(targets.len()).unwrap_or(2))?;
        for (h, _) in targets {
            // Look again: the other half's bucket may share a list with the first.
            let found = self.present(h.vertex(), key_of(h));
            self.write_value(found, (width, h.position()), value);
        }
        Ok(Some(old))
    }

    /// Writes `value` at `position` of `found`, giving the bucket a value plane if it has none.
    /// Within reserved memory.
    fn write_value(&mut self, found: Found, (width, position): (Width, Position), value: &[u8]) {
        let mut bucket = if found.bucket.values.is_none() {
            self.with_values(found.bucket, width)
        } else {
            found.bucket
        };
        if let Some(cells) = bucket.values.as_mut() {
            cells.plane = reserved(values::write(
                &mut self.edges,
                cells.plane,
                width,
                position.index(),
                value,
            ));
        }
        self.publish(found.entry, bucket);
    }

    /// The value at `position` of `found` (a bucket of `label`), whether or not an edge is live
    /// there: its cell, zeros before any write, empty for a label without a width.
    pub(super) fn read_value(&self, found: &Found, label: Label, position: Position) -> Vec<u8> {
        let Some(cells) = found.bucket.values else {
            let width = self.widths.get(label);
            return vec![0; width.map_or(0, |w| usize::from(w.get()))];
        };
        let mut out = vec![0; usize::from(cells.width.get())];
        values::read(
            &self.edges.pages(),
            cells.plane,
            cells.width,
            position.index(),
            &mut out,
        );
        out
    }

    /// The value of the live edge `half` belongs to, as `half`'s bucket holds it; `None` when
    /// `half` is not live.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn value(&self, half: EdgeRef) -> Result<Option<Vec<u8>>, Error> {
        let Some(found) = self.find_bucket(half.vertex(), key_of(half))? else {
            return Ok(None);
        };
        let view = self.edges.pages();
        Ok(bucket::get(&found.bucket.slots, half.position(), &view)
            .map(|_| self.read_value(&found, half.label(), half.position())))
    }

    /// Appends the values at `positions` of `v`'s `(label, side)` bucket to `out`, one width
    /// each, in the order given: zeros for a hole, a position past the end or no write yet.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn values_at(
        &self,
        v: VertexId,
        (label, side): (Label, Side),
        positions: &[Position],
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        out.clear();
        let width = self.widths.get(label).map_or(0, |w| usize::from(w.get()));
        let found = self.find_bucket(v, BucketKey { label, side })?;
        for &p in positions {
            match &found {
                Some(found) => out.extend(self.read_value(found, label, p)),
                None => out.resize(out.len() + width, 0),
            }
        }
        Ok(())
    }

    /// Calls `f` with the value of every position of `v`'s `(label, side)` bucket in `order`,
    /// until it breaks: the first phase of a values-first traversal. Reads the value plane only,
    /// skipping pages the slot index marks all dead; other removed positions come with their last
    /// value, so the second phase ([`Self::targets_at`]) drops them.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn visit_values<B>(
        &self,
        v: VertexId,
        (label, side): (Label, Side),
        order: Order,
        mut f: impl FnMut(Position, &[u8]) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let Some(found) = self.find_bucket(v, BucketKey { label, side })? else {
            return Ok(None);
        };
        let Some(width) = self.widths.get(label) else {
            return Ok(None);
        };
        let view = self.edges.pages();
        let w = usize::from(width.get());
        let mut ranges = live_ranges(&view, &found.bucket.slots);
        if order == Order::Descending {
            ranges.reverse();
        }
        let plane = found
            .bucket
            .values
            .map_or(values::Plane::Empty, |c| c.plane);
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
                return Ok(flow);
            }
        }
        Ok(None)
    }

    /// Calls `f` with each live edge of `v`'s `(label, side)` bucket and its value, in `order`,
    /// until it breaks.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn visit_neighbors_with_values<B>(
        &self,
        v: VertexId,
        (label, side): (Label, Side),
        order: Order,
        mut f: impl FnMut(Edge, &[u8]) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let Some(found) = self.find_bucket(v, BucketKey { label, side })? else {
            return Ok(None);
        };
        let view = self.edges.pages();
        let slots = found.bucket.slots;
        let width = self.widths.get(label);
        let plane = found
            .bucket
            .values
            .map_or(values::Plane::Empty, |c| c.plane);
        // A slot page's worth of positions at a time: its live edges, then its values in one read.
        let mut windows: Vec<u32> = (0..slots.stored().div_ceil(PAGE_SLOTS)).collect();
        if order == Order::Descending {
            windows.reverse();
        }
        let mut join = Join {
            view: &view,
            slots: &slots,
            plane,
            width,
            edges: Vec::new(),
            cells: Vec::new(),
        };
        for k in windows {
            let span = (k * PAGE_SLOTS, ((k + 1) * PAGE_SLOTS).min(slots.stored()));
            if let ControlFlow::Break(done) = join.window(span, order, &mut f) {
                return Ok(Some(done));
            }
        }
        Ok(None)
    }
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
        bucket::visit(self.slots, self.view, span, Window::Page, |e| {
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
