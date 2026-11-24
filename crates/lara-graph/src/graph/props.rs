//! Edge values: writing, point and batch reads, values-first scans and combined scans.

use super::pair::{Found, key_of, opposite};
use super::{BatchError, Error, Graph, reserved};
use crate::descriptor::{Descriptor, PAGE_SLOTS};
use crate::id::{Direction, Edge, EdgeRef, Label, NodeId, Order, Position, Width};
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

/// One edge value to write in [`Graph::set_edge_values`].
#[derive(Clone, Copy, Debug)]
pub struct EdgeValue<'a> {
    half: EdgeRef,
    value: &'a [u8],
}

impl<'a> EdgeValue<'a> {
    /// Writes `value` to the edge `half` belongs to.
    #[must_use]
    pub const fn new(half: EdgeRef, value: &'a [u8]) -> Self {
        Self { half, value }
    }
}

impl<M: Memory> Graph<M> {
    /// Replaces the value of the edge `half` belongs to, in both halves, and returns the value
    /// the canonical half held; `None` when `half` is not live. A bucket created before its
    /// label's width was declared gets its value plane here.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], [`Error::WrongValueLength`], [`Error::OutOfMemory`]. On error
    /// nothing changes.
    pub fn set_edge_value(
        &mut self,
        half: EdgeRef,
        value: &[u8],
    ) -> Result<Option<Vec<u8>>, Error> {
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
        let first = half.node();
        for (i, (h, found)) in targets.into_iter().enumerate() {
            // The second half is looked up again only on the first's node, the one list the
            // first write can change; elsewhere the lookup from `pair` still holds.
            let found = if i > 0 && h.node() == first {
                self.present(h.node(), key_of(h))
            } else {
                found
            };
            self.write_value(found, (width, h.position()), value);
        }
        Ok(Some(old))
    }

    /// Writes each edge value in `values`, in order, and returns the old canonical value of
    /// each edge (`None` for a hole). Each edge is all-or-none, but the batch is not atomic:
    /// an error leaves the values before it written.
    ///
    /// All nodes and value lengths are checked before the first write.
    ///
    /// # Errors
    ///
    /// As [`Self::set_edge_value`], for the value that fails.
    pub fn set_edge_values(
        &mut self,
        values: &[EdgeValue<'_>],
    ) -> Result<Vec<Option<Vec<u8>>>, BatchError> {
        for (index, edge) in values.iter().enumerate() {
            self.check(edge.half.node())
                .and_then(|()| self.value_width(edge.half.label(), edge.value))
                .map_err(|error| BatchError::unchanged(index, error))?;
        }
        let mut out = Vec::with_capacity(values.len());
        let mut own_cache: Option<(NodeId, LabelBucketKey, Option<Found>)> = None;
        let mut theirs_cache: Option<(NodeId, LabelBucketKey, Found)> = None;
        for (index, edge) in values.iter().enumerate() {
            let half = edge.half;
            let node = half.node();
            let key = key_of(half);
            let own = match own_cache {
                Some((last_node, last_key, found)) if last_node == node && last_key == key => found,
                _ => {
                    let found = self
                        .find_label_bucket(node, key)
                        .map_err(|error| BatchError::after_prefix(index, error))?;
                    own_cache = Some((node, key, found));
                    found
                }
            };
            let Some(own) = own else {
                out.push(None);
                continue;
            };
            let view = self.edges.pages();
            let Some(other) = label_bucket::get(&own.bucket.slots, half.position(), &view) else {
                out.push(None);
                continue;
            };
            let theirs_key = LabelBucketKey {
                label: half.label(),
                direction: opposite(half.direction()),
            };
            let theirs = match theirs_cache {
                Some((last_node, last_key, found))
                    if last_node == other && last_key == theirs_key =>
                {
                    found
                }
                _ => {
                    let found = self.present(other, theirs_key);
                    theirs_cache = Some((other, theirs_key, found));
                    found
                }
            };
            let pair = self.pair_with_theirs(half, own, other, theirs_key, theirs);
            let halves = super::pair::order(half, pair.other);
            let old = if halves.canonical() == half {
                self.read_value(&pair.own, half.label(), half.position())
            } else {
                self.read_value(&pair.theirs, pair.other.label(), pair.other.position())
            };
            let Some(width) = self
                .value_width(half.label(), edge.value)
                .map_err(|error| BatchError::after_prefix(index, error))?
            else {
                out.push(Some(old));
                continue;
            };
            let mut targets = [(half, pair.own), (pair.other, pair.theirs)];
            let count = if pair.other == half { 1 } else { 2 };
            self.edges
                .reserve((2 + values::PAGES_PER_WRITE) * u32::try_from(count).unwrap_or(2))
                .map_err(|error| BatchError::after_prefix(index, error.into()))?;
            for (h, found) in targets.iter_mut().take(count) {
                let updated = self.write_value(*found, (width, h.position()), edge.value);
                if *h == half {
                    own_cache = Some((node, key, Some(updated)));
                } else {
                    theirs_cache = Some((other, theirs_key, updated));
                }
            }
            out.push(Some(old));
        }
        Ok(out)
    }

    /// Writes `value` at `position` of `found`, giving the bucket a value plane if it has none.
    /// Within reserved memory.
    fn write_value(
        &mut self,
        found: Found,
        (width, position): (Width, Position),
        value: &[u8],
    ) -> Found {
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
        let bucket = self.publish(found.entry, bucket);
        Found {
            entry: found.entry,
            bucket,
        }
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
    /// [`Error::UnknownNode`].
    pub fn edge_value(&self, half: EdgeRef) -> Result<Option<Vec<u8>>, Error> {
        let Some(found) = self.find_label_bucket(half.node(), key_of(half))? else {
            return Ok(None);
        };
        let view = self.edges.pages();
        Ok(
            label_bucket::get(&found.bucket.slots, half.position(), &view)
                .map(|_| self.read_value(&found, half.label(), half.position())),
        )
    }

    /// Appends the values at `positions` of `v`'s `(label, direction)` bucket to `out`, one width
    /// each, in the order given: zeros for a hole, a position past the end or no write yet.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn values_at(
        &self,
        v: NodeId,
        (label, direction): (Label, Direction),
        positions: &[Position],
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        out.clear();
        let width = self.widths.get(label).map_or(0, |w| usize::from(w.get()));
        let found = self.find_label_bucket(v, LabelBucketKey { label, direction })?;
        for &p in positions {
            match &found {
                Some(found) => out.extend(self.read_value(found, label, p)),
                None => out.resize(out.len() + width, 0),
            }
        }
        Ok(())
    }

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
    pub(super) fn visit_values_in<B>(
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
pub(super) struct Join<'a, 'v, M: Memory> {
    pub(super) view: &'a Pages<'v, M>,
    pub(super) slots: &'a Descriptor,
    pub(super) plane: values::Plane,
    pub(super) width: Option<Width>,
    pub(super) edges: Vec<Edge>,
    pub(super) cells: Vec<u8>,
}

impl<M: Memory> Join<'_, '_, M> {
    /// Calls `f` with each live edge of positions `low..high` and its value.
    pub(super) fn window<B>(
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
