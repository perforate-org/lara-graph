//! Reads beyond a whole bucket in position order: either order, windows and chosen positions.

use super::{Error, Graph};
use crate::bucket::{self, Span, Window};
use crate::id::{Cursor, Edge, Label, Order, Position, Side, VertexId};
use crate::labels::BucketKey;
use crate::memory::Memory;
use alloc::vec::Vec;
use core::ops::ControlFlow;

impl<M: Memory> Graph<M> {
    /// Calls `f` with each live edge of `v`'s `(label, side)` bucket in `order`, until it breaks;
    /// returns the break value. Reads like [`Self::neighbors`] without collecting.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn visit_neighbors<B>(
        &self,
        v: VertexId,
        label: Label,
        side: Side,
        order: Order,
        f: impl FnMut(Edge) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        Ok(self.bucket(v, BucketKey { label, side })?.and_then(|d| {
            bucket::visit(
                &d,
                &self.edges.pages(),
                Span::all(&d, order),
                Window::Page,
                f,
            )
        }))
    }

    /// Calls `f` with each live edge in `cursor`'s window of `v`'s `(label, side)` bucket, in its
    /// order, and returns the next window, or `None` when the bucket has no positions past this
    /// one. Holes count toward the span, so each call reads at most `span` slots.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn visit_window(
        &self,
        v: VertexId,
        label: Label,
        side: Side,
        cursor: Cursor,
        mut f: impl FnMut(Edge),
    ) -> Result<Option<Cursor>, Error> {
        let Some(d) = self.bucket(v, BucketKey { label, side })? else {
            return Ok(None);
        };
        let stored = d.stored();
        let span = cursor.span().get();
        let (window, next) = match cursor.order() {
            Order::Ascending => {
                let low = cursor.from().map_or(0, Position::index);
                let high = low.saturating_add(span).min(stored);
                (
                    Span::new(low, high, Order::Ascending),
                    (high < stored).then_some(high),
                )
            }
            Order::Descending => {
                // The window ends at `from` (included), clamped to the last position.
                let Some(last) = stored.checked_sub(1) else {
                    return Ok(None);
                };
                let top = cursor.from().map_or(last, |p| p.index().min(last));
                let low = (top + 1).saturating_sub(span);
                (
                    Span::new(low, top + 1, Order::Descending),
                    low.checked_sub(1),
                )
            }
        };
        bucket::visit(&d, &self.edges.pages(), window, Window::Page, |e| {
            f(e);
            ControlFlow::<()>::Continue(())
        });
        Ok(next.map(|at| Cursor::at(cursor.order(), Position::new(at), cursor.span())))
    }

    /// Replaces the contents of `out` with the targets at `positions` of `v`'s `(label, side)`
    /// bucket, in the order given: `None` for a hole or a position past the end. Reads each
    /// touched run or page once.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn targets_at(
        &self,
        v: VertexId,
        label: Label,
        side: Side,
        positions: &[Position],
        out: &mut Vec<Option<VertexId>>,
    ) -> Result<(), Error> {
        out.clear();
        let Some(d) = self.bucket(v, BucketKey { label, side })? else {
            out.resize(positions.len(), None);
            return Ok(());
        };
        let mut sorted = positions.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let mut found = Vec::with_capacity(sorted.len());
        bucket::read_positions(&d, &self.edges.pages(), &sorted, |_, target| {
            found.push(target);
        });
        out.extend(positions.iter().map(|p| {
            sorted
                .binary_search(p)
                .ok()
                .and_then(|i| found.get(i).copied().flatten())
        }));
        Ok(())
    }
}
