//! Reads beyond a whole bucket in position order: either order, windows and chosen positions.

use super::{Error, Graph, RecordLines};
use crate::corrupt::corrupt;
use crate::descriptor::Descriptor;
use crate::id::{Cursor, Direction, Edge, Label, NodeId, Order, Position};
use crate::label_bucket::{self, Span, Window};
use crate::label_bucket_list::{self, LabelBucketKey};
use crate::memory::Memory;
use crate::stored::Stored;
use alloc::vec::Vec;
use core::ops::ControlFlow;

impl<M: Memory> Graph<M> {
    /// Calls `f` with each live edge of `v`'s `(label, direction)` bucket in `order`, until it
    /// breaks; returns the break value. Reads like [`Self::neighbors_directed`] without collecting.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn visit_neighbors<B>(
        &self,
        v: NodeId,
        label: Label,
        direction: Direction,
        order: Order,
        f: impl FnMut(Edge) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        Ok(self
            .bucket(v, LabelBucketKey { label, direction })?
            .and_then(|d| {
                label_bucket::visit(
                    &d,
                    &self.edges.pages(),
                    Span::all(&d, order),
                    Window::Page,
                    f,
                )
            }))
    }

    /// Calls `f` with `(node, edge)` for every live edge of each node's `(label, direction)`
    /// bucket, in `nodes` order and `order` within each bucket, until `f` breaks.
    ///
    /// The batch holds both `Pages` views and an eight-record line of node records across the
    /// nodes, so dense ids read one 64-byte line per eight nodes instead of one record per node. All nodes are checked before the first callback, so an unknown node
    /// leaves the callbacks untouched.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn visit_neighbors_each<B>(
        &self,
        nodes: &[NodeId],
        label: Label,
        direction: Direction,
        order: Order,
        mut f: impl FnMut(NodeId, Edge) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        for &v in nodes {
            self.check(v)?;
        }
        let view = self.edges.pages();
        let lists = self.label_bucket_lists.pages();
        let key = LabelBucketKey { label, direction };
        // A line only pays off when the ids are dense; a scattered batch would read 64 bytes
        // for each single record, so it keeps the one-record path. The first and eighth ids are
        // enough to tell: a false positive still reads the right line, only without a gain.
        let dense = nodes.len() >= 8
            && nodes
                .first()
                .map(|v| v.index())
                .zip(nodes.get(7).map(|v| v.index()))
                .is_some_and(|(first, eighth)| eighth == first + 7);
        let mut records = dense.then(RecordLines::default);
        for &v in nodes {
            let record = records.as_mut().map_or_else(
                || self.record(v).unwrap_or_else(|_| corrupt("a known node")),
                |lines| lines.record(self, v),
            );
            let Some(list) = record.list else { continue };
            let Some(stored) = label_bucket_list::lookup_stored(&lists, Some(list), key) else {
                continue;
            };
            let slots = match stored {
                Stored::Plain(slots) => slots,
                Stored::Extended(header) => Descriptor::decode(view.read_array(header.offset())),
            };
            if let Some(found) =
                label_bucket::visit(&slots, &view, Span::all(&slots, order), Window::Page, |e| {
                    f(v, e)
                })
            {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// Calls `f` with each live edge in `cursor`'s window of `v`'s `(label, direction)` bucket, in
    /// its order, and returns the next window, or `None` when the bucket has no positions past this
    /// one. Holes count toward the span, so each call reads at most `span` slots.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn visit_window(
        &self,
        v: NodeId,
        label: Label,
        direction: Direction,
        cursor: Cursor,
        mut f: impl FnMut(Edge),
    ) -> Result<Option<Cursor>, Error> {
        let Some(d) = self.bucket(v, LabelBucketKey { label, direction })? else {
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
        label_bucket::visit(&d, &self.edges.pages(), window, Window::Page, |e| {
            f(e);
            ControlFlow::<()>::Continue(())
        });
        Ok(next.map(|at| Cursor::at(cursor.order(), Position::new(at), cursor.span())))
    }

    /// Replaces the contents of `out` with the targets at `positions` of `v`'s `(label, direction)`
    /// bucket, in the order given: `None` for a hole or a position past the end. Reads each
    /// touched run or page once.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn targets_at(
        &self,
        v: NodeId,
        label: Label,
        direction: Direction,
        positions: &[Position],
        out: &mut Vec<Option<NodeId>>,
    ) -> Result<(), Error> {
        out.clear();
        let Some(d) = self.bucket(v, LabelBucketKey { label, direction })? else {
            out.resize(positions.len(), None);
            return Ok(());
        };
        // Positions already ascending and distinct, as a reader of windows passes them, go
        // straight through; others are sorted for the one pass and put back in their order.
        if positions.is_sorted_by(|a, b| a < b) {
            label_bucket::read_positions(&d, &self.edges.pages(), positions, |_, target| {
                out.push(target);
            });
            return Ok(());
        }
        let mut sorted = positions.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let mut found = Vec::with_capacity(sorted.len());
        label_bucket::read_positions(&d, &self.edges.pages(), &sorted, |_, target| {
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
