//! Reading one node's edges: its bucket whole or in either order, windows of it, chosen
//! positions, its degree and holes, and an edge's target or other half.

use crate::bucket::LabelBucketKey;
use crate::bucket::descriptor::Descriptor;
use crate::bucket::slots::{self, Span, Window};
use crate::error::Error;
use crate::graph::Graph;
use crate::id::{Direction, EdgeRef, Label, Neighbor, NodeId, Position};
use crate::input::{Cursor, Order};
use crate::node::label_bucket_list;
use crate::store::memory::Memory;
use alloc::vec::Vec;
use core::ops::ControlFlow;

impl<M: Memory> Graph<M> {
    /// The other half of the edge `half` belongs to, or `None` when `half` is not live.
    ///
    /// When the other bucket is small and holds one entry back, that entry is the answer;
    /// otherwise the halves are matched by rank. A paged bucket answers rank and selection from
    /// its target index (logarithmic, then one block scan); a smaller one is scanned.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn counterpart(&self, half: EdgeRef) -> Result<Option<EdgeRef>, Error> {
        Ok(self.pair(half)?.map(|pair| pair.other))
    }

    /// Calls `f` with each live edge of `v`'s `(label, direction)` bucket in `order`, until it
    /// breaks; returns the break value. Reads like [`Self::neighbors`] without collecting.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn visit_neighbors<B>(
        &self,
        v: NodeId,
        (label, direction): (Label, Direction),
        order: Order,
        f: impl FnMut(Neighbor) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        Ok(self
            .bucket(v, LabelBucketKey { label, direction })?
            .and_then(|d| {
                slots::visit(
                    &d,
                    &self.edge_view().pages(),
                    Span::all(&d, order),
                    Window::Page,
                    f,
                )
            }))
    }

    /// Calls `f` with `(label, edge)` for each live edge of `v` on `direction`, whatever its label,
    /// until it breaks; returns the break value. `Ascending` goes through the labels from low to
    /// high and each bucket in position order; `Descending` reverses both. Reads `v`'s label
    /// bucket list once, where a loop of [`Self::visit_neighbors`] over [`Self::edge_labels`]
    /// would look each bucket up again.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn visit_all_neighbors<B>(
        &self,
        v: NodeId,
        direction: Direction,
        order: Order,
        mut f: impl FnMut(Label, Neighbor) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let Some(list) = self.label_bucket_list(v)? else {
            return Ok(None);
        };
        let lists = self.label_bucket_lists.pages();
        let edges = self.edge_view().pages();
        let mut visit = |label: Label, slots: Descriptor| {
            slots::visit(
                &slots,
                &edges,
                Span::all(&slots, order),
                Window::Page,
                |e| f(label, e),
            )
        };
        if order == Order::Ascending {
            return Ok(label_bucket_list::for_each(
                &lists,
                Some(list),
                |key, listed| {
                    if key.direction != direction {
                        return ControlFlow::Continue(());
                    }
                    visit(key.label, listed.stored().slots(&edges))
                        .map_or(ControlFlow::Continue(()), ControlFlow::Break)
                },
            ));
        }
        let mut buckets = Vec::new();
        label_bucket_list::for_each(&lists, Some(list), |key, listed| {
            if key.direction == direction {
                buckets.push((key.label, listed.stored()));
            }
            ControlFlow::<()>::Continue(())
        });
        Ok(buckets
            .into_iter()
            .rev()
            .find_map(|(label, stored)| visit(label, stored.slots(&edges))))
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
        (label, direction): (Label, Direction),
        cursor: Cursor,
        mut f: impl FnMut(Neighbor),
    ) -> Result<Option<Cursor>, Error> {
        let Some(d) = self.bucket(v, LabelBucketKey { label, direction })? else {
            return Ok(None);
        };
        let stored = d.stored();
        let span = cursor.span().get();
        let (window, next) = match cursor.order() {
            Order::Ascending => {
                let low = cursor.start().map_or(0, Position::index);
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
                let top = cursor.start().map_or(last, |p| p.index().min(last));
                let low = (top + 1).saturating_sub(span);
                (
                    Span::new(low, top + 1, Order::Descending),
                    low.checked_sub(1),
                )
            }
        };
        slots::visit(&d, &self.edge_view().pages(), window, Window::Page, |e| {
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
        (label, direction): (Label, Direction),
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
            slots::read_positions(&d, &self.edge_view().pages(), positions, |_, target| {
                out.push(target);
            });
            return Ok(());
        }
        let mut sorted = positions.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let mut found = Vec::with_capacity(sorted.len());
        slots::read_positions(&d, &self.edge_view().pages(), &sorted, |_, target| {
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

impl<M: Memory> Graph<M> {
    /// The position of the live edge that `skip` live edges precede in `v`'s `(label,
    /// direction)` bucket, counted in `order`: where a read that skips the first `skip` edges
    /// starts, as [`Self::visit_window`] reads on from it. `None` when the bucket has no more.
    /// A large bucket skips whole pages by the live counts its page index keeps, so the cost
    /// follows the pages, not `skip`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn nth_position(
        &self,
        v: NodeId,
        (label, direction): (Label, Direction),
        order: Order,
        skip: u32,
    ) -> Result<Option<Position>, Error> {
        Ok(self
            .bucket(v, LabelBucketKey { label, direction })?
            .and_then(|d| slots::nth_live(&d, &self.edge_view().pages(), order, skip)))
    }

    /// Number of live edges in `v`'s `(label, direction)` bucket.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn degree(&self, v: NodeId, (label, direction): (Label, Direction)) -> Result<u32, Error> {
        Ok(self
            .bucket(v, LabelBucketKey { label, direction })?
            .map_or(0, |d| d.live()))
    }

    /// Replaces the contents of `out` with the live edges of `v`'s `(label, direction)` bucket, in
    /// position order. Each edge's target is the node at its other end.
    /// [`Self::visit_neighbors`] reads the same edges in about half the instructions when they
    /// need not be kept.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn neighbors(
        &self,
        v: NodeId,
        (label, direction): (Label, Direction),
        out: &mut Vec<Neighbor>,
    ) -> Result<(), Error> {
        out.clear();
        if let Some(d) = self.bucket(v, LabelBucketKey { label, direction })? {
            slots::read(&d, &self.edge_view().pages(), out);
        }
        Ok(())
    }

    /// The node at the other end of `half`, if it is live.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn edge_target(&self, half: EdgeRef) -> Result<Option<NodeId>, Error> {
        Ok(self
            .bucket(half.node(), LabelBucketKey::of(half))?
            .and_then(|d| slots::get(&d, half.position(), &self.edge_view().pages())))
    }

    /// Removed edges `v`'s `(label, direction)` bucket still holds positions for: the work
    /// compaction can do.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn holes(&self, v: NodeId, (label, direction): (Label, Direction)) -> Result<u32, Error> {
        Ok(self
            .bucket(v, LabelBucketKey { label, direction })?
            .map_or(0, |d| d.stored() - d.live()))
    }

    /// A bucket's slot descriptor, without building the bucket's values or index: a read that
    /// only scans slots does not pay for the rest of an extended header.
    pub(in crate::graph) fn bucket(
        &self,
        v: NodeId,
        key: LabelBucketKey,
    ) -> Result<Option<Descriptor>, Error> {
        let list = self.label_bucket_list(v)?;
        let lists = self.label_bucket_lists.pages();
        Ok(label_bucket_list::lookup_stored(&lists, list, key)
            .map(|stored| stored.slots(&self.edge_view().pages())))
    }
}
