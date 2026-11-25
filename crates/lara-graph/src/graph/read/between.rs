//! Edges between two given nodes: `u`'s edges to `v` with one label and direction.
//!
//! Every edge has a half at each end, so the edges between `u` and `v` are found in whichever of
//! `u`'s bucket and `v`'s opposite bucket is smaller: the cost follows the smaller end, not the
//! hub. A bucket too small for a target index is scanned, a paged bucket is read only in the
//! blocks its index lists for the other end. Parallel edges keep the same order at both ends
//! (their halves pair by rank), so either end lists them in the same order.

use crate::bucket::LabelBucket;
use crate::bucket::LabelBucketKey;
use crate::bucket::slots::{self, Span, Window};
use crate::error::Error;
use crate::graph::Graph;
use crate::id::{Direction, EdgeRef, Label, NodeId, Position};
use crate::input::Order;
use crate::store::memory::Memory;
use core::ops::ControlFlow;

impl<M: Memory> Graph<M> {
    /// Calls `f` with each live edge between `u` and `v` that `u` holds in its
    /// `(label, direction)` bucket, in `order`, until it breaks; returns the break value. Each
    /// edge comes as the half at whichever end is cheaper to read, `u`'s or `v`'s: both name the
    /// same edge, and [`Self::counterpart`] gives the other.
    ///
    /// Reads the smaller of the two ends' buckets; in a hub's, only the blocks its target index
    /// lists for the other end.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for `u` or `v`.
    pub fn visit_edges_between<B>(
        &self,
        u: NodeId,
        (label, direction): (Label, Direction),
        v: NodeId,
        order: Order,
        mut f: impl FnMut(EdgeRef) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let Some(End {
            node,
            direction,
            other,
            bucket,
        }) = self.smaller_end(u, (label, direction), v)?
        else {
            return Ok(None);
        };
        let half = |at: Position| EdgeRef::new(node, label, direction, at);
        let slots = &bucket.slots;
        let view = self.edge_view().pages();
        let mut visit_span = |&(low, high): &(u32, u32)| {
            slots::visit(
                slots,
                &view,
                Span::new(low, high, order),
                Window::Page,
                |e| {
                    if e.target() == other {
                        f(half(e.position()))
                    } else {
                        ControlFlow::Continue(())
                    }
                },
            )
        };
        // A bucket without an index is one span, read with no list of spans.
        let Some(spans) = self.edge_view().indexed_spans_to(&bucket, other) else {
            return Ok(visit_span(&(0, slots.stored())));
        };
        Ok(match order {
            Order::Ascending => spans.iter().find_map(&mut visit_span),
            Order::Descending => spans.iter().rev().find_map(&mut visit_span),
        })
    }

    /// Whether `u` has an edge to `v` in its `(label, direction)` bucket. Reads as
    /// [`Self::visit_edges_between`] does, and stops at the first edge.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for `u` or `v`.
    pub fn contains_edge(
        &self,
        u: NodeId,
        (label, direction): (Label, Direction),
        v: NodeId,
    ) -> Result<bool, Error> {
        Ok(self
            .visit_edges_between(u, (label, direction), v, Order::Ascending, |_| {
                ControlFlow::Break(())
            })?
            .is_some())
    }

    /// The smaller of `u`'s `(label, direction)` bucket and `v`'s opposite one, as the end that
    /// holds it, the node its entries point to, and the bucket; `None` when either is absent, as
    /// then no edge joins them. A bucket at `u` of at most [`READ_AS_IS`] positions is taken as it
    /// is, without looking up `v`'s.
    fn smaller_end(
        &self,
        u: NodeId,
        (label, direction): (Label, Direction),
        v: NodeId,
    ) -> Result<Option<End>, Error> {
        let own_end = |bucket| End {
            node: u,
            direction,
            other: v,
            bucket,
        };
        let own = self.read_label_bucket(u, LabelBucketKey { label, direction })?;
        let own = match own {
            Some(own) if own.slots.stored() > READ_AS_IS => own,
            tiny_or_none => {
                self.check(v)?;
                return Ok(tiny_or_none.map(own_end));
            }
        };
        let theirs_direction = direction.opposite();
        let theirs = self.read_label_bucket(
            v,
            LabelBucketKey {
                label,
                direction: theirs_direction,
            },
        )?;
        Ok(theirs.map(|theirs| {
            if theirs.slots.stored() < own.slots.stored() {
                End {
                    node: v,
                    direction: theirs_direction,
                    other: u,
                    bucket: theirs,
                }
            } else {
                own_end(own)
            }
        }))
    }
}

/// The most positions of `u`'s bucket that are read without looking up `v`'s: the lookup costs
/// about as much as reading this many slots, so it cannot pay off below it.
const READ_AS_IS: u32 = 32;

/// The end of an edge set that is read: its node and bucket, and the node at the other end.
struct End {
    node: NodeId,
    direction: Direction,
    other: NodeId,
    bucket: LabelBucket,
}
