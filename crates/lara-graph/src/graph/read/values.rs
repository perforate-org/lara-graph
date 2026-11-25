//! Reading edge values: one edge's, chosen positions', and scans of a bucket's values alone or
//! with its edges.

use crate::bucket::slots::{self};
use crate::bucket::{LabelBucketKey, Migration};
use crate::edge::visit_joined;
use crate::error::Error;
use crate::graph::Graph;
use crate::id::{Direction, EdgeRef, Label, Neighbor, NodeId, Position};
use crate::id::{ValueVersion, Width};
use crate::input::Order;
use crate::store::memory::Memory;
use alloc::vec::Vec;
use core::ops::ControlFlow;

impl<M: Memory> Graph<M> {
    /// The width of `label`'s values, if it has been declared.
    #[must_use]
    pub fn width(&self, label: Label) -> Option<Width> {
        self.widths.get(label)
    }

    /// The version of `label`'s current value layout ([`Self::change_width`]), if it has a width.
    #[must_use]
    pub fn value_version(&self, label: Label) -> Option<ValueVersion> {
        self.widths.schema(label).map(|s| s.version)
    }

    /// The version the values of `v`'s `(label, direction)` bucket are in, and so how to decode
    /// what reads return from it: an older one until [`Self::migrate_values`] has rewritten them,
    /// the label's current one for a bucket that holds none yet. `None` for a label without a
    /// width or a node without the bucket.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn bucket_value_version(
        &self,
        v: NodeId,
        (label, direction): (Label, Direction),
    ) -> Result<Option<ValueVersion>, Error> {
        let Some(schema) = self.widths.schema(label) else {
            self.check(v)?;
            return Ok(None);
        };
        let bucket = self.read_label_bucket(v, LabelBucketKey { label, direction })?;
        Ok(bucket.map(|b| {
            b.cells()
                .map_or(schema.version, |c| c.version_given(schema.version))
        }))
    }

    /// Whether [`Self::migrate_values`] has work in `v`'s `(label, direction)` bucket: it holds
    /// values in an older version, or a migration is under way.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn needs_migration(
        &self,
        v: NodeId,
        (label, direction): (Label, Direction),
    ) -> Result<bool, Error> {
        let Some(schema) = self.widths.schema(label) else {
            self.check(v)?;
            return Ok(false);
        };
        let bucket = self.read_label_bucket(v, LabelBucketKey { label, direction })?;
        Ok(bucket
            .and_then(|b| b.cells())
            .is_some_and(|c| !c.is_current(schema.version) || c.migration == Migration::UnderWay))
    }

    /// The value of the live edge `half` belongs to, as `half`'s bucket holds it; `None` when
    /// `half` is not live.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn edge_value(&self, half: EdgeRef) -> Result<Option<Vec<u8>>, Error> {
        let Some(found) = self.find_label_bucket(half.node(), LabelBucketKey::of(half))? else {
            return Ok(None);
        };
        let view = self.edge_view().pages();
        Ok(
            slots::get(&found.bucket.slots, half.position(), &view).map(|_| {
                self.edge_view()
                    .read_value(&found.bucket, half.label(), half.position())
            }),
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
                Some(found) => out.extend(self.edge_view().read_value(&found.bucket, label, p)),
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
        let Some(bucket) = self.read_label_bucket(v, LabelBucketKey { label, direction })? else {
            return Ok(None);
        };
        Ok(self
            .edge_view()
            .visit_values_in(&bucket, label, order, &mut f))
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
        mut f: impl FnMut(Neighbor, &[u8]) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let Some(bucket) = self.read_label_bucket(v, LabelBucketKey { label, direction })? else {
            return Ok(None);
        };
        let width = self.widths.get(label);
        Ok(visit_joined(
            &self.edge_view().pages(),
            &bucket,
            width,
            order,
            &mut f,
        ))
    }
}
