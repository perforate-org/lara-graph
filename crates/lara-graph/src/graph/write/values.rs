//! Writing edge values, one edge or a batch that is all-or-none.

use crate::bucket::LabelBucketKey;
use crate::bucket::slots;
use crate::edge::Writes;
use crate::error::{BatchError, Error};
use crate::graph::locate::Found;
use crate::graph::locate::LastBucket;
use crate::graph::{Graph, corrupt};
use crate::id::{EdgeRef, Halves};
use crate::id::{Label, Position, Width};
use crate::input::EdgeValue;
use crate::store::memory::Memory;
use alloc::vec;
use alloc::vec::Vec;

/// What one write of a batch did to its edge, with the old canonical value.
enum SetValue {
    /// The half is not live.
    Absent,
    /// The label has no width, so nothing was written.
    Kept(Vec<u8>),
    Written(Vec<u8>),
}

/// The buckets of the batch's last edge: its own half's and the other half's.
#[derive(Default)]
struct LastPair {
    own: LastBucket<Option<Found>>,
    theirs: LastBucket<Found>,
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
        let halves = Halves::of(half, pair.other);
        let mut targets = vec![(half, pair.own)];
        if pair.other != half {
            targets.push((pair.other, pair.theirs));
        }
        let old = targets
            .iter()
            .find(|(h, _)| *h == halves.canonical())
            .map_or_else(Vec::new, |(h, found)| {
                self.edge_view()
                    .read_value(&found.bucket, h.label(), h.position())
            });
        let Some(width) = width else {
            return Ok(Some(old));
        };
        self.edges.reserve(Writes::Values {
            count: u32::try_from(targets.len()).unwrap_or(2),
        })?;
        let first = half.node();
        for (i, (h, found)) in targets.into_iter().enumerate() {
            // The second half is looked up again only on the first's node, the one list the
            // first write can change; elsewhere the lookup from `pair` still holds.
            let found = if i > 0 && h.node() == first {
                self.present(h.node(), LabelBucketKey::of(h))
            } else {
                found
            };
            self.write_value(found, (width, h.position()), value);
        }
        Ok(Some(old))
    }

    /// Writes each edge value in `values`, in order, and returns the old canonical value of
    /// each edge (`None` for a hole). All nodes and value lengths are checked before the first
    /// write, and a failure after that restores every value written so far, so the batch is
    /// all-or-none.
    ///
    /// # Errors
    ///
    /// As [`Self::set_edge_value`], for the value that fails.
    pub fn set_edge_values(
        &mut self,
        values: &[EdgeValue<'_>],
    ) -> Result<Vec<Option<Vec<u8>>>, BatchError> {
        for (index, edge) in values.iter().enumerate() {
            self.check(edge.half().node())
                .and_then(|()| self.value_width(edge.half().label(), edge.value()))
                .map_err(|error| BatchError::unchanged(index, error))?;
        }
        let mut out = Vec::with_capacity(values.len());
        let mut journal: Vec<(EdgeRef, usize)> = Vec::new();
        let mut last = LastPair::default();
        for (index, edge) in values.iter().enumerate() {
            match self.set_value_reusing(edge, &mut last) {
                Ok(SetValue::Absent) => out.push(None),
                Ok(SetValue::Kept(old)) => out.push(Some(old)),
                Ok(SetValue::Written(old)) => {
                    journal.push((edge.half(), out.len()));
                    out.push(Some(old));
                }
                Err(error) => {
                    self.rollback_values(&journal, &out);
                    return Err(BatchError::unchanged(index, error));
                }
            }
        }
        Ok(out)
    }

    /// Writes one edge value of a batch, reusing the buckets `last` holds from the edge before.
    fn set_value_reusing(
        &mut self,
        edge: &EdgeValue<'_>,
        last: &mut LastPair,
    ) -> Result<SetValue, Error> {
        let half = edge.half();
        let node = half.node();
        let key = LabelBucketKey::of(half);
        let own = last
            .own
            .get_or_try(node, key, || self.find_label_bucket(node, key))?;
        let Some(own) = own else {
            return Ok(SetValue::Absent);
        };
        let view = self.edge_view().pages();
        let Some(other) = slots::get(&own.bucket.slots, half.position(), &view) else {
            return Ok(SetValue::Absent);
        };
        let theirs_key = LabelBucketKey::of(half).opposite();
        let theirs = last
            .theirs
            .get_or_else(other, theirs_key, || self.present(other, theirs_key));
        let pair = self.pair_with_theirs(half, own, other, theirs_key, theirs);
        let halves = Halves::of(half, pair.other);
        let old = if halves.canonical() == half {
            self.edge_view()
                .read_value(&pair.own.bucket, half.label(), half.position())
        } else {
            self.edge_view().read_value(
                &pair.theirs.bucket,
                pair.other.label(),
                pair.other.position(),
            )
        };
        let Some(width) = self.value_width(half.label(), edge.value())? else {
            return Ok(SetValue::Kept(old));
        };
        let mut targets = [(half, pair.own), (pair.other, pair.theirs)];
        let count = if pair.other == half { 1 } else { 2 };
        self.edges.reserve(Writes::Values {
            count: u32::try_from(count).unwrap_or(2),
        })?;
        for (h, found) in targets.iter_mut().take(count) {
            let updated = self.write_value(*found, (width, h.position()), edge.value());
            if *h == half {
                last.own.set(node, key, Some(updated));
            } else {
                last.theirs.set(other, theirs_key, updated);
            }
        }
        Ok(SetValue::Written(old))
    }

    /// Restores the old values written before a failed batch, newest first.
    fn rollback_values(&mut self, journal: &[(EdgeRef, usize)], out: &[Option<Vec<u8>>]) {
        for &(half, index) in journal.iter().rev() {
            let Some(Some(old)) = out.get(index) else {
                continue;
            };
            self.restore_value(half, old);
        }
    }

    /// Restores `old` to both halves of `half` without reserving: the failed write already made
    /// the value cell, so this only overwrites it.
    fn restore_value(&mut self, half: EdgeRef, old: &[u8]) {
        let pair = self
            .pair(half)
            .unwrap_or_else(|_| corrupt("an edge whose value is being restored"));
        let Some(pair) = pair else {
            return;
        };
        let width = self
            .widths
            .get(half.label())
            .unwrap_or_else(|| corrupt("a value being restored without a width"));
        let targets = [(half, pair.own), (pair.other, pair.theirs)];
        let count = if pair.other == half { 1 } else { 2 };
        for (h, found) in targets.iter().take(count) {
            self.write_value(*found, (width, h.position()), old);
        }
    }
}

impl<M: Memory> Graph<M> {
    /// `label`'s width, checking that `value` matches it.
    pub(in crate::graph) fn value_width(
        &self,
        label: Label,
        value: &[u8],
    ) -> Result<Option<Width>, Error> {
        self.widths.check(label, value)
    }

    /// Writes `value` at `position` of `found`, giving the bucket a value plane if it has none.
    /// Within reserved memory.
    pub(in crate::graph) fn write_value(
        &mut self,
        found: Found,
        (width, position): (Width, Position),
        value: &[u8],
    ) -> Found {
        let bucket = self.edges.put_value(found.bucket, (width, position), value);
        let bucket = self.publish(found.entry, bucket);
        Found {
            entry: found.entry,
            bucket,
        }
    }
}

impl<M: Memory> Graph<M> {
    /// Declares that `label`'s edges carry a value of `width` bytes, on every side. Constant
    /// time: existing edges read as zeros until a value is written to them. Declaring the same
    /// width again changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::WidthConflict`] if `label` already has another width,
    /// past 1023 labels, [`Error::OutOfMemory`]. On error nothing changes.
    pub fn declare_width(&mut self, label: Label, width: Width) -> Result<(), Error> {
        if let Some(root) = self
            .widths
            .declare(&mut self.label_bucket_lists, label, width)?
        {
            self.nodes.set_widths_root(root);
        }
        Ok(())
    }
}
