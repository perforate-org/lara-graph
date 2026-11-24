//! Edge values: writing and point and batch reads.

mod scan;

pub(super) use scan::visit_joined;

use super::pair::{Found, LastBucket, key_of, opposite};
use super::{BatchError, Error, Graph, corrupt, reserved};
use crate::id::{Direction, EdgeRef, Label, NodeId, Position, Width};
use crate::label_bucket;
use crate::label_bucket_list::LabelBucketKey;
use crate::memory::Memory;
use crate::values;
use alloc::vec;
use alloc::vec::Vec;

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
            self.check(edge.half.node())
                .and_then(|()| self.value_width(edge.half.label(), edge.value))
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
                    journal.push((edge.half, out.len()));
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
        let half = edge.half;
        let node = half.node();
        let key = key_of(half);
        let own = last
            .own
            .get_or_try(node, key, || self.find_label_bucket(node, key))?;
        let Some(own) = own else {
            return Ok(SetValue::Absent);
        };
        let view = self.edges.pages();
        let Some(other) = label_bucket::get(&own.bucket.slots, half.position(), &view) else {
            return Ok(SetValue::Absent);
        };
        let theirs_key = LabelBucketKey {
            label: half.label(),
            direction: opposite(half.direction()),
        };
        let theirs = last
            .theirs
            .get_or_else(other, theirs_key, || self.present(other, theirs_key));
        let pair = self.pair_with_theirs(half, own, other, theirs_key, theirs);
        let halves = super::pair::order(half, pair.other);
        let old = if halves.canonical() == half {
            self.read_value(&pair.own, half.label(), half.position())
        } else {
            self.read_value(&pair.theirs, pair.other.label(), pair.other.position())
        };
        let Some(width) = self.value_width(half.label(), edge.value)? else {
            return Ok(SetValue::Kept(old));
        };
        let mut targets = [(half, pair.own), (pair.other, pair.theirs)];
        let count = if pair.other == half { 1 } else { 2 };
        self.edges
            .reserve((2 + values::PAGES_PER_WRITE) * u32::try_from(count).unwrap_or(2))?;
        for (h, found) in targets.iter_mut().take(count) {
            let updated = self.write_value(*found, (width, h.position()), edge.value);
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

    /// Writes `value` at `position` of `found`, giving the bucket a value plane if it has none.
    /// Within reserved memory.
    fn write_value(
        &mut self,
        found: Found,
        (width, position): (Width, Position),
        value: &[u8],
    ) -> Found {
        let mut bucket = if found.bucket.cells().is_none() {
            self.with_values(found.bucket, width)
        } else {
            found.bucket
        };
        if let Some(cells) = bucket.cells() {
            let plane = reserved(values::write(
                &mut self.edges,
                cells.plane,
                width,
                position.index(),
                value,
            ));
            bucket.set_plane(plane);
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
        let Some(cells) = found.bucket.cells() else {
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
}
