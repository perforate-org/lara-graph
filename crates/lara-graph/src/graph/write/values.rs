//! Writing edge values, one edge or a batch that is all-or-none.

use crate::bucket::LabelBucketKey;
use crate::bucket::slots;
use crate::bucket::{LabelBucket, Migration, Schema};
use crate::edge::Writes;
use crate::error::{BatchError, Error};
use crate::graph::GraphMemories;
use crate::graph::locate::Found;
use crate::graph::locate::LastBucket;
use crate::graph::locate::Pair;
use crate::graph::{Graph, corrupt};
use crate::id::{Adjacency, EdgeRef, Halves, NodeId};
use crate::id::{EdgeLabel, Position, ValueVersion, Width};
use crate::input::EdgeValue;
use crate::outcome::Progress;
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

impl<S: GraphMemories> Graph<S> {
    /// Replaces the value of the edge `half` belongs to, in both halves; whether `half` is live,
    /// and so had a value to replace. A bucket created before its label's width was declared gets
    /// its value plane here.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], [`Error::WrongValueLength`], [`Error::OutOfMemory`]. On error
    /// nothing changes.
    pub fn set_edge_value(&mut self, half: EdgeRef, value: &[u8]) -> Result<bool, Error> {
        let width = self.value_schema(half.label(), value)?;
        let Some(pair) = self.pair(half)? else {
            return Ok(false);
        };
        let Some(width) = width else {
            return Ok(true);
        };
        let mut targets = vec![(half, pair.own)];
        if pair.other != half {
            targets.push((pair.other, pair.theirs));
        }
        for (h, found) in &targets {
            Self::ensure_current(*h, &found.bucket, width)?;
        }
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
        Ok(true)
    }

    /// Writes each edge value in `values`, in order, and returns for each whether its edge was
    /// live ([`Self::set_edge_value`]). All nodes and value lengths are checked before the first
    /// write, and a failure after that restores every value written so far, so the batch is
    /// all-or-none.
    ///
    /// # Errors
    ///
    /// As [`Self::set_edge_value`], for the value that fails.
    pub fn set_edge_values(&mut self, values: &[EdgeValue<'_>]) -> Result<Vec<bool>, BatchError> {
        for (index, edge) in values.iter().enumerate() {
            self.check(edge.half().node())
                .and_then(|()| self.value_schema(edge.half().label(), edge.value()))
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
        Ok(out.iter().map(Option::is_some).collect())
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
        let Some(width) = self.value_schema(half.label(), edge.value())? else {
            return Ok(SetValue::Kept(old));
        };
        if width.version != ValueVersion::FIRST {
            Self::ensure_pair_current(half, &pair, width)?;
        }
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
        let schema = self
            .widths
            .schema(half.label())
            .unwrap_or_else(|| corrupt("a value being restored without a width"));
        let targets = [(half, pair.own), (pair.other, pair.theirs)];
        let count = if pair.other == half { 1 } else { 2 };
        for (h, found) in targets.iter().take(count) {
            self.write_value(*found, (schema, h.position()), old);
        }
    }
}

impl<S: GraphMemories> Graph<S> {
    /// [`Self::ensure_current`] for both buckets of `half`'s edge, or its one bucket for a loop.
    /// Kept out of line and called only for a changed label: checked over the array of targets
    /// instead, the value write built that array early, 340 bytes of copies per write (batched
    /// `set_edge_value` 9 %).
    #[cold]
    #[inline(never)]
    fn ensure_pair_current(half: EdgeRef, pair: &Pair, schema: Schema) -> Result<(), Error> {
        Self::ensure_current(half, &pair.own.bucket, schema)?;
        if pair.other != half {
            Self::ensure_current(pair.other, &pair.theirs.bucket, schema)?;
        }
        Ok(())
    }

    /// Fails with [`Error::StaleValues`] when `bucket`, `half`'s, holds values in an older
    /// version than `schema`'s or is being migrated: they take no writes until migrated.
    pub(in crate::graph) fn ensure_current(
        half: EdgeRef,
        bucket: &LabelBucket,
        schema: Schema,
    ) -> Result<(), Error> {
        // A label never changed has every value in its first version.
        if schema.version == ValueVersion::FIRST {
            return Ok(());
        }
        match bucket.cells() {
            Some(cells)
                if !cells.is_current(schema.version) || cells.migration == Migration::UnderWay =>
            {
                Err(Error::StaleValues {
                    node: half.node(),
                    label: half.label(),
                    direction: half.direction(),
                })
            }
            Some(_) | None => Ok(()),
        }
    }

    /// `label`'s width and value version, checking that `value` matches the width.
    pub(in crate::graph) fn value_schema(
        &self,
        label: EdgeLabel,
        value: &[u8],
    ) -> Result<Option<Schema>, Error> {
        self.widths.check(label, value)
    }

    /// Writes `value` at `position` of `found`, giving the bucket a value plane if it has none.
    /// Within reserved memory.
    pub(in crate::graph) fn write_value(
        &mut self,
        found: Found,
        (schema, position): (Schema, Position),
        value: &[u8],
    ) -> Found {
        let bucket = self
            .edges
            .put_value(found.bucket, (schema, position), value);
        let bucket = self.publish(found.entry, bucket);
        Found {
            entry: found.entry,
            bucket,
        }
    }
}

impl<S: GraphMemories> Graph<S> {
    /// Declares that `label`'s edges carry a value of `width` bytes, on every side. Constant
    /// time: existing edges read as zeros until a value is written to them. Declaring the same
    /// width again changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::WidthConflict`] if `label` already has another width, [`Error::OutOfMemory`]. On
    /// error nothing changes.
    pub fn declare_width(&mut self, label: EdgeLabel, width: Width) -> Result<(), Error> {
        self.declare(label, width)
    }

    /// Gives `label`'s values a new layout of `width` bytes (the same width included, for a
    /// layout that changes only its meaning) and returns its value version: the next one, or the
    /// first if it had no width. Constant time: the buckets that hold values keep them in their
    /// old version, still read in it ([`Self::bucket_value_version`]), and take no writes
    /// ([`Error::StaleValues`]) until [`Self::migrate_values`] rewrites them; the others read zeros
    /// of the new width.
    ///
    /// A small bucket records only its version's low three bits, so a label changes width at most
    /// seven more times before every bucket of its earlier version is migrated.
    ///
    /// # Errors
    ///
    /// [`Error::AddressSpaceExhausted`] past 255 changes, or [`Error::OutOfMemory`]. On error
    /// nothing changes.
    pub fn change_width(&mut self, label: EdgeLabel, width: Width) -> Result<ValueVersion, Error> {
        let (version, root) = self
            .widths
            .change(&mut self.edges.lists_mut(), label, width)?;
        self.nodes.set_widths_root(root);
        Ok(version)
    }

    /// One bounded step of rewriting the values of `v`'s `adjacency` bucket in `label`'s
    /// current layout ([`Self::change_width`]): `transform` turns each old value (in the bucket's
    /// version, [`Self::bucket_value_version`]) into a new one, given zeroed and of the new width.
    /// A step rewrites at most one value page (small buckets first move their values beside the
    /// slots into a plane), and reads keep the old values until the last step puts the new ones in
    /// place, so the bucket is in one version throughout. It takes no writes meanwhile
    /// ([`Error::StaleValues`]); compacting it starts the migration over.
    ///
    /// [`Progress::Done`] once nothing is left: the bucket holds no values, or holds them in the
    /// current version; a missing bucket or a label without a width is done too. The bucket at
    /// each end of an edge is migrated apart, so both halves' values go through `transform`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], or [`Error::OutOfMemory`]. On error nothing changed.
    pub fn migrate_values(
        &mut self,
        v: NodeId,
        adjacency: Adjacency,
        mut transform: impl FnMut(&[u8], &mut [u8]),
    ) -> Result<Progress, Error> {
        let (label, direction) = (adjacency.label(), adjacency.direction());
        let Some(schema) = self.widths.schema(label) else {
            self.check(v)?;
            return Ok(Progress::Done);
        };
        let Some(found) = self.find_label_bucket(v, LabelBucketKey { label, direction })? else {
            return Ok(Progress::Done);
        };
        let (bucket, progress) = self.edges.migrate(found.bucket, schema, &mut transform)?;
        if bucket != found.bucket {
            self.publish(found.entry, bucket);
        }
        Ok(progress)
    }

    fn declare(&mut self, label: EdgeLabel, width: Width) -> Result<(), Error> {
        if let Some(root) = self
            .widths
            .declare(&mut self.edges.lists_mut(), label, width)?
        {
            self.nodes.set_widths_root(root);
        }
        Ok(())
    }
}
