//! One half going into its bucket: where it goes, then the slot, its value and its index
//! posting. The caller keeps the bucket's list entry and node record in step.

use crate::bucket::LabelBucketKey;
use crate::bucket::descriptor::Descriptor;
use crate::bucket::reserved;
use crate::bucket::slots::{self, Spot};
use crate::bucket::stored::Stored;
use crate::bucket::{LabelBucket, Migration, Schema, Values};
use crate::edge::EdgeView;
use crate::edge::{Batch, Between, EdgeStore};
use crate::error::Error;
use crate::id::{NodeId, Position};
use crate::input::Placement;
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;

/// A batch-local cursor into a paged bucket's holes. It is only used for fresh targets, so the
/// holes before `next` have been filled by this run.
pub(crate) struct HoleCursor {
    key: Option<(NodeId, LabelBucketKey)>,
    next: Position,
}

impl Default for HoleCursor {
    fn default() -> Self {
        Self {
            key: None,
            next: Position::new(0),
        }
    }
}

impl<M: Memory> EdgeStore<M> {
    /// The next hole `cursor` finds in `b`, the bucket `key` names.
    fn next_hole(
        &self,
        b: &LabelBucket,
        key: (NodeId, LabelBucketKey),
        cursor: &mut HoleCursor,
    ) -> Spot {
        if cursor.key != Some(key) {
            cursor.key = Some(key);
            cursor.next = Position::new(0);
        }
        let Some(position) = slots::hole_from(&b.slots, &self.store.pages(), cursor.next) else {
            cursor.next = Position::new(b.slots.stored());
            return Spot::End;
        };
        cursor.next = Position::new(position.index().saturating_add(1));
        Spot::Hole(position)
    }

    /// Puts an edge to `target` at its planned spot in the bucket `stored` (a new one for
    /// `None`), with `value` for a label with a width, and returns its position and the bucket,
    /// not yet published. Within reserved memory.
    pub(crate) fn insert(
        &mut self,
        stored: Option<Stored>,
        target: NodeId,
        (spot, batch): (Spot, Option<Batch>),
        value: Option<(Schema, &[u8])>,
    ) -> (Position, LabelBucket) {
        match (stored, value) {
            // A plain bucket of a label without values takes the slot alone.
            (Some(Stored::Plain(slots)), None) => self.insert_plain(slots, target, (spot, batch)),
            (None, None) => self.insert_plain(Descriptor::EMPTY, target, (spot, batch)),
            (stored, value) => self.insert_resolved(stored, target, (spot, batch), value),
        }
    }

    /// [`Self::insert`] into a bucket with a header, or of a label with values. Kept out of line,
    /// so that the plain path stays small: inlined, it cost each plain insert about 40
    /// instructions.
    #[inline(never)]
    fn insert_resolved(
        &mut self,
        stored: Option<Stored>,
        target: NodeId,
        (spot, batch): (Spot, Option<Batch>),
        value: Option<(Schema, &[u8])>,
    ) -> (Position, LabelBucket) {
        let mut bucket = match stored {
            Some(stored) => LabelBucket::resolve(&self.store.pages(), stored),
            None => LabelBucket::plain(Descriptor::EMPTY),
        };
        bucket = match value {
            // A bucket of a label with values but no value plane yet (new, or made before the
            // width was declared) gets one.
            Some((schema, _)) if bucket.cells().is_none() => self.with_values(bucket, schema),
            Some((schema, _)) if spot == Spot::End => self.room_to_append(bucket, schema),
            Some(_) | None => bucket,
        };
        let (slots, at) = reserved(slots::insert(bucket.slots, target, spot, &mut self.store));
        let was = core::mem::replace(&mut bucket.slots, slots);
        if let Some((schema, bytes)) = value {
            bucket = self.put_value(bucket, (schema, at), bytes);
        }
        (at, self.index_inserted(bucket, &was, (target, at), batch))
    }

    /// [`Self::insert`] into the slots of a plain bucket, which stay plain unless they become
    /// paged and take an index.
    fn insert_plain(
        &mut self,
        slots: Descriptor,
        target: NodeId,
        (spot, batch): (Spot, Option<Batch>),
    ) -> (Position, LabelBucket) {
        let (next, at) = reserved(slots::insert(slots, target, spot, &mut self.store));
        (
            at,
            self.index_inserted(LabelBucket::plain(next), &slots, (target, at), batch),
        )
    }

    /// `bucket` with room to append: a full page-sized run with values beside its slots moves them
    /// into a value plane, so that it can grow on into pages. Within reserved memory.
    fn room_to_append(&mut self, bucket: LabelBucket, schema: Schema) -> LabelBucket {
        let Descriptor::Run(run) = bucket.slots else {
            return bucket;
        };
        if !slots::full_beside(&bucket.slots) {
            return bucket;
        }
        let width = run
            .width()
            .unwrap_or_else(|| corrupt("a full run without values beside its slots"));
        let (slots, plane) = reserved(slots::values_out(run, &mut self.store));
        // Inserts reach only buckets in the label's current version.
        LabelBucket {
            slots,
            values: Some(Values {
                plane,
                width,
                version: schema.version,
                migration: Migration::Idle,
            }),
            ..bucket
        }
    }
}

impl<M: Memory> EdgeView<'_, M> {
    /// Where an edge to `target` goes in the bucket `stored`, and the block batch its append
    /// completes, if any; fails for a bucket without a free position. `bucket` names the bucket
    /// for `cursor`, which a run of fresh targets into one bucket shares.
    #[inline]
    pub(crate) fn plan(
        &self,
        stored: Stored,
        bucket: (NodeId, LabelBucketKey),
        (target, placement): (NodeId, Placement),
        between: Between,
        cursor: Option<&mut HoleCursor>,
    ) -> Result<(Spot, Option<Batch>), Error> {
        let b = LabelBucket::resolve(&self.pages(), stored);
        if b.slots.stored() == u32::MAX {
            return Err(Error::AddressSpaceExhausted);
        }
        let spot = match cursor {
            Some(cursor)
                if placement == Placement::ReuseHoles
                    && between == Between::Nothing
                    && matches!(b.slots, Descriptor::Paged(_)) =>
            {
                self.edges.next_hole(&b, bucket, cursor)
            }
            _ => self.edges.spot_for(&b, target, placement, between),
        };
        let batch = match spot {
            Spot::End => self.edges.batch_for(&b, target),
            Spot::Hole(_) => None,
        };
        Ok((spot, batch))
    }
}
