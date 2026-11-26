//! Rewriting a bucket's values in their label's current layout (`Graph::migrate_values`), a page
//! at a time, while reads keep the old values until the last step.
//!
//! Values beside a run's slots first move into a value plane of their own, in their old version.
//! A plane's values are then rewritten into a new plane, one value page of it per step, through
//! the caller's transform. A record in a 16-byte extent, which the bucket's header points to,
//! holds the new plane, its width and version, and the first position not yet rewritten. Reads
//! use the old plane throughout, so a bucket's values are in one version at a time; the last step
//! puts the new plane in its place and releases the old one and the record.
//!
//! The bucket takes no writes meanwhile (`Error::StaleValues`), and it keeps its positions:
//! compaction cancels a migration under way before it moves anything, and the next migration
//! starts over.
//!
//! # Record
//!
//! ```text
//! ---------------------------------------- <- Offset 0
//! New value plane root            ↕ 6 bytes   (see `bucket::values`)
//! ----------------------------------------
//! New width                       ↕ 2 bytes
//! ----------------------------------------
//! New value version               ↕ 1 byte
//! ----------------------------------------
//! First position not rewritten    ↕ 4 bytes
//! ----------------------------------------
//! Reserved space                  ↕ 3 bytes
//! ----------------------------------------
//! ```

use crate::bucket::descriptor::Descriptor;
use crate::bucket::stored::{migration_record, set_migration_record};
use crate::bucket::values::{self, Plane, ROOT_BYTES};
use crate::bucket::{LabelBucket, Migration, Schema, Values, reserved, slots};
use crate::edge::EdgeStore;
use crate::error::Error;
use crate::id::{ValueVersion, Width};
use crate::outcome::Progress;
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use crate::store::page_store::{ByteClass, Extent, Pages};
use alloc::vec;
#[cfg(test)]
use alloc::vec::Vec;

/// Bytes of a record.
const RECORD_BYTES: usize = ROOT_BYTES + 2 + 1 + 4;
const _: () = assert!(RECORD_BYTES <= 16);

/// A migration under way: the new plane and its layout, and how far it has got.
#[derive(Clone, Copy)]
struct Record {
    plane: Plane,
    schema: Schema,
    next: u32,
}

impl Record {
    fn read<M: Memory>(view: &Pages<'_, M>, at: Extent) -> Self {
        let bytes: [u8; RECORD_BYTES] = view.read_array(at.offset());
        let (root, rest) = bytes.split_at(ROOT_BYTES);
        let [w0, w1, version, n0, n1, n2, n3] =
            <[u8; 7]>::try_from(rest).unwrap_or_else(|_| corrupt("migration record"));
        Self {
            plane: Plane::decode(
                root.try_into()
                    .unwrap_or_else(|_| corrupt("migration record")),
            ),
            schema: Schema {
                width: Width::new(u16::from_le_bytes([w0, w1]))
                    .unwrap_or_else(|| corrupt("a migration to no width")),
                version: ValueVersion::new(version),
            },
            next: u32::from_le_bytes([n0, n1, n2, n3]),
        }
    }

    fn write<M: Memory, D: Memory>(
        self,
        store: &crate::store::page_store::PageStore<M, D>,
        at: Extent,
    ) {
        let mut bytes = [0u8; RECORD_BYTES];
        let (root, rest) = bytes.split_at_mut(ROOT_BYTES);
        root.copy_from_slice(&self.plane.encode());
        let [w0, w1] = self.schema.width.get().to_le_bytes();
        let [n0, n1, n2, n3] = self.next.to_le_bytes();
        rest.copy_from_slice(&[w0, w1, self.schema.version.get(), n0, n1, n2, n3]);
        store.write(at.offset(), &bytes);
    }
}

impl<M: Memory, D: Memory> EdgeStore<M, D> {
    /// One step of rewriting `bucket`'s values in `schema`, the label's current layout, through
    /// `transform` (from an old value to a new one, given zeroed): moving values beside a run's
    /// slots into a plane, or rewriting one value page of the new plane, the last step putting it
    /// in place. Returns the bucket, to publish, and whether more remains. On error nothing
    /// changed.
    pub(crate) fn migrate(
        &mut self,
        bucket: LabelBucket,
        schema: Schema,
        transform: &mut impl FnMut(&[u8], &mut [u8]),
    ) -> Result<(LabelBucket, Progress), Error> {
        let Some(cells) = bucket.cells() else {
            return Ok((bucket, Progress::Done));
        };
        if cells.migration == Migration::Idle && cells.is_current(schema.version) {
            return Ok((bucket, Progress::Done));
        }
        if let (Descriptor::Run(run), Plane::Beside { .. }) = (bucket.slots, cells.plane) {
            // A run of the slots alone, a plane of the values, and the header that holds it.
            self.store.reserve(3)?;
            let (slots, plane) = reserved(slots::values_out(run, &mut self.store));
            let values = Values {
                plane,
                width: cells.width,
                version: cells.version_given(schema.version),
                migration: Migration::Idle,
            };
            let bucket = LabelBucket {
                slots,
                values: Some(values),
                ..bucket
            };
            return Ok((bucket, Progress::Remaining));
        }
        // A record, and one value page of the new plane with its path.
        self.store.reserve(1 + values::PAGES_PER_WRITE)?;
        let header = bucket
            .header
            .unwrap_or_else(|| corrupt("values in a plane without a header"));
        let (at, record) = self.record((cells, header), schema);
        let stored = bucket.slots.stored();
        let record = self.rewrite_page(record, (cells, stored), transform);
        if record.next < stored {
            record.write(&self.store, at);
            let values = Values {
                migration: Migration::UnderWay,
                ..cells
            };
            return Ok((
                LabelBucket {
                    values: Some(values),
                    ..bucket
                },
                Progress::Remaining,
            ));
        }
        values::release(&mut self.store, cells.plane);
        self.store.release(at);
        set_migration_record(&self.store, header, None);
        let values = Values {
            plane: record.plane,
            width: schema.width,
            version: schema.version,
            migration: Migration::Idle,
        };
        Ok((
            LabelBucket {
                values: Some(values),
                ..bucket
            },
            Progress::Done,
        ))
    }

    /// The record of `cells`, whose bucket `header` heads, and where it is: the one under way, or
    /// a fresh one, which the header then points to, when none is or it was for an older layout
    /// (its plane released). Within reserved memory.
    fn record(&mut self, (cells, header): (Values, Extent), schema: Schema) -> (Extent, Record) {
        let fresh = Record {
            plane: Plane::Empty,
            schema,
            next: 0,
        };
        let under_way = match cells.migration {
            Migration::Idle => None,
            Migration::UnderWay => migration_record(&self.store.pages(), header),
        };
        match under_way {
            None => {
                let at = reserved(self.store.allocate(ByteClass::SMALLEST));
                set_migration_record(&self.store, header, Some(at));
                (at, fresh)
            }
            Some(at) => {
                let record = Record::read(&self.store.pages(), at);
                if record.schema == schema {
                    return (at, record);
                }
                values::release(&mut self.store, record.plane);
                (at, fresh)
            }
        }
    }

    /// `record` one value page of the new plane further: the old values of its positions, read
    /// from `cells`' plane, transformed and written. Within reserved memory.
    fn rewrite_page(
        &mut self,
        mut record: Record,
        (cells, stored): (Values, u32),
        transform: &mut impl FnMut(&[u8], &mut [u8]),
    ) -> Record {
        let per_page = values::cells_per_page(record.schema.width);
        let end = (record.next / per_page + 1)
            .saturating_mul(per_page)
            .min(stored);
        if record.next >= end {
            return record;
        }
        let old = values::read_cells(
            &self.store.pages(),
            cells.plane,
            cells.width,
            (record.next, end),
        );
        let (old_width, new_width) = (
            usize::from(cells.width.get()),
            usize::from(record.schema.width.get()),
        );
        let mut new = vec![0u8; old.len() / old_width * new_width];
        for (from, to) in old
            .chunks_exact(old_width)
            .zip(new.chunks_exact_mut(new_width))
        {
            transform(from, to);
        }
        record.plane = reserved(values::write_cells(
            &mut self.store,
            record.plane,
            record.schema.width,
            record.next,
            &new,
        ));
        record.next = end;
        record
    }

    /// `bucket`'s values without the migration under way, if any: its record and new plane
    /// released, and the header pointing at none.
    pub(crate) fn cancel_migration(&mut self, bucket: &LabelBucket) -> Option<Values> {
        let values = bucket.values?;
        let Some(header) = bucket.header else {
            return Some(values);
        };
        if let Some(at) = migration_record(&self.store.pages(), header) {
            let record = Record::read(&self.store.pages(), at);
            values::release(&mut self.store, record.plane);
            self.store.release(at);
            set_migration_record(&self.store, header, None);
        }
        Some(Values {
            migration: Migration::Idle,
            ..values
        })
    }
}

/// The extents a migration under way owns: its record and its new plane's (test invariant
/// checks).
#[cfg(test)]
pub(crate) fn extents<M: Memory>(view: &Pages<'_, M>, bucket: &LabelBucket) -> Vec<Extent> {
    let Some(at) = bucket.header.and_then(|h| migration_record(view, h)) else {
        return Vec::new();
    };
    let mut out = values::extents(view, Record::read(view, at).plane);
    out.push(at);
    out
}
