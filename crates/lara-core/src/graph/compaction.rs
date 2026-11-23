//! Compacting a bucket: its slots, and the values and target index that follow them.

use super::index::{indexed, moved};
use super::{Compaction, Error, Graph, Step, reserved};
use crate::compact::{self, Inline, Outcome};
use crate::corrupt::corrupt;
use crate::descriptor::{Descriptor, PAGE_SLOTS};
use crate::id::{Label, Progress, Side, VertexId};
use crate::labels::{self, BucketKey};
use crate::memory::Memory;
use crate::stored::{Bucket, Index, Values};
use crate::targets::{self, Change, Plan};
use crate::values;
use alloc::vec::Vec;

impl<M: Memory> Graph<M> {
    /// Reclaims removed edges' positions in `v`'s `(label, side)` bucket, one bounded step at a
    /// time: at most one page of edges moves. Live edges keep their relative order, so the
    /// halves stay paired; each one that moves is reported. Call again while
    /// [`Compaction::progress`] is [`Progress::Remaining`].
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`], or [`Error::OutOfMemory`] when a smaller run, moved values or the
    /// target index's changes cannot be allocated. On error the graph is unchanged.
    pub fn compact(&mut self, v: VertexId, label: Label, side: Side) -> Result<Compaction, Error> {
        let Some(found) = self.find_bucket(v, BucketKey { label, side })? else {
            return Ok(Step::new(Vec::new(), Progress::Done));
        };
        // Plan everything that allocates before the step changes anything.
        let planned = compact::plan(found.bucket.slots, &self.edges);
        let plan = self.plan_index(&found.bucket, &planned)?;
        let inline = match found.bucket.values {
            Some(values) => {
                // Moved values may land in value pages not yet allocated.
                self.edges.reserve(values::pages_per_moves(values.width))?;
                Inline::Forbidden
            }
            None => Inline::Allowed,
        };
        let step = compact::apply(planned, &mut self.edges, inline)?;
        let old_end = found.bucket.slots.stored();
        let values = found
            .bucket
            .values
            .map(|cells| self.move_values(cells, &step, old_end));
        match step.outcome {
            Outcome::Bucket(slots) => {
                let bucket = Bucket {
                    slots,
                    values,
                    ..found.bucket
                };
                let bucket = self.reindex(bucket, plan);
                self.publish(found.entry, bucket);
            }
            Outcome::Empty => {
                let bucket = self.drop_index(found.bucket, Index::Empty);
                if let Some(gone) = values {
                    values::release(&mut self.edges, gone.plane);
                }
                if let Some(header) = bucket.header {
                    self.edges.release(header);
                }
                let list = self
                    .list(v)?
                    .unwrap_or_else(|| corrupt("present bucket without a list"));
                let list = labels::remove(&mut self.lists, list, found.entry);
                self.set_list(v, list);
            }
        }
        Ok(Step::new(step.moves, step.progress))
    }

    /// A compacted bucket's index: the plan committed while it stays paged, dropped when it was
    /// demoted. Within reserved memory.
    fn reindex(&mut self, bucket: Bucket, plan: Option<Plan>) -> Bucket {
        if !matches!(bucket.slots, Descriptor::Paged(_)) {
            return self.drop_index(bucket, Index::Empty);
        }
        let Some(plan) = plan else {
            return bucket;
        };
        Bucket {
            index: Index::of(reserved(targets::commit(&mut self.edges, plan))),
            ..bucket
        }
    }

    /// The plan of the index changes a step's moves make in a bucket that stays paged, with its
    /// pages reserved.
    fn plan_index(
        &mut self,
        bucket: &Bucket,
        step: &compact::Planned,
    ) -> Result<Option<Plan>, Error> {
        if indexed(bucket).is_none() || !step.stays_paged() || step.moves().is_empty() {
            return Ok(None);
        }
        let changes: Vec<Change> = step
            .moves()
            .iter()
            .flat_map(|m| moved(m.target(), (m.from(), m.to())))
            .collect();
        let plan = targets::plan(&self.edges.pages(), bucket.index.tree(), &changes);
        self.edges.reserve(plan.pages())?;
        Ok(Some(plan))
    }

    /// Moves the values of moved edges, and releases the value pages of positions that hold no
    /// edges any more: past the new end, or in a released slot page.
    fn move_values(&mut self, cells: Values, step: &compact::Step, old_end: u32) -> Values {
        let mut plane = cells.plane;
        // Moves go to lower positions in ascending order, so no source is overwritten first.
        for m in &step.moves {
            let pair = (m.from().index(), m.to().index());
            plane = reserved(values::copy(&mut self.edges, plane, cells.width, pair));
        }
        if let Outcome::Bucket(slots) = &step.outcome {
            let ranges = [
                (slots.stored() < old_end).then_some((slots.stored(), u32::MAX)),
                step.hollowed
                    .map(|page| (page * PAGE_SLOTS, (page + 1) * PAGE_SLOTS)),
            ];
            for range in ranges.into_iter().flatten() {
                plane = values::release_range(&mut self.edges, plane, cells.width, range);
            }
        }
        Values { plane, ..cells }
    }
}
