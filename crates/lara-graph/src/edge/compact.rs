//! One bounded compaction step of a bucket: its slots, and the values and target index that
//! follow them.

use crate::bucket::compact::{self, Inline, Outcome};
use crate::bucket::descriptor::{Descriptor, PAGE_SLOTS};
use crate::bucket::reserved;
use crate::bucket::target_index::{self, Change, Merged, Plan};
use crate::bucket::values;
use crate::bucket::{Index, LabelBucket, Values};
use crate::edge::EdgeStore;
use crate::error::Error;
use crate::outcome::{Move, Progress};
use crate::store::memory::Memory;
use alloc::vec::Vec;

/// What a compaction step left of a bucket.
pub(crate) enum Compacted {
    /// Nothing changed.
    Unchanged,
    /// The bucket, to publish.
    Kept(LabelBucket),
    /// No position is left: the bucket's storage is released and its list entry is to go.
    Emptied,
}

impl<M: Memory> EdgeStore<M> {
    /// One bounded compaction step of `bucket`. Plans everything that allocates before it changes
    /// anything, so on error nothing changed.
    pub(crate) fn compact(
        &mut self,
        bucket: LabelBucket,
    ) -> Result<(Compacted, Vec<Move>, Progress), Error> {
        let planned = compact::plan(bucket.slots, &self.store);
        // With nothing to do in the slots, a paged bucket's index gives back sparse leaves.
        if planned.is_idle()
            && matches!(bucket.slots, Descriptor::Paged(_))
            && let Some(root) = bucket.index.tree()
            && let Merged::Into(root) = target_index::merge_step(&mut self.store, root)
        {
            let bucket = LabelBucket {
                index: Index::Tree(root),
                ..bucket
            };
            return Ok((Compacted::Kept(bucket), Vec::new(), Progress::Remaining));
        }
        // An idle step changes nothing, so it reserves and writes nothing: before, the last call
        // of every compaction loop still reserved value pages and rewrote the bucket. A run that
        // still carries a dropped index goes on, so that its header is released.
        if planned.is_idle()
            && (matches!(bucket.slots, Descriptor::Paged(_))
                || matches!(bucket.index, Index::Empty))
        {
            return Ok((Compacted::Unchanged, Vec::new(), Progress::Done));
        }
        let plan = self.plan_index(&bucket, &planned)?;
        // Values beside a run's slots move with them; values in a plane move below.
        let inline = match (bucket.values, bucket.cells()) {
            (Some(values), _) => {
                // Moved values may land in value pages not yet allocated.
                self.store.reserve(values::pages_per_moves(values.width))?;
                Inline::Forbidden
            }
            (None, Some(_)) => Inline::Forbidden,
            (None, None) => Inline::Allowed,
        };
        let step = compact::apply(planned, &mut self.store, inline)?;
        let old_end = bucket.slots.stored();
        let values = bucket
            .values
            .map(|cells| self.move_values(cells, &step, old_end));
        let compacted = match step.outcome {
            Outcome::LabelBucket(slots) => {
                let kept = LabelBucket {
                    slots,
                    values,
                    ..bucket
                };
                Compacted::Kept(self.reindex(kept, plan))
            }
            Outcome::Empty => {
                let gone = self.drop_index(bucket, Index::Empty);
                if let Some(cells) = values {
                    values::release(&mut self.store, cells.plane);
                }
                if let Some(header) = gone.header {
                    self.store.release(header);
                }
                Compacted::Emptied
            }
        };
        Ok((compacted, step.moves, step.progress))
    }

    /// A compacted bucket's index: the plan committed while it stays paged, dropped when it was
    /// demoted. Within reserved memory.
    fn reindex(&mut self, bucket: LabelBucket, plan: Option<Plan>) -> LabelBucket {
        if !matches!(bucket.slots, Descriptor::Paged(_)) {
            return self.drop_index(bucket, Index::Empty);
        }
        let Some(plan) = plan else {
            return bucket;
        };
        LabelBucket {
            index: Index::of(reserved(target_index::commit(&mut self.store, plan))),
            ..bucket
        }
    }

    /// The plan of the index changes a step's moves make in a bucket that stays paged, with its
    /// pages reserved.
    fn plan_index(
        &mut self,
        bucket: &LabelBucket,
        step: &compact::Planned,
    ) -> Result<Option<Plan>, Error> {
        if bucket.indexed().is_none() || !step.stays_paged() || step.moves().is_empty() {
            return Ok(None);
        }
        let changes: Vec<Change> = step
            .moves()
            .iter()
            .flat_map(|m| Change::moved(m.target(), (m.from(), m.to())))
            .collect();
        let plan = target_index::plan(&self.store.pages(), bucket.index.tree(), &changes);
        self.store.reserve(plan.pages())?;
        Ok(Some(plan))
    }

    /// Moves the values of moved edges, and releases the value pages of positions that hold no
    /// edges any more: past the new end, or in a released slot page.
    fn move_values(&mut self, cells: Values, step: &compact::Step, old_end: u32) -> Values {
        let mut plane = cells.plane;
        // Moves go to lower positions in ascending order, so no source is overwritten first.
        for m in &step.moves {
            let pair = (m.from().index(), m.to().index());
            plane = reserved(values::copy(&mut self.store, plane, cells.width, pair));
        }
        if let Outcome::LabelBucket(slots) = &step.outcome {
            let ranges = [
                (slots.stored() < old_end).then_some((slots.stored(), u32::MAX)),
                step.hollowed
                    .map(|page| (page * PAGE_SLOTS, (page + 1) * PAGE_SLOTS)),
            ];
            for range in ranges.into_iter().flatten() {
                plane = values::release_range(&mut self.store, plane, cells.width, range);
            }
        }
        Values { plane, ..cells }
    }
}
