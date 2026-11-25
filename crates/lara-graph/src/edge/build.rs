//! Building one label bucket whole from its halves in position order: a bulk load's.

use crate::bucket::descriptor::{self, Descriptor, PAGE_SLOTS, Paged, Run, Slot, TINY_SLOTS, Tiny};
use crate::bucket::paged::{self, Leaf};
use crate::bucket::slots;
use crate::bucket::stored::Stored;
use crate::bucket::target_index::{self, BLOCK, Change};
use crate::bucket::values::{self, Plane};
use crate::bucket::{Index, LabelBucket, Values};
use crate::edge::EdgeStore;
use crate::error::Error;
use crate::id::{NodeId, Width};
use crate::store::memory::Memory;
use crate::store::page_store::{ByteClass, PageStore};
use crate::store::units::ByteOffset;
use alloc::vec::Vec;

/// A half a bulk build puts in a bucket: the node its entry points at.
pub(crate) trait BuiltHalf {
    fn target(&self) -> NodeId;
}

impl<M: Memory> EdgeStore<M> {
    /// Writes a bucket of `halves`, with `value` of each when the label has a `width`, and
    /// returns its list entry.
    pub(crate) fn build_bucket<'e, H: BuiltHalf>(
        &mut self,
        halves: &[H],
        width: Option<Width>,
        value: impl Fn(&H) -> &'e [u8],
    ) -> Result<Stored, Error> {
        let count = u32::try_from(halves.len()).map_err(|_| Error::AddressSpaceExhausted)?;
        let tiny_slots = u32::try_from(TINY_SLOTS).unwrap_or(2);
        if width.is_none() && count <= tiny_slots {
            return Ok(tiny_bucket(halves));
        }
        // A small enough bucket keeps its values beside its slots; others in a value plane.
        let beside = width.filter(|&w| descriptor::beside(w) && count <= descriptor::max_beside(w));
        let mut bucket = self.build_slots(halves, count, beside)?;
        match (bucket.cells(), width) {
            (Some(cells), _) => {
                let bytes: Vec<u8> = halves
                    .iter()
                    .flat_map(|half| value(half).iter().copied())
                    .collect();
                if let (Descriptor::Run(run), Plane::Beside { .. }) = (bucket.slots, cells.plane) {
                    self.store.write(run.cells_offset(), &bytes);
                }
            }
            (None, Some(width)) => {
                let plane = self.build_values(halves, &value, width)?;
                bucket.values = Some(Values { plane, width });
            }
            (None, None) => {}
        }
        if matches!(bucket.slots, Descriptor::Paged(_)) {
            self.build_target_index(&mut bucket, halves, count)?;
        }
        if bucket.extended() {
            bucket.header = Some(self.store.allocate(crate::bucket::stored::HEADER_CLASS)?);
            bucket.write_header(&self.store);
        }
        Ok(bucket.stored())
    }

    /// Writes the slots of a bucket too large to be tiny: one run (with room for values of
    /// `beside` beside them if set), or pages.
    fn build_slots<H: BuiltHalf>(
        &mut self,
        halves: &[H],
        count: u32,
        beside: Option<Width>,
    ) -> Result<LabelBucket, Error> {
        if count <= PAGE_SLOTS {
            let extent = self.store.allocate(descriptor::run_class(count, beside))?;
            write_halves(&self.store, extent.offset(), halves);
            Ok(LabelBucket::plain(Descriptor::Run(Run {
                extent,
                stored: slots::narrow(count),
                live: slots::narrow(count),
                values: beside,
            })))
        } else {
            let paged = self.build_paged(halves)?;
            Ok(LabelBucket::plain(Descriptor::Paged(paged)))
        }
    }

    /// Writes each half's edge value at its position.
    fn build_values<'e, H: BuiltHalf>(
        &mut self,
        halves: &[H],
        value: &impl Fn(&H) -> &'e [u8],
        width: Width,
    ) -> Result<Plane, Error> {
        let mut plane = Plane::Empty;
        for (position, half) in halves.iter().enumerate() {
            plane = values::write(
                &mut self.store,
                plane,
                width,
                u32::try_from(position).map_err(|_| Error::AddressSpaceExhausted)?,
                value(half),
            )?;
        }
        Ok(plane)
    }

    /// Builds the target index of a paged bucket over its full blocks.
    #[expect(
        clippy::indexing_slicing,
        reason = "positions stay below the slot count"
    )]
    fn build_target_index<H: BuiltHalf>(
        &mut self,
        bucket: &mut LabelBucket,
        halves: &[H],
        count: u32,
    ) -> Result<(), Error> {
        let end = count / BLOCK * BLOCK;
        if end == 0 {
            return Ok(());
        }
        let changes: Vec<Change> = (0..end)
            .map(|at| Change {
                key: (
                    halves[usize::try_from(at).unwrap_or(0)].target().index(),
                    at / BLOCK,
                ),
                delta: 1,
            })
            .collect();
        let plan = target_index::plan(&self.store.pages(), None, &changes);
        self.store.reserve(plan.pages())?;
        bucket.index = Index::of(target_index::commit(&mut self.store, plan)?);
        Ok(())
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "page ranges come from the slot count"
    )]
    fn build_paged<H: BuiltHalf>(&mut self, halves: &[H]) -> Result<Paged, Error> {
        let count = u32::try_from(halves.len()).map_err(|_| Error::AddressSpaceExhausted)?;
        let page_slots = usize::try_from(PAGE_SLOTS).unwrap_or(1024);
        let first = self.store.allocate(ByteClass::PAGE)?.page();
        let (mut index, second) =
            paged::start(&mut self.store, Leaf::data(first, 0, paged::NO_HOLE))?;
        write_halves(&self.store, first.offset(), &halves[..page_slots]);
        write_halves(
            &self.store,
            second.offset(),
            &halves
                [page_slots..usize::try_from(count.min(2 * PAGE_SLOTS)).unwrap_or(2 * page_slots)],
        );
        let page_count = paged::data_pages(count);
        for page in 2..page_count {
            let before = page * PAGE_SLOTS;
            let paged = Paged {
                index,
                stored: before,
                live: before,
            };
            let (next, data) = paged::append_page(&mut self.store, &paged)?;
            write_halves(
                &self.store,
                data.offset(),
                &halves[usize::try_from(before).unwrap_or(0)
                    ..usize::try_from(count.min(before + PAGE_SLOTS)).unwrap_or(0)],
            );
            index = next;
        }
        Ok(Paged {
            index,
            stored: count,
            live: count,
        })
    }
}

/// A bucket of at most [`TINY_SLOTS`] halves without values, held inline.
fn tiny_bucket<H: BuiltHalf>(halves: &[H]) -> Stored {
    let mut tiny = Tiny::EMPTY;
    for half in halves {
        tiny = tiny.push(Slot::Live(half.target()));
    }
    LabelBucket::plain(Descriptor::Tiny(tiny)).stored()
}

#[expect(
    clippy::indexing_slicing,
    reason = "one page of slots bounds the slice"
)]
fn write_halves<M: Memory, H: BuiltHalf>(store: &PageStore<M>, at: ByteOffset, halves: &[H]) {
    let mut bytes = [0u8; 4096];
    let len = halves.len() * 4;
    for (dst, half) in bytes[..len].as_chunks_mut::<4>().0.iter_mut().zip(halves) {
        *dst = Slot::Live(half.target()).encode();
    }
    store.write(at, &bytes[..len]);
}
