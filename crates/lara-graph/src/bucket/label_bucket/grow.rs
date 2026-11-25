//! Appending at the end of a bucket: a tiny bucket becomes a run, a run grows a class or becomes
//! the first data page, a paged bucket fills its last page or opens a new one.

use super::{byte_len, first_tombstone, narrow, position, slot_offset, stored_u32};
use crate::bucket::descriptor::{
    Descriptor, PAGE_SLOTS, Paged, Run, Slot, TINY_SLOTS, Tiny, run_class,
};
use crate::bucket::paged::{self, Leaf};
use crate::id::{NodeId, Position, Width};
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::PageStore;
use crate::store::units::Overflow;

// --- runs --------------------------------------------------------------------------------------

#[inline]
pub(super) fn tiny_to_run<M: Memory>(
    tiny: Tiny,
    edge: Slot,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Position), WriteError> {
    let stored = TINY_SLOTS + 1;
    let extent = store.allocate(run_class(stored_u32(stored), None))?;
    let mut bytes = [0u8; (TINY_SLOTS + 1) * 4];
    for (dst, slot) in bytes
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(tiny.slots().iter().chain([&edge]))
    {
        *dst = slot.encode();
    }
    store.write(extent.offset(), &bytes);
    let live = Descriptor::Tiny(tiny).live() + 1;
    let run = Run {
        extent,
        stored: narrow(stored_u32(stored)),
        live: narrow(live),
        values: None,
    };
    Ok((Descriptor::Run(run), position(TINY_SLOTS)))
}

#[inline]
pub(super) fn run_append<M: Memory>(
    run: Run,
    target: NodeId,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Position), WriteError> {
    let edge = Slot::Live(target);
    let stored = u32::from(run.stored);
    let live = run.live + 1;
    let next = Run {
        stored: run.stored + 1,
        live,
        ..run
    };
    if stored < run.capacity() {
        store.write(slot_offset(run.extent.offset(), stored), &edge.encode());
        return Ok((Descriptor::Run(next), Position::new(stored)));
    }
    if run.can_grow() {
        // Move to the next class. One allocation: a failure leaves nothing behind.
        let extent = store.allocate(run_class(stored + 1, run.values))?;
        let grown = Run { extent, ..next };
        let bytes = store.pages().read(run.extent.offset(), byte_len(stored));
        store.write(extent.offset(), &bytes);
        store.write(slot_offset(extent.offset(), stored), &edge.encode());
        if let Some(width) = run.values {
            let cells = store
                .pages()
                .read(run.cells_offset(), cells_len(stored, width));
            store.write(grown.cells_offset(), &cells);
        }
        store.release(run.extent);
        return Ok((Descriptor::Run(grown), Position::new(stored)));
    }
    if run.values.is_some() {
        corrupt("a full run with values beside its slots");
    }
    // A full page: it becomes data page 0 as it is, and the new edge opens page 1.
    let tombstones = narrow(stored - u32::from(run.live));
    let first_hole = if tombstones == 0 {
        paged::NO_HOLE
    } else {
        let all = 0..PAGE_SLOTS;
        first_tombstone(&store.pages(), run.extent.page(), all).map_or(paged::NO_HOLE, narrow)
    };
    let first = Leaf::data(run.extent.page(), tombstones, first_hole);
    let (index, page) = paged::start(store, first)?;
    store.write(page.offset(), &edge.encode());
    let paged = Paged {
        index,
        stored: stored + 1,
        live: u32::from(live),
    };
    Ok((Descriptor::Paged(paged), Position::new(stored)))
}

/// Bytes of `cells` values of `width`.
pub(super) fn cells_len(cells: u32, width: Width) -> usize {
    usize::try_from(cells * u32::from(width.get())).unwrap_or_else(|_| corrupt("cells length"))
}

// --- pages -------------------------------------------------------------------------------------

pub(super) fn paged_append<M: Memory>(
    paged: Paged,
    target: NodeId,
    store: &mut PageStore<M>,
) -> Result<(Descriptor, Position), WriteError> {
    let edge = Slot::Live(target);
    let live = paged
        .live
        .checked_add(1)
        .ok_or(WriteError::Overflow(Overflow))?;
    let last = paged::data_pages(paged.stored) - 1;
    let first = last * PAGE_SLOTS;
    let in_last = paged.stored - first;
    let leaf = paged::data_page(&store.pages(), &paged, last);
    // The last data page is never hollow: compaction keeps it.
    let last_page = leaf.present();
    let stored = paged
        .stored
        .checked_add(1)
        .ok_or(WriteError::Overflow(Overflow))?;
    let at = Position::new(paged.stored);
    if in_last < PAGE_SLOTS {
        store.write(slot_offset(last_page.offset(), in_last), &edge.encode());
        return Ok((
            Descriptor::Paged(Paged {
                stored,
                live,
                ..paged
            }),
            at,
        ));
    }
    let (index, page) = paged::append_page(store, &paged)?;
    store.write(page.offset(), &edge.encode());
    Ok((
        Descriptor::Paged(Paged {
            index,
            stored,
            live,
        }),
        at,
    ))
}
