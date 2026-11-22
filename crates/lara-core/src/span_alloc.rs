//! The slab span allocator: external boundary tags for O(1) coalescing, TLSF classes for O(1)
//! good fit, all of it persisted in its own [`Memory`] so reopening reads only the header.
//!
//! - Allocation takes the low end of a fitting free segment (low addresses first, never a roving
//!   pointer), or extends the slab at the frontier.
//! - Release coalesces with both neighbours immediately. It never creates a record, so it never
//!   grows memory and cannot fail; only allocation creates records.
//! - The allocator manages slot numbers only. Backing the slab itself is its owner's job.
//!
//! See `DESIGN.md` for why this structure was chosen.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "measurement-only allocator: reopen and recovery run in its tests"
    )
)]

mod class;
mod tag;

use crate::memory::{Memory, Region, WriteError};
use crate::units::{ByteOffset, Slot, SlotRange, SpanLen};
use class::{CLASS_COUNT, Class, FL_COUNT};
pub(crate) use tag::TagId;
use tag::{RECORD_BYTES, Record, Segment, SegmentUse, corrupt, decode_link, encode_link};

const MAGIC: [u8; 3] = *b"LSA";
const VERSION: u8 = 1;

// Header layout. Records start on the second page of 4 KiB so the header can grow.
const AT_MAGIC: u64 = 0;
const AT_FRONTIER: u64 = 8;
const AT_FIRST: u64 = 16;
const AT_LAST: u64 = 20;
const AT_RECORDS: u64 = 24;
const AT_RECYCLED: u64 = 28;
const AT_FIRST_LEVEL: u64 = 32;
const AT_SECOND_LEVEL: u64 = 36;
#[expect(
    clippy::as_conversions,
    reason = "lossless u32 to u64 widening; `From` is not const"
)]
const AT_HEADS: u64 = AT_SECOND_LEVEL + 4 * FL_COUNT as u64;
#[expect(
    clippy::as_conversions,
    reason = "lossless u32 to u64 widening; `From` is not const"
)]
const HEADER_END: u64 = AT_HEADS + 4 * CLASS_COUNT as u64;
const RECORDS_START: u64 = 4096;
const _: () = assert!(HEADER_END <= RECORDS_START);

/// Ownership of one allocated span. Not `Clone`: it is handed back to [`TagAllocator::release`]
/// exactly once. Its owner persists [`SpanTag::id`] and recovers the tag after a reopen with
/// [`TagAllocator::recover`].
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SpanTag {
    id: TagId,
    range: SlotRange,
}

impl SpanTag {
    pub(crate) const fn id(&self) -> TagId {
        self.id
    }

    pub(crate) const fn range(&self) -> SlotRange {
        self.range
    }
}

/// Opening an allocator memory failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpenError {
    /// The memory does not hold an allocator.
    NotAnAllocator,
    UnsupportedVersion(u8),
}

/// Creating an allocator failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CreateError {
    /// The memory already holds data; open it instead.
    NotEmpty,
    Write(WriteError),
}

/// An in-place extension was blocked by an allocated neighbour (or the address space); the
/// span is unchanged and its tag is handed back.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Blocked {
    tag: SpanTag,
}

impl Blocked {
    pub(crate) fn into_tag(self) -> SpanTag {
        self.tag
    }
}

/// Recovering a tag by id failed: the id does not name an allocated segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NotAllocated;

/// The span allocator over one [`Memory`].
pub(crate) struct TagAllocator<M: Memory> {
    memory: M,
}

impl<M: Memory> TagAllocator<M> {
    /// Formats an empty memory.
    pub(crate) fn create(memory: M) -> Result<Self, CreateError> {
        if memory.size() != 0 {
            return Err(CreateError::NotEmpty);
        }
        let region = Region::new(&memory);
        region
            .reserve(ByteOffset::new(RECORDS_START))
            .map_err(CreateError::Write)?;
        let mut magic = [0u8; 4];
        for (dst, src) in magic.iter_mut().zip(MAGIC.iter().chain(&[VERSION])) {
            *dst = *src;
        }
        region.overwrite(ByteOffset::new(AT_MAGIC), &magic);
        // Every link field starts as "none"; counters and bitmaps start at zero.
        for at in [AT_FIRST, AT_LAST, AT_RECYCLED] {
            region.overwrite(ByteOffset::new(at), &encode_link(None));
        }
        for index in 0..CLASS_COUNT {
            region.overwrite(head_offset_at(index), &encode_link(None));
        }
        Ok(Self { memory })
    }

    /// Opens a memory written by [`Self::create`]. Reads the header only.
    pub(crate) fn open(memory: M) -> Result<Self, OpenError> {
        if memory.size() == 0 {
            return Err(OpenError::NotAnAllocator);
        }
        let [m0, m1, m2, version] = Region::new(&memory).read_array(ByteOffset::new(AT_MAGIC));
        if [m0, m1, m2] != MAGIC {
            return Err(OpenError::NotAnAllocator);
        }
        if version != VERSION {
            return Err(OpenError::UnsupportedVersion(version));
        }
        Ok(Self { memory })
    }

    /// The memory the allocator lives in (measurement only).
    #[cfg(feature = "bench-internals")]
    pub(crate) const fn memory(&self) -> &M {
        &self.memory
    }

    /// Bytes of tag records ever created (measurement only).
    #[cfg(feature = "bench-internals")]
    pub(crate) fn tag_bytes(&self) -> u64 {
        u64::from(self.record_count()) * RECORD_BYTES
    }

    /// One past the highest slot ever handed out. The owner keeps the slab backed up to here.
    pub(crate) fn frontier(&self) -> Slot {
        Slot::new(u64::from_le_bytes(
            self.region().read_array(ByteOffset::new(AT_FRONTIER)),
        ))
    }

    /// Allocates `len` slots: from the best-fitting free segment, else at the frontier.
    pub(crate) fn allocate(&mut self, len: SpanLen) -> Result<SpanTag, WriteError> {
        match self.find_free(len) {
            Some(id) => self.allocate_from(id, len),
            None => self.allocate_at_frontier(len),
        }
    }

    /// Returns a span and coalesces it with free neighbours. Never grows memory.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "`SpanTag` is an ownership token: taking it by value is what ends the allocation"
    )]
    pub(crate) fn release(&mut self, tag: SpanTag) {
        let SpanTag {
            id: released,
            range,
        } = tag;
        let mut seg = self.segment(released);
        if seg.use_ != SegmentUse::Allocated || seg.range() != range {
            corrupt("released tag does not match its segment");
        }
        let mut id = released;
        if let Some(before) = seg.before {
            let prev = self.segment(before);
            if let (true, Some(len)) = (prev.is_free(), prev.len.join(seg.len)) {
                self.unlink_free(prev);
                self.recycle(id);
                seg = Segment {
                    start: prev.start,
                    len,
                    before: prev.before,
                    ..seg
                };
                id = before;
            }
        }
        if let Some(after) = seg.after {
            let next = self.segment(after);
            if let (true, Some(len)) = (next.is_free(), seg.len.join(next.len)) {
                self.unlink_free(next);
                self.recycle(after);
                seg = Segment {
                    len,
                    after: next.after,
                    ..seg
                };
            }
        }
        self.set_neighbours(id, seg);
        self.push_free(id, seg);
    }

    /// Grows an allocated span in place to `len`, without moving it: into the free segment right
    /// after it, or past the frontier when the span (or the free segment after it) ends there.
    /// Never creates a record. A `len` that is not larger returns the tag unchanged.
    pub(crate) fn try_extend(&mut self, tag: SpanTag, len: SpanLen) -> Result<SpanTag, Blocked> {
        let seg = self.segment(tag.id);
        if seg.use_ != SegmentUse::Allocated || seg.range() != tag.range {
            corrupt("extended tag does not match its segment");
        }
        let Some(extra) = len.get().checked_sub(seg.len.get()).and_then(SpanLen::new) else {
            return Ok(tag);
        };
        let grown = Segment { len, ..seg };
        let next = seg.after.map(|id| (id, self.segment(id)));
        match next {
            None => {
                let Ok(frontier) = self.frontier().advance(extra.count()) else {
                    return Err(Blocked { tag });
                };
                self.write_segment(tag.id, grown);
                self.set_frontier(frontier);
            }
            Some((next_id, next)) if next.is_free() && next.len >= extra => {
                self.unlink_free(next);
                if let Some(rest_len) = SpanLen::new(next.len.get() - extra.get()) {
                    // The neighbour keeps its record and shrinks from the front.
                    let rest_start = grown.range().end();
                    let rest = Segment {
                        start: rest_start,
                        len: rest_len,
                        ..next
                    };
                    self.push_free(next_id, rest);
                    self.write_segment(tag.id, grown);
                } else {
                    // The neighbour is consumed exactly.
                    self.recycle(next_id);
                    let grown = Segment {
                        after: next.after,
                        ..grown
                    };
                    self.set_neighbours(tag.id, grown);
                    self.write_segment(tag.id, grown);
                }
            }
            // A free tail shorter than needed: absorb it and extend past the frontier.
            Some((next_id, next)) if next.is_free() && next.after.is_none() => {
                let short = extra.count().saturating_sub(next.len.count());
                let Ok(frontier) = self.frontier().advance(short) else {
                    return Err(Blocked { tag });
                };
                self.unlink_free(next);
                self.recycle(next_id);
                let grown = Segment {
                    after: None,
                    ..grown
                };
                self.set_neighbours(tag.id, grown);
                self.write_segment(tag.id, grown);
                self.set_frontier(frontier);
            }
            Some(_) => return Err(Blocked { tag }),
        }
        Ok(SpanTag {
            id: tag.id,
            range: grown.range(),
        })
    }

    /// The tag of an allocated segment, for an owner that persisted `id`.
    pub(crate) fn recover(&self, id: TagId) -> Result<SpanTag, NotAllocated> {
        if id.get() >= self.record_count() {
            return Err(NotAllocated);
        }
        match self.record(id) {
            Record::Segment(seg) if seg.use_ == SegmentUse::Allocated => Ok(SpanTag {
                id,
                range: seg.range(),
            }),
            Record::Segment(_) | Record::Recycled { .. } => Err(NotAllocated),
        }
    }

    // --- allocation paths --------------------------------------------------------------------

    /// A free segment of at least `len`: the head of `len`'s own class if it fits, else the
    /// head of the lowest class that guarantees a fit.
    fn find_free(&self, len: SpanLen) -> Option<TagId> {
        let own = self.head(Class::containing(len));
        if let Some(id) = own.filter(|&id| self.segment(id).len >= len) {
            return Some(id);
        }
        let from = Class::guaranteeing(len)?;
        let first = self.read_u32(AT_FIRST_LEVEL);
        let found = class::find_at_or_above(first, from, |fl| self.second_level(fl))?;
        self.head(found)
    }

    fn allocate_from(&mut self, id: TagId, len: SpanLen) -> Result<SpanTag, WriteError> {
        let seg = self.segment(id);
        let rest = seg.len.get() - len.get();
        // Take the record for the remainder before changing anything, so a failure leaves the
        // allocator untouched.
        let spare = match SpanLen::new(rest) {
            Some(rest) => Some((self.new_record()?, rest)),
            None => None,
        };
        self.unlink_free(seg);
        let mut taken = Segment {
            len,
            use_: SegmentUse::Allocated,
            ..seg
        };
        if let Some((rest_id, rest_len)) = spare {
            let rest_start = taken.range().end();
            let rest = Segment {
                start: rest_start,
                len: rest_len,
                before: Some(id),
                after: seg.after,
                use_: SegmentUse::Allocated,
            };
            taken.after = Some(rest_id);
            self.set_neighbours(rest_id, rest);
            self.push_free(rest_id, rest);
        }
        self.write_segment(id, taken);
        Ok(SpanTag {
            id,
            range: taken.range(),
        })
    }

    fn allocate_at_frontier(&mut self, len: SpanLen) -> Result<SpanTag, WriteError> {
        let frontier = self.frontier();
        // A free segment touching the frontier is extended rather than left behind.
        let last = self.link(AT_LAST).map(|id| (id, self.segment(id)));
        if let Some((id, seg)) = last.filter(|(_, seg)| seg.is_free()) {
            // The class search can miss a fitting segment that is not at its list's head.
            if seg.len >= len {
                return self.allocate_from(id, len);
            }
            let grow = len.count().saturating_sub(seg.len.count());
            let new_frontier = frontier.advance(grow)?;
            self.unlink_free(seg);
            let taken = Segment {
                len,
                use_: SegmentUse::Allocated,
                ..seg
            };
            self.write_segment(id, taken);
            self.set_frontier(new_frontier);
            return Ok(SpanTag {
                id,
                range: taken.range(),
            });
        }
        let new_frontier = frontier.advance(len.count())?;
        let id = self.new_record()?;
        let seg = Segment {
            start: frontier,
            len,
            before: last.map(|(id, _)| id),
            after: None,
            use_: SegmentUse::Allocated,
        };
        self.set_neighbours(id, seg);
        self.write_segment(id, seg);
        self.set_frontier(new_frontier);
        Ok(SpanTag {
            id,
            range: seg.range(),
        })
    }

    // --- records -----------------------------------------------------------------------------

    fn record_count(&self) -> u32 {
        self.read_u32(AT_RECORDS)
    }

    /// A record for a new segment: a recycled one, else a fresh one (which may grow memory).
    fn new_record(&mut self) -> Result<TagId, WriteError> {
        if let Some(id) = self.link(AT_RECYCLED) {
            let Record::Recycled { next } = self.record(id) else {
                corrupt("recycled list holds a live segment")
            };
            self.set_link(AT_RECYCLED, next);
            return Ok(id);
        }
        let count = self.record_count();
        let id = TagId::new(count);
        let end = record_offset(id).advance(RECORD_BYTES)?;
        self.region().reserve(end)?;
        self.write_u32(AT_RECORDS, count + 1);
        Ok(id)
    }

    fn recycle(&mut self, id: TagId) {
        let next = self.link(AT_RECYCLED);
        self.write_record(id, Record::Recycled { next });
        self.set_link(AT_RECYCLED, Some(id));
    }

    fn record(&self, id: TagId) -> Record {
        Record::decode(self.region().read_array(record_offset(id)))
    }

    fn segment(&self, id: TagId) -> Segment {
        match self.record(id) {
            Record::Segment(seg) => seg,
            Record::Recycled { .. } => corrupt("link to a recycled record"),
        }
    }

    fn write_record(&self, id: TagId, record: Record) {
        self.region().overwrite(record_offset(id), &record.encode());
    }

    fn write_segment(&self, id: TagId, seg: Segment) {
        self.write_record(id, Record::Segment(seg));
    }

    /// Points `seg`'s address-order neighbours (or the first/last links) at `id`.
    fn set_neighbours(&self, id: TagId, seg: Segment) {
        match seg.before {
            Some(before) => self.update(before, |s| s.after = Some(id)),
            None => self.set_link(AT_FIRST, Some(id)),
        }
        match seg.after {
            Some(after) => self.update(after, |s| s.before = Some(id)),
            None => self.set_link(AT_LAST, Some(id)),
        }
    }

    fn update(&self, id: TagId, edit: impl FnOnce(&mut Segment)) {
        let mut seg = self.segment(id);
        edit(&mut seg);
        self.write_segment(id, seg);
    }

    // --- free lists --------------------------------------------------------------------------

    fn push_free(&self, id: TagId, seg: Segment) {
        let class = Class::containing(seg.len);
        let next = self.head(class);
        if let Some(next) = next {
            self.set_free_links(next, |prev, _| *prev = Some(id));
        }
        self.write_segment(
            id,
            Segment {
                use_: SegmentUse::Free { prev: None, next },
                ..seg
            },
        );
        self.set_head(class, Some(id));
    }

    fn unlink_free(&self, seg: Segment) {
        let SegmentUse::Free { prev, next } = seg.use_ else {
            corrupt("unlinking an allocated segment")
        };
        let class = Class::containing(seg.len);
        match prev {
            Some(prev) => self.set_free_links(prev, |_, n| *n = next),
            None => self.set_head(class, next),
        }
        if let Some(next) = next {
            self.set_free_links(next, |p, _| *p = prev);
        }
    }

    fn set_free_links(&self, id: TagId, edit: impl FnOnce(&mut Option<TagId>, &mut Option<TagId>)) {
        self.update(id, |seg| match &mut seg.use_ {
            SegmentUse::Free { prev, next } => edit(prev, next),
            SegmentUse::Allocated => corrupt("free list links an allocated segment"),
        });
    }

    fn head(&self, class: Class) -> Option<TagId> {
        decode_link(self.region().read_array(head_offset_at(class.index())))
    }

    /// Sets a free-list head and keeps the occupancy bitmaps in step with it.
    fn set_head(&self, class: Class, head: Option<TagId>) {
        self.region()
            .overwrite(head_offset_at(class.index()), &encode_link(head));
        let fl = class.first_level();
        let bit = 1u32 << class.second_level();
        let second = match head {
            Some(_) => self.second_level(fl) | bit,
            None => self.second_level(fl) & !bit,
        };
        self.write_u32(second_level_offset(fl), second);
        let first = self.read_u32(AT_FIRST_LEVEL);
        let first = if second == 0 {
            first & !(1 << fl)
        } else {
            first | (1 << fl)
        };
        self.write_u32(AT_FIRST_LEVEL, first);
    }

    fn second_level(&self, fl: u32) -> u32 {
        self.read_u32(second_level_offset(fl))
    }

    // --- header fields -----------------------------------------------------------------------

    fn region(&self) -> Region<'_, M> {
        Region::new(&self.memory)
    }

    fn set_frontier(&self, frontier: Slot) {
        self.region()
            .overwrite(ByteOffset::new(AT_FRONTIER), &frontier.get().to_le_bytes());
    }

    fn link(&self, at: u64) -> Option<TagId> {
        decode_link(self.region().read_array(ByteOffset::new(at)))
    }

    fn set_link(&self, at: u64, link: Option<TagId>) {
        self.region()
            .overwrite(ByteOffset::new(at), &encode_link(link));
    }

    fn read_u32(&self, at: u64) -> u32 {
        u32::from_le_bytes(self.region().read_array(ByteOffset::new(at)))
    }

    fn write_u32(&self, at: u64, value: u32) {
        self.region()
            .overwrite(ByteOffset::new(at), &value.to_le_bytes());
    }
}

fn record_offset(id: TagId) -> ByteOffset {
    ByteOffset::new(RECORDS_START + u64::from(id.get()) * RECORD_BYTES)
}

fn head_offset_at(index: u32) -> ByteOffset {
    ByteOffset::new(AT_HEADS + 4 * u64::from(index))
}

fn second_level_offset(fl: u32) -> u64 {
    AT_SECOND_LEVEL + 4 * u64::from(fl)
}

#[cfg(test)]
mod tests;
