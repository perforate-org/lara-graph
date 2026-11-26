//! The page store: every byte of adjacency lives in fixed 4096-byte pages.
//!
//! - **pages** holds a header, then page data at `64 + page_id × 4096`, so an extent reads as one
//!   contiguous range.
//! - **directory** holds a header, then one 32-byte record per page, addressed by page id:
//!   free-list links for free pages, and the class, pool, use count and slot bitmap for pages
//!   split into chunks (for the 16-byte class, a slab's free list instead: see [`Slots`]).
//!
//! Label buckets and the label bucket lists share the store. Whole pages serve either, but a
//! chunk page serves one [`Pool`]: list extents keep to pages of their own, so that the lists of
//! nodes added together stay near each other.
//!
//! Each memory starts as ic-stable-structures' memories do: a 3-byte magic and a version byte.
//!
//! An allocation is an [`Extent`] of a byte class: a power of two from 16 to 4096, or three
//! quarters of one from 48 to 768 (see [`ByteClass`]). Classes below a page are carved from chunk
//! pages (one class per page, a bitmap of slots); a full-page extent is the page itself. Every
//! operation is O(1). A page that empties goes back to one free list, so space freed by one class
//! or use serves any other.
//!
//! Allocation fails only when a memory cannot grow, and grows before changing anything. Release
//! never fails.
//!
//! Chosen over a TLSF span allocator with boundary tags (32 bytes per span) and over PMA leaves:
//! per-bucket runs are mostly 8-16 slots, and counted with their metadata the page store used the
//! least memory of the three, with fewer memory calls and the same copying.
//!
//! # Directory memory
//!
//! ```text
//! ---------------------------------------- <- Address 0
//! Magic "LPS"                     ↕ 3 bytes
//! ----------------------------------------
//! Layout version (1)              ↕ 1 byte
//! ----------------------------------------
//! Pages minted = N                ↕ 4 bytes
//! ----------------------------------------
//! Free list head (page id)        ↕ 4 bytes
//! ---------------------------------------- <- Address 12 (AT_PARTIAL_HEADS)
//! Partial list head, edges, 16 B  ↕ 4 bytes
//! ----------------------------------------
//! ...                                             (thirteen heads, one per chunk class)
//! ----------------------------------------
//! Partial list head, lists, 16 B  ↕ 4 bytes
//! ----------------------------------------
//! ...                                             (thirteen more, for the list pool)
//! ----------------------------------------
//! Reserved space                  ↕ 12 bytes
//! ---------------------------------------- <- Address 128 (RECORDS_START)
//! Record of page 0                ↕ 32 bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Record of page N - 1            ↕ 32 bytes
//! ----------------------------------------
//! Unallocated space
//! ```
//!
//! A page id of `u32::MAX` stands for none.
//!
//! # Pages memory
//!
//! ```text
//! ---------------------------------------- <- Address 0
//! Magic "LPD"                     ↕ 3 bytes
//! ----------------------------------------
//! Layout version (1)              ↕ 1 byte
//! ----------------------------------------
//! Reserved space                  ↕ 60 bytes
//! ---------------------------------------- <- Address 64 (PAGES_START)
//! Page 0                          ↕ 4096 bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Page N - 1                      ↕ 4096 bytes
//! ----------------------------------------
//! Unallocated space
//! ```
//!
//! # Extent
//!
//! Extent `index` of a class of `b` bytes in a page starts at `64 + page × 4096 + index × b` in
//! the pages memory. It encodes in 40 bits, little-endian:
//!
//! ```text
//! ---------------------------------------- <- Bit 0
//! Page id                         ↕ 27 bits       (512 GiB of pages per memory, past the IC's
//!                                                  stable memory)
//! ---------------------------------------- <- Bit 27
//! Index in the page, low 7 bits   ↕ 7 bits        (below the class's extents per page)
//! ---------------------------------------- <- Bit 34
//! Class code                      ↕ 4 bits        (0 and 1: 16 B, the code being the index's
//!                                                  8th bit; 2 to 14: the 32 B to 4 KiB classes)
//! ---------------------------------------- <- Bit 38
//! Spare, for the holder           ↕ 2 bits
//! ----------------------------------------
//! ```

mod class;
mod record;

use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, Region, WriteError};
use crate::store::txn::Allocator;
use crate::store::units::ByteOffset;
use alloc::vec::Vec;
use class::Code;
pub(crate) use class::{ByteClass, Slots};
use class::{SLAB_LINK_BYTES, decode_slab_link, encode_slab_link};
pub(crate) use record::Pool;
use record::{PageRecord, RECORD_BYTES, decode_link, encode_link};

/// Bytes per page.
pub(crate) const PAGE_BYTES: u32 = 4096;
/// Bytes a stack buffer covers in [`Pages::with_read`]: three label bucket list entries.
const SMALL_READ: usize = 64;

/// The directory's magic: "lara page store".
const DIRECTORY_MAGIC: [u8; 3] = *b"LPS";
/// The pages memory's magic: "lara page data".
const PAGES_MAGIC: [u8; 3] = *b"LPD";
const VERSION: u8 = 1;
const AT_MINTED: u64 = 4;
const AT_FREE_HEAD: u64 = 8;
const AT_PARTIAL_HEADS: u64 = 12;
/// Bytes the directory's header takes before its records: the partial list heads of both pools.
const RECORDS_START: u64 = 128;
/// Bytes the pages memory's header takes before its pages.
const PAGES_START: u64 = 64;

/// One page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PageId(u32);

impl PageId {
    pub(crate) const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub(crate) const fn get(self) -> u32 {
        self.0
    }

    /// Where the page's data starts.
    pub(crate) fn offset(self) -> ByteOffset {
        ByteOffset::new(PAGES_START + u64::from(self.0) * u64::from(PAGE_BYTES))
    }
}

/// An allocation: the `index`-th extent of `class` bytes in `page`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Extent {
    page: PageId,
    /// Where it starts in its page: its index times its class's bytes, so a read adds it alone.
    start: u16,
    class: ByteClass,
}

/// Bytes an encoded [`Extent`] takes: 40 bits, of which the extent uses the low 38.
pub(crate) const EXTENT_BYTES: usize = 5;
/// Bits of a page id in an encoded extent.
pub(crate) const PAGE_BITS: u32 = 27;
/// Bits of the index within the page: 128 extents of the 32-byte class. The 16-byte class's 256
/// take an eighth bit from the class code.
const INDEX_BITS: u32 = 7;
/// Bits of the class code.
const CLASS_BITS: u32 = 4;
/// Page ids an encoded extent can name: 512 GiB of pages per memory.
pub(crate) const MAX_PAGES: u32 = 1 << PAGE_BITS;
/// The top bits of an encoded extent's last byte, which the extent leaves zero for its holder (a
/// node record keeps the node's state there).
pub(crate) const EXTENT_SPARE_BITS: u8 = 0b1100_0000;
const _: () = assert!(PAGE_BITS + INDEX_BITS + CLASS_BITS + 2 == 40 && EXTENT_BYTES == 5);

impl Extent {
    /// The `index`-th extent of `class` in `page`.
    #[cfg(test)]
    pub(crate) fn new(page: PageId, index: u16, class: ByteClass) -> Self {
        if index >= class.per_page() {
            corrupt("extent index");
        }
        Self::at(page, index, class)
    }

    /// The `index`-th extent of `class` in `page`, `index` below the class's extents per page.
    fn at(page: PageId, index: u16, class: ByteClass) -> Self {
        let start = u32::from(index) * class.bytes();
        Self {
            page,
            start: u16::try_from(start).unwrap_or_else(|_| corrupt("extent start")),
            class,
        }
    }

    /// Which extent of its class in its page it is.
    fn index(self) -> u16 {
        let index = u32::from(self.start) / self.class.bytes();
        u16::try_from(index).unwrap_or(u16::MAX)
    }

    /// A whole page as an extent.
    pub(crate) const fn whole(page: PageId) -> Self {
        Self {
            page,
            start: 0,
            class: ByteClass::PAGE,
        }
    }

    pub(crate) const fn page(self) -> PageId {
        self.page
    }

    pub(crate) const fn class(self) -> ByteClass {
        self.class
    }

    pub(crate) fn len(self) -> u32 {
        self.class.bytes()
    }

    /// Where the extent starts in the pages memory.
    pub(crate) fn offset(self) -> ByteOffset {
        ByteOffset::new(self.page.offset().get() + u64::from(self.start))
    }

    /// As the module's Extent diagram shows.
    pub(crate) fn encode(self) -> [u8; EXTENT_BYTES] {
        let code = if self.class.is_slab() {
            u64::from(self.index() >> INDEX_BITS)
        } else {
            u64::from(self.class.0) + 1
        };
        let word = u64::from(self.page.0)
            | (u64::from(self.index() & ((1 << INDEX_BITS) - 1)) << PAGE_BITS)
            | (code << (PAGE_BITS + INDEX_BITS));
        let [b0, b1, b2, b3, b4, ..] = word.to_le_bytes();
        [b0, b1, b2, b3, b4]
    }

    /// Ignores the spare bits ([`EXTENT_SPARE_BITS`]). One table load, a minimum and a product,
    /// as every extent read goes through here; any 38 bits name a valid extent, so there is
    /// nothing to check.
    pub(crate) fn decode(bytes: [u8; EXTENT_BYTES]) -> Self {
        let [b0, b1, b2, b3, b4] = bytes;
        let word = u64::from_le_bytes([b0, b1, b2, b3, b4, 0, 0, 0]);
        let field = |at: u32, bits: u32| (word >> at) & ((1 << bits) - 1);
        let code = Code::of(field(PAGE_BITS + INDEX_BITS, CLASS_BITS));
        // An index past its class's last extent is no extent's: it clamps.
        let index =
            (u16::try_from(field(PAGE_BITS, INDEX_BITS)).unwrap_or(0) | code.high).min(code.last);
        Self {
            page: PageId(u32::try_from(field(0, PAGE_BITS)).unwrap_or(0)),
            start: index * code.bytes,
            class: code.class,
        }
    }
}

/// Opening a page store failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpenError {
    NotAPageStore,
    UnsupportedVersion(u8),
}

/// The store owns its memories exclusively, so it keeps two header values in the heap: how many
/// pages were ever taken, and how many both memories can hold without growing.
pub(crate) struct PageStore<M: Memory, D: Memory = M> {
    pages: M,
    directory: D,
    /// Mirrors the directory's minted count.
    minted: u32,
    /// Pages both memories hold at their current size: taking pages below this never grows.
    room: u32,
    /// The pool new chunk extents come from ([`Self::lists`] lends the list pool).
    pool: Pool,
}

impl<M: Memory, D: Memory> PageStore<M, D> {
    /// Formats two memories as an empty store, whatever they held: every header field is
    /// written, and nothing past the headers is read before it is written. Constant time.
    pub(crate) fn format(pages: M, directory: D) -> Result<Self, WriteError> {
        let region = Region::new(&directory);
        let [d0, d1, d2] = DIRECTORY_MAGIC;
        region.write(ByteOffset::new(0), &[d0, d1, d2, VERSION])?;
        region.overwrite(ByteOffset::new(AT_MINTED), &0u32.to_le_bytes());
        region.overwrite(ByteOffset::new(AT_FREE_HEAD), &encode_link(None));
        for lane in 0..Pool::COUNT * ByteClass::CHUNK_CLASSES {
            region.overwrite(partial_head_at(lane), &encode_link(None));
        }
        let [p0, p1, p2] = PAGES_MAGIC;
        Region::new(&pages).write(ByteOffset::new(0), &[p0, p1, p2, VERSION])?;
        Ok(Self::with_header(pages, directory, 0))
    }

    /// Opens memories written by [`Self::format`]. Reads the headers only.
    pub(crate) fn open(pages: M, directory: D) -> Result<Self, OpenError> {
        let version = header(&directory, DIRECTORY_MAGIC).ok_or(OpenError::NotAPageStore)?;
        if header(&pages, PAGES_MAGIC) != Some(version) {
            return Err(OpenError::NotAPageStore);
        }
        if version != VERSION {
            return Err(OpenError::UnsupportedVersion(version));
        }
        let minted =
            u32::from_le_bytes(Region::new(&directory).read_array(ByteOffset::new(AT_MINTED)));
        Ok(Self::with_header(pages, directory, minted))
    }

    /// The memories back, as they were given.
    pub(crate) fn into_memories(self) -> (M, D) {
        (self.pages, self.directory)
    }

    fn with_header(pages: M, directory: D, minted: u32) -> Self {
        let mut store = Self {
            pages,
            directory,
            minted,
            room: 0,
            pool: Pool::Edges,
        };
        store.measure_room();
        store
    }

    /// Recomputes [`Self::room`] from the memories' sizes.
    fn measure_room(&mut self) {
        let bytes = |pages: u64| pages.saturating_mul(crate::store::memory::PAGE_SIZE);
        let pages = bytes(self.pages.size()).saturating_sub(PAGES_START) / u64::from(PAGE_BYTES);
        let records = bytes(self.directory.size()).saturating_sub(RECORDS_START) / RECORD_BYTES;
        self.room = u32::try_from(pages.min(records)).unwrap_or(u32::MAX);
    }

    /// Read-only access to page data, without the directory: what a scan is given.
    pub(crate) fn pages(&self) -> Pages<'_, M> {
        Pages {
            memory: &self.pages,
        }
    }

    /// Writes inside an allocated extent. Never grows memory.
    pub(crate) fn write(&self, at: ByteOffset, bytes: &[u8]) {
        Region::new(&self.pages).overwrite(at, bytes);
    }

    /// Allocates an extent of `class`.
    pub(crate) fn allocate(&mut self, class: ByteClass) -> Result<Extent, WriteError> {
        if class.is_page() {
            let page = self.take_page()?;
            self.set_record(page, PageRecord::Whole);
            return Ok(Extent::whole(page));
        }
        let mut chunk = self.open_chunk(class)?;
        let index = self.take_slot(class, &mut chunk);
        self.store_chunk(class, &chunk);
        if chunk.used == class.per_page() {
            self.unlink_partial(lane(chunk.pool, class), chunk.page);
        }
        Ok(Extent::at(chunk.page, index, class))
    }

    /// Allocates up to `count` extents of `class`, batching the chunk-page bitmap updates.
    pub(crate) fn allocate_many(
        &mut self,
        class: ByteClass,
        count: u32,
    ) -> Result<Vec<Extent>, WriteError> {
        let mut out = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
        if class.is_page() {
            for _ in 0..count {
                out.push(self.allocate(class)?);
            }
            return Ok(out);
        }
        while below(&out, count) {
            self.fill_chunk(class, count, &mut out)?;
        }
        Ok(out)
    }

    /// Takes free extents of one partial (or fresh) chunk page into `out` until it holds `count`
    /// or the page is full, writing the page's record once.
    fn fill_chunk(
        &mut self,
        class: ByteClass,
        count: u32,
        out: &mut Vec<Extent>,
    ) -> Result<(), WriteError> {
        let mut chunk = self.open_chunk(class)?;
        let page = chunk.page;
        let mut taken = 0u32;
        while chunk.used < class.per_page() && below(out, count) {
            let index = self.take_slot(class, &mut chunk);
            out.push(Extent::at(page, index, class));
            taken += 1;
        }
        self.store_chunk(class, &chunk);
        if chunk.used >= class.per_page() {
            self.unlink_partial(lane(chunk.pool, class), page);
        }
        if taken == 0 {
            corrupt("chunk page without a free extent");
        }
        Ok(())
    }

    /// The head of `class`'s partial list, or a fresh empty chunk page pushed onto it, with its
    /// record read.
    #[inline]
    fn open_chunk(&mut self, class: ByteClass) -> Result<OpenChunk, WriteError> {
        let pool = self.pool;
        let page = if let Some(page) = self.partial_head(lane(pool, class)) {
            page
        } else {
            let page = self.take_page()?;
            let fresh = OpenChunk {
                page,
                pool,
                used: 0,
                prev: None,
                next: None,
                slots: Slots::fresh(class),
            };
            self.store_chunk(class, &fresh);
            self.push_partial(lane(pool, class), page);
            page
        };
        let PageRecord::Chunk {
            used,
            bitmap,
            prev,
            next,
            ..
        } = self.record(page)
        else {
            corrupt("partial list holds a page that is not a chunk")
        };
        Ok(OpenChunk {
            page,
            pool,
            used,
            prev,
            next,
            slots: Slots::decode(class, bitmap),
        })
    }

    /// Takes a free extent of an open chunk page that has one: the first clear bit, the slab's
    /// first freed extent (one two-byte read), or its next untouched one.
    fn take_slot(&self, class: ByteClass, chunk: &mut OpenChunk) -> u16 {
        let index = match &mut chunk.slots {
            Slots::Bitmap(bits) => {
                let index = bits.trailing_ones();
                *bits |= 1 << index;
                u16::try_from(index).unwrap_or(u16::MAX)
            }
            Slots::Slab {
                head: Some(index), ..
            } => {
                let index = *index;
                let link = Extent::at(chunk.page, index, class).offset();
                let next = decode_slab_link(u16::from_le_bytes(self.pages().read_array(link)));
                chunk.slots = Slots::Slab {
                    head: next,
                    bump: slab_bump(chunk.slots),
                };
                index
            }
            Slots::Slab { head: None, bump } => {
                *bump += 1;
                *bump - 1
            }
        };
        if index >= class.per_page() {
            corrupt("partial chunk has no free slot");
        }
        chunk.used += 1;
        index
    }

    /// Returns extent `index` of an open chunk page to its free slots.
    fn free_slot(&self, class: ByteClass, chunk: &mut OpenChunk, index: u16) {
        match &mut chunk.slots {
            Slots::Bitmap(bits) => {
                let bit = 1u128 << index;
                if *bits & bit == 0 {
                    corrupt("released extent is not allocated");
                }
                *bits &= !bit;
            }
            Slots::Slab { head, bump } => {
                if index >= *bump {
                    corrupt("released extent is not allocated");
                }
                let link = Extent::at(chunk.page, index, class).offset();
                let bytes: [u8; SLAB_LINK_BYTES] = encode_slab_link(*head).to_le_bytes();
                self.write(link, &bytes);
                *head = Some(index);
            }
        }
        chunk.used -= 1;
    }

    /// Writes back the record of a chunk page opened by [`Self::open_chunk`].
    #[inline]
    fn store_chunk(&self, class: ByteClass, chunk: &OpenChunk) {
        self.set_record(
            chunk.page,
            PageRecord::Chunk {
                class,
                pool: chunk.pool,
                used: chunk.used,
                prev: chunk.prev,
                next: chunk.next,
                bitmap: chunk.slots.encode(),
            },
        );
    }

    /// Grows both memories so that `pages` more pages can be taken without growing them: after
    /// this, allocations that take at most `pages` pages cannot fail. Changes no state.
    pub(crate) fn reserve(&mut self, pages: u32) -> Result<(), WriteError> {
        // Beyond the page ids an extent can name, nothing more is taken.
        let end = self
            .minted
            .checked_add(pages)
            .filter(|&end| end <= MAX_PAGES)
            .ok_or(WriteError::Overflow(crate::store::units::Overflow))?;
        if end > self.room {
            Region::new(&self.pages).reserve(PageId(end).offset())?;
            self.directory().reserve(record_at(PageId(end)))?;
            self.measure_room();
        }
        Ok(())
    }

    /// Returns an extent. A chunk page that empties goes back to the free pages.
    pub(crate) fn release(&mut self, extent: Extent) {
        let Extent { page, class, .. } = extent;
        let index = extent.index();
        match self.record(page) {
            PageRecord::Whole if class.is_page() => self.give_page(page),
            PageRecord::Chunk {
                class: c,
                pool,
                used,
                prev,
                next,
                bitmap,
            } if c == class => {
                let mut chunk = OpenChunk {
                    page,
                    pool,
                    used,
                    prev,
                    next,
                    slots: Slots::decode(class, bitmap),
                };
                let was_full = used == class.per_page();
                self.free_slot(class, &mut chunk, index);
                self.store_chunk(class, &chunk);
                let used = chunk.used;
                if used == 0 {
                    if !was_full {
                        self.unlink_partial(lane(pool, class), page);
                    }
                    self.give_page(page);
                } else if was_full {
                    self.push_partial(lane(pool, class), page);
                }
            }
            PageRecord::Free { .. } | PageRecord::Whole | PageRecord::Chunk { .. } => {
                corrupt("released extent does not match its page")
            }
        }
    }

    // --- pages -------------------------------------------------------------------------------

    /// Pages ever taken from the memory.
    pub(crate) const fn minted(&self) -> u32 {
        self.minted
    }

    /// A free page, or a new one. Grows both memories before changing anything.
    fn take_page(&mut self) -> Result<PageId, WriteError> {
        if let Some(page) = self.link(AT_FREE_HEAD) {
            let PageRecord::Free { next } = self.record(page) else {
                corrupt("free list holds a page in use")
            };
            self.set_link(AT_FREE_HEAD, next);
            return Ok(page);
        }
        let page = PageId(self.minted);
        self.reserve(1)?;
        let next = self.minted + 1;
        self.directory()
            .overwrite(ByteOffset::new(AT_MINTED), &next.to_le_bytes());
        self.minted = next;
        Ok(page)
    }

    fn give_page(&mut self, page: PageId) {
        let next = self.link(AT_FREE_HEAD);
        self.set_record(page, PageRecord::Free { next });
        self.set_link(AT_FREE_HEAD, Some(page));
    }

    // --- partial chunk lists -----------------------------------------------------------------

    fn partial_head(&self, lane: u64) -> Option<PageId> {
        decode_link(self.directory().read_array(partial_head_at(lane)))
    }

    fn set_partial_head(&self, lane: u64, head: Option<PageId>) {
        self.directory()
            .overwrite(partial_head_at(lane), &encode_link(head));
    }

    fn push_partial(&self, lane: u64, page: PageId) {
        let head = self.partial_head(lane);
        if let Some(head) = head {
            self.relink(head, |prev, _| *prev = Some(page));
        }
        self.relink(page, |prev, next| {
            *prev = None;
            *next = head;
        });
        self.set_partial_head(lane, Some(page));
    }

    fn unlink_partial(&self, lane: u64, page: PageId) {
        let PageRecord::Chunk { prev, next, .. } = self.record(page) else {
            corrupt("unlinking a page that is not a chunk")
        };
        match prev {
            Some(prev) => self.relink(prev, |_, n| *n = next),
            None => self.set_partial_head(lane, next),
        }
        if let Some(next) = next {
            self.relink(next, |p, _| *p = prev);
        }
        self.relink(page, |p, n| {
            *p = None;
            *n = None;
        });
    }

    fn relink(&self, page: PageId, edit: impl FnOnce(&mut Option<PageId>, &mut Option<PageId>)) {
        let PageRecord::Chunk {
            class,
            pool,
            used,
            mut prev,
            mut next,
            bitmap,
        } = self.record(page)
        else {
            corrupt("partial list links a page that is not a chunk")
        };
        edit(&mut prev, &mut next);
        self.set_record(
            page,
            PageRecord::Chunk {
                class,
                pool,
                used,
                prev,
                next,
                bitmap,
            },
        );
    }

    // --- directory ---------------------------------------------------------------------------

    fn directory(&self) -> Region<'_, D> {
        Region::new(&self.directory)
    }

    fn record(&self, page: PageId) -> PageRecord {
        if page.0 >= self.minted() {
            corrupt("page beyond the minted range");
        }
        PageRecord::decode(self.directory().read_array(record_at(page)))
    }

    fn set_record(&self, page: PageId, record: PageRecord) {
        self.directory()
            .overwrite(record_at(page), &record.encode());
    }

    fn link(&self, at: u64) -> Option<PageId> {
        decode_link(self.directory().read_array(ByteOffset::new(at)))
    }

    fn set_link(&self, at: u64, link: Option<PageId>) {
        self.directory()
            .overwrite(ByteOffset::new(at), &encode_link(link));
    }
}

/// Read-only access to page data. Holds no directory, so a reader cannot consult allocation
/// metadata.
pub(crate) struct Pages<'a, M: Memory> {
    memory: &'a M,
}

impl<M: Memory> Pages<'_, M> {
    /// Reads `len` bytes at `at`, with one memory read.
    pub(crate) fn read(&self, at: ByteOffset, len: usize) -> Vec<u8> {
        let mut buf = alloc::vec![0u8; len];
        Region::new(self.memory).read(at, &mut buf);
        buf
    }

    pub(crate) fn read_array<const N: usize>(&self, at: ByteOffset) -> [u8; N] {
        Region::new(self.memory).read_array(at)
    }

    /// Calls `f` with `len` bytes read at `at`: from a small stack buffer when they fit (the usual
    /// short label bucket list), else from the heap.
    pub(crate) fn with_read<R>(&self, at: ByteOffset, len: usize, f: impl FnOnce(&[u8]) -> R) -> R {
        // Never zeroed: the read fills it in place (see `Memory::read_uninit`).
        let mut small = [core::mem::MaybeUninit::<u8>::uninit(); SMALL_READ];
        if let Some(buf) = small.get_mut(..len) {
            return f(self.read_uninit(at, buf));
        }
        f(&self.read(at, len))
    }

    /// Fills `buf` from `at`, with one memory read and no allocation.
    pub(crate) fn read_into(&self, at: ByteOffset, buf: &mut [u8]) {
        Region::new(self.memory).read(at, buf);
    }

    /// Fills `buf`, which need not be initialized, from `at`, and returns it.
    pub(crate) fn read_uninit<'b>(
        &self,
        at: ByteOffset,
        buf: &'b mut [core::mem::MaybeUninit<u8>],
    ) -> &'b mut [u8] {
        Region::new(self.memory).read_uninit(at, buf)
    }
}

impl<M: Memory, D: Memory> PageStore<M, D> {
    /// The store, allocating chunk extents from the list pool until the guard drops.
    pub(crate) fn lists(&mut self) -> ListPool<'_, M, D> {
        self.pool = Pool::Lists;
        ListPool { store: self }
    }
}

/// A page store lent for label bucket lists and node label data: its chunk extents come from
/// pages of their own ([`Pool::Lists`]).
pub(crate) struct ListPool<'a, M: Memory, D: Memory> {
    store: &'a mut PageStore<M, D>,
}

impl<M: Memory, D: Memory> core::ops::Deref for ListPool<'_, M, D> {
    type Target = PageStore<M, D>;

    fn deref(&self) -> &Self::Target {
        self.store
    }
}

impl<M: Memory, D: Memory> core::ops::DerefMut for ListPool<'_, M, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.store
    }
}

impl<M: Memory, D: Memory> Drop for ListPool<'_, M, D> {
    fn drop(&mut self) {
        self.store.pool = Pool::Edges;
    }
}

impl<M: Memory, D: Memory> Allocator for PageStore<M, D> {
    type Request = ByteClass;
    type Extent = Extent;
    type Error = WriteError;

    fn allocate(&mut self, class: ByteClass) -> Result<Extent, WriteError> {
        Self::allocate(self, class)
    }

    fn release(&mut self, extent: Extent) {
        Self::release(self, extent);
    }
}

#[cfg(test)]
impl<M: Memory, D: Memory> PageStore<M, D> {
    /// The allocated extents' indexes of chunk `page` of `class` (test invariant checks).
    pub(crate) fn allocated_in(&self, page: PageId, class: ByteClass, slots: Slots) -> Vec<u16> {
        match slots {
            Slots::Bitmap(bits) => (0..class.per_page())
                .filter(|&i| bits & (1 << i) != 0)
                .collect(),
            Slots::Slab { head, bump } => {
                let mut free = alloc::collections::BTreeSet::new();
                let mut cursor = head;
                while let Some(index) = cursor {
                    assert!(index < bump && free.insert(index), "slab free list");
                    let link = Extent::at(page, index, class).offset();
                    cursor = decode_slab_link(u16::from_le_bytes(self.pages().read_array(link)));
                }
                (0..bump).filter(|i| !free.contains(i)).collect()
            }
        }
    }

    /// Every allocated extent (test invariant checks).
    pub(crate) fn allocated(&self) -> Vec<Extent> {
        let mut out = Vec::new();
        for page in (0..self.minted()).map(PageId) {
            match self.record(page) {
                PageRecord::Free { .. } => {}
                PageRecord::Whole => out.push(Extent::whole(page)),
                PageRecord::Chunk { class, bitmap, .. } => {
                    let used = self.allocated_in(page, class, Slots::decode(class, bitmap));
                    out.extend(used.into_iter().map(|index| Extent::at(page, index, class)));
                }
            }
        }
        out
    }
}

/// The version a memory's header gives, when it starts with `magic`; `None` for an empty memory or
/// another magic.
pub(crate) fn header<M: Memory>(memory: &M, magic: [u8; 3]) -> Option<u8> {
    if memory.size() == 0 {
        return None;
    }
    let [m0, m1, m2, version] = Region::new(memory).read_array(ByteOffset::new(0));
    ([m0, m1, m2] == magic).then_some(version)
}

fn record_at(page: PageId) -> ByteOffset {
    ByteOffset::new(RECORDS_START + u64::from(page.0) * RECORD_BYTES)
}

/// A chunk page's record while extents are taken from it.
struct OpenChunk {
    page: PageId,
    pool: Pool,
    used: u16,
    prev: Option<PageId>,
    next: Option<PageId>,
    slots: Slots,
}

/// A slab's bump index (0 for a bitmap).
const fn slab_bump(slots: Slots) -> u16 {
    match slots {
        Slots::Slab { bump, .. } => bump,
        Slots::Bitmap(_) => 0,
    }
}

/// Whether `out` holds fewer than `count` extents.
fn below(out: &[Extent], count: u32) -> bool {
    u32::try_from(out.len()).unwrap_or(u32::MAX) < count
}

const _: () =
    assert!(AT_PARTIAL_HEADS + 4 * Pool::COUNT * ByteClass::CHUNK_CLASSES <= RECORDS_START);

/// The partial list of `pool`'s chunk pages of `class`.
fn lane(pool: Pool, class: ByteClass) -> u64 {
    u64::from(pool.id()) * ByteClass::CHUNK_CLASSES + class.lane()
}

fn partial_head_at(lane: u64) -> ByteOffset {
    ByteOffset::new(AT_PARTIAL_HEADS + 4 * lane)
}

#[cfg(test)]
mod tests;

/// Proofs over every input, checked by Kani (`cargo kani -p lara-graph`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Any five bytes decode to an extent, and it lies within its page.
    #[kani::proof]
    fn any_bytes_decode_to_an_extent_within_its_page() {
        let extent = Extent::decode(kani::any());
        let len = u64::from(extent.len());
        assert!(u64::from(extent.start) + len <= u64::from(PAGE_BYTES));
        assert_eq!(u64::from(extent.start), u64::from(extent.index()) * len);
    }

    /// An extent encodes to bytes that decode back to it.
    #[kani::proof]
    fn extents_round_trip() {
        let extent = Extent::decode(kani::any());
        assert_eq!(Extent::decode(extent.encode()), extent);
    }
}
