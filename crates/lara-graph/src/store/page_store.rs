//! The page store: every byte of adjacency lives in fixed 4096-byte pages.
//!
//! - **pages** holds a header, then page data at `64 + page_id × 4096`, so an extent reads as one
//!   contiguous range.
//! - **directory** holds a header, then one 32-byte record per page, addressed by page id:
//!   free-list links for free pages, and the class, use count and slot bitmap for pages split
//!   into chunks.
//!
//! Each memory starts as ic-stable-structures' memories do: a 3-byte magic and a version byte.
//!
//! An allocation is an [`Extent`] of a power-of-two byte class from 32 to 4096. Classes below a
//! page are carved from chunk pages (one class per page, a bitmap of slots); a full-page extent
//! is the page itself. Every operation is O(1). A page that empties goes back to one free list,
//! so space freed by one class or use serves any other.
//!
//! Allocation fails only when a memory cannot grow, and grows before changing anything. Release
//! never fails.
//!
//! Chosen over a TLSF span allocator with boundary tags (32 bytes per span) and over PMA leaves:
//! per-bucket runs are mostly 8-16 slots, and counted with their metadata the page store used the
//! least memory of the three, with fewer memory calls and the same copying. The TLSF allocator is
//! in the history under the tag `lara-graph-legacy`.
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
//! Partial list head, 32 B class   ↕ 4 bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Partial list head, 2 KiB class  ↕ 4 bytes
//! ---------------------------------------- <- Address 40
//! Reserved space                  ↕ 24 bytes
//! ---------------------------------------- <- Address 64 (RECORDS_START)
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
//! Extent `index` of class `2^log2` bytes in a page starts at `64 + page × 4096 + index ×
//! 2^log2` in the pages memory. It encodes in 40 bits, little-endian:
//!
//! ```text
//! ---------------------------------------- <- Bit 0
//! Page id                         ↕ 28 bits       (1 TiB of pages per memory)
//! ---------------------------------------- <- Bit 28
//! Start in the page, 32 B units  ↕ 7 bits       (index << (log2 - 5))
//! ---------------------------------------- <- Bit 35
//! Class log2 - 5 (0 to 7)         ↕ 3 bits
//! ---------------------------------------- <- Bit 38
//! Spare, for the holder           ↕ 2 bits
//! ----------------------------------------
//! ```

mod record;

use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, Region, WriteError};
use crate::store::txn::Allocator;
use crate::store::units::ByteOffset;
use alloc::vec::Vec;
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
/// Bytes each memory's header takes before its records or pages.
const HEADER_BYTES: u64 = 64;
const RECORDS_START: u64 = HEADER_BYTES;
const PAGES_START: u64 = HEADER_BYTES;

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

/// A power-of-two allocation size from 32 bytes to a whole page, stored as its log2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ByteClass(u8);

impl ByteClass {
    const MIN_LOG2: u8 = 5;
    const PAGE_LOG2: u8 = 12;
    /// Chunk classes, i.e. classes below a page.
    const CHUNK_CLASSES: usize = 7;
    pub(crate) const PAGE: Self = Self(Self::PAGE_LOG2);
    /// The smallest class: 32 bytes.
    pub(crate) const SMALLEST: Self = Self(5);

    /// The smallest class holding `bytes`; `None` past a page.
    pub(crate) fn for_bytes(bytes: u32) -> Option<Self> {
        if bytes > PAGE_BYTES {
            return None;
        }
        let log2 = bytes.max(1).next_power_of_two().trailing_zeros();
        u8::try_from(log2)
            .ok()
            .map(|log2| Self(log2.max(Self::MIN_LOG2)))
    }

    pub(crate) fn from_log2(log2: u8) -> Option<Self> {
        (Self::MIN_LOG2..=Self::PAGE_LOG2)
            .contains(&log2)
            .then_some(Self(log2))
    }

    pub(crate) const fn log2(self) -> u8 {
        self.0
    }

    pub(crate) const fn bytes(self) -> u32 {
        1 << self.0
    }

    pub(crate) const fn is_page(self) -> bool {
        self.0 == Self::PAGE_LOG2
    }

    /// Extents of this class per chunk page.
    const fn per_page(self) -> u16 {
        1 << (Self::PAGE_LOG2 - self.0)
    }

    /// Position of this chunk class among the partial-list heads.
    fn lane(self) -> u64 {
        u64::from(self.0 - Self::MIN_LOG2)
    }
}

/// An allocation: the `index`-th extent of `class` bytes in `page`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Extent {
    page: PageId,
    index: u16,
    class: ByteClass,
}

/// Bytes an encoded [`Extent`] takes: 40 bits, of which the extent uses the low 38.
pub(crate) const EXTENT_BYTES: usize = 5;
/// Bits of a page id in an encoded extent.
const PAGE_BITS: u32 = 28;
/// Bits of the start within the page, in 32-byte units: 128 per page.
const INDEX_BITS: u32 = 7;
/// Bits of the class, as `log2 - 5`.
const CLASS_BITS: u32 = 3;
/// Page ids an encoded extent can name: 1 TiB of pages per memory.
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
        Self { page, index, class }
    }

    /// A whole page as an extent.
    pub(crate) const fn whole(page: PageId) -> Self {
        Self {
            page,
            index: 0,
            class: ByteClass::PAGE,
        }
    }

    pub(crate) const fn page(self) -> PageId {
        self.page
    }

    pub(crate) const fn class(self) -> ByteClass {
        self.class
    }

    pub(crate) const fn len(self) -> u32 {
        self.class.bytes()
    }

    /// Where the extent starts in the pages memory.
    pub(crate) fn offset(self) -> ByteOffset {
        let within = u64::from(self.index) * u64::from(self.class.bytes());
        ByteOffset::new(self.page.offset().get() + within)
    }

    /// As the module's Extent diagram shows.
    pub(crate) fn encode(self) -> [u8; EXTENT_BYTES] {
        let [p0, p1, p2, p3] = self.page.0.to_le_bytes();
        // Bits 28-34 the start within the page in 32-byte units, 35-37 the class: over the page
        // id's top byte and one more.
        let shift = self.class.0 - ByteClass::MIN_LOG2;
        let [i0, i1] = ((self.index << shift << 4) | (u16::from(shift) << 11)).to_le_bytes();
        [p0, p1, p2, p3 | i0, i1]
    }

    /// Ignores the spare bits ([`EXTENT_SPARE_BITS`]). Shifts and masks only, as every extent
    /// read goes through here; any 38 bits name a valid extent, so there is nothing to check.
    pub(crate) fn decode(bytes: [u8; EXTENT_BYTES]) -> Self {
        let [b0, b1, b2, b3, b4] = bytes;
        let page = u32::from_le_bytes([b0, b1, b2, b3]) & (MAX_PAGES - 1);
        let units = (u16::from_le_bytes([b3, b4]) >> 4) & ((1 << INDEX_BITS) - 1);
        let shift = (b4 >> 3) & 0b111;
        Self {
            page: PageId(page),
            // A start of fewer units than the class's size is no extent's: its low bits drop.
            index: units >> shift,
            class: ByteClass(ByteClass::MIN_LOG2 + shift),
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
pub(crate) struct PageStore<M: Memory> {
    pages: M,
    directory: M,
    /// Mirrors the directory's minted count.
    minted: u32,
    /// Pages both memories hold at their current size: taking pages below this never grows.
    room: u32,
}

impl<M: Memory> PageStore<M> {
    /// Formats two memories as an empty store, whatever they held: every header field is
    /// written, and nothing past the headers is read before it is written. Constant time.
    pub(crate) fn format(pages: M, directory: M) -> Result<Self, WriteError> {
        let region = Region::new(&directory);
        let [d0, d1, d2] = DIRECTORY_MAGIC;
        region.write(ByteOffset::new(0), &[d0, d1, d2, VERSION])?;
        region.overwrite(ByteOffset::new(AT_MINTED), &0u32.to_le_bytes());
        region.overwrite(ByteOffset::new(AT_FREE_HEAD), &encode_link(None));
        for lane in 0..ByteClass::CHUNK_CLASSES {
            region.overwrite(
                partial_head_at(u64::try_from(lane).unwrap_or(0)),
                &encode_link(None),
            );
        }
        let [p0, p1, p2] = PAGES_MAGIC;
        Region::new(&pages).write(ByteOffset::new(0), &[p0, p1, p2, VERSION])?;
        Ok(Self::with_header(pages, directory, 0))
    }

    /// Opens memories written by [`Self::format`]. Reads the headers only.
    pub(crate) fn open(pages: M, directory: M) -> Result<Self, OpenError> {
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
    pub(crate) fn into_memories(self) -> (M, M) {
        (self.pages, self.directory)
    }

    fn with_header(pages: M, directory: M, minted: u32) -> Self {
        let mut store = Self {
            pages,
            directory,
            minted,
            room: 0,
        };
        store.measure_room();
        store
    }

    /// Recomputes [`Self::room`] from the memories' sizes.
    fn measure_room(&mut self) {
        let bytes = |m: &M| m.size().saturating_mul(crate::store::memory::PAGE_SIZE);
        let pages = bytes(&self.pages).saturating_sub(PAGES_START) / u64::from(PAGE_BYTES);
        let records = bytes(&self.directory).saturating_sub(RECORDS_START) / RECORD_BYTES;
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
            return Ok(Extent {
                page,
                index: 0,
                class,
            });
        }
        let mut chunk = self.open_chunk(class)?;
        let index =
            u16::try_from(chunk.bitmap.trailing_ones()).unwrap_or_else(|_| corrupt("bitmap"));
        if index >= class.per_page() {
            corrupt("partial chunk has no free slot");
        }
        chunk.used += 1;
        chunk.bitmap |= 1 << index;
        self.store_chunk(class, &chunk);
        if chunk.used == class.per_page() {
            self.unlink_partial(class, chunk.page);
        }
        Ok(Extent {
            page: chunk.page,
            index,
            class,
        })
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
        let mut free = !chunk.bitmap;
        let mut taken = 0u32;
        while free != 0 && below(out, count) {
            let index = free.trailing_zeros();
            free &= free - 1;
            chunk.bitmap |= 1u128 << index;
            out.push(Extent {
                page,
                index: u16::try_from(index).unwrap_or_else(|_| corrupt("chunk index")),
                class,
            });
            taken += 1;
        }
        chunk.used += u16::try_from(taken).unwrap_or_else(|_| corrupt("chunk used"));
        self.store_chunk(class, &chunk);
        if chunk.used >= class.per_page() {
            self.unlink_partial(class, page);
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
        let page = if let Some(page) = self.partial_head(class) {
            page
        } else {
            let page = self.take_page()?;
            let chunk = PageRecord::Chunk {
                class,
                used: 0,
                prev: None,
                next: None,
                bitmap: 0,
            };
            self.set_record(page, chunk);
            self.push_partial(class, page);
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
            used,
            prev,
            next,
            bitmap,
        })
    }

    /// Writes back the record of a chunk page opened by [`Self::open_chunk`].
    #[inline]
    fn store_chunk(&self, class: ByteClass, chunk: &OpenChunk) {
        self.set_record(
            chunk.page,
            PageRecord::Chunk {
                class,
                used: chunk.used,
                prev: chunk.prev,
                next: chunk.next,
                bitmap: chunk.bitmap,
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
        let Extent { page, index, class } = extent;
        match self.record(page) {
            PageRecord::Whole if class.is_page() => self.give_page(page),
            PageRecord::Chunk {
                class: c,
                used,
                prev,
                next,
                bitmap,
            } if c == class => {
                let bit = 1u128 << index;
                if bitmap & bit == 0 {
                    corrupt("released extent is not allocated");
                }
                let was_full = used == class.per_page();
                let used = used - 1;
                self.set_record(
                    page,
                    PageRecord::Chunk {
                        class,
                        used,
                        prev,
                        next,
                        bitmap: bitmap & !bit,
                    },
                );
                if used == 0 {
                    if !was_full {
                        self.unlink_partial(class, page);
                    }
                    self.give_page(page);
                } else if was_full {
                    self.push_partial(class, page);
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

    fn partial_head(&self, class: ByteClass) -> Option<PageId> {
        decode_link(self.directory().read_array(partial_head_at(class.lane())))
    }

    fn set_partial_head(&self, class: ByteClass, head: Option<PageId>) {
        self.directory()
            .overwrite(partial_head_at(class.lane()), &encode_link(head));
    }

    fn push_partial(&self, class: ByteClass, page: PageId) {
        let head = self.partial_head(class);
        if let Some(head) = head {
            self.relink(head, |prev, _| *prev = Some(page));
        }
        self.relink(page, |prev, next| {
            *prev = None;
            *next = head;
        });
        self.set_partial_head(class, Some(page));
    }

    fn unlink_partial(&self, class: ByteClass, page: PageId) {
        let PageRecord::Chunk { prev, next, .. } = self.record(page) else {
            corrupt("unlinking a page that is not a chunk")
        };
        match prev {
            Some(prev) => self.relink(prev, |_, n| *n = next),
            None => self.set_partial_head(class, next),
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
                used,
                prev,
                next,
                bitmap,
            },
        );
    }

    // --- directory ---------------------------------------------------------------------------

    fn directory(&self) -> Region<'_, M> {
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
        let mut small = [0u8; SMALL_READ];
        if let Some(buf) = small.get_mut(..len) {
            self.read_into(at, buf);
            return f(buf);
        }
        f(&self.read(at, len))
    }

    /// Fills `buf` from `at`, with one memory read and no allocation.
    pub(crate) fn read_into(&self, at: ByteOffset, buf: &mut [u8]) {
        Region::new(self.memory).read(at, buf);
    }
}

impl<M: Memory> Allocator for PageStore<M> {
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
impl<M: Memory> PageStore<M> {
    /// Every allocated extent (test invariant checks).
    pub(crate) fn allocated(&self) -> Vec<Extent> {
        let mut out = Vec::new();
        for page in (0..self.minted()).map(PageId) {
            match self.record(page) {
                PageRecord::Free { .. } => {}
                PageRecord::Whole => out.push(Extent::whole(page)),
                PageRecord::Chunk { class, bitmap, .. } => {
                    let used = (0..class.per_page()).filter(|i| bitmap & (1 << i) != 0);
                    out.extend(used.map(|index| Extent { page, index, class }));
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
    used: u16,
    prev: Option<PageId>,
    next: Option<PageId>,
    bitmap: u128,
}

/// Whether `out` holds fewer than `count` extents.
fn below(out: &[Extent], count: u32) -> bool {
    u32::try_from(out.len()).unwrap_or(u32::MAX) < count
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
        assert!(u64::from(extent.index) * len + len <= u64::from(PAGE_BYTES));
    }

    /// An extent encodes to bytes that decode back to it.
    #[kani::proof]
    fn extents_round_trip() {
        let extent = Extent::decode(kani::any());
        assert_eq!(Extent::decode(extent.encode()), extent);
    }
}
