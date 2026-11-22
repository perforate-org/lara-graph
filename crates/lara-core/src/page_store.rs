//! The page store: every byte of adjacency lives in fixed 4096-byte pages.
//!
//! - **pages** holds page data only, at `page_id × 4096`: no headers, so an extent reads as one
//!   contiguous range.
//! - **directory** holds one 32-byte record per page, addressed by page id: free-list links for
//!   free pages, and the class, use count and slot bitmap for pages split into chunks.
//!
//! An allocation is an [`Extent`] of a power-of-two byte class from 32 to 4096. Classes below a
//! page are carved from chunk pages (one class per page, a bitmap of slots); a full-page extent
//! is the page itself. Every operation is O(1). A page that empties goes back to one free list,
//! so space freed by one class or use serves any other.
//!
//! Allocation fails only when a memory cannot grow, and grows before changing anything. Release
//! never fails.

mod record;

use crate::corrupt::corrupt;
use crate::memory::{Memory, Region, WriteError};
use crate::txn::Allocator;
use crate::units::ByteOffset;
use alloc::vec::Vec;
use record::{PageRecord, RECORD_BYTES, decode_link, encode_link};

/// Bytes per page.
pub(crate) const PAGE_BYTES: u32 = 4096;

const MAGIC: [u8; 3] = *b"LPS";
const VERSION: u8 = 1;
const AT_MINTED: u64 = 4;
const AT_FREE_HEAD: u64 = 8;
const AT_PARTIAL_HEADS: u64 = 12;
const RECORDS_START: u64 = 64;

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
        ByteOffset::new(u64::from(self.0) * u64::from(PAGE_BYTES))
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

/// Bytes an encoded [`Extent`] takes.
pub(crate) const EXTENT_BYTES: usize = 7;

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

    /// Layout: page u32 | index u16 | class log2 u8.
    pub(crate) fn encode(self) -> [u8; EXTENT_BYTES] {
        let [p0, p1, p2, p3] = self.page.0.to_le_bytes();
        let [i0, i1] = self.index.to_le_bytes();
        [p0, p1, p2, p3, i0, i1, self.class.0]
    }

    pub(crate) fn decode(bytes: [u8; EXTENT_BYTES]) -> Self {
        let [p0, p1, p2, p3, i0, i1, class] = bytes;
        let class = ByteClass::from_log2(class).unwrap_or_else(|| corrupt("extent class"));
        let index = u16::from_le_bytes([i0, i1]);
        if index >= class.per_page() {
            corrupt("extent index");
        }
        Self {
            page: PageId(u32::from_le_bytes([p0, p1, p2, p3])),
            index,
            class,
        }
    }
}

/// Opening a page store failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpenError {
    NotAPageStore,
    UnsupportedVersion(u8),
}

/// Creating a page store failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CreateError {
    NotEmpty,
    Write(WriteError),
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
    /// Formats two empty memories.
    pub(crate) fn create(pages: M, directory: M) -> Result<Self, CreateError> {
        if pages.size() != 0 || directory.size() != 0 {
            return Err(CreateError::NotEmpty);
        }
        let region = Region::new(&directory);
        let [m0, m1, m2] = MAGIC;
        region
            .write(ByteOffset::new(0), &[m0, m1, m2, VERSION])
            .map_err(CreateError::Write)?;
        region.overwrite(ByteOffset::new(AT_FREE_HEAD), &encode_link(None));
        for lane in 0..ByteClass::CHUNK_CLASSES {
            region.overwrite(
                partial_head_at(u64::try_from(lane).unwrap_or(0)),
                &encode_link(None),
            );
        }
        Ok(Self::with_header(pages, directory, 0))
    }

    /// Opens memories written by [`Self::create`]. Reads the header only.
    pub(crate) fn open(pages: M, directory: M) -> Result<Self, OpenError> {
        if directory.size() == 0 {
            return Err(OpenError::NotAPageStore);
        }
        let [m0, m1, m2, version] = Region::new(&directory).read_array(ByteOffset::new(0));
        if [m0, m1, m2] != MAGIC {
            return Err(OpenError::NotAPageStore);
        }
        if version != VERSION {
            return Err(OpenError::UnsupportedVersion(version));
        }
        let minted =
            u32::from_le_bytes(Region::new(&directory).read_array(ByteOffset::new(AT_MINTED)));
        Ok(Self::with_header(pages, directory, minted))
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
        let bytes = |m: &M| m.size().saturating_mul(crate::memory::PAGE_SIZE);
        let pages = bytes(&self.pages) / u64::from(PAGE_BYTES);
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
        let index = u16::try_from(bitmap.trailing_ones()).unwrap_or_else(|_| corrupt("bitmap"));
        if index >= class.per_page() {
            corrupt("partial chunk has no free slot");
        }
        let used = used + 1;
        self.set_record(
            page,
            PageRecord::Chunk {
                class,
                used,
                prev,
                next,
                bitmap: bitmap | (1 << index),
            },
        );
        if used == class.per_page() {
            self.unlink_partial(class, page);
        }
        Ok(Extent { page, index, class })
    }

    /// Grows both memories so that `pages` more pages can be taken without growing them: after
    /// this, allocations that take at most `pages` pages cannot fail. Changes no state.
    pub(crate) fn reserve(&mut self, pages: u32) -> Result<(), WriteError> {
        let end = self
            .minted
            .checked_add(pages)
            .ok_or(WriteError::Overflow(crate::units::Overflow))?;
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

fn record_at(page: PageId) -> ByteOffset {
    ByteOffset::new(RECORDS_START + u64::from(page.0) * RECORD_BYTES)
}

fn partial_head_at(lane: u64) -> ByteOffset {
    ByteOffset::new(AT_PARTIAL_HEADS + 4 * lane)
}

#[cfg(test)]
mod tests;
