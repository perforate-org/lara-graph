//! The nodes and proxies memories: the node and proxy counts, and each one's record, read one at
//! a time or, for local nodes, a line of eight at a time for batches. A proxy's record sits at its
//! ordinal in the proxies memory (see `crate::NodeId::proxy`), laid out as local nodes' are.
//!
//! # Nodes memory
//!
//! ```text
//! ---------------------------------------- <- Address 0
//! Magic "LGN"                     ↕ 3 bytes
//! ----------------------------------------
//! Layout version                  ↕ 1 byte
//! ----------------------------------------
//! Node count = N                  ↕ 4 bytes
//! ---------------------------------------- <- Address 8 (AT_WIDTHS)
//! Width table root                ↕ 6 bytes   (a value plane root, see `edge::widths`)
//! ----------------------------------------
//! Reserved space                  ↕ 2 bytes
//! ---------------------------------------- <- Address 16 (RECORDS_START)
//! Record of node 0                ↕ 6 bytes   (see `record`)
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Record of node N - 1            ↕ 6 bytes
//! ----------------------------------------
//! Unallocated space
//! ```
//!
//! # Proxies memory
//!
//! Left empty, with no header, until the first proxy is added: a graph without proxies takes no
//! space for them.
//!
//! ```text
//! ---------------------------------------- <- Address 0
//! Magic "LGP"                     ↕ 3 bytes
//! ----------------------------------------
//! Layout version                  ↕ 1 byte
//! ----------------------------------------
//! Proxy count = P                 ↕ 4 bytes
//! ----------------------------------------
//! Reserved space                  ↕ 8 bytes
//! ---------------------------------------- <- Address 16 (RECORDS_START)
//! Record of proxy 0               ↕ 6 bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Record of proxy P - 1           ↕ 6 bytes
//! ----------------------------------------
//! Unallocated space
//! ```

use crate::bucket::values::Plane;
use crate::id::NodeId;
use crate::node::record::{RECORD_BYTES, Record};
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, Region, WriteError};
use crate::store::units::ByteOffset;

/// The nodes memory's magic: "lara graph nodes".
pub(crate) const MAGIC: [u8; 3] = *b"LGN";
/// The proxies memory's magic: "lara graph proxies".
pub(crate) const PROXIES_MAGIC: [u8; 3] = *b"LGP";
const AT_COUNT: u64 = 4;
const AT_WIDTHS: u64 = 8;
const RECORDS_START: u64 = 16;

/// Eight node records: 48 bytes, read at once.
const RECORDS_PER_LINE: u32 = 8;
const LINE_BYTES: usize = RECORD_BYTES * 8;

/// The nodes and proxies memories, with both counts mirrored.
pub(crate) struct NodeStore<M: Memory, P: Memory = M> {
    memory: M,
    count: u32,
    /// `None` for a graph given no proxies memory: it holds no proxy.
    proxies: Option<P>,
    proxy_count: u32,
    /// The layout version a first proxy's header is written with.
    version: u8,
}

impl<M: Memory, P: Memory> NodeStore<M, P> {
    /// Formats `memory` as a store with no nodes and no width table, and `proxies` as holding no
    /// proxy, overwriting whatever they held. An empty proxies memory stays empty.
    pub(crate) fn format(memory: M, proxies: Option<P>, version: u8) -> Result<Self, WriteError> {
        let region = Region::new(&memory);
        let [m0, m1, m2] = MAGIC;
        region.write(ByteOffset::new(0), &[m0, m1, m2, version])?;
        region.overwrite(ByteOffset::new(AT_COUNT), &0u32.to_le_bytes());
        region.overwrite(ByteOffset::new(AT_WIDTHS), &Plane::Empty.encode());
        if let Some(proxies) = proxies.as_ref().filter(|p| p.size() > 0) {
            write_proxies_header(proxies, version)?;
        }
        Ok(Self::open(memory, proxies, version))
    }

    /// The store `memory` and `proxies` hold; their headers were checked (an empty proxies
    /// memory holds no proxy).
    pub(crate) fn open(memory: M, proxies: Option<P>, version: u8) -> Self {
        Self {
            count: count_in(&memory),
            proxy_count: proxies.as_ref().map_or(0, count_in),
            memory,
            proxies,
            version,
        }
    }

    /// The nodes memory and the proxies memory back.
    pub(crate) fn into_memories(self) -> (M, Option<P>) {
        (self.memory, self.proxies)
    }

    /// Number of local nodes; ids are `0..count`.
    pub(crate) const fn count(&self) -> u32 {
        self.count
    }

    /// Number of proxies; ids are `NodeId::proxy(0..proxy_count)`.
    pub(crate) const fn proxy_count(&self) -> u32 {
        self.proxy_count
    }

    /// Whether `v` was added, as a local node or a proxy.
    pub(crate) const fn contains(&self, v: NodeId) -> bool {
        let index = v.index();
        if index < NodeId::FIRST_PROXY {
            index < self.count
        } else {
            index - NodeId::FIRST_PROXY < self.proxy_count
        }
    }

    /// The line of eight local records from `base`, when the store holds all eight.
    fn line(&self, base: u32) -> Option<[u8; LINE_BYTES]> {
        (self.count.saturating_sub(base) >= RECORDS_PER_LINE)
            .then(|| Region::new(&self.memory).read_array(record_at(base)))
    }

    /// The proxies memory, given since `v`, a known proxy, was added.
    fn proxies(&self) -> &P {
        self.proxies
            .as_ref()
            .unwrap_or_else(|| corrupt("a proxy without a proxies memory"))
    }

    /// A known node's record, undecoded. A proxy's is read out of line, so that a local node's
    /// read stays small enough to inline: with both reads here, finding a bucket cost about 20
    /// instructions more (`degree_1024`).
    pub(crate) fn record_bytes(&self, v: NodeId) -> [u8; RECORD_BYTES] {
        let index = v.index();
        if index < NodeId::FIRST_PROXY {
            Region::new(&self.memory).read_array(record_at(index))
        } else {
            self.proxy_record_bytes(index - NodeId::FIRST_PROXY)
        }
    }

    /// The record of the proxy `ordinal`; a zero record without a proxies memory, which has no
    /// proxy to ask for.
    #[cold]
    #[inline(never)]
    fn proxy_record_bytes(&self, ordinal: u32) -> [u8; RECORD_BYTES] {
        self.proxies.as_ref().map_or([0; RECORD_BYTES], |proxies| {
            Region::new(proxies).read_array(record_at(ordinal))
        })
    }

    /// Rewrites an existing node's record. Cannot fail.
    pub(crate) fn write(&self, v: NodeId, record: Record) {
        let index = v.index();
        match &self.proxies {
            _ if index < NodeId::FIRST_PROXY => {
                Region::new(&self.memory).overwrite(record_at(index), &record.encode());
            }
            Some(proxies) => Region::new(proxies)
                .overwrite(record_at(index - NodeId::FIRST_PROXY), &record.encode()),
            // No proxy is known without a proxies memory.
            None => {}
        }
    }

    /// Whether the store was given a proxies memory, so that proxies can be added.
    pub(crate) const fn has_proxies_memory(&self) -> bool {
        self.proxies.is_some()
    }

    /// Rewrites the records of nodes `0..records.len()`, all known, a line at a time: a bulk
    /// load's. Cannot fail.
    pub(crate) fn write_all(&self, records: &[[u8; RECORD_BYTES]]) {
        let region = Region::new(&self.memory);
        for (line, at) in records.chunks(8).zip((0u32..).step_by(8)) {
            let mut bytes = [0u8; LINE_BYTES];
            for (to, record) in bytes.chunks_mut(RECORD_BYTES).zip(line) {
                to.copy_from_slice(record);
            }
            region.overwrite(
                record_at(at),
                bytes.get(..line.len() * RECORD_BYTES).unwrap_or(&[]),
            );
        }
    }

    /// Adds new nodes until there are `next`, each with [`Record::NEW`]. On error none is visible.
    #[inline]
    pub(crate) fn extend(&mut self, next: u32) -> Result<(), WriteError> {
        extend_records(&self.memory, self.count, next)?;
        self.set_count(next);
        Ok(())
    }

    /// Adds new proxies until there are `next`, each with [`Record::NEW`], writing the proxies
    /// memory's header first if this is the first. On error none is visible.
    /// The store has a proxies memory ([`Self::has_proxies_memory`]).
    pub(crate) fn extend_proxies(&mut self, next: u32) -> Result<(), WriteError> {
        let proxies = self.proxies();
        if proxies.size() == 0 {
            write_proxies_header(proxies, self.version)?;
        }
        extend_records(proxies, self.proxy_count, next)?;
        Region::new(proxies).overwrite(ByteOffset::new(AT_COUNT), &next.to_le_bytes());
        self.proxy_count = next;
        Ok(())
    }

    /// The root of the label width table's value plane.
    pub(crate) fn widths_root(&self) -> Plane {
        Plane::decode(Region::new(&self.memory).read_array(ByteOffset::new(AT_WIDTHS)))
    }

    /// Points the header at the label width table's new root.
    pub(crate) fn set_widths_root(&self, root: Plane) {
        Region::new(&self.memory).overwrite(ByteOffset::new(AT_WIDTHS), &root.encode());
    }

    fn set_count(&mut self, count: u32) {
        Region::new(&self.memory).overwrite(ByteOffset::new(AT_COUNT), &count.to_le_bytes());
        self.count = count;
    }
}

/// The count a nodes or proxies memory holds; none in an empty one.
fn count_in<M: Memory>(memory: &M) -> u32 {
    if memory.size() == 0 {
        return 0;
    }
    u32::from_le_bytes(Region::new(memory).read_array(ByteOffset::new(AT_COUNT)))
}

/// Writes an empty proxies memory's header: no proxy yet.
fn write_proxies_header<M: Memory>(proxies: &M, version: u8) -> Result<(), WriteError> {
    let region = Region::new(proxies);
    let [m0, m1, m2] = PROXIES_MAGIC;
    region.write(ByteOffset::new(0), &[m0, m1, m2, version])?;
    region.write(ByteOffset::new(AT_COUNT), &0u32.to_le_bytes())
}

/// Writes [`Record::NEW`] for records `first..next` of `memory`, a line of eight at a time.
#[inline]
fn extend_records<M: Memory>(memory: &M, first: u32, next: u32) -> Result<(), WriteError> {
    let region = Region::new(memory);
    if next == first + 1 {
        return region.write(record_at(first), &Record::NEW.encode());
    }
    region.reserve(record_at(next))?;
    let new = Record::NEW.encode();
    let line: [u8; LINE_BYTES] =
        core::array::from_fn(|i| new.get(i % RECORD_BYTES).copied().unwrap_or(0));
    let mut index = first;
    while index < next {
        let count = (next - index).min(RECORDS_PER_LINE);
        let len = usize::try_from(count).unwrap_or(0) * RECORD_BYTES;
        region.overwrite(record_at(index), line.get(..len).unwrap_or(&[]));
        index += count;
    }
    Ok(())
}

/// Where record `index` starts, and so where the records before it end.
fn record_at(index: u32) -> ByteOffset {
    ByteOffset::new(RECORDS_START + u64::from(index) * u64::try_from(RECORD_BYTES).unwrap_or(8))
}

/// A batch read's line of node records. Local to one call, so no invalidation is needed.
pub(crate) struct RecordLines {
    base: u32,
    bytes: [u8; LINE_BYTES],
}

impl Default for RecordLines {
    fn default() -> Self {
        Self {
            base: u32::MAX,
            bytes: [0; LINE_BYTES],
        }
    }
}

impl RecordLines {
    /// `v`'s record's bytes, reading one 48-byte line per eight dense ids: a whole record or the
    /// state alone decodes from them. `v` is known. Always inlined: a batch's per-node loop is
    /// little more than this, and left a call it cost those loops 8-13 %.
    #[expect(clippy::inline_always, reason = "measured: see the doc")]
    #[inline(always)]
    pub(crate) fn bytes_of<M: Memory, P: Memory>(
        &mut self,
        nodes: &NodeStore<M, P>,
        v: NodeId,
    ) -> [u8; RECORD_BYTES] {
        let base = v.index() / RECORDS_PER_LINE * RECORDS_PER_LINE;
        if base != self.base {
            // A line past the local records (the last partial one, or a proxy's) takes the
            // single-record path: a 48-byte read would run past the records the memory holds.
            let Some(line) = nodes.line(base) else {
                return nodes.record_bytes(v);
            };
            self.bytes = line;
            self.base = base;
        }
        let offset = usize::try_from(v.index() - base).unwrap_or_else(|_| corrupt("record offset"))
            * RECORD_BYTES;
        line_record(&self.bytes, offset)
    }
}

fn line_record(bytes: &[u8; LINE_BYTES], offset: usize) -> [u8; RECORD_BYTES] {
    let (_, rest) = bytes.split_at(offset);
    let (record, _) = rest.split_at(RECORD_BYTES);
    record.try_into().unwrap_or_else(|_| corrupt("record line"))
}
