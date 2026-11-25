//! The nodes memory: the node count and each node's record, read one at a time or a line of eight
//! at a time for batches.
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

use crate::bucket::values::Plane;
use crate::id::{NodeId, NodeState};
use crate::node::record::{RECORD_BYTES, Record};
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, Region, WriteError};
use crate::store::units::ByteOffset;

/// The nodes memory's magic: "lara graph nodes".
pub(crate) const MAGIC: [u8; 3] = *b"LGN";
const AT_COUNT: u64 = 4;
const AT_WIDTHS: u64 = 8;
const RECORDS_START: u64 = 16;

/// Eight node records: 48 bytes, read at once.
const RECORDS_PER_LINE: u32 = 8;
const LINE_BYTES: usize = RECORD_BYTES * 8;

/// The nodes memory, with the node count mirrored.
pub(crate) struct NodeStore<M: Memory> {
    memory: M,
    count: u32,
}

impl<M: Memory> NodeStore<M> {
    /// Formats `memory` as a store with no nodes and no width table, overwriting whatever it held.
    pub(crate) fn format(memory: M, version: u8) -> Result<Self, WriteError> {
        let region = Region::new(&memory);
        let [m0, m1, m2] = MAGIC;
        region.write(ByteOffset::new(0), &[m0, m1, m2, version])?;
        region.overwrite(ByteOffset::new(AT_COUNT), &0u32.to_le_bytes());
        region.overwrite(ByteOffset::new(AT_WIDTHS), &Plane::Empty.encode());
        Ok(Self::open(memory))
    }

    /// The store `memory` holds; its header was checked.
    pub(crate) fn open(memory: M) -> Self {
        let count = u32::from_le_bytes(Region::new(&memory).read_array(ByteOffset::new(AT_COUNT)));
        Self { memory, count }
    }

    pub(crate) fn into_memory(self) -> M {
        self.memory
    }

    /// Number of nodes; ids are `0..count`.
    pub(crate) const fn count(&self) -> u32 {
        self.count
    }

    /// `v`'s record. `v` is known.
    #[inline]
    pub(crate) fn record(&self, v: NodeId) -> Record {
        Record::decode(self.record_bytes(v))
    }

    /// A known node's state alone, without decoding its list's extent.
    pub(crate) fn state(&self, v: NodeId) -> NodeState {
        Record::state_of(self.record_bytes(v))
    }

    /// A known node's record, undecoded.
    pub(crate) fn record_bytes(&self, v: NodeId) -> [u8; RECORD_BYTES] {
        Region::new(&self.memory).read_array(record_at(v.index()))
    }

    /// Rewrites an existing node's record. Cannot fail.
    pub(crate) fn write(&self, v: NodeId, record: Record) {
        Region::new(&self.memory).overwrite(record_at(v.index()), &record.encode());
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
        let first = self.count;
        let region = Region::new(&self.memory);
        if next == first + 1 {
            region.write(record_at(first), &Record::NEW.encode())?;
        } else {
            region.reserve(record_at(next))?;
            // A line of new records at a time: one write per eight nodes.
            let line: [u8; LINE_BYTES] = core::array::from_fn(|i| {
                Record::NEW
                    .encode()
                    .get(i % RECORD_BYTES)
                    .copied()
                    .unwrap_or(0)
            });
            let mut index = first;
            while index < next {
                let count = (next - index).min(RECORDS_PER_LINE);
                let len = usize::try_from(count).unwrap_or(0) * RECORD_BYTES;
                region.overwrite(record_at(index), line.get(..len).unwrap_or(&[]));
                index += count;
            }
        }
        self.set_count(next);
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

/// Where node `index`'s record starts, and so where the records before it end.
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
    pub(crate) fn bytes_of<M: Memory>(
        &mut self,
        nodes: &NodeStore<M>,
        v: NodeId,
    ) -> [u8; RECORD_BYTES] {
        let base = v.index() / RECORDS_PER_LINE * RECORDS_PER_LINE;
        // The last partial line still takes the single-record path: a 48-byte read would run
        // past the records the memory holds.
        if nodes.count.saturating_sub(base) < RECORDS_PER_LINE {
            return nodes.record_bytes(v);
        }
        if base != self.base {
            self.bytes = Region::new(&nodes.memory).read_array(record_at(base));
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
