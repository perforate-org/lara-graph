//! The graph: label buckets of edge halves per node and direction, over five memories.
//!
//! # Nodes memory
//!
//! ```text
//! ---------------------------------------- <- Address 0
//! Magic "LGN"                     ↕ 3 bytes
//! ----------------------------------------
//! Layout version (1)              ↕ 1 byte
//! ----------------------------------------
//! Node count = N                  ↕ 4 bytes
//! ---------------------------------------- <- Address 8 (AT_WIDTHS)
//! Width table present (0 or 1)    ↕ 1 byte
//! ----------------------------------------
//! Width table extent              ↕ 7 bytes
//! ---------------------------------------- <- Address 16 (RECORDS_START)
//! Record of node 0                ↕ 8 bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Record of node N - 1            ↕ 8 bytes
//! ----------------------------------------
//! Unallocated space
//! ```

use crate::corrupt::{corrupt, reservation_exceeded};
use crate::descriptor::Descriptor;
use crate::id::{
    Direction, Edge, EdgeRef, Label, Move, NodeId, NodeState, Position, Progress, RemovedEdge,
    Width,
};
use crate::label_bucket;
use crate::label_bucket_list::{self, LabelBucketKey, LabelBucketList};
use crate::memory::{GrowFailed, Memory, Region, WriteError};
use crate::node::{RECORD_BYTES, Record};
use crate::page_store::PageStore;
use crate::stored::{LabelBucket, Stored};
use crate::units::{ByteOffset, Overflow};
use alloc::vec::Vec;
use core::fmt;
use core::ops::ControlFlow;

/// The nodes memory's magic: "lara graph nodes".
const MAGIC: [u8; 3] = *b"LGN";
const VERSION: u8 = 1;
const AT_COUNT: u64 = 4;
const RECORDS_START: u64 = 16;

/// A multigraph with dense node ids and labeled edges, directed or undirected.
///
/// Every edge is stored at both ends (see [`EdgeRef`]), so traversal runs either way. Each
/// `(node, label, direction)` has a bucket addressed by [`Position`]. Positions are stable:
/// removing an edge leaves a hole that a later insert may fill, and only [`Graph::compact`] moves
/// edges.
///
/// The two halves of an edge are matched by rank: the k-th live entry to `v` in `u`'s out
/// bucket pairs with the k-th live entry to `u` in `v`'s in bucket (undirected alike). Inserts
/// and compaction keep entries to one node in insertion order, so the ranks stay aligned
/// without links between the halves.
///
/// Reading one bucket costs a node read, a label bucket list read and one data read (a few more
/// past 1024 edges). An insert or removal copies at most one run of under 1024 edges or one page of
/// index.
pub struct Graph<M: Memory> {
    nodes: M,
    edges: PageStore<M>,
    label_bucket_lists: PageStore<M>,
    /// Mirrors the node count in the header: the graph owns its memories.
    count: u32,
    /// Mirrors the label width table.
    widths: Widths,
}

/// An operation failed; the graph is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A memory could not grow.
    OutOfMemory(GrowFailed),
    /// The graph outgrew its address space or a bucket outgrew 2^32 positions.
    AddressSpaceExhausted,
    /// The node was never added.
    UnknownNode(NodeId),
    /// The node already has the maximum number of `(label, direction)` buckets (215).
    TooManyLabelBuckets(NodeId),
    /// The node is being detached ([`Graph::detach`]) and takes no new edges.
    Detaching(NodeId),
    /// [`Graph::detach_step`] on a node that is not being detached.
    NotDetaching(NodeId),
    /// The node was deleted ([`Graph::detach`]); ids are never reused.
    Deleted(NodeId),
    /// The label already has a different width ([`Graph::declare_width`]).
    WidthConflict {
        /// The label.
        label: Label,
        /// Its width.
        declared: Width,
    },
    /// 1023 labels already have a width.
    TooManyWidths,
    /// A value's length is not its label's width (0 for a label without one).
    WrongValueLength {
        /// The label.
        label: Label,
        /// The label's width in bytes.
        expected: u16,
        /// The value's length.
        actual: usize,
    },
}

/// A batch operation failed at `index`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchError {
    /// Nothing before `index` was applied; the graph is unchanged.
    Unchanged {
        /// The failing element.
        index: usize,
        /// The failure itself.
        error: Error,
    },
    /// Elements before `index` were applied; `index` itself is all-or-none.
    PartiallyApplied {
        /// The failing element.
        index: usize,
        /// The failure itself.
        error: Error,
    },
}

impl BatchError {
    pub(crate) const fn unchanged(index: usize, error: Error) -> Self {
        Self::Unchanged { index, error }
    }

    pub(crate) const fn after_prefix(index: usize, error: Error) -> Self {
        if index == 0 {
            Self::Unchanged { index, error }
        } else {
            Self::PartiallyApplied { index, error }
        }
    }

    /// Where in the batch the failure happened.
    #[must_use]
    pub const fn index(&self) -> usize {
        match *self {
            Self::Unchanged { index, .. } | Self::PartiallyApplied { index, .. } => index,
        }
    }

    /// The failure itself.
    #[must_use]
    pub const fn error(&self) -> &Error {
        match self {
            Self::Unchanged { error, .. } | Self::PartiallyApplied { error, .. } => error,
        }
    }

    /// The failure itself, without the batch position.
    #[must_use]
    pub const fn into_error(self) -> Error {
        match self {
            Self::Unchanged { error, .. } | Self::PartiallyApplied { error, .. } => error,
        }
    }
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = match self {
            Self::Unchanged { .. } => "nothing applied",
            Self::PartiallyApplied { .. } => "earlier elements applied",
        };
        write!(f, "batch edge {}: {} ({state})", self.index(), self.error())
    }
}

impl core::error::Error for BatchError {}

/// A graph could not be made or opened ([`Graph::try_new`], [`Graph::try_load`],
/// [`Graph::try_init`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitError {
    /// A memory could not grow.
    OutOfMemory(GrowFailed),
    /// The memories do not hold a graph.
    NotAGraph,
    /// The graph was written by an unsupported layout version.
    UnsupportedVersion(u8),
}

/// What one bounded step of a longer operation changed, and whether more remains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step<T> {
    changes: Vec<T>,
    progress: Progress,
}

impl<T> Step<T> {
    const fn new(changes: Vec<T>, progress: Progress) -> Self {
        Self { changes, progress }
    }

    /// What this step changed, in the order it changed it.
    #[must_use]
    pub fn changes(&self) -> &[T] {
        &self.changes
    }

    /// Whether another step has work to do.
    #[must_use]
    pub const fn progress(&self) -> Progress {
        self.progress
    }
}

/// One [`Graph::compact`] step: the edges that moved. Every other edge of the bucket kept its
/// position.
pub type Compaction = Step<Move>;

/// One [`Graph::sweep`]: the edges removed, both halves of each, with their values, and where the
/// sweep goes on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sweep {
    removed: Vec<RemovedEdge>,
    next: Option<Position>,
}

impl Sweep {
    const fn new(removed: Vec<RemovedEdge>, next: Option<Position>) -> Self {
        Self { removed, next }
    }

    /// The edges removed, in position order.
    #[must_use]
    pub fn changes(&self) -> &[RemovedEdge] {
        &self.removed
    }

    /// Where the next call goes on; `None` once the bucket's end was reached.
    #[must_use]
    pub const fn next(&self) -> Option<Position> {
        self.next
    }
}

/// One [`Graph::detach_step`]: the edges removed, both halves of each, with their values.
pub type Detachment = Step<RemovedEdge>;

/// What [`Graph::remove_edge`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Removal {
    /// The edge was removed; its halves' positions are now holes.
    Removed(RemovedEdge),
    /// No live edge was at that position.
    Absent,
}

impl<M: Memory> Graph<M> {
    /// A graph over formatted memories, with its header mirrors loaded.
    fn assemble(nodes: M, edges: PageStore<M>, label_bucket_lists: PageStore<M>) -> Self {
        let count = u32::from_le_bytes(Region::new(&nodes).read_array(ByteOffset::new(AT_COUNT)));
        let widths = Widths::load(&nodes, &label_bucket_lists);
        Self {
            nodes,
            edges,
            label_bucket_lists,
            count,
            widths,
        }
    }

    /// Number of nodes; ids are `0..node_count`.
    #[must_use]
    pub const fn node_count(&self) -> u32 {
        self.count
    }

    /// Adds a node with no edges and returns its id.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`], or [`Error::AddressSpaceExhausted`] past `u32::MAX - 1` nodes
    /// (the last id is reserved as the removed-edge marker).
    pub fn add_node(&mut self) -> Result<NodeId, Error> {
        let id = NodeId::new(self.node_count());
        let next = id
            .index()
            .checked_add(1)
            .filter(|&n| n < u32::MAX)
            .ok_or(Error::AddressSpaceExhausted)?;
        self.region().write(record_at(id), &Record::NEW.encode())?;
        self.region()
            .overwrite(ByteOffset::new(AT_COUNT), &next.to_le_bytes());
        self.count = next;
        Ok(id)
    }

    /// Whether `v` is live, being detached or deleted.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn node_state(&self, v: NodeId) -> Result<NodeState, Error> {
        Ok(self.record(v)?.state)
    }

    /// The labels `v` has buckets under on `direction`, in order.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn labels(
        &self,
        v: NodeId,
        direction: Direction,
        out: &mut Vec<Label>,
    ) -> Result<(), Error> {
        let Some(list) = self.label_bucket_list(v)? else {
            out.clear();
            return Ok(());
        };
        out.clear();
        label_bucket_list::for_each(&self.label_bucket_lists.pages(), Some(list), |key, _| {
            if key.direction == direction {
                out.push(key.label);
            }
            ControlFlow::<()>::Continue(())
        });
        Ok(())
    }

    /// Whether `v` has a live edge under any label, on any side. A bucket whose edges were all
    /// removed stays listed by [`Self::labels`] until compaction drops it; this looks past it.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn has_edges(&self, v: NodeId) -> Result<bool, Error> {
        let Some(list) = self.label_bucket_list(v)? else {
            return Ok(false);
        };
        let view = self.edges.pages();
        Ok(label_bucket_list::for_each(
            &self.label_bucket_lists.pages(),
            Some(list),
            |_, stored| {
                if LabelBucket::resolve(&view, stored).slots.live() > 0 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::<()>::Continue(())
                }
            },
        )
        .is_some())
    }

    /// Number of live edges in `v`'s `(label, direction)` bucket.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn degree(&self, v: NodeId, label: Label, direction: Direction) -> Result<u32, Error> {
        Ok(self
            .bucket(v, LabelBucketKey { label, direction })?
            .map_or(0, |d| d.live()))
    }

    /// Replaces the contents of `out` with the live edges of `v`'s `(label, direction)` bucket, in
    /// position order. Each edge's target is the node at its other end.
    /// [`Self::visit_neighbors`] reads the same edges in about half the instructions when they
    /// need not be kept.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn neighbors_directed(
        &self,
        v: NodeId,
        label: Label,
        direction: Direction,
        out: &mut Vec<Edge>,
    ) -> Result<(), Error> {
        out.clear();
        if let Some(d) = self.bucket(v, LabelBucketKey { label, direction })? {
            label_bucket::read(&d, &self.edges.pages(), out);
        }
        Ok(())
    }

    /// The node at the other end of `half`, if it is live.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn edge_target(&self, half: EdgeRef) -> Result<Option<NodeId>, Error> {
        Ok(self
            .bucket(half.node(), key_of(half))?
            .and_then(|d| label_bucket::get(&d, half.position(), &self.edges.pages())))
    }

    /// Removed edges `v`'s `(label, direction)` bucket still holds positions for: the work
    /// compaction can do.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn holes(&self, v: NodeId, label: Label, direction: Direction) -> Result<u32, Error> {
        Ok(self
            .bucket(v, LabelBucketKey { label, direction })?
            .map_or(0, |d| d.stored() - d.live()))
    }

    fn region(&self) -> Region<'_, M> {
        Region::new(&self.nodes)
    }

    fn check(&self, v: NodeId) -> Result<(), Error> {
        if v.index() < self.node_count() {
            Ok(())
        } else {
            Err(Error::UnknownNode(v))
        }
    }

    fn record(&self, v: NodeId) -> Result<Record, Error> {
        self.check(v)?;
        Ok(Record::decode(self.region().read_array(record_at(v))))
    }

    fn label_bucket_list(&self, v: NodeId) -> Result<Option<LabelBucketList>, Error> {
        Ok(self.record(v)?.list)
    }

    /// Rewrites an existing node's record. Cannot fail.
    fn write_record(&self, v: NodeId, record: Record) {
        self.region().overwrite(record_at(v), &record.encode());
    }

    /// Publishes `v`'s new label bucket list, keeping its state.
    fn set_label_bucket_list(&self, v: NodeId, list: Option<LabelBucketList>) {
        let record = self.label_bucket_list_owner(v);
        self.write_record(v, Record { list, ..record });
    }

    fn label_bucket_list_owner(&self, v: NodeId) -> Record {
        self.record(v).unwrap_or_else(|_| corrupt("a known node"))
    }

    /// A bucket's slot descriptor, without building the bucket's values or index: a read that
    /// only scans slots does not pay for the rest of an extended header.
    fn bucket(&self, v: NodeId, key: LabelBucketKey) -> Result<Option<Descriptor>, Error> {
        let list = self.label_bucket_list(v)?;
        let view = self.edges.pages();
        Ok(
            label_bucket_list::lookup_stored(&self.label_bucket_lists.pages(), list, key).map(
                |stored| match stored {
                    Stored::Plain(slots) => slots,
                    Stored::Extended(header) => {
                        Descriptor::decode(view.read_array(header.offset()))
                    }
                },
            ),
        )
    }
}

/// The result of an update that runs within reserved memory.
fn reserved<T>(result: Result<T, WriteError>) -> T {
    result.unwrap_or_else(|_| reservation_exceeded())
}

fn record_at(v: NodeId) -> ByteOffset {
    ByteOffset::new(RECORDS_START + u64::from(v.index()) * u64::try_from(RECORD_BYTES).unwrap_or(8))
}

/// Eight node records: 64 bytes, one cache line on the IC's host as well.
const RECORDS_PER_LINE: u32 = 8;
const RECORD_LINE_BYTES: usize = 64;

/// A batch read's line of node records. Local to one call, so no invalidation is needed.
pub(super) struct RecordLines {
    base: u32,
    bytes: [u8; RECORD_LINE_BYTES],
}

impl Default for RecordLines {
    fn default() -> Self {
        Self {
            base: u32::MAX,
            bytes: [0; RECORD_LINE_BYTES],
        }
    }
}

impl RecordLines {
    /// `v`'s record, reading one 64-byte line per eight dense ids.
    #[inline]
    pub(super) fn record<M: Memory>(&mut self, graph: &Graph<M>, v: NodeId) -> Record {
        let base = v.index() / RECORDS_PER_LINE * RECORDS_PER_LINE;
        // The last partial line still takes the single-record path: a 64-byte read would run
        // past the records the memory holds.
        if graph.count.saturating_sub(base) < RECORDS_PER_LINE {
            return graph.record(v).unwrap_or_else(|_| corrupt("a known node"));
        }
        if base != self.base {
            self.bytes = graph.region().read_array(record_at(NodeId::new(base)));
            self.base = base;
        }
        let offset = usize::try_from(v.index() - base).unwrap_or_else(|_| corrupt("record offset"))
            * RECORD_BYTES;
        Record::decode(line_record(&self.bytes, offset))
    }
}

fn line_record(bytes: &[u8; RECORD_LINE_BYTES], offset: usize) -> [u8; RECORD_BYTES] {
    let (_, rest) = bytes.split_at(offset);
    let (record, _) = rest.split_at(RECORD_BYTES);
    record.try_into().unwrap_or_else(|_| corrupt("record line"))
}

impl From<WriteError> for Error {
    fn from(error: WriteError) -> Self {
        match error {
            WriteError::Grow(grow) => Self::OutOfMemory(grow),
            WriteError::Overflow(Overflow) => Self::AddressSpaceExhausted,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory(grow) => write!(f, "out of memory: {grow}"),
            Self::AddressSpaceExhausted => f.write_str("address space exhausted"),
            Self::UnknownNode(v) => write!(f, "unknown node {}", v.index()),
            Self::TooManyLabelBuckets(v) => {
                write!(f, "node {} has too many label buckets", v.index())
            }
            Self::Detaching(v) => write!(f, "node {} is being detached", v.index()),
            Self::NotDetaching(v) => write!(f, "node {} is not being detached", v.index()),
            Self::Deleted(v) => write!(f, "node {} is deleted", v.index()),
            Self::WidthConflict { label, declared } => write!(
                f,
                "label {} already has width {}",
                label.get(),
                declared.get()
            ),
            Self::TooManyWidths => f.write_str("too many labels with a width"),
            Self::WrongValueLength {
                label,
                expected,
                actual,
            } => write!(
                f,
                "label {} takes {expected}-byte values, not {actual}",
                label.get()
            ),
        }
    }
}

impl fmt::Display for InitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory(grow) => write!(f, "out of memory: {grow}"),
            Self::NotAGraph => f.write_str("the memories do not hold a graph"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported layout version {v}"),
        }
    }
}

impl core::error::Error for Error {}
impl core::error::Error for InitError {}

mod compaction;
mod detach;
mod index;
mod insert;

pub use insert::{EdgeKind, NewEdge};
mod lifecycle;
mod pair;
mod props;

pub use props::EdgeValue;
mod read;

pub use read::ReadBatch;
mod widths;

use widths::Widths;

use pair::key_of;
#[cfg(test)]
use pair::{opposite, order};

#[cfg(test)]
mod tests;
