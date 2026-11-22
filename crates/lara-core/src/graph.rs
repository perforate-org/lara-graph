//! The graph: labeled buckets of edge halves per vertex and side, over five memories.

use crate::bucket::{self, Window};
use crate::compact::{self, Outcome};
use crate::corrupt::corrupt;
use crate::descriptor::Descriptor;
use crate::id::{Edge, EdgeRef, Halves, Label, Move, Position, Progress, Side, VertexId};
use crate::labels::{self, BucketKey, BucketList, Lookup};
use crate::memory::{GrowFailed, Memory, Region, WriteError};
use crate::page_store::{self, PageStore};
use crate::units::{ByteOffset, Overflow};
use crate::vertex::{RECORD_BYTES, Record};
use alloc::vec::Vec;
use core::fmt;
use core::ops::ControlFlow;

const MAGIC: [u8; 3] = *b"LGV";
const VERSION: u8 = 4;
const AT_COUNT: u64 = 4;
const RECORDS_START: u64 = 16;

/// The memories a graph lives in. Each must be used by this graph only.
#[derive(Debug)]
pub struct Memories<M> {
    vertices: M,
    edge_pages: M,
    edge_directory: M,
    list_pages: M,
    list_directory: M,
}

impl<M> Memories<M> {
    /// `vertices` holds one 8-byte record per vertex; `edge_pages` and `edge_directory` hold
    /// edges; `list_pages` and `list_directory` hold each vertex's per-label buckets.
    pub const fn new(
        vertices: M,
        edge_pages: M,
        edge_directory: M,
        list_pages: M,
        list_directory: M,
    ) -> Self {
        Self {
            vertices,
            edge_pages,
            edge_directory,
            list_pages,
            list_directory,
        }
    }
}

/// A multigraph with dense vertex ids and labeled edges, directed or undirected.
///
/// Every edge is stored at both ends (see [`EdgeRef`]), so traversal runs either way. Each
/// `(vertex, label, side)` has a bucket addressed by [`Position`]. Positions are stable: removing
/// an edge leaves a hole that a later insert may fill, and only [`Graph::compact`] moves edges.
///
/// The two halves of an edge are matched by rank: the k-th live entry to `v` in `u`'s out
/// bucket pairs with the k-th live entry to `u` in `v`'s in bucket (undirected alike). Inserts
/// and compaction keep entries to one vertex in insertion order, so the ranks stay aligned
/// without links between the halves.
///
/// Reading one bucket costs a vertex read, a list read and one data read (a few more past 1024
/// edges). An insert or removal copies at most one run of under 1024 edges or one page of index.
pub struct Graph<M: Memory> {
    vertices: M,
    edges: PageStore<M>,
    lists: PageStore<M>,
    /// Mirrors the vertex count in the header: the graph owns its memories.
    count: u32,
}

/// An operation failed; the graph is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A memory could not grow.
    OutOfMemory(GrowFailed),
    /// The graph outgrew its address space or a bucket outgrew 2^32 positions.
    AddressSpaceExhausted,
    /// The vertex was never added.
    UnknownVertex(VertexId),
    /// The vertex already has the maximum number of `(label, side)` buckets (215).
    TooManyBuckets(VertexId),
    /// The vertex is being detached ([`Graph::detach`]) and takes no new edges.
    Detaching(VertexId),
    /// [`Graph::detach_step`] on a vertex that is not being detached.
    NotDetaching(VertexId),
}

/// Creating a graph failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreateError {
    /// A memory already holds data; open it instead.
    NotEmpty,
    /// A memory could not grow.
    OutOfMemory(GrowFailed),
}

/// Opening a graph failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenError {
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

/// One [`Graph::detach_step`]: the edges removed, both halves of each.
pub type Detachment = Step<Halves>;

/// What [`Graph::remove_edge`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Removal {
    /// The edge with these halves was removed; their positions are now holes.
    Removed(Halves),
    /// No live edge was at that position.
    Absent,
}

impl<M: Memory> Graph<M> {
    /// Formats empty memories as an empty graph.
    ///
    /// # Errors
    ///
    /// [`CreateError::NotEmpty`] if a memory holds data; [`CreateError::OutOfMemory`] if one
    /// cannot grow.
    pub fn create(memories: Memories<M>) -> Result<Self, CreateError> {
        let Memories {
            vertices,
            edge_pages,
            edge_directory,
            list_pages,
            list_directory,
        } = memories;
        if vertices.size() != 0 {
            return Err(CreateError::NotEmpty);
        }
        let edges = PageStore::create(edge_pages, edge_directory).map_err(create_error)?;
        let lists = PageStore::create(list_pages, list_directory).map_err(create_error)?;
        let [m0, m1, m2] = MAGIC;
        Region::new(&vertices)
            .write(ByteOffset::new(0), &[m0, m1, m2, VERSION])
            .map_err(|e| create_error(page_store::CreateError::Write(e)))?;
        let count =
            u32::from_le_bytes(Region::new(&vertices).read_array(ByteOffset::new(AT_COUNT)));
        Ok(Self {
            vertices,
            edges,
            lists,
            count,
        })
    }

    /// Opens a graph written by [`Self::create`]. Reads headers only.
    ///
    /// # Errors
    ///
    /// [`OpenError::NotAGraph`] if the memories do not hold one;
    /// [`OpenError::UnsupportedVersion`] for an unknown layout.
    pub fn open(memories: Memories<M>) -> Result<Self, OpenError> {
        let Memories {
            vertices,
            edge_pages,
            edge_directory,
            list_pages,
            list_directory,
        } = memories;
        if vertices.size() == 0 {
            return Err(OpenError::NotAGraph);
        }
        let [m0, m1, m2, version] = Region::new(&vertices).read_array(ByteOffset::new(0));
        if [m0, m1, m2] != MAGIC {
            return Err(OpenError::NotAGraph);
        }
        if version != VERSION {
            return Err(OpenError::UnsupportedVersion(version));
        }
        let edges = PageStore::open(edge_pages, edge_directory).map_err(open_error)?;
        let lists = PageStore::open(list_pages, list_directory).map_err(open_error)?;
        let count =
            u32::from_le_bytes(Region::new(&vertices).read_array(ByteOffset::new(AT_COUNT)));
        Ok(Self {
            vertices,
            edges,
            lists,
            count,
        })
    }

    /// Number of vertices; ids are `0..vertex_count`.
    #[must_use]
    pub const fn vertex_count(&self) -> u32 {
        self.count
    }

    /// Adds a vertex with no edges and returns its id.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`], or [`Error::AddressSpaceExhausted`] past `u32::MAX - 1` vertices
    /// (the last id is reserved as the removed-edge marker).
    pub fn add_vertex(&mut self) -> Result<VertexId, Error> {
        let id = VertexId::new(self.vertex_count());
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

    /// The labels `v` has buckets under on `side`, in order.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn labels(&self, v: VertexId, side: Side, out: &mut Vec<Label>) -> Result<(), Error> {
        let list = self.list(v)?;
        out.clear();
        out.extend(
            labels::keys(&self.lists.pages(), list)
                .into_iter()
                .filter(|k| k.side == side)
                .map(|k| k.label),
        );
        Ok(())
    }

    /// Number of live edges in `v`'s `(label, side)` bucket.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn degree(&self, v: VertexId, label: Label, side: Side) -> Result<u32, Error> {
        Ok(self
            .bucket(v, BucketKey { label, side })?
            .map_or(0, |d| d.live()))
    }

    /// Replaces the contents of `out` with the live edges of `v`'s `(label, side)` bucket, in
    /// position order. Each edge's target is the vertex at its other end.
    /// [`Self::visit_neighbors`] reads the same edges in about half the instructions when they
    /// need not be kept.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn neighbors(
        &self,
        v: VertexId,
        label: Label,
        side: Side,
        out: &mut Vec<Edge>,
    ) -> Result<(), Error> {
        out.clear();
        if let Some(d) = self.bucket(v, BucketKey { label, side })? {
            bucket::read(&d, &self.edges.pages(), out);
        }
        Ok(())
    }

    /// Calls `f` with each live edge of `v`'s `(label, side)` bucket in position order, until
    /// it breaks; returns the break value. Reads like [`Self::neighbors`] without collecting.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn visit_neighbors<B>(
        &self,
        v: VertexId,
        label: Label,
        side: Side,
        f: impl FnMut(Edge) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        Ok(self.bucket(v, BucketKey { label, side })?.and_then(|d| {
            bucket::visit(&d, &self.edges.pages(), Position::new(0), Window::Page, f)
        }))
    }

    /// The vertex at the other end of `half`, if it is live.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn edge(&self, half: EdgeRef) -> Result<Option<VertexId>, Error> {
        Ok(self
            .bucket(half.vertex(), key_of(half))?
            .and_then(|d| bucket::get(&d, half.position(), &self.edges.pages())))
    }

    /// Removed edges `v`'s `(label, side)` bucket still holds positions for: the work compaction
    /// can do.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`].
    pub fn holes(&self, v: VertexId, label: Label, side: Side) -> Result<u32, Error> {
        Ok(self
            .bucket(v, BucketKey { label, side })?
            .map_or(0, |d| d.stored() - d.live()))
    }

    /// Reclaims removed edges' positions in `v`'s `(label, side)` bucket, one bounded step at a
    /// time: at most one page of edges moves. Live edges keep their relative order, so the
    /// halves stay paired; each one that moves is reported. Call again while
    /// [`Compaction::progress`] is [`Progress::Remaining`].
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVertex`], or [`Error::OutOfMemory`] when a smaller run cannot be
    /// allocated. On error the graph is unchanged.
    pub fn compact(&mut self, v: VertexId, label: Label, side: Side) -> Result<Compaction, Error> {
        let list = self.list(v)?;
        let Lookup::Present { index, descriptor } =
            labels::lookup(&self.lists.pages(), list, BucketKey { label, side })
        else {
            return Ok(Step::new(Vec::new(), Progress::Done));
        };
        let list = list.unwrap_or_else(|| corrupt("present bucket without a list"));
        let step = compact::step(descriptor, &mut self.edges)?;
        match step.outcome {
            Outcome::Bucket(next) => labels::set(&self.lists, list, index, &next),
            Outcome::Empty => {
                let list = labels::remove(&mut self.lists, list, index);
                self.set_list(v, list);
            }
        }
        Ok(Step::new(step.moves, step.progress))
    }

    fn region(&self) -> Region<'_, M> {
        Region::new(&self.vertices)
    }

    fn check(&self, v: VertexId) -> Result<(), Error> {
        if v.index() < self.vertex_count() {
            Ok(())
        } else {
            Err(Error::UnknownVertex(v))
        }
    }

    fn record(&self, v: VertexId) -> Result<Record, Error> {
        self.check(v)?;
        Ok(Record::decode(self.region().read_array(record_at(v))))
    }

    fn list(&self, v: VertexId) -> Result<Option<BucketList>, Error> {
        Ok(self.record(v)?.list)
    }

    /// Rewrites an existing vertex's record. Cannot fail.
    fn write_record(&self, v: VertexId, record: Record) {
        self.region().overwrite(record_at(v), &record.encode());
    }

    /// Publishes `v`'s new bucket list, keeping its state.
    fn set_list(&self, v: VertexId, list: Option<BucketList>) {
        let record = self.list_owner(v);
        self.write_record(v, Record { list, ..record });
    }

    fn list_owner(&self, v: VertexId) -> Record {
        self.record(v).unwrap_or_else(|_| corrupt("a known vertex"))
    }

    fn bucket(&self, v: VertexId, key: BucketKey) -> Result<Option<Descriptor>, Error> {
        Ok(self.find_bucket(v, key)?.map(|found| found.descriptor))
    }
}

fn record_at(v: VertexId) -> ByteOffset {
    ByteOffset::new(RECORDS_START + u64::from(v.index()) * u64::try_from(RECORD_BYTES).unwrap_or(8))
}

fn create_error(error: page_store::CreateError) -> CreateError {
    match error {
        page_store::CreateError::NotEmpty => CreateError::NotEmpty,
        page_store::CreateError::Write(WriteError::Grow(grow)) => CreateError::OutOfMemory(grow),
        page_store::CreateError::Write(WriteError::Overflow(Overflow)) => {
            unreachable!("a header at offset 0 cannot overflow the address space")
        }
    }
}

fn open_error(error: page_store::OpenError) -> OpenError {
    match error {
        page_store::OpenError::NotAPageStore => OpenError::NotAGraph,
        page_store::OpenError::UnsupportedVersion(v) => OpenError::UnsupportedVersion(v),
    }
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
            Self::UnknownVertex(v) => write!(f, "unknown vertex {}", v.index()),
            Self::TooManyBuckets(v) => write!(f, "vertex {} has too many buckets", v.index()),
            Self::Detaching(v) => write!(f, "vertex {} is being detached", v.index()),
            Self::NotDetaching(v) => write!(f, "vertex {} is not being detached", v.index()),
        }
    }
}

impl fmt::Display for CreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotEmpty => f.write_str("memory is not empty"),
            Self::OutOfMemory(grow) => write!(f, "out of memory: {grow}"),
        }
    }
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAGraph => f.write_str("memories do not hold a graph"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported layout version {v}"),
        }
    }
}

impl core::error::Error for Error {}
impl core::error::Error for CreateError {}
impl core::error::Error for OpenError {}

mod detach;
mod insert;
mod pair;

use pair::key_of;
#[cfg(test)]
use pair::{opposite, order};

#[cfg(test)]
mod tests;
