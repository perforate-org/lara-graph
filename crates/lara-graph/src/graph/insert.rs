//! Inserting edges: every half or none, each with its value when the label has a width.
//!
//! [`admit`] checks an insert before anything changes, [`apply`] plans, reserves and writes its
//! halves, [`batch`] runs [`Graph::add_edges`] over the caches in [`cache`].

mod admit;
mod apply;
mod batch;
mod cache;

use super::{Error, Graph, reserved};
use crate::descriptor::{self, Descriptor};
use crate::id::{Direction, EdgeRef, Halves, Label, NodeId, Placement, Width};
use crate::label_bucket;
use crate::label_bucket_list::{EntryAt, LabelBucketKey};
use crate::memory::Memory;
use crate::node::Record;
use crate::stored::{LabelBucket, Stored, Values};
use crate::targets;
use crate::values::{self, Plane};
use apply::HoleCursor;
use cache::{ManyCache, NoCache, OneCache};

/// Whether [`NewEdge`] is directed or undirected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// `from -[label]-> to`.
    Directed,
    /// `a -[label]- b`, stored at both ends with one undirected half each.
    Undirected,
}

/// One edge for [`Graph::add_edges`].
#[derive(Clone, Copy, Debug)]
pub struct NewEdge<'a> {
    kind: EdgeKind,
    from: NodeId,
    label: Label,
    to: NodeId,
    placement: Placement,
    value: &'a [u8],
}

impl<'a> NewEdge<'a> {
    /// The directed edge `from -[label]-> to`, placed by `placement` in both buckets.
    #[must_use]
    pub const fn directed(
        from: NodeId,
        label: Label,
        to: NodeId,
        placement: Placement,
        value: &'a [u8],
    ) -> Self {
        Self {
            kind: EdgeKind::Directed,
            from,
            label,
            to,
            placement,
            value,
        }
    }

    /// The undirected edge `a -[label]- b`, placed by `placement` in both buckets.
    #[must_use]
    pub const fn undirected(
        a: NodeId,
        label: Label,
        b: NodeId,
        placement: Placement,
        value: &'a [u8],
    ) -> Self {
        Self {
            kind: EdgeKind::Undirected,
            from: a,
            label,
            to: b,
            placement,
            value,
        }
    }
}

impl NewEdge<'_> {
    pub(crate) const fn kind(&self) -> EdgeKind {
        self.kind
    }

    pub(crate) const fn from(&self) -> NodeId {
        self.from
    }

    pub(crate) const fn label(&self) -> Label {
        self.label
    }

    pub(crate) const fn to(&self) -> NodeId {
        self.to
    }

    pub(crate) const fn value(&self) -> &[u8] {
        self.value
    }

    /// A directed edge's halves: outgoing at `from`, then incoming at `to`.
    const fn directed_halves(&self) -> [Half; 2] {
        [
            Half::new(
                self.from,
                (self.label, Direction::Outgoing),
                self.to,
                self.placement,
            ),
            Half::new(
                self.to,
                (self.label, Direction::Incoming),
                self.from,
                self.placement,
            ),
        ]
    }

    /// An undirected edge's halves, normalized by node id: at the high end, then at the low end
    /// unless the edge is a self-loop, which has one half.
    fn undirected_halves(&self) -> (Half, Option<Half>) {
        let (low, high) = (self.from.min(self.to), self.from.max(self.to));
        let at_high = Half::new(
            high,
            (self.label, Direction::Undirected),
            low,
            self.placement,
        );
        if low == high {
            return (at_high, None);
        }
        let at_low = Half::new(
            low,
            (self.label, Direction::Undirected),
            high,
            self.placement,
        );
        (at_high, Some(at_low))
    }
}

impl<M: Memory> Graph<M> {
    /// Adds one edge, placed by its placement in both buckets, with its value in both halves: a
    /// [`NewEdge::directed`] edge `from -[label]-> to`, or a [`NewEdge::undirected`] one, which is
    /// normalized by node id. A directed self-loop has two halves; an undirected self-loop has
    /// one. Parallel edges keep their relative order either way. [`Self::add_edges`] takes the
    /// same edges in a batch.
    ///
    /// The value is exactly the label's width ([`Self::declare_width`]) long, or empty for a label
    /// without one.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for either endpoint, [`Error::Detaching`] or [`Error::Deleted`]
    /// for an endpoint being or already deleted, [`Error::WrongValueLength`],
    /// [`Error::TooManyLabelBuckets`], [`Error::OutOfMemory`] or [`Error::AddressSpaceExhausted`].
    /// On error the graph is unchanged.
    pub fn add_edge(&mut self, edge: NewEdge<'_>) -> Result<Halves, Error> {
        match edge.kind {
            EdgeKind::Directed => {
                let [canonical, other] = self.insert_halves(edge.directed_halves(), edge.value)?;
                Ok(Halves::new(canonical, other))
            }
            EdgeKind::Undirected => match edge.undirected_halves() {
                (at_high, None) => {
                    let [half] = self.insert_halves([at_high], edge.value)?;
                    Ok(Halves::new(half, half))
                }
                (at_high, Some(at_low)) => {
                    let [canonical, other] = self.insert_halves([at_high, at_low], edge.value)?;
                    Ok(Halves::new(canonical, other))
                }
            },
        }
    }

    /// Inserts every half or none: checks and reserves memory first, then applies them.
    fn insert_halves<const N: usize>(
        &mut self,
        halves: [Half; N],
        value: &[u8],
    ) -> Result<[EdgeRef; N], Error> {
        Ok(self.insert_halves_with(halves, value, &NoCache, None)?.0)
    }

    /// [`insert_halves`] with an optional already admitted half.
    fn insert_halves_cached<const N: usize>(
        &mut self,
        halves: [Half; N],
        value: &[u8],
        cached: Option<(usize, Place)>,
        cursor: Option<(usize, &mut HoleCursor)>,
    ) -> Result<([EdgeRef; N], [Option<Place>; N]), Error> {
        self.insert_halves_with(halves, value, &OneCache(cached), cursor)
    }

    /// [`insert_halves`] with both halves' places already admitted.
    fn insert_halves_two<const N: usize>(
        &mut self,
        halves: [Half; N],
        value: &[u8],
        cached: [Option<Place>; N],
    ) -> Result<([EdgeRef; N], [Option<Place>; N]), Error> {
        self.insert_halves_with(halves, value, &ManyCache(cached), None)
    }

    /// The label's width, checking that `value` matches it.
    pub(super) fn value_width(&self, label: Label, value: &[u8]) -> Result<Option<Width>, Error> {
        let width = self.widths.get(label);
        let expected = width.map_or(0, Width::get);
        if usize::from(expected) == value.len() {
            Ok(width)
        } else {
            Err(Error::WrongValueLength {
                label,
                expected,
                actual: value.len(),
            })
        }
    }

    /// `bucket`, which has no values, ready for values of `width`, all zero: a small bucket as a
    /// run with them beside its slots, others (never tiny) with an empty value plane, which
    /// [`Self::publish`] gives a header. Within reserved memory.
    pub(super) fn with_values(&mut self, bucket: LabelBucket, width: Width) -> LabelBucket {
        let small = match bucket.slots {
            Descriptor::Tiny(_) => true,
            Descriptor::Run(run) => u32::from(run.stored) < descriptor::max_beside(width),
            Descriptor::Paged(_) => false,
        };
        if small && descriptor::beside(width) && bucket.header.is_none() {
            let slots = reserved(label_bucket::with_values_beside(
                bucket.slots,
                width,
                &mut self.edges,
            ));
            return LabelBucket { slots, ..bucket };
        }
        let slots = reserved(label_bucket::without_tiny(bucket.slots, &mut self.edges));
        LabelBucket {
            slots,
            values: Some(Values {
                plane: Plane::Empty,
                width,
            }),
            ..bucket
        }
    }
}

/// Pages one half can take from the edge store: its slots, a header, the target index's change,
/// and for a label with a width, a run replacing a tiny bucket and the value's write.
pub(super) fn pages_per_half(width: Option<Width>) -> u32 {
    label_bucket::PAGES_PER_INSERT
        + 1
        + targets::PAGES_PER_CHANGE
        + width.map_or(0, |_| 1 + values::PAGES_PER_WRITE)
}

/// Where a half goes: its node's record and its bucket, or where the bucket belongs.
#[derive(Clone, Copy)]
struct Place {
    /// The node's record when the half was admitted.
    record: Record,
    target: Target,
}

#[derive(Clone, Copy)]
enum Target {
    /// The bucket's list entry, resolved only where its header matters.
    Present(EntryAt, Stored),
    Absent(EntryAt),
}

/// One half to insert: an entry to `target` in `node`'s bucket `key`.
#[derive(Clone, Copy)]
struct Half {
    node: NodeId,
    key: LabelBucketKey,
    target: NodeId,
    placement: Placement,
}

impl Half {
    const fn new(
        node: NodeId,
        (label, direction): (Label, Direction),
        target: NodeId,
        placement: Placement,
    ) -> Self {
        Self {
            node,
            key: LabelBucketKey { label, direction },
            target,
            placement,
        }
    }
}
