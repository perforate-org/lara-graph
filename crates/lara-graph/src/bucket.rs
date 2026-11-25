//! One label bucket: [`LabelBucket`], named by its [`LabelBucketKey`], and its storage: its
//! descriptor, its slots in a run or pages, its value plane, its target index, the header that
//! ties them, and compaction. Knows nothing of nodes or graphs.

pub(crate) mod compact;
pub(crate) mod descriptor;
pub(crate) mod paged;
pub(crate) mod slots;
pub(crate) mod stored;
pub(crate) mod target_index;
pub(crate) mod values;

use crate::bucket::descriptor::Descriptor;
use crate::bucket::target_index::Root;
use crate::bucket::values::Plane;
use crate::id::{Direction, EdgeRef, Label, Width};
use crate::store::page_store::Extent;

/// Which label bucket of a node: a label in one direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct LabelBucketKey {
    pub(crate) label: Label,
    pub(crate) direction: Direction,
}

impl LabelBucketKey {
    /// The key of the bucket that holds `half`.
    pub(crate) const fn of(half: EdgeRef) -> Self {
        Self {
            label: half.label(),
            direction: half.direction(),
        }
    }

    /// The key of the bucket that holds the other halves of this one's edges.
    pub(crate) const fn opposite(self) -> Self {
        Self {
            label: self.label,
            direction: self.direction.opposite(),
        }
    }
}

/// A bucket's values: the plane and the width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Values {
    pub(crate) plane: Plane,
    pub(crate) width: Width,
}

/// A bucket's target index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Index {
    /// No postings: a bucket too small for an index, or a paged one whose complete blocks hold
    /// no live edges.
    Empty,
    /// The tree of postings.
    Tree(Root),
    /// Released while its node detaches, as all its edges are going: lookups scan the bucket.
    Dropped,
}

impl Index {
    /// The tree, if there is one.
    pub(crate) const fn tree(self) -> Option<Root> {
        match self {
            Self::Tree(root) => Some(root),
            Self::Empty | Self::Dropped => None,
        }
    }

    /// The index holding `root`.
    pub(crate) const fn of(root: Option<Root>) -> Self {
        match root {
            Some(root) => Self::Tree(root),
            None => Self::Empty,
        }
    }
}

/// A label bucket resolved from its label bucket list entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LabelBucket {
    pub(crate) slots: Descriptor,
    /// The header, while the bucket has values in a plane or an index.
    pub(crate) header: Option<Extent>,
    /// Values in a value plane; values beside a run's slots are not here (see [`Self::cells`]).
    pub(crate) values: Option<Values>,
    pub(crate) index: Index,
}

impl LabelBucket {
    /// The end of the positions the index covers; `None` when lookups scan the whole bucket (too
    /// small for an index, or its index dropped).
    pub(crate) fn indexed(&self) -> Option<u32> {
        match self.index {
            Index::Dropped => None,
            Index::Empty | Index::Tree(_) => self.slots.indexed_end(),
        }
    }

    /// A bucket with only slots.
    pub(crate) const fn plain(slots: Descriptor) -> Self {
        Self {
            slots,
            header: None,
            values: None,
            index: Index::Empty,
        }
    }

    /// The bucket's values: beside a run's slots, or in the value plane its header holds.
    pub(crate) fn cells(&self) -> Option<Values> {
        match self.slots {
            Descriptor::Run(run) => run.values.map_or(self.values, |width| {
                Some(Values {
                    plane: Plane::beside(run.extent.page(), run.cells_offset(), run.stored),
                    width,
                })
            }),
            Descriptor::Tiny(_) | Descriptor::Paged(_) => self.values,
        }
    }

    /// Records the plane a value write left: values beside a run's slots never move.
    pub(crate) fn set_plane(&mut self, plane: Plane) {
        if let Some(values) = self.values.as_mut() {
            values.plane = plane;
        }
    }

    /// Whether the bucket needs a header: it has values in a plane, or an index.
    pub(crate) const fn extended(&self) -> bool {
        self.values.is_some() || !matches!(self.index, Index::Empty)
    }
}

/// Pages one half can take from the edge store: its slots, a header, the target index's change,
/// and for a label with a width, a run replacing a tiny bucket and the value's write.
pub(crate) fn pages_per_half(width: Option<Width>) -> u32 {
    slots::PAGES_PER_INSERT
        + 1
        + target_index::PAGES_PER_CHANGE
        + width.map_or(0, |_| 1 + values::PAGES_PER_WRITE)
}

/// The result of an update that runs within reserved memory: [`pages_per_half`] and the other
/// reservations cover it, so running out is a bug.
pub(crate) fn reserved<T, E>(result: Result<T, E>) -> T {
    result.unwrap_or_else(|_| crate::store::corrupt::reservation_exceeded())
}
