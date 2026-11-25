//! The errors a graph's operations return.

#[cfg(doc)]
use crate::graph::Graph;
use crate::id::{Label, NodeId, NodeState, Width};
use crate::store::memory::{GrowFailed, WriteError};
use crate::store::units::Overflow;
use core::fmt;

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
    /// [`Graph::bulk_load`] was called on a graph that already holds edges.
    NotEmpty,
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

/// A batch operation failed at `index`, with whether earlier elements were applied.
///
/// Only `add_edges` returns [`Self::PartiallyApplied`]; `set_edge_values`, `remove_edges` and
/// `detach_many` restore or avoid earlier changes on error. On the Internet Computer, a
/// canister can trap on [`Self::PartiallyApplied`] to make the update call all-or-none; the IC
/// discards the call's state changes when the call traps.
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
            Self::NotEmpty => f.write_str("the graph already holds edges"),
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

impl Error {
    /// `Ok` for a live node `v` in `state`; otherwise the error an edge to it fails with.
    pub(crate) const fn unless_live(state: NodeState, v: NodeId) -> Result<(), Self> {
        match state {
            NodeState::Live => Ok(()),
            NodeState::Detaching => Err(Self::Detaching(v)),
            NodeState::Deleted => Err(Self::Deleted(v)),
        }
    }
}

impl core::error::Error for Error {}
impl core::error::Error for BatchError {}
impl core::error::Error for InitError {}
