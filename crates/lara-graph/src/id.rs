//! Public identifiers: of nodes, labels, positions, edges and their halves, value widths, and a
//! node's state.

use core::cmp::Ordering;
use core::fmt;
use core::num::{NonZeroU16, NonZeroU32};

/// A node: a local node, or a proxy for a node another graph holds. Local nodes are dense: a
/// graph with `n` of them has ids `0..n`, below 2^31. Proxies have the top bit set: the graph's
/// proxies are `2^31 + 0..2^31 + p` ([`Self::proxy`]), so a read tells a proxy from its id alone.
///
/// It holds its index's complement, which is never zero, so that `Option<NodeId>` and a slot
/// (an edge or a tombstone) take four bytes, not eight.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(NonZeroU32);

impl NodeId {
    /// The node with dense index `index`.
    ///
    /// # Panics
    ///
    /// When `index` is `u32::MAX`, which no node has.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        match NonZeroU32::new(!index) {
            Some(complement) => Self(complement),
            None => panic!("u32::MAX is no node's index"),
        }
    }

    /// The dense index of this node: below 2^31 for a local node, 2^31 and above for a proxy.
    #[must_use]
    pub const fn index(self) -> u32 {
        !self.0.get()
    }

    /// The `ordinal`-th proxy, index `2^31 + ordinal`.
    ///
    /// # Panics
    ///
    /// When `ordinal` is `2^31 - 1` or more: the last index, `u32::MAX`, is no node's.
    #[must_use]
    pub const fn proxy(ordinal: u32) -> Self {
        assert!(ordinal < Self::PROXIES, "past the last proxy");
        Self::new(Self::FIRST_PROXY + ordinal)
    }

    /// Whether this is a proxy: its index has the top bit set.
    #[must_use]
    pub const fn is_proxy(self) -> bool {
        self.index() >= Self::FIRST_PROXY
    }

    /// The proxy's ordinal (its index without the top bit), or `None` for a local node.
    #[must_use]
    pub const fn proxy_ordinal(self) -> Option<u32> {
        if self.is_proxy() {
            Some(self.index() - Self::FIRST_PROXY)
        } else {
            None
        }
    }

    /// The first proxy's index, and how many local nodes a graph can hold.
    pub(crate) const FIRST_PROXY: u32 = 1 << 31;
    /// How many proxies a graph can hold: every index from [`Self::FIRST_PROXY`] but `u32::MAX`.
    pub(crate) const PROXIES: u32 = u32::MAX - Self::FIRST_PROXY;

    /// The node whose index's complement is `complement`: how a slot stores it.
    pub(crate) const fn from_complement(complement: NonZeroU32) -> Self {
        Self(complement)
    }

    /// The complement of this node's index, never zero.
    pub(crate) const fn complement(self) -> NonZeroU32 {
        self.0
    }
}

impl Ord for NodeId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index().cmp(&other.index())
    }
}

impl PartialOrd for NodeId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("NodeId").field(&self.index()).finish()
    }
}

/// An edge label. Each `(node, label)` pair has its own bucket of edges; labels order
/// buckets by their raw value (callers may pack flags such as directedness into it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Label(u16);

impl Label {
    /// The label with raw value `raw`.
    #[must_use]
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    /// The raw value.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Bytes in each value of a label's edges: 1 to 4096.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Width(NonZeroU16);

impl Width {
    /// The widest value: one per value page.
    pub const MAX: u16 = 4096;

    /// A width of `bytes`, if it is 1 to [`Self::MAX`].
    #[must_use]
    pub const fn new(bytes: u16) -> Option<Self> {
        match NonZeroU16::new(bytes) {
            Some(bytes) if bytes.get() <= Self::MAX => Some(Self(bytes)),
            Some(_) | None => None,
        }
    }

    /// A width as stored, 0 for none: one comparison, as stored widths were checked when written.
    pub(crate) const fn stored(bytes: u16) -> Option<Self> {
        match NonZeroU16::new(bytes) {
            Some(bytes) => Some(Self(bytes)),
            None => None,
        }
    }

    /// The width in bytes.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

/// Where a node is in its life. Node ids are never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeState {
    /// Takes and holds edges.
    Live,
    /// Being deleted: its edges are removed step by step and it takes no new ones.
    Detaching,
    /// Deleted: it has no edges and takes none.
    Deleted,
}

/// Which of a node's adjacencies a bucket holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Direction {
    /// Directed edges leaving the node.
    Outgoing,
    /// Directed edges entering the node.
    Incoming,
    /// Undirected edges at the node.
    Undirected,
}

impl Direction {
    /// The side the other half of an edge is on.
    pub(crate) const fn opposite(self) -> Self {
        match self {
            Self::Outgoing => Self::Incoming,
            Self::Incoming => Self::Outgoing,
            Self::Undirected => Self::Undirected,
        }
    }
}

/// A position in one `(node, label)` bucket. Positions count removed edges, so an edge keeps
/// its position until the bucket is compacted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Position(u32);

impl Position {
    /// The position `index`.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The index of this position.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// One live entry of a bucket, as a read from its node sees it: the neighbour it points to and
/// its position. The node, label and direction are the read's own; [`EdgeRef`] names the same
/// half on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Neighbor {
    position: Position,
    target: NodeId,
}

impl Neighbor {
    pub(crate) const fn new(position: Position, target: NodeId) -> Self {
        Self { position, target }
    }

    /// The entry's position in its bucket.
    #[must_use]
    pub const fn position(&self) -> Position {
        self.position
    }

    /// The neighbour: the node at the edge's other end.
    #[must_use]
    pub const fn target(&self) -> NodeId {
        self.target
    }
}

/// One half of an edge: its entry in one bucket. A directed edge `u → v` has an
/// [`Direction::Outgoing`] half at `u` and an [`Direction::Incoming`] half at `v`; an undirected
/// edge `u — v` has a [`Direction::Undirected`] half at each end, and a single one when `u = v`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeRef {
    node: NodeId,
    label: Label,
    direction: Direction,
    position: Position,
}

impl EdgeRef {
    /// The entry at `position` in `node`'s `(label, direction)` bucket.
    #[must_use]
    pub const fn new(node: NodeId, label: Label, direction: Direction, position: Position) -> Self {
        Self {
            node,
            label,
            direction,
            position,
        }
    }

    /// The node whose bucket holds this half.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// The edge's label.
    #[must_use]
    pub const fn label(&self) -> Label {
        self.label
    }

    /// Which bucket of the node.
    #[must_use]
    pub const fn direction(&self) -> Direction {
        self.direction
    }

    /// The position in that bucket.
    #[must_use]
    pub const fn position(&self) -> Position {
        self.position
    }

    /// Whether this is its edge's canonical half (see [`Halves`]), `other_end` being the node at
    /// the edge's other end: the half at a local node when the other end is a proxy; otherwise
    /// the out half of a directed edge, or the undirected half at the larger node.
    #[must_use]
    pub const fn is_canonical(&self, other_end: NodeId) -> bool {
        match (self.node.is_proxy(), other_end.is_proxy()) {
            // A proxy's id means nothing outside its graph: the local half is the handle.
            (false, true) => true,
            (true, false) => false,
            (false, false) | (true, true) => match self.direction {
                Direction::Outgoing => true,
                Direction::Incoming => false,
                Direction::Undirected => self.node.index() >= other_end.index(),
            },
        }
    }
}

/// The halves of one edge, the canonical one first: where a caller keys what belongs to the
/// edge (properties, index postings). For an edge to a proxy it is the half at the local node,
/// as a proxy's id means nothing outside its graph; otherwise, for a directed edge it is the out
/// half, and for an undirected edge, the half at the larger node id. Ids grow in creation order and hubs tend to be old, so
/// this puts a hub's undirected edges' canonical halves in its newer neighbours' small buckets:
/// the keyed data spreads out, and compacting a hub moves few canonical halves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Halves {
    /// An edge stored at both ends.
    Pair {
        /// The out half, or the undirected half at the larger node.
        canonical: EdgeRef,
        /// The in half, or the undirected half at the smaller node.
        other: EdgeRef,
    },
    /// An undirected self-loop: its one entry is both halves.
    Single(EdgeRef),
}

impl Halves {
    /// An edge's halves from any half and its counterpart.
    pub(crate) fn of(half: EdgeRef, other: EdgeRef) -> Self {
        if half.is_canonical(other.node()) {
            Self::new(half, other)
        } else {
            Self::new(other, half)
        }
    }

    pub(crate) fn new(canonical: EdgeRef, other: EdgeRef) -> Self {
        if canonical == other {
            Self::Single(canonical)
        } else {
            Self::Pair { canonical, other }
        }
    }

    /// The canonical half: the out half, or the undirected half at the larger node.
    #[must_use]
    pub const fn canonical(&self) -> EdgeRef {
        match *self {
            Self::Pair { canonical, .. } | Self::Single(canonical) => canonical,
        }
    }

    /// The other half (the canonical one again for an undirected self-loop).
    #[must_use]
    pub const fn other(&self) -> EdgeRef {
        match *self {
            Self::Pair { other, .. } | Self::Single(other) => other,
        }
    }
}

/// Proofs over every input, checked by Kani (`cargo kani -p lara-graph`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Every proxy ordinal names a proxy that gives it back; every other index is local.
    #[kani::proof]
    fn proxies_round_trip() {
        let ordinal: u32 = kani::any();
        kani::assume(ordinal < NodeId::PROXIES);
        let proxy = NodeId::proxy(ordinal);
        assert!(proxy.is_proxy());
        assert_eq!(proxy.proxy_ordinal(), Some(ordinal));
        let index: u32 = kani::any();
        kani::assume(index < NodeId::FIRST_PROXY);
        assert_eq!(NodeId::new(index).proxy_ordinal(), None);
    }

    /// Exactly one half of an edge between two nodes is canonical, and it is never a proxy's
    /// when the other end is local.
    #[kani::proof]
    fn one_half_is_canonical() {
        let (a, b): (u32, u32) = (kani::any(), kani::any());
        kani::assume(a != u32::MAX && b != u32::MAX && a != b);
        let (a, b) = (NodeId::new(a), NodeId::new(b));
        kani::assume(!(a.is_proxy() && b.is_proxy()));
        let (label, at) = (Label::new(kani::any()), Position::new(kani::any()));
        let directed = kani::any::<bool>();
        let (here, there) = if directed {
            (Direction::Outgoing, Direction::Incoming)
        } else {
            (Direction::Undirected, Direction::Undirected)
        };
        let at_a = EdgeRef::new(a, label, here, at);
        let at_b = EdgeRef::new(b, label, there, at);
        let (a_canonical, b_canonical) = (at_a.is_canonical(b), at_b.is_canonical(a));
        assert!(a_canonical != b_canonical);
        if a.is_proxy() != b.is_proxy() {
            assert_eq!(a_canonical, !a.is_proxy());
        }
    }
}
