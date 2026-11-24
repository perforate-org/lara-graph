//! Inserting edges: every half or none, each with its value when the label has a width.

use super::index::{Batch, Between};
use super::{BatchError, Error, Graph, reserved};
use crate::corrupt::corrupt;
use crate::descriptor::{Descriptor, Slot};
use crate::id::{Direction, EdgeRef, Halves, Label, NodeId, NodeState, Placement, Position, Width};
use crate::label_bucket::{self, Spot};
use crate::label_bucket_list::{self, EntryAt, LabelBucketKey, LabelBucketList, Lookup};
use crate::memory::Memory;
use crate::node::Record;
use crate::page_store::Pages;
use crate::stored::{LabelBucket, Stored, Values};
use crate::targets;
use crate::values::{self, Plane};
use alloc::vec::Vec;

/// Target places an `add_edges` batch remembers. Eight covers the common fan-in/fan-out set.
const TARGET_CACHE: usize = 8;
/// A mixed batch keeps both directions and both undirected ends, so it needs a few more.
const MIXED_CACHE: usize = 16;

/// A batch-local cursor into a paged bucket's holes. It is only used for fresh targets, so the
/// holes before `next` have been filled by this run.
struct HoleCursor {
    key: Option<(NodeId, LabelBucketKey)>,
    next: Position,
}

impl Default for HoleCursor {
    fn default() -> Self {
        Self {
            key: None,
            next: Position::new(0),
        }
    }
}

/// Whether a target repeats near the start of the batch. The probe is capped so a long unique
/// fan-out does not pay for a set it will not use; a repeated run of edges is visible early.
#[inline(never)]
fn repeated_targets(edges: &[NewEdge<'_>]) -> bool {
    let sample = edges.get(..64).unwrap_or(edges);
    let wanted = sample.len().saturating_mul(2).clamp(16, 128);
    let size = wanted.next_power_of_two();
    let mask = size - 1;
    let mut slots: Vec<u32> = Vec::new();
    slots.resize(size, u32::MAX);
    for edge in sample {
        let index = usize::try_from(edge.to.index()).unwrap_or(usize::MAX);
        let mut at = index & mask;
        loop {
            match slots.get(at).copied() {
                Some(u32::MAX) => {
                    if let Some(slot) = slots.get_mut(at) {
                        *slot = edge.to.index();
                    }
                    break;
                }
                Some(seen) if seen == edge.to.index() => return true,
                Some(_) => at = (at + 1) & mask,
                None => return true,
            }
        }
    }
    false
}

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

impl<M: Memory> Graph<M> {
    /// Adds every edge in `edges`, in order. Each edge is all-or-none like [`Self::add_edge`],
    /// but the batch is not atomic: an error leaves the edges before it inserted.
    ///
    /// All endpoints are checked and every value length is validated before the first write.
    /// Edges with the same `from` and `label` should be adjacent; the batch can then reuse that
    /// node's bucket state.
    ///
    /// # Errors
    ///
    /// As [`Self::add_edge`], for the edge that fails.
    pub fn add_edges(&mut self, edges: &[NewEdge<'_>]) -> Result<Vec<Halves>, BatchError> {
        for (index, edge) in edges.iter().enumerate() {
            self.check(edge.from)
                .and_then(|()| self.check(edge.to))
                .and_then(|()| self.value_width(edge.label, edge.value))
                .map_err(|error| BatchError::unchanged(index, error))?;
        }
        let all_directed = edges.iter().all(|edge| edge.kind == EdgeKind::Directed);
        if all_directed && repeated_targets(edges) {
            return self.add_edges_repeated(edges);
        }
        if edges.iter().all(|edge| edge.kind == EdgeKind::Undirected) {
            return self.add_edges_undirected(edges);
        }
        if edges.iter().any(|edge| edge.kind == EdgeKind::Undirected) {
            return self.add_edges_mixed(edges);
        }
        let mut out = Vec::with_capacity(edges.len());
        let mut cache: Option<(NodeId, LabelBucketKey, Place)> = None;
        let mut holes = HoleCursor::default();
        for (index, edge) in edges.iter().enumerate() {
            if edge.kind == EdgeKind::Undirected {
                self.add_undirected_batch_edge(edge, &mut out)
                    .map_err(|error| BatchError::after_prefix(index, error))?;
                cache = None;
                continue;
            }
            let key = LabelBucketKey {
                label: edge.label,
                direction: Direction::Outgoing,
            };
            let cached = cache
                .filter(|(last_node, last_key, _)| *last_node == edge.from && *last_key == key)
                .map(|(_, _, place)| (0, place));
            let from = Half::new(
                edge.from,
                (edge.label, Direction::Outgoing),
                edge.to,
                edge.placement,
            );
            let to = Half::new(
                edge.to,
                (edge.label, Direction::Incoming),
                edge.from,
                edge.placement,
            );
            let (halves, updated) = self
                .insert_halves_cached([from, to], edge.value, cached, Some((0, &mut holes)))
                .map_err(|error| BatchError::after_prefix(index, error))?;
            let [canonical, other] = halves;
            out.push(Halves::new(canonical, other));
            cache = updated
                .first()
                .and_then(|place| *place)
                .map(|place| (edge.from, key, place));
        }
        Ok(out)
    }

    /// Undirected batches cache both endpoints by `(node, key)`. A hub is usually the low or
    /// the high end of every edge, so one side stays hot while the other moves.
    /// A batch with both edge kinds: every half is cached by its endpoint, independent of the
    /// half's role in a directed or undirected edge.
    #[inline(never)]
    fn add_edges_mixed(&mut self, edges: &[NewEdge<'_>]) -> Result<Vec<Halves>, BatchError> {
        let mut out = Vec::with_capacity(edges.len());
        let mut cache: [Option<(NodeId, LabelBucketKey, Place)>; MIXED_CACHE] = [None; MIXED_CACHE];
        for (index, edge) in edges.iter().enumerate() {
            match edge.kind {
                EdgeKind::Directed => {
                    let from = Half::new(
                        edge.from,
                        (edge.label, Direction::Outgoing),
                        edge.to,
                        edge.placement,
                    );
                    let to = Half::new(
                        edge.to,
                        (edge.label, Direction::Incoming),
                        edge.from,
                        edge.placement,
                    );
                    let cached = [
                        cached_place(&cache, from.node, from.key),
                        cached_place(&cache, to.node, to.key),
                    ];
                    let (halves, updated) = self
                        .insert_halves_with([from, to], edge.value, &ManyCache(cached), None)
                        .map_err(|error| BatchError::after_prefix(index, error))?;
                    remember_halves(&mut cache, [from, to], updated);
                    let [canonical, other] = halves;
                    out.push(Halves::new(canonical, other));
                }
                EdgeKind::Undirected => {
                    let (low, high) = (edge.from.min(edge.to), edge.from.max(edge.to));
                    let at_high = Half::new(
                        high,
                        (edge.label, Direction::Undirected),
                        low,
                        edge.placement,
                    );
                    if low == high {
                        let cached = [cached_place(&cache, at_high.node, at_high.key)];
                        let (halves, updated) = self
                            .insert_halves_with([at_high], edge.value, &ManyCache(cached), None)
                            .map_err(|error| BatchError::after_prefix(index, error))?;
                        remember_halves(&mut cache, [at_high], updated);
                        let [half] = halves;
                        out.push(Halves::new(half, half));
                        continue;
                    }
                    let at_low = Half::new(
                        low,
                        (edge.label, Direction::Undirected),
                        high,
                        edge.placement,
                    );
                    let cached = [
                        cached_place(&cache, at_high.node, at_high.key),
                        cached_place(&cache, at_low.node, at_low.key),
                    ];
                    let (halves, updated) = self
                        .insert_halves_with([at_high, at_low], edge.value, &ManyCache(cached), None)
                        .map_err(|error| BatchError::after_prefix(index, error))?;
                    remember_halves(&mut cache, [at_high, at_low], updated);
                    let [canonical, other] = halves;
                    out.push(Halves::new(canonical, other));
                }
            }
        }
        Ok(out)
    }

    fn add_edges_undirected(&mut self, edges: &[NewEdge<'_>]) -> Result<Vec<Halves>, BatchError> {
        let mut out = Vec::with_capacity(edges.len());
        let mut high_cache: [Option<(NodeId, LabelBucketKey, Place)>; TARGET_CACHE] =
            [None; TARGET_CACHE];
        let mut low_cache: [Option<(NodeId, LabelBucketKey, Place)>; TARGET_CACHE] =
            [None; TARGET_CACHE];
        for (index, edge) in edges.iter().enumerate() {
            let (low, high) = (edge.from.min(edge.to), edge.from.max(edge.to));
            let key = LabelBucketKey {
                label: edge.label,
                direction: Direction::Undirected,
            };
            let cache_high = cached_place(&high_cache, high, key);
            let cache_low = cached_place(&low_cache, low, key);
            let at_high = Half::new(
                high,
                (edge.label, Direction::Undirected),
                low,
                edge.placement,
            );
            if low == high {
                let (halves, updated) = self
                    .insert_halves_two([at_high], edge.value, [cache_high])
                    .map_err(|error| BatchError::after_prefix(index, error))?;
                let [half] = halves;
                out.push(Halves::new(half, half));
                if let Some(place) = updated.first().and_then(|place| *place) {
                    cache_place(&mut high_cache, high, key, Some(place));
                } else {
                    forget_cache_node(&mut high_cache, high);
                    forget_cache_node(&mut low_cache, low);
                }
                continue;
            }
            let at_low = Half::new(
                low,
                (edge.label, Direction::Undirected),
                high,
                edge.placement,
            );
            let (halves, updated) = self
                .insert_halves_two([at_high, at_low], edge.value, [cache_high, cache_low])
                .map_err(|error| BatchError::after_prefix(index, error))?;
            let [canonical, other] = halves;
            out.push(Halves::new(canonical, other));
            if updated.first().and_then(|place| *place).is_none() {
                forget_cache_node(&mut high_cache, high);
                forget_cache_node(&mut low_cache, high);
            }
            if updated.get(1).and_then(|place| *place).is_none() {
                forget_cache_node(&mut high_cache, low);
                forget_cache_node(&mut low_cache, low);
            }
            if let Some(place) = updated.first().and_then(|place| *place) {
                cache_place(&mut high_cache, high, key, Some(place));
            }
            if let Some(place) = updated.get(1).and_then(|place| *place) {
                cache_place(&mut low_cache, low, key, Some(place));
            }
        }
        Ok(out)
    }

    fn add_undirected_batch_edge(
        &mut self,
        edge: &NewEdge<'_>,
        out: &mut Vec<Halves>,
    ) -> Result<(), Error> {
        let (low, high) = (edge.from.min(edge.to), edge.from.max(edge.to));
        let at_high = Half::new(
            high,
            (edge.label, Direction::Undirected),
            low,
            edge.placement,
        );
        if low == high {
            let [half] = self.insert_halves([at_high], edge.value)?;
            out.push(Halves::new(half, half));
            return Ok(());
        }
        let at_low = Half::new(
            low,
            (edge.label, Direction::Undirected),
            high,
            edge.placement,
        );
        let [canonical, other] = self.insert_halves([at_high, at_low], edge.value)?;
        out.push(Halves::new(canonical, other));
        Ok(())
    }

    /// [`Self::add_edges`] with both endpoint places cached. Targets that repeat within the
    /// batch make the target-side reads worth keeping; unique fan-out uses the one-sided path.
    #[inline(never)]
    fn add_edges_repeated(&mut self, edges: &[NewEdge<'_>]) -> Result<Vec<Halves>, BatchError> {
        let mut out = Vec::with_capacity(edges.len());
        let mut from_cache: Option<(LabelBucketKey, Place)> = None;
        let mut to_cache: [Option<(NodeId, LabelBucketKey, Place)>; TARGET_CACHE] =
            [None; TARGET_CACHE];
        for (index, edge) in edges.iter().enumerate() {
            let from_key = LabelBucketKey {
                label: edge.label,
                direction: Direction::Outgoing,
            };
            let to_key = LabelBucketKey {
                label: edge.label,
                direction: Direction::Incoming,
            };
            let self_loop = edge.from == edge.to;
            let cache_from = (!self_loop)
                .then(|| {
                    from_cache
                        .filter(|(key, _)| *key == from_key)
                        .map(|(_, place)| place)
                })
                .flatten();
            let cache_to = (!self_loop)
                .then(|| {
                    to_cache
                        .iter()
                        .flatten()
                        .find(|(node, key, _)| *node == edge.to && *key == to_key)
                        .map(|(_, _, place)| *place)
                })
                .flatten();
            let from = Half::new(
                edge.from,
                (edge.label, Direction::Outgoing),
                edge.to,
                edge.placement,
            );
            let to = Half::new(
                edge.to,
                (edge.label, Direction::Incoming),
                edge.from,
                edge.placement,
            );
            let (halves, updated) = if cache_to.is_some() {
                self.insert_halves_two([from, to], edge.value, [cache_from, cache_to])
            } else {
                self.insert_halves_cached(
                    [from, to],
                    edge.value,
                    cache_from.map(|place| (0, place)),
                    None,
                )
            }
            .map_err(|error| BatchError::after_prefix(index, error))?;
            let [canonical, other] = halves;
            out.push(Halves::new(canonical, other));
            if self_loop {
                from_cache = None;
                to_cache = [None; TARGET_CACHE];
            } else {
                for (half, place) in [from, to].iter().zip(updated) {
                    if place.is_none() {
                        forget_cache_node(&mut to_cache, half.node);
                    }
                }
                from_cache = updated
                    .first()
                    .and_then(|place| *place)
                    .map(|place| (from_key, place));
                cache_place(
                    &mut to_cache,
                    edge.to,
                    to_key,
                    updated.get(1).and_then(|place| *place),
                );
            }
        }
        Ok(out)
    }

    /// Adds one edge of `kind`, placed by `placement` in both buckets, with `value` in both
    /// halves. A directed edge is `from -[label]-> to`; an undirected edge is `from -[label]- to`
    /// and is normalized by node id. A directed self-loop has two halves; an undirected self-loop
    /// has one. Parallel edges keep their relative order either way.
    ///
    /// `value` is exactly the label's width ([`Self::declare_width`]) long, or empty for a label
    /// without one.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for either endpoint, [`Error::Detaching`] or [`Error::Deleted`]
    /// for an endpoint being or already deleted, [`Error::WrongValueLength`],
    /// [`Error::TooManyLabelBuckets`], [`Error::OutOfMemory`] or [`Error::AddressSpaceExhausted`].
    /// On error the graph is unchanged.
    pub fn add_edge(
        &mut self,
        kind: EdgeKind,
        from: NodeId,
        label: Label,
        to: NodeId,
        placement: Placement,
        value: &[u8],
    ) -> Result<Halves, Error> {
        match kind {
            EdgeKind::Directed => {
                let out = Half::new(from, (label, Direction::Outgoing), to, placement);
                let into = Half::new(to, (label, Direction::Incoming), from, placement);
                let [canonical, other] = self.insert_halves([out, into], value)?;
                Ok(Halves::new(canonical, other))
            }
            EdgeKind::Undirected => {
                let (low, high) = (from.min(to), from.max(to));
                let at_high = Half::new(high, (label, Direction::Undirected), low, placement);
                if low == high {
                    let [half] = self.insert_halves([at_high], value)?;
                    return Ok(Halves::new(half, half));
                }
                let at_low = Half::new(low, (label, Direction::Undirected), high, placement);
                let [canonical, other] = self.insert_halves([at_high, at_low], value)?;
                Ok(Halves::new(canonical, other))
            }
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

    fn spot_for_cursor(
        &self,
        bucket: &LabelBucket,
        half: Half,
        between: Between,
        cursor: &mut HoleCursor,
    ) -> Spot {
        if half.placement != Placement::ReuseHoles
            || between != Between::Nothing
            || !matches!(bucket.slots, Descriptor::Paged(_))
        {
            return self.spot_for(bucket, half.target, half.placement, between);
        }
        let key = (half.node, half.key);
        if cursor.key != Some(key) {
            cursor.key = Some(key);
            cursor.next = Position::new(0);
        }
        let Some(position) =
            label_bucket::hole_from(&bucket.slots, &self.edges.pages(), cursor.next)
        else {
            cursor.next = Position::new(bucket.slots.stored());
            return Spot::End;
        };
        cursor.next = Position::new(position.index().saturating_add(1));
        Spot::Hole(position)
    }

    /// The one insert path: a cache supplies any half already admitted, then the batch plans
    /// spots, reserves once and applies each half.
    fn insert_halves_with<const N: usize, C: Cached<N>>(
        &mut self,
        halves: [Half; N],
        value: &[u8],
        cached: &C,
        mut cursor: Option<(usize, &mut HoleCursor)>,
    ) -> Result<([EdgeRef; N], [Option<Place>; N]), Error> {
        let label = halves
            .first()
            .map_or_else(|| corrupt("an edge without halves"), |h| h.key.label);
        let width = self.value_width(label, value)?;
        let places = self.admit_with(&halves, cached)?;
        let n = u32::try_from(N).map_err(|_| Error::AddressSpaceExhausted)?;
        // Each half's spot, and the block batch of an append that completes a block of a hub's
        // bucket: planned now, so that its pages are reserved with the rest.
        let spots: [(Spot, Option<Batch>); N] = core::array::from_fn(|i| {
            let (Some(half), Some(place)) = (halves.get(i), places.get(i)) else {
                corrupt("an admitted half");
            };
            match place.target {
                Target::Present(_, stored) => {
                    let bucket = LabelBucket::resolve(&self.edges.pages(), stored);
                    // The other half's bucket, as it stands, tells whether the two ends have
                    // edges between them already.
                    let partner = (N == 2)
                        .then(|| halves.get(1 - i).zip(places.get(1 - i)))
                        .flatten();
                    let between = partner.map_or(Between::Unknown, |(h, p)| between(h, p));
                    let spot = match cursor.as_mut() {
                        Some((at, cursor)) if *at == i => {
                            self.spot_for_cursor(&bucket, *half, between, cursor)
                        }
                        _ => self.spot_for(&bucket, half.target, half.placement, between),
                    };
                    let batch = match spot {
                        Spot::End => self.batch_for(&bucket, half.target),
                        Spot::Hole(_) => None,
                    };
                    (spot, batch)
                }
                Target::Absent(_) => (Spot::End, None),
            }
        });
        let batch_pages: u32 = spots
            .iter()
            .filter_map(|(_, b)| b.as_ref())
            .map(Batch::pages)
            .sum();
        self.edges
            .reserve(pages_per_half(width) * n + batch_pages)?;
        self.label_bucket_lists
            .reserve(label_bucket_list::PAGES_PER_ADD * n)?;
        let mut places = places.into_iter();
        let mut spots = spots.into_iter();
        let mut touched: Option<NodeId> = None;
        let mut updated: [Option<Place>; N] = [None; N];
        let out = core::array::from_fn(|i| {
            let half = halves
                .get(i)
                .copied()
                .unwrap_or_else(|| corrupt("an admitted half"));
            let planned = places.next().unwrap_or_else(|| corrupt("an admitted half"));
            // An earlier half at the same node may have changed its list: look again.
            let place = if touched == Some(half.node) {
                self.place(half)
                    .unwrap_or_else(|_| corrupt("an admitted node"))
            } else {
                planned
            };
            touched = Some(half.node);
            let (spot, batch) = spots.next().unwrap_or_else(|| corrupt("an admitted half"));
            let (edge, place) =
                self.insert_half_place(half, place, (spot, batch), width.map(|w| (w, value)));
            if let Some(slot) = updated.get_mut(i) {
                *slot = place;
            }
            edge
        });
        Ok((out, updated))
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

    /// Fails as an insert of `halves` would, before anything changes; otherwise returns where
    /// each half goes. A cached half skips its node and list read.
    fn admit_with<const N: usize, C: Cached<N>>(
        &self,
        halves: &[Half; N],
        cached: &C,
    ) -> Result<[Place; N], Error> {
        let mut records = Records::default();
        let mut places: [Option<Place>; N] = [None; N];
        let mut fresh: [Option<(NodeId, Option<LabelBucketList>)>; N] = [None; N];
        for i in 0..N {
            let half = halves
                .get(i)
                .copied()
                .unwrap_or_else(|| corrupt("an admitted half"));
            let planned = if let Some(place) = cached.get(i) {
                match place.record.state {
                    NodeState::Live => place,
                    NodeState::Detaching => return Err(Error::Detaching(half.node)),
                    NodeState::Deleted => return Err(Error::Deleted(half.node)),
                }
            } else {
                self.place_checked(&mut records, half)?
            };
            match planned.target {
                Target::Present(_, stored) if self.stored_positions(stored) == u32::MAX => {
                    return Err(Error::AddressSpaceExhausted);
                }
                Target::Present(..) => {}
                Target::Absent(_) => {
                    if let Some(slot) = fresh.get_mut(i) {
                        *slot = Some((half.node, planned.record.list));
                    }
                }
            }
            if let Some(slot) = places.get_mut(i) {
                *slot = Some(planned);
            }
        }
        check_room(&self.label_bucket_lists.pages(), &fresh)?;
        Ok(places.map(|p| p.unwrap_or_else(|| corrupt("an admitted half"))))
    }

    fn place_checked(&self, records: &mut Records, half: Half) -> Result<Place, Error> {
        let own = records.get(self, half.node)?;
        records.get(self, half.target)?;
        Ok(self.place_in(own, half))
    }

    fn place(&self, half: Half) -> Result<Place, Error> {
        Ok(self.place_in(self.record(half.node)?, half))
    }

    fn place_in(&self, record: Record, half: Half) -> Place {
        let target = match label_bucket_list::lookup(
            &self.label_bucket_lists.pages(),
            record.list,
            half.key,
        ) {
            Lookup::Present { entry, stored } => Target::Present(entry, stored),
            Lookup::Absent { gap } => Target::Absent(gap),
        };
        Place { record, target }
    }

    /// Applies one admitted half within reserved memory. Cannot fail.
    /// A half of a label without values. A plain bucket that stays below a page takes the slot
    /// alone; anything else goes through [`Self::insert_extended`].
    /// A half into a bucket with a header (values, an index or both), or one that needs one.
    /// Publishes a half's bucket: in place when it existed, else as a new list entry.
    fn insert_half_place(
        &mut self,
        half: Half,
        place: Place,
        spot: (Spot, Option<Batch>),
        value: Option<(Width, &[u8])>,
    ) -> (EdgeRef, Option<Place>) {
        let (at, updated) = match value {
            None => self.insert_plain_place(half, place, spot),
            Some(value) => self.insert_extended_place(half, place, spot, Some(value)),
        };
        (
            EdgeRef::new(half.node, half.key.label, half.key.direction, at),
            updated,
        )
    }

    /// A half of a label without values. A plain bucket that stays below a page takes the slot
    /// alone; anything else goes through [`Self::insert_extended`].
    fn insert_plain_place(
        &mut self,
        half: Half,
        place: Place,
        (destination, batch): (Spot, Option<Batch>),
    ) -> (Position, Option<Place>) {
        let (slot, slots) = match place.target {
            Target::Present(entry, Stored::Plain(slots)) => (Ok(entry), slots),
            Target::Absent(gap) => (Err(gap), Descriptor::EMPTY),
            Target::Present(_, Stored::Extended(_)) => {
                return self.insert_extended_place(half, place, (destination, batch), None);
            }
        };
        let (next, at) = reserved(label_bucket::insert(
            slots,
            half.target,
            destination,
            &mut self.edges,
        ));
        // A run that just became paged gets its target index.
        let bucket =
            self.index_inserted(LabelBucket::plain(next), &slots, (half.target, at), batch);
        let updated = self.finish_place(half, place, slot, bucket);
        (at, updated)
    }

    /// A half into a bucket with a header (values, an index or both), or one that needs one:
    /// the slot, its value cell, and its index posting. A bucket of a label with values but no
    /// value plane yet (new, or made before the width was declared) gets one.
    fn insert_extended_place(
        &mut self,
        half: Half,
        place: Place,
        (destination, batch): (Spot, Option<Batch>),
        value: Option<(Width, &[u8])>,
    ) -> (Position, Option<Place>) {
        let (slot, bucket) = match place.target {
            Target::Present(entry, stored) => {
                (Ok(entry), LabelBucket::resolve(&self.edges.pages(), stored))
            }
            Target::Absent(gap) => (Err(gap), LabelBucket::plain(Descriptor::EMPTY)),
        };
        let mut bucket = match value {
            Some((width, _)) if bucket.values.is_none() => self.with_values(bucket, width),
            Some(_) | None => bucket,
        };
        let (slots, at) = reserved(label_bucket::insert(
            bucket.slots,
            half.target,
            destination,
            &mut self.edges,
        ));
        let was = core::mem::replace(&mut bucket.slots, slots);
        if let (Some(cells), Some((width, bytes))) = (bucket.values.as_mut(), value) {
            cells.plane = reserved(values::write(
                &mut self.edges,
                cells.plane,
                width,
                at.index(),
                bytes,
            ));
        }
        let bucket = self.index_inserted(bucket, &was, (half.target, at), batch);
        let updated = self.finish_place(half, place, slot, bucket);
        (at, updated)
    }

    /// Publishes a half's bucket: in place when it existed, else as a new list entry.
    fn finish_place(
        &mut self,
        half: Half,
        place: Place,
        slot: Result<EntryAt, EntryAt>,
        bucket: LabelBucket,
    ) -> Option<Place> {
        match slot {
            Ok(entry) => {
                let bucket = self.publish(entry, bucket);
                Some(Place {
                    record: place.record,
                    target: Target::Present(entry, bucket.stored()),
                })
            }
            Err(gap) => {
                let bucket = self.settle(bucket);
                bucket.write_header(&self.edges);
                let list = reserved(label_bucket_list::add(
                    &mut self.label_bucket_lists,
                    place.record.list,
                    gap,
                    half.key,
                    &bucket.stored(),
                ));
                self.set_record_label_bucket_list(half.node, place.record, list);
                None
            }
        }
    }

    /// How many positions a bucket stores, from its list entry.
    fn stored_positions(&self, stored: Stored) -> u32 {
        match stored {
            Stored::Plain(d) => d.stored(),
            Stored::Extended(_) => LabelBucket::resolve(&self.edges.pages(), stored)
                .slots
                .stored(),
        }
    }

    /// Publishes `node`'s new label bucket list, keeping the rest of the record it was admitted
    /// with.
    fn set_record_label_bucket_list(&self, node: NodeId, record: Record, list: LabelBucketList) {
        self.write_record(
            node,
            Record {
                list: Some(list),
                ..record
            },
        );
    }

    /// `bucket` with an empty value plane, never tiny; [`Self::publish`] gives it a header.
    /// Within reserved memory.
    pub(super) fn with_values(&mut self, bucket: LabelBucket, width: Width) -> LabelBucket {
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

/// Whether `half`'s bucket, at `place`, shows the two ends without edges between them: absent, or
/// tiny without an entry to `half`'s target. Larger buckets are not read for this.
fn between(half: &Half, place: &Place) -> Between {
    match place.target {
        Target::Absent(_) => Between::Nothing,
        Target::Present(_, Stored::Plain(Descriptor::Tiny(tiny)))
            if !tiny.slots().contains(&Slot::Live(half.target)) =>
        {
            Between::Nothing
        }
        Target::Present(..) => Between::Unknown,
    }
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

/// Places a batch edge may already have admitted.
trait Cached<const N: usize> {
    fn get(&self, index: usize) -> Option<Place>;
}

struct NoCache;

impl<const N: usize> Cached<N> for NoCache {
    fn get(&self, _index: usize) -> Option<Place> {
        None
    }
}

struct OneCache(Option<(usize, Place)>);

impl<const N: usize> Cached<N> for OneCache {
    fn get(&self, index: usize) -> Option<Place> {
        self.0
            .and_then(|(at, place)| (at == index).then_some(place))
    }
}

struct ManyCache<const N: usize>([Option<Place>; N]);

impl<const N: usize> Cached<N> for ManyCache<N> {
    fn get(&self, index: usize) -> Option<Place> {
        self.0.get(index).copied().flatten()
    }
}

fn cached_place<const N: usize>(
    cache: &[Option<(NodeId, LabelBucketKey, Place)>; N],
    node: NodeId,
    key: LabelBucketKey,
) -> Option<Place> {
    cache
        .iter()
        .flatten()
        .find(|(n, k, _)| *n == node && *k == key)
        .map(|(_, _, place)| *place)
}

fn cache_place<const N: usize>(
    cache: &mut [Option<(NodeId, LabelBucketKey, Place)>; N],
    node: NodeId,
    key: LabelBucketKey,
    place: Option<Place>,
) {
    let matches = |(n, k, _): &(NodeId, LabelBucketKey, Place)| *n == node && *k == key;
    if let Some(place) = place {
        if let Some(slot) = cache
            .iter_mut()
            .find(|entry| entry.as_ref().is_some_and(matches))
        {
            *slot = Some((node, key, place));
        } else if let Some(slot) = cache.iter_mut().find(|entry| entry.is_none()) {
            *slot = Some((node, key, place));
        } else if let Some(slot) = cache.get_mut(0) {
            *slot = Some((node, key, place));
        }
    } else if let Some(slot) = cache
        .iter_mut()
        .find(|entry| entry.as_ref().is_some_and(matches))
    {
        *slot = None;
    }
}

fn forget_cache_node<const N: usize>(
    cache: &mut [Option<(NodeId, LabelBucketKey, Place)>; N],
    node: NodeId,
) {
    for entry in cache.iter_mut() {
        if entry.as_ref().is_some_and(|(n, _, _)| *n == node) {
            *entry = None;
        }
    }
}

/// Remembers each half's new place, and drops every entry for a node whose label bucket list
/// moved because the half added a new bucket.
fn remember_halves<const N: usize>(
    cache: &mut [Option<(NodeId, LabelBucketKey, Place)>; MIXED_CACHE],
    halves: [Half; N],
    updated: [Option<Place>; N],
) {
    for (half, place) in halves.iter().zip(updated) {
        if place.is_none() {
            for entry in cache.iter_mut() {
                if entry
                    .as_ref()
                    .is_some_and(|(node, _, _)| *node == half.node)
                {
                    *entry = None;
                }
            }
        }
    }
    for (half, place) in halves.iter().zip(updated) {
        if place.is_some() {
            cache_place(cache, half.node, half.key, place);
        }
    }
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

/// The records of an insert's nodes (an edge touches at most two), each read once; a
/// detaching or deleted node fails the insert.
#[derive(Default)]
struct Records([Option<(NodeId, Record)>; 2]);

impl Records {
    fn get<M: Memory>(&mut self, g: &Graph<M>, v: NodeId) -> Result<Record, Error> {
        if let Some((_, record)) = self.0.iter().flatten().find(|(w, _)| *w == v) {
            return Ok(*record);
        }
        let record = g.record(v)?;
        match record.state {
            NodeState::Live => {}
            NodeState::Detaching => return Err(Error::Detaching(v)),
            NodeState::Deleted => return Err(Error::Deleted(v)),
        }
        let free = self
            .0
            .iter_mut()
            .find(|slot| slot.is_none())
            .unwrap_or_else(|| corrupt("an edge with more than two nodes"));
        *free = Some((v, record));
        Ok(record)
    }
}

/// Fails when new buckets would not fit a node's label bucket list. `fresh` lists each new bucket's
/// node with its current list.
fn check_room<M: Memory>(
    pages: &Pages<'_, M>,
    fresh: &[Option<(NodeId, Option<LabelBucketList>)>],
) -> Result<(), Error> {
    for &(v, list) in fresh.iter().flatten() {
        let adding = fresh.iter().flatten().filter(|(w, _)| *w == v).count();
        if !label_bucket_list::has_room(pages, list, adding) {
            return Err(Error::TooManyLabelBuckets(v));
        }
    }
    Ok(())
}
