//! [`Graph::add_edges`]: one path per kind of batch, each with the cache that suits it.

use super::cache::{EdgeCache, EndCache, MIXED_CACHE, PlaceCache};
use super::{Half, Place};
use crate::bucket::LabelBucketKey;
use crate::edge::HoleCursor;
use crate::error::{BatchError, Error};
use crate::graph::Graph;
use crate::graph::GraphMemories;
use crate::id::{EdgeRef, Halves, NodeId};
use crate::input::{Edge, EdgeKind, Placement};
use alloc::vec::Vec;

/// Whether a target repeats near the start of the batch. The probe is capped so a long unique
/// fan-out does not pay for a set it will not use; a repeated run of edges is visible early.
#[inline(never)]
fn repeated_targets(edges: &[Edge<'_>]) -> bool {
    let sample = edges.get(..64).unwrap_or(edges);
    let wanted = sample.len().saturating_mul(2).clamp(16, 128);
    let size = wanted.next_power_of_two();
    let mask = size - 1;
    let mut slots: Vec<u32> = Vec::new();
    slots.resize(size, u32::MAX);
    for edge in sample {
        if seen_before(&mut slots, mask, edge.to()) {
            return true;
        }
    }
    false
}

/// Probes the open-addressed set `slots` for `node`, adding it when absent. A full set counts as
/// a repeat.
fn seen_before(slots: &mut [u32], mask: usize, node: NodeId) -> bool {
    let index = usize::try_from(node.index()).unwrap_or(usize::MAX);
    let mut at = index & mask;
    loop {
        match slots.get(at).copied() {
            Some(u32::MAX) => {
                if let Some(slot) = slots.get_mut(at) {
                    *slot = node.index();
                }
                return false;
            }
            Some(seen) if seen == node.index() => return true,
            Some(_) => at = (at + 1) & mask,
            None => return true,
        }
    }
}

impl<S: GraphMemories> Graph<S> {
    /// Adds every edge in `edges`, in order, placed as [`Self::add_edge`] places one. Each edge is all-or-none like [`Self::add_edge`],
    /// but the batch is not atomic: an error leaves the edges before it inserted.
    ///
    /// All endpoints are checked and every value length is validated before the first write.
    /// Edges with the same `from` and `label` should be adjacent; the batch can then reuse that
    /// node's bucket state.
    ///
    /// # Errors
    ///
    /// As [`Self::add_edge`], for the edge that fails.
    pub fn add_edges(&mut self, edges: &[Edge<'_>]) -> Result<Vec<Halves>, BatchError> {
        add_edges_reusing_holes(self, edges)
    }

    /// [`Self::add_edges`], placing each half by `placement`.
    ///
    /// # Errors
    ///
    /// As [`Self::add_edge`], for the edge that fails.
    pub fn add_edges_with(
        &mut self,
        edges: &[Edge<'_>],
        placement: Placement,
    ) -> Result<Vec<Halves>, BatchError> {
        add_edges(self, edges, placement)
    }
}

/// [`Graph::add_edges`]: [`add_edges`] by the default placement.
fn add_edges_reusing_holes<S: GraphMemories>(
    graph: &mut Graph<S>,
    edges: &[Edge<'_>],
) -> Result<Vec<Halves>, BatchError> {
    add_edges(graph, edges, Placement::ReuseHoles)
}

/// [`Graph::add_edges`]: checks every edge, then takes the path that suits the batch.
fn add_edges<S: GraphMemories>(
    graph: &mut Graph<S>,
    edges: &[Edge<'_>],
    placement: Placement,
) -> Result<Vec<Halves>, BatchError> {
    for (index, edge) in edges.iter().enumerate() {
        graph
            .check(edge.from())
            .and_then(|()| graph.check(edge.to()))
            .and_then(|()| graph.value_schema(edge.label(), edge.bytes()))
            .map_err(|error| BatchError::unchanged(index, error))?;
    }
    let all_directed = edges.iter().all(|edge| edge.kind() == EdgeKind::Directed);
    if all_directed && repeated_targets(edges) {
        return add_repeated(graph, edges, placement);
    }
    if edges.iter().all(|edge| edge.kind() == EdgeKind::Undirected) {
        return add_undirected(graph, edges, placement);
    }
    if !all_directed {
        return add_mixed(graph, edges, placement);
    }
    add_directed(graph, edges, placement)
}

/// A directed batch with unique targets: the source's last place, and a cursor into its holes.
fn add_directed<S: GraphMemories>(
    graph: &mut Graph<S>,
    edges: &[Edge<'_>],
    placement: Placement,
) -> Result<Vec<Halves>, BatchError> {
    let mut out = Vec::with_capacity(edges.len());
    let mut cache: Option<(NodeId, LabelBucketKey, Place)> = None;
    let mut holes = HoleCursor::default();
    for (index, edge) in edges.iter().enumerate() {
        let [from, to] = Half::directed(edge, placement);
        let cached = cache
            .filter(|(last_node, last_key, _)| *last_node == edge.from() && *last_key == from.key)
            .map(|(_, _, place)| place);
        let (halves, updated) = graph
            .insert_halves_with(
                [from, to],
                edge.bytes(),
                &[cached, None],
                Some((0, &mut holes)),
            )
            .map_err(|error| BatchError::after_prefix(index, error))?;
        let [first, second] = halves;
        out.push(Halves::of(first, second));
        cache = updated
            .first()
            .and_then(|place| *place)
            .map(|place| (edge.from(), from.key, place));
    }
    Ok(out)
}

/// A batch with both edge kinds: every half is cached by its endpoint, independent of the
/// half's role in a directed or undirected edge.
#[inline(never)]
fn add_mixed<S: GraphMemories>(
    graph: &mut Graph<S>,
    edges: &[Edge<'_>],
    placement: Placement,
) -> Result<Vec<Halves>, BatchError> {
    let mut out = Vec::with_capacity(edges.len());
    let mut cache = PlaceCache::<MIXED_CACHE>::EMPTY;
    for (index, edge) in edges.iter().enumerate() {
        let halves = match edge.kind() {
            EdgeKind::Directed => insert_remembered(
                graph,
                Half::directed(edge, placement),
                edge.bytes(),
                &mut cache,
            )
            .map(|[first, second]| Halves::of(first, second)),
            EdgeKind::Undirected => match Half::undirected(edge, placement) {
                (at_high, None) => insert_remembered(graph, [at_high], edge.bytes(), &mut cache)
                    .map(|[half]| Halves::new(half, half)),
                (at_high, Some(at_low)) => {
                    insert_remembered(graph, [at_high, at_low], edge.bytes(), &mut cache)
                        .map(|[first, second]| Halves::of(first, second))
                }
            },
        };
        out.push(halves.map_err(|error| BatchError::after_prefix(index, error))?);
    }
    Ok(out)
}

/// Inserts `halves` with the places `cache` remembers, then remembers their new places.
fn insert_remembered<S: GraphMemories, const N: usize>(
    graph: &mut Graph<S>,
    halves: [Half; N],
    value: &[u8],
    cache: &mut PlaceCache<MIXED_CACHE>,
) -> Result<[EdgeRef; N], Error> {
    let cached = cache.places(halves);
    let (edges, updated) = graph.insert_halves_with(halves, value, &cached, None)?;
    cache.remember(halves, updated);
    Ok(edges)
}

/// Undirected batches cache both endpoints by `(node, key)`.
fn add_undirected<S: GraphMemories>(
    graph: &mut Graph<S>,
    edges: &[Edge<'_>],
    placement: Placement,
) -> Result<Vec<Halves>, BatchError> {
    let mut out = Vec::with_capacity(edges.len());
    let mut cache = EndCache::EMPTY;
    for (index, edge) in edges.iter().enumerate() {
        let (at_high, at_low) = Half::undirected(edge, placement);
        let ends = (at_high.node, at_high.target);
        let key = at_high.key;
        let [cache_high, cache_low] = cache.places(ends, key);
        let Some(at_low) = at_low else {
            let (halves, updated) = graph
                .insert_halves_with([at_high], edge.bytes(), &[cache_high], None)
                .map_err(|error| BatchError::after_prefix(index, error))?;
            let [half] = halves;
            out.push(Halves::new(half, half));
            let [updated] = updated;
            cache.remember_loop(at_high.node, key, updated);
            continue;
        };
        let (halves, updated) = graph
            .insert_halves_with(
                [at_high, at_low],
                edge.bytes(),
                &[cache_high, cache_low],
                None,
            )
            .map_err(|error| BatchError::after_prefix(index, error))?;
        let [first, second] = halves;
        out.push(Halves::of(first, second));
        cache.remember(ends, key, updated);
    }
    Ok(out)
}

/// [`Graph::add_edges`] with both endpoint places cached. Targets that repeat within the batch
/// make the target-side reads worth keeping.
#[inline(never)]
fn add_repeated<S: GraphMemories>(
    graph: &mut Graph<S>,
    edges: &[Edge<'_>],
    placement: Placement,
) -> Result<Vec<Halves>, BatchError> {
    let mut out = Vec::with_capacity(edges.len());
    let mut cache = EdgeCache::EMPTY;
    for (index, edge) in edges.iter().enumerate() {
        let halves = Half::directed(edge, placement);
        let cached = cache.places(halves);
        let (inserted, updated) = graph
            .insert_halves_with(halves, edge.bytes(), &cached, None)
            .map_err(|error| BatchError::after_prefix(index, error))?;
        let [first, second] = inserted;
        out.push(Halves::of(first, second));
        cache.remember(halves, updated);
    }
    Ok(out)
}
