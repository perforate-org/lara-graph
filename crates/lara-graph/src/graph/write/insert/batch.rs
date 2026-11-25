//! [`Graph::add_edges`]: one path per kind of batch, each with the cache that suits it.

use super::apply::HoleCursor;
use super::cache::{EdgeCache, EndCache, MIXED_CACHE, ManyCache, PlaceCache};
use super::{EdgeKind, Half, NewEdge, Place};
use crate::graph::{BatchError, Error, Graph};
use crate::id::{EdgeRef, Halves, NodeId};
use crate::node::label_bucket_list::LabelBucketKey;
use crate::store::memory::Memory;
use alloc::vec::Vec;

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
        if seen_before(&mut slots, mask, edge.to) {
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
                out.push(
                    self.add_edge(*edge)
                        .map_err(|error| BatchError::after_prefix(index, error))?,
                );
                cache = None;
                continue;
            }
            let [from, to] = edge.directed_halves();
            let cached = cache
                .filter(|(last_node, last_key, _)| *last_node == edge.from && *last_key == from.key)
                .map(|(_, _, place)| (0, place));
            let (halves, updated) = self
                .insert_halves_cached([from, to], edge.value, cached, Some((0, &mut holes)))
                .map_err(|error| BatchError::after_prefix(index, error))?;
            let [canonical, other] = halves;
            out.push(Halves::new(canonical, other));
            cache = updated
                .first()
                .and_then(|place| *place)
                .map(|place| (edge.from, from.key, place));
        }
        Ok(out)
    }

    /// A batch with both edge kinds: every half is cached by its endpoint, independent of the
    /// half's role in a directed or undirected edge.
    #[inline(never)]
    fn add_edges_mixed(&mut self, edges: &[NewEdge<'_>]) -> Result<Vec<Halves>, BatchError> {
        let mut out = Vec::with_capacity(edges.len());
        let mut cache = PlaceCache::<MIXED_CACHE>::EMPTY;
        for (index, edge) in edges.iter().enumerate() {
            let halves = match edge.kind {
                EdgeKind::Directed => self
                    .insert_remembered(edge.directed_halves(), edge.value, &mut cache)
                    .map(|[canonical, other]| Halves::new(canonical, other)),
                EdgeKind::Undirected => match edge.undirected_halves() {
                    (at_high, None) => self
                        .insert_remembered([at_high], edge.value, &mut cache)
                        .map(|[half]| Halves::new(half, half)),
                    (at_high, Some(at_low)) => self
                        .insert_remembered([at_high, at_low], edge.value, &mut cache)
                        .map(|[canonical, other]| Halves::new(canonical, other)),
                },
            };
            out.push(halves.map_err(|error| BatchError::after_prefix(index, error))?);
        }
        Ok(out)
    }

    /// Inserts `halves` with the places `cache` remembers, then remembers their new places.
    fn insert_remembered<const N: usize>(
        &mut self,
        halves: [Half; N],
        value: &[u8],
        cache: &mut PlaceCache<MIXED_CACHE>,
    ) -> Result<[EdgeRef; N], Error> {
        let cached = cache.places(halves);
        let (edges, updated) = self.insert_halves_with(halves, value, &ManyCache(cached), None)?;
        cache.remember(halves, updated);
        Ok(edges)
    }

    /// Undirected batches cache both endpoints by `(node, key)`.
    fn add_edges_undirected(&mut self, edges: &[NewEdge<'_>]) -> Result<Vec<Halves>, BatchError> {
        let mut out = Vec::with_capacity(edges.len());
        let mut cache = EndCache::EMPTY;
        for (index, edge) in edges.iter().enumerate() {
            let (at_high, at_low) = edge.undirected_halves();
            let ends = (at_high.node, at_high.target);
            let key = at_high.key;
            let [cache_high, cache_low] = cache.places(ends, key);
            let Some(at_low) = at_low else {
                let (halves, updated) = self
                    .insert_halves_two([at_high], edge.value, [cache_high])
                    .map_err(|error| BatchError::after_prefix(index, error))?;
                let [half] = halves;
                out.push(Halves::new(half, half));
                let [updated] = updated;
                cache.remember_loop(at_high.node, key, updated);
                continue;
            };
            let (halves, updated) = self
                .insert_halves_two([at_high, at_low], edge.value, [cache_high, cache_low])
                .map_err(|error| BatchError::after_prefix(index, error))?;
            let [canonical, other] = halves;
            out.push(Halves::new(canonical, other));
            cache.remember(ends, key, updated);
        }
        Ok(out)
    }

    /// [`Self::add_edges`] with both endpoint places cached. Targets that repeat within the
    /// batch make the target-side reads worth keeping; unique fan-out uses the one-sided path.
    #[inline(never)]
    fn add_edges_repeated(&mut self, edges: &[NewEdge<'_>]) -> Result<Vec<Halves>, BatchError> {
        let mut out = Vec::with_capacity(edges.len());
        let mut cache = EdgeCache::EMPTY;
        for (index, edge) in edges.iter().enumerate() {
            let halves = edge.directed_halves();
            let [cache_from, cache_to] = cache.places(halves);
            let (inserted, updated) = if cache_to.is_some() {
                self.insert_halves_two(halves, edge.value, [cache_from, cache_to])
            } else {
                self.insert_halves_cached(
                    halves,
                    edge.value,
                    cache_from.map(|place| (0, place)),
                    None,
                )
            }
            .map_err(|error| BatchError::after_prefix(index, error))?;
            let [canonical, other] = inserted;
            out.push(Halves::new(canonical, other));
            cache.remember(halves, updated);
        }
        Ok(out)
    }
}
