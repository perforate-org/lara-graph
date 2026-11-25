//! The places an insert batch remembers between edges, so a hot bucket is not read again.

use super::{Half, Place};
use crate::id::NodeId;
use crate::node::label_bucket_list::LabelBucketKey;

/// Target places an `add_edges` batch remembers. Eight covers the common fan-in/fan-out set.
pub(super) const TARGET_CACHE: usize = 8;
/// A mixed batch keeps both directions and both undirected ends, so it needs a few more.
pub(super) const MIXED_CACHE: usize = 16;

/// Places a batch edge may already have admitted.
pub(super) trait Cached<const N: usize> {
    fn get(&self, index: usize) -> Option<Place>;
}

pub(super) struct NoCache;

impl<const N: usize> Cached<N> for NoCache {
    fn get(&self, _index: usize) -> Option<Place> {
        None
    }
}

pub(super) struct OneCache(pub(super) Option<(usize, Place)>);

impl<const N: usize> Cached<N> for OneCache {
    fn get(&self, index: usize) -> Option<Place> {
        self.0
            .and_then(|(at, place)| (at == index).then_some(place))
    }
}

pub(super) struct ManyCache<const N: usize>(pub(super) [Option<Place>; N]);

impl<const N: usize> Cached<N> for ManyCache<N> {
    fn get(&self, index: usize) -> Option<Place> {
        self.0.get(index).copied().flatten()
    }
}

/// Up to `N` places, each by its node and bucket key.
#[derive(Clone, Copy)]
pub(super) struct PlaceCache<const N: usize>([Option<(NodeId, LabelBucketKey, Place)>; N]);

impl<const N: usize> PlaceCache<N> {
    pub(super) const EMPTY: Self = Self([None; N]);

    /// The place of `node`'s bucket `key`, if remembered.
    #[inline]
    pub(super) fn get(&self, node: NodeId, key: LabelBucketKey) -> Option<Place> {
        self.0
            .iter()
            .flatten()
            .find(|(n, k, _)| *n == node && *k == key)
            .map(|(_, _, place)| *place)
    }

    /// Remembers `place` for `node`'s bucket `key`, in its entry, a free one or the first; `None`
    /// forgets the entry.
    #[inline]
    pub(super) fn set(&mut self, node: NodeId, key: LabelBucketKey, place: Option<Place>) {
        let matches = |(n, k, _): &(NodeId, LabelBucketKey, Place)| *n == node && *k == key;
        let Some(place) = place else {
            if let Some(slot) = self
                .0
                .iter_mut()
                .find(|entry| entry.as_ref().is_some_and(matches))
            {
                *slot = None;
            }
            return;
        };
        if let Some(slot) = self
            .0
            .iter_mut()
            .find(|entry| entry.as_ref().is_some_and(matches))
        {
            *slot = Some((node, key, place));
        } else if let Some(slot) = self.0.iter_mut().find(|entry| entry.is_none()) {
            *slot = Some((node, key, place));
        } else if let Some(slot) = self.0.get_mut(0) {
            *slot = Some((node, key, place));
        }
    }

    /// Forgets every bucket of `node`.
    #[inline]
    pub(super) fn forget_node(&mut self, node: NodeId) {
        for entry in &mut self.0 {
            if entry.as_ref().is_some_and(|(n, _, _)| *n == node) {
                *entry = None;
            }
        }
    }

    /// The remembered place of each half.
    #[inline]
    pub(super) fn places<const H: usize>(&self, halves: [Half; H]) -> [Option<Place>; H] {
        halves.map(|half| self.get(half.node, half.key))
    }

    /// Remembers each half's new place, and drops every entry for a node whose label bucket list
    /// moved because the half added a new bucket.
    #[inline]
    pub(super) fn remember<const H: usize>(
        &mut self,
        halves: [Half; H],
        updated: [Option<Place>; H],
    ) {
        for (half, place) in halves.iter().zip(updated) {
            if place.is_none() {
                self.forget_node(half.node);
            }
        }
        for (half, place) in halves.iter().zip(updated) {
            if place.is_some() {
                self.set(half.node, half.key, place);
            }
        }
    }
}

/// An undirected batch's places by end. A hub is usually the low or the high end of every edge,
/// so one side stays hot while the other moves.
pub(super) struct EndCache {
    high: PlaceCache<TARGET_CACHE>,
    low: PlaceCache<TARGET_CACHE>,
}

impl EndCache {
    pub(super) const EMPTY: Self = Self {
        high: PlaceCache::EMPTY,
        low: PlaceCache::EMPTY,
    };

    /// The remembered places of `high`'s and `low`'s buckets `key`.
    #[inline]
    pub(super) fn places(
        &self,
        (high, low): (NodeId, NodeId),
        key: LabelBucketKey,
    ) -> [Option<Place>; 2] {
        [self.high.get(high, key), self.low.get(low, key)]
    }

    /// Forgets every bucket of `node`, at either end.
    fn forget_node(&mut self, node: NodeId) {
        self.high.forget_node(node);
        self.low.forget_node(node);
    }

    /// Remembers a self-loop's new place, or forgets its node when its list moved.
    pub(super) fn remember_loop(
        &mut self,
        node: NodeId,
        key: LabelBucketKey,
        updated: Option<Place>,
    ) {
        if let Some(place) = updated {
            self.high.set(node, key, Some(place));
        } else {
            self.forget_node(node);
        }
    }

    /// Remembers both ends' new places, after forgetting each end whose list moved.
    #[inline]
    pub(super) fn remember(
        &mut self,
        (high, low): (NodeId, NodeId),
        key: LabelBucketKey,
        [at_high, at_low]: [Option<Place>; 2],
    ) {
        if at_high.is_none() {
            self.forget_node(high);
        }
        if at_low.is_none() {
            self.forget_node(low);
        }
        if let Some(place) = at_high {
            self.high.set(high, key, Some(place));
        }
        if let Some(place) = at_low {
            self.low.set(low, key, Some(place));
        }
    }
}

/// A directed batch with repeated targets: the source's last place, and the targets' places.
pub(super) struct EdgeCache {
    from: Option<(LabelBucketKey, Place)>,
    to: PlaceCache<TARGET_CACHE>,
}

impl EdgeCache {
    pub(super) const EMPTY: Self = Self {
        from: None,
        to: PlaceCache::EMPTY,
    };

    /// The remembered places of a directed edge's halves; none for a self-loop.
    #[inline]
    pub(super) fn places(&self, [from, to]: [Half; 2]) -> [Option<Place>; 2] {
        if from.node == from.target {
            return [None, None];
        }
        let at_from = self
            .from
            .filter(|(key, _)| *key == from.key)
            .map(|(_, place)| place);
        [at_from, self.to.get(to.node, to.key)]
    }

    /// Remembers both halves' new places, after forgetting each node whose list moved. A
    /// self-loop forgets everything.
    #[inline]
    pub(super) fn remember(&mut self, halves: [Half; 2], updated: [Option<Place>; 2]) {
        let [from, to] = halves;
        if from.node == from.target {
            *self = Self::EMPTY;
            return;
        }
        for (half, place) in halves.iter().zip(updated) {
            if place.is_none() {
                self.to.forget_node(half.node);
            }
        }
        let [at_from, at_to] = updated;
        self.from = at_from.map(|place| (from.key, place));
        self.to.set(to.node, to.key, at_to);
    }
}
