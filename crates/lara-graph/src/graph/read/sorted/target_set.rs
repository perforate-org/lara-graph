//! A set of node indexes built once from one bucket's targets, to intersect with many others: a
//! dense bitmap over the targets' range when it is narrow ([`bits`]), else an open-addressing
//! hash table ([`table`]). Each intersection is a query that marks the targets it finds, so the
//! set itself never changes and repeats in the other bucket (parallel edges) are found once.

mod bits;
mod table;

use super::sort_indexes;
use crate::id::NodeId;
use alloc::vec::Vec;
use bits::Bits;
use table::Table;

/// Ids of range a target may stand for in a bitmap. A bitmap and its marks then take up to 64
/// bytes a key, four times the hash table, but a lookup is a shift and a mask: intersecting 65536
/// edges with a set of 1024 targets spread over 131072 ids cost 4.6M instructions against 8.7M
/// through the hash table (`common_65536_1024`), while at 64 ids a key it stayed a hash table.
const BITMAP_RANGE_PER_KEY: u64 = 256;

/// Targets a set is filled from or intersected with, which may repeat: a bucket's.
pub(super) trait Targets {
    /// How many targets, repeats counted.
    fn count(&self) -> usize;

    /// Whether looking each of `candidates` set targets up in these is cheaper than reading them
    /// all.
    fn probes(&self, candidates: usize) -> bool;

    /// Calls `f` with each target.
    fn visit(&self, f: impl FnMut(u32));

    /// Whether `t` is among the targets.
    fn holds(&self, t: u32) -> bool;
}

impl Targets for [u32] {
    fn count(&self) -> usize {
        self.len()
    }

    fn probes(&self, _: usize) -> bool {
        false
    }

    fn visit(&self, f: impl FnMut(u32)) {
        self.iter().copied().for_each(f);
    }

    fn holds(&self, t: u32) -> bool {
        self.contains(&t)
    }
}

/// The targets one bucket holds, each once, ready to intersect with others
/// ([`crate::Graph::common_with`]). Built by [`crate::Graph::neighbor_set`], or refilled by
/// [`crate::Graph::neighbor_set_into`] in the memory it already took; it borrows nothing, so it may
/// outlive writes to the graph, and then describes the bucket as it was.
pub struct NeighborSet {
    len: usize,
    lookup: Lookup,
    /// The current query's matches.
    matches: Vec<u32>,
    /// The kind of lookup not in use, kept for a refill that needs it.
    spare_words: Vec<[u64; 2]>,
    spare_slots: Vec<[u32; 2]>,
    /// The bucket's targets as read, before the set is built from them.
    read: Vec<u32>,
}

enum Lookup {
    Bits(Bits),
    Hash(Table),
}

impl Default for NeighborSet {
    fn default() -> Self {
        Self::new()
    }
}

impl NeighborSet {
    /// An empty set, to fill with [`crate::Graph::neighbor_set_into`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            len: 0,
            lookup: Lookup::Bits(Bits::new()),
            matches: Vec::new(),
            spare_words: Vec::new(),
            spare_slots: Vec::new(),
            read: Vec::new(),
        }
    }

    /// The set of `targets`, which may repeat.
    #[cfg(test)]
    pub(super) fn of(targets: &[u32]) -> Self {
        let mut set = Self::new();
        set.fill(targets);
        set
    }

    /// Refills the set with `targets`, read into the buffer it keeps for them.
    pub(super) fn refill<T: Targets + ?Sized>(&mut self, targets: &T) {
        let mut read = core::mem::take(&mut self.read);
        read.clear();
        read.reserve(targets.count());
        targets.visit(|t| read.push(t));
        self.fill(&read);
        self.read = read;
    }

    /// Appends the targets both the set and `targets` hold from index `from` on, ascending, to
    /// `out`, and leaves the set as it was.
    pub(super) fn common_from<T: Targets + ?Sized>(
        &mut self,
        targets: &T,
        from: u32,
        out: &mut Vec<u32>,
    ) {
        self.mark(targets, from);
        self.finish_from(from, out);
    }

    /// Appends the targets the set and every one of `rest` hold, ascending, to `out`: each of
    /// `rest` in turn keeps in the set those it holds, the set narrowed in place, so it is left
    /// holding what they share, or less once none is left.
    pub(super) fn narrow_to_common<T: Targets>(
        &mut self,
        rest: impl IntoIterator<Item = T>,
        out: &mut Vec<u32>,
    ) {
        let mut rest = rest.into_iter().peekable();
        if rest.peek().is_none() {
            let mut keys = self.keys();
            sort_indexes(&mut keys);
            out.append(&mut keys);
            return;
        }
        // Each round keeps in the set what the bucket holds; the last one lists it.
        while let Some(targets) = rest.next() {
            self.mark(&targets, 0);
            if rest.peek().is_none() {
                self.finish(out);
                return;
            }
            self.narrow();
            if self.is_empty() {
                return;
            }
        }
    }

    /// Starts a query matching the targets `targets` holds from index `from` on: `targets` read
    /// whole and looked up in the set, or, if it [probes](Targets::probes), each of the set's
    /// targets looked up in it.
    fn mark<T: Targets + ?Sized>(&mut self, targets: &T, from: u32) {
        self.begin();
        if targets.probes(self.len) {
            for t in self.keys() {
                if t >= from && targets.holds(t) {
                    self.hit(t);
                }
            }
        } else {
            self.mark_read(targets, from);
        }
    }

    /// [`Self::mark`] reading `targets` whole.
    fn mark_read<T: Targets + ?Sized>(&mut self, targets: &T, from: u32) {
        if from == 0 {
            // Unbounded, the loop keeps no comparison: with it, intersections cost 2-11 % more.
            targets.visit(|t| self.hit(t));
        } else {
            targets.visit(|t| {
                if t >= from {
                    self.hit(t);
                }
            });
        }
    }

    /// Refills the set with `targets`, which may repeat, in the memory it already took.
    fn fill(&mut self, targets: &[u32]) {
        match core::mem::replace(&mut self.lookup, Lookup::Bits(Bits::new())) {
            Lookup::Bits(bits) => self.spare_words = bits.into_words(),
            Lookup::Hash(table) => self.spare_slots = table.into_slots(),
        }
        self.matches.clear();
        self.len = 0;
        let Some((min, max)) = range(targets) else {
            return;
        };
        let (lookup, len) = if u64::from(max - min) < width(targets.len()) {
            let words = core::mem::take(&mut self.spare_words);
            let (bits, len) = Bits::of(words, targets, (min, max));
            (Lookup::Bits(bits), len)
        } else {
            let slots = core::mem::take(&mut self.spare_slots);
            let (table, len) = Table::of(slots, targets);
            (Lookup::Hash(table), len)
        };
        self.lookup = lookup;
        self.len = len;
    }

    /// How many targets the set holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the set holds no target.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the set holds `t`.
    #[must_use]
    pub fn contains(&self, t: NodeId) -> bool {
        match &self.lookup {
            Lookup::Bits(bits) => bits.contains(t.index()),
            Lookup::Hash(table) => table.contains(t.index()),
        }
    }

    /// The targets, each once: ascending from a bitmap, in no order from a hash table.
    fn keys(&self) -> Vec<u32> {
        let mut keys = Vec::with_capacity(self.len);
        match &self.lookup {
            Lookup::Bits(bits) => bits.push_keys(&mut keys),
            Lookup::Hash(table) => table.push_keys(&mut keys),
        }
        keys
    }

    /// Starts a query: no target is matched yet.
    fn begin(&mut self) {
        self.matches.clear();
        if let Lookup::Hash(table) = &mut self.lookup {
            table.begin();
        }
    }

    /// Matches `t` if the set holds it and this query has not matched it yet.
    #[inline]
    fn hit(&mut self, t: u32) {
        let new = match &mut self.lookup {
            Lookup::Bits(bits) => bits.hit(t),
            Lookup::Hash(table) => table.hit(t),
        };
        if new {
            self.matches.push(t);
        }
    }

    /// Ends the query: its matches, ascending, appended to `out`, and the marks cleared.
    fn finish(&mut self, out: &mut Vec<u32>) {
        self.finish_from(0, out);
    }

    /// [`Self::finish`] for a query that matched nothing below index `from`: a bitmap's scan
    /// starts at `from`'s word.
    fn finish_from(&mut self, from: u32, out: &mut Vec<u32>) {
        match &mut self.lookup {
            Lookup::Bits(bits) => bits.finish_from(from, &mut self.matches, out),
            Lookup::Hash(_) => {
                sort_indexes(&mut self.matches);
                out.append(&mut self.matches);
            }
        }
    }

    /// Ends the query by keeping only its matches: a k-way intersection's next round, without
    /// building a new set.
    fn narrow(&mut self) {
        self.len = self.matches.len();
        self.matches.clear();
        match &mut self.lookup {
            Lookup::Bits(bits) => bits.narrow(),
            Lookup::Hash(table) => table.narrow(),
        }
    }
}

/// The smallest and largest of `targets`; `None` for none.
fn range(targets: &[u32]) -> Option<(u32, u32)> {
    let (first, rest) = targets.split_first()?;
    Some(
        rest.iter()
            .fold((*first, *first), |(min, max), &t| (min.min(t), max.max(t))),
    )
}

/// The widest range of ids `n` targets make a bitmap for.
fn width(n: usize) -> u64 {
    u64::try_from(n)
        .unwrap_or(u64::MAX)
        .saturating_mul(BITMAP_RANGE_PER_KEY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn query(set: &mut NeighborSet, targets: &[u32]) -> Vec<u32> {
        set.begin();
        for &t in targets {
            set.hit(t);
        }
        let mut out = Vec::new();
        set.finish(&mut out);
        out
    }

    fn narrowed(set: &mut NeighborSet, targets: &[u32]) {
        set.begin();
        for &t in targets {
            set.hit(t);
        }
        set.narrow();
    }

    #[test]
    fn narrowing_keeps_only_a_query_s_matches() {
        for keys in [vec![10, 12, 70, 11, 40], vec![5, 1 << 20, 77, 9, 1 << 30]] {
            let mut set = NeighborSet::of(&keys);
            let (a, b, c) = (keys[0], keys[1], keys[4]);
            // A reused query between rounds changes nothing.
            assert_eq!(query(&mut set, &[a]), [a]);
            narrowed(&mut set, &[c, a, b, a, 999_999]);
            assert_eq!(set.len(), 3);
            let mut left = set.keys();
            left.sort_unstable();
            let mut want = vec![a, b, c];
            want.sort_unstable();
            assert_eq!(left, want);
            assert!(!set.contains(NodeId::new(keys[2])));
            assert_eq!(query(&mut set, &keys), want, "only the kept ones match");
            narrowed(&mut set, &[b, keys[3]]);
            assert_eq!((set.len(), set.keys()), (1, vec![b]));
            narrowed(&mut set, &[a]);
            assert!(set.is_empty() && set.keys().is_empty());
        }
    }

    #[test]
    fn a_refill_replaces_the_set_whatever_its_kind() {
        let mut set = NeighborSet::of(&[10, 12, 70]);
        assert_eq!(query(&mut set, &[12, 70]), [12, 70]);
        for keys in [
            vec![5, 1 << 20, 9],
            vec![300, 301],
            vec![],
            vec![7, 1 << 30],
            vec![1, 2, 3],
        ] {
            set.fill(&keys);
            let mut want = keys.clone();
            want.sort_unstable();
            assert_eq!(set.len(), want.len());
            let mut asked = want.clone();
            asked.extend([12, 70, 11]);
            assert_eq!(query(&mut set, &asked), want, "{keys:?}");
            assert!(!set.contains(NodeId::new(12)));
        }
    }

    #[test]
    fn dense_targets_make_a_bitmap_and_sparse_ones_a_hash_table() {
        let dense = NeighborSet::of(&[10, 12, 10, 70, 11]);
        assert!(matches!(dense.lookup, Lookup::Bits(_)));
        assert_eq!(dense.keys(), [10, 11, 12, 70]);
        let sparse = NeighborSet::of(&[5, 1 << 20, 5, NodeId::FIRST_PROXY]);
        assert!(matches!(sparse.lookup, Lookup::Hash(_)));
        let mut keys = sparse.keys();
        keys.sort_unstable();
        assert_eq!(keys, [5, 1 << 20, NodeId::FIRST_PROXY]);
        assert!(NeighborSet::of(&[]).is_empty());
    }

    #[test]
    fn a_query_finds_each_held_target_once_in_order_and_leaves_no_mark() {
        let wide: Vec<u32> = (0..2000).map(|i| i * 3).collect();
        for keys in [vec![10, 12, 70, 11], vec![5, 1 << 20, 77, 9], wide] {
            let mut set = NeighborSet::of(&keys);
            let mut want: Vec<u32> = keys.iter().copied().filter(|k| k % 2 == 1).collect();
            want.sort_unstable();
            // Few matches (sorted), then many (scanned, for a bitmap), each asked twice.
            let few: Vec<u32> = want.iter().take(1).flat_map(|&k| [k, k, 1 << 30]).collect();
            assert_eq!(query(&mut set, &few), want.get(..1).unwrap_or(&[]));
            let mut all: Vec<u32> = keys.iter().flat_map(|&k| [k, k + 1, k]).collect();
            all.retain(|k| k % 2 == 1);
            assert_eq!(query(&mut set, &all), want);
            assert_eq!(query(&mut set, &all), want, "the marks were cleared");
            assert_eq!(query(&mut set, &[]), Vec::<u32>::new());
        }
    }

    #[test]
    fn a_hash_table_survives_its_query_count_wrapping() {
        let mut set = NeighborSet::of(&[3, 1 << 20, 1 << 25]);
        if let Lookup::Hash(table) = &mut set.lookup {
            table.set_query(u32::MAX - 2);
        }
        for _ in 0..2 {
            assert_eq!(query(&mut set, &[1 << 25, 3, 3]), [3, 1 << 25]);
        }
        narrowed(&mut set, &[3, 1 << 25]);
        for _ in 0..3 {
            assert_eq!(query(&mut set, &[1 << 25, 3, 1 << 20]), [3, 1 << 25]);
        }
        assert_eq!(set.len(), 2);
        assert!(!set.contains(NodeId::new(1 << 20)));
    }
}
