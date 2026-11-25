//! A set of node indexes built once from one bucket's targets, to intersect with many others: a
//! dense bitmap over the targets' range when it is narrow, else an open-addressing hash table.
//! Each intersection is a query that marks the targets it finds, so the set itself never changes
//! and repeats in the other bucket (parallel edges) are found once.

use super::sort_indexes;
use crate::id::NodeId;
use alloc::vec::Vec;

/// Ids of range a target may stand for in a bitmap. A bitmap and its marks then take up to 64
/// bytes a key, four times the hash table, but a lookup is a shift and a mask: intersecting 65536
/// edges with a set of 1024 targets spread over 131072 ids cost 4.6M instructions against 8.7M
/// through the hash table (`common_65536_1024`), while at 64 ids a key it stayed a hash table.
const BITMAP_RANGE_PER_KEY: u64 = 256;

/// A query's matches past which the bitmap's marks are scanned in order rather than sorted: one
/// word of marks per this many matches.
const SCAN_WORDS_PER_MATCH: usize = 8;

/// A slot no key occupies. No node has the index `u32::MAX`: the last proxy's is one less.
const EMPTY: u32 = u32::MAX;

/// The targets one bucket holds, each once, ready to intersect with others
/// ([`crate::Graph::common_with`]). Built by [`crate::Graph::target_set`]; it borrows nothing, so
/// it may outlive writes to the graph, and then describes the bucket as it was.
pub struct TargetSet {
    len: usize,
    lookup: Lookup,
    /// The current query's matches.
    matches: Vec<u32>,
}

enum Lookup {
    /// For each 64 ids from `base`, the bits of the targets and the marks of the current query's
    /// matches, side by side: one allocation, one place a lookup reads.
    Bits { base: u32, words: Vec<[u64; 2]> },
    /// Linear probing, at most half full: each slot's key and the last query that matched it.
    Hash {
        slots: Vec<[u32; 2]>,
        mask: usize,
        shift: u32,
        query: u32,
    },
}

impl TargetSet {
    /// The set of `targets`, which may repeat.
    pub(super) fn of(targets: &[u32]) -> Self {
        let (mut min, mut max) = (u32::MAX, 0);
        for &t in targets {
            min = min.min(t);
            max = max.max(t);
        }
        let mut len = 0;
        let lookup = if targets.is_empty() {
            Lookup::Bits {
                base: 0,
                words: Vec::new(),
            }
        } else if u64::from(max - min) < width(targets.len()) {
            let range = u64::from(max - min) + 1;
            let mut words =
                alloc::vec![[0u64; 2]; usize::try_from(range.div_ceil(64)).unwrap_or(0)];
            for &t in targets {
                let off = t - min;
                if let Some([word, _]) = words.get_mut(word_of(off)) {
                    let bit = bit_of(off);
                    len += usize::from(*word & bit == 0);
                    *word |= bit;
                }
            }
            Lookup::Bits { base: min, words }
        } else {
            let size = (2 * targets.len()).next_power_of_two().max(16);
            let mut slots = alloc::vec![[EMPTY, 0]; size];
            let mask = size - 1;
            let shift = 32 - size.trailing_zeros().min(32);
            for &t in targets {
                let at = slot_of(&slots, mask, shift, t);
                if let Some(slot) = slots.get_mut(at)
                    && slot[0] == EMPTY
                {
                    slot[0] = t;
                    len += 1;
                }
            }
            Lookup::Hash {
                slots,
                mask,
                shift,
                query: 0,
            }
        };
        Self {
            len,
            lookup,
            matches: Vec::new(),
        }
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
        let t = t.index();
        match &self.lookup {
            Lookup::Bits { base, words } => t
                .checked_sub(*base)
                .and_then(|off| words.get(word_of(off)).map(|[w, _]| w & bit_of(off) != 0))
                .unwrap_or(false),
            Lookup::Hash {
                slots, mask, shift, ..
            } => slots
                .get(slot_of(slots, *mask, *shift, t))
                .is_some_and(|s| s[0] == t),
        }
    }

    /// The targets, each once: ascending from a bitmap, in no order from a hash table.
    pub(super) fn keys(&self) -> Vec<u32> {
        let mut keys = Vec::with_capacity(self.len);
        match &self.lookup {
            Lookup::Bits { base, words } => {
                for (at, [word, _]) in (0u32..).step_by(64).zip(words) {
                    push_bits(*word, *base + at, &mut keys);
                }
            }
            Lookup::Hash { slots, .. } => {
                keys.extend(slots.iter().map(|s| s[0]).filter(|&k| k != EMPTY));
            }
        }
        keys
    }

    /// Starts a query: no target is matched yet.
    pub(super) fn begin(&mut self) {
        self.matches.clear();
        if let Lookup::Hash { slots, query, .. } = &mut self.lookup {
            *query = query.wrapping_add(1);
            if *query == 0 {
                // Stamps start at 0, so a wrapped query clears them first.
                for slot in slots.iter_mut() {
                    slot[1] = 0;
                }
                *query = 1;
            }
        }
    }

    /// Matches `t` if the set holds it and this query has not matched it yet.
    #[inline]
    pub(super) fn hit(&mut self, t: u32) {
        let new = match &mut self.lookup {
            Lookup::Bits { base, words } => {
                let Some(off) = t.checked_sub(*base) else {
                    return;
                };
                let bit = bit_of(off);
                match words.get_mut(word_of(off)) {
                    Some([held, seen]) if *held & bit != 0 && *seen & bit == 0 => {
                        *seen |= bit;
                        true
                    }
                    _ => false,
                }
            }
            Lookup::Hash {
                slots,
                mask,
                shift,
                query,
            } => {
                let at = slot_of(slots, *mask, *shift, t);
                match slots.get_mut(at) {
                    Some(slot) if slot[0] == t && slot[1] != *query => {
                        slot[1] = *query;
                        true
                    }
                    _ => false,
                }
            }
        };
        if new {
            self.matches.push(t);
        }
    }

    /// Ends the query: its matches, ascending, appended to `out`, and the marks cleared.
    pub(super) fn finish(&mut self, out: &mut Vec<u32>) {
        match &mut self.lookup {
            Lookup::Bits { base, words }
                if self.matches.len().saturating_mul(SCAN_WORDS_PER_MATCH) >= words.len() =>
            {
                for (at, [_, seen]) in (0u32..).step_by(64).zip(words.iter_mut()) {
                    push_bits(*seen, *base + at, out);
                    *seen = 0;
                }
                self.matches.clear();
                return;
            }
            Lookup::Bits { base, words } => {
                for &t in &self.matches {
                    let off = t - *base;
                    if let Some([_, seen]) = words.get_mut(word_of(off)) {
                        *seen &= !bit_of(off);
                    }
                }
            }
            Lookup::Hash { .. } => {}
        }
        sort_indexes(&mut self.matches);
        out.append(&mut self.matches);
    }
}

/// The widest range of ids `n` targets make a bitmap for.
fn width(n: usize) -> u64 {
    u64::try_from(n)
        .unwrap_or(u64::MAX)
        .saturating_mul(BITMAP_RANGE_PER_KEY)
}

/// Appends `from + i` for each set bit `i` of `word`, ascending.
fn push_bits(word: u64, from: u32, out: &mut Vec<u32>) {
    let mut rest = word;
    while rest != 0 {
        out.push(from + rest.trailing_zeros());
        rest &= rest - 1;
    }
}

fn word_of(off: u32) -> usize {
    usize::try_from(off >> 6).unwrap_or(usize::MAX)
}

fn bit_of(off: u32) -> u64 {
    1 << (off & 63)
}

/// The slot holding `key`, or the empty one where it would go: linear probing from its home, the
/// top bits of the key times the golden ratio. The table is at most half full, so an empty slot
/// ends every probe.
#[inline]
fn slot_of(slots: &[[u32; 2]], mask: usize, shift: u32, key: u32) -> usize {
    let top = key
        .wrapping_mul(0x9E37_79B9)
        .checked_shr(shift)
        .unwrap_or(0);
    let mut at = usize::try_from(top).unwrap_or(0) & mask;
    loop {
        match slots.get(at) {
            Some(&[k, _]) if k != key && k != EMPTY => at = (at + 1) & mask,
            Some(_) | None => return at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn query(set: &mut TargetSet, targets: &[u32]) -> Vec<u32> {
        set.begin();
        for &t in targets {
            set.hit(t);
        }
        let mut out = Vec::new();
        set.finish(&mut out);
        out
    }

    #[test]
    fn dense_targets_make_a_bitmap_and_sparse_ones_a_hash_table() {
        let dense = TargetSet::of(&[10, 12, 10, 70, 11]);
        assert!(matches!(dense.lookup, Lookup::Bits { .. }));
        assert_eq!(dense.keys(), [10, 11, 12, 70]);
        let sparse = TargetSet::of(&[5, 1 << 20, 5, NodeId::FIRST_PROXY]);
        assert!(matches!(sparse.lookup, Lookup::Hash { .. }));
        let mut keys = sparse.keys();
        keys.sort_unstable();
        assert_eq!(keys, [5, 1 << 20, NodeId::FIRST_PROXY]);
        assert!(TargetSet::of(&[]).is_empty());
    }

    #[test]
    fn a_query_finds_each_held_target_once_in_order_and_leaves_no_mark() {
        let wide: Vec<u32> = (0..2000).map(|i| i * 3).collect();
        for keys in [vec![10, 12, 70, 11], vec![5, 1 << 20, 77, 9], wide] {
            let mut set = TargetSet::of(&keys);
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
        let mut set = TargetSet::of(&[3, 1 << 20, 1 << 25]);
        if let Lookup::Hash { query, .. } = &mut set.lookup {
            *query = u32::MAX - 1;
        }
        for _ in 0..4 {
            assert_eq!(query(&mut set, &[1 << 25, 3, 3]), [3, 1 << 25]);
        }
    }
}
