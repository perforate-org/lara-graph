//! A target set as a bitmap over its range: a lookup is a shift and a mask.

use super::super::sort_indexes;
use alloc::vec::Vec;

/// A query's matches past which the bitmap's marks are scanned in order rather than sorted: one
/// word of marks per this many matches.
const SCAN_WORDS_PER_MATCH: usize = 8;

/// For each 64 ids from `base`, the bits of the targets and the marks of the current query's
/// matches, side by side: one allocation, one place a lookup reads.
pub(super) struct Bits {
    base: u32,
    words: Vec<[u64; 2]>,
}

impl Bits {
    /// An empty bitmap, holding no memory.
    pub(super) const fn new() -> Self {
        Self {
            base: 0,
            words: Vec::new(),
        }
    }

    /// The bitmap of `targets`, which lie in `min..=max` and may repeat, in `words`' memory, and
    /// how many distinct targets it holds.
    pub(super) fn of(
        mut words: Vec<[u64; 2]>,
        targets: &[u32],
        (min, max): (u32, u32),
    ) -> (Self, usize) {
        let range = u64::from(max - min) + 1;
        words.clear();
        words.resize(usize::try_from(range.div_ceil(64)).unwrap_or(0), [0; 2]);
        let mut len = 0;
        for &t in targets {
            let off = t - min;
            if let Some([word, _]) = words.get_mut(word_of(off)) {
                let bit = bit_of(off);
                len += usize::from(*word & bit == 0);
                *word |= bit;
            }
        }
        (Self { base: min, words }, len)
    }

    /// The memory the bitmap took, for a later refill.
    pub(super) fn into_words(self) -> Vec<[u64; 2]> {
        self.words
    }

    /// Whether the bitmap holds `t`.
    pub(super) fn contains(&self, t: u32) -> bool {
        t.checked_sub(self.base)
            .and_then(|off| {
                self.words
                    .get(word_of(off))
                    .map(|[w, _]| w & bit_of(off) != 0)
            })
            .unwrap_or(false)
    }

    /// Appends the targets, ascending, to `out`.
    pub(super) fn push_keys(&self, out: &mut Vec<u32>) {
        for (i, [word, _]) in self.words.iter().enumerate() {
            push_bits(*word, self.base + word_start(i), out);
        }
    }

    /// Marks `t` if the bitmap holds it and it is not marked yet; whether it was marked now.
    #[inline]
    pub(super) fn hit(&mut self, t: u32) -> bool {
        let Some(off) = t.checked_sub(self.base) else {
            return false;
        };
        let bit = bit_of(off);
        match self.words.get_mut(word_of(off)) {
            Some([held, seen]) if *held & bit != 0 && *seen & bit == 0 => {
                *seen |= bit;
                true
            }
            _ => false,
        }
    }

    /// Ends a query whose marks are `matches`, none below `from`: they are appended to `out`,
    /// ascending, and cleared, with `matches`. Many matches are read off the marks in order, from
    /// `from`'s word; few are sorted.
    pub(super) fn finish_from(&mut self, from: u32, matches: &mut Vec<u32>, out: &mut Vec<u32>) {
        let base = self.base;
        let skip = word_of(from.saturating_sub(base)).min(self.words.len());
        let tail = self.words.get_mut(skip..).unwrap_or(&mut []);
        if matches.len().saturating_mul(SCAN_WORDS_PER_MATCH) >= tail.len() {
            for (i, [_, seen]) in (skip..).zip(tail.iter_mut()) {
                push_bits(*seen, base + word_start(i), out);
                *seen = 0;
            }
            matches.clear();
            return;
        }
        for &t in matches.iter() {
            let off = t - base;
            if let Some([_, seen]) = self.words.get_mut(word_of(off)) {
                *seen &= !bit_of(off);
            }
        }
        sort_indexes(matches);
        out.append(matches);
    }

    /// Keeps only the marked targets, and clears the marks.
    pub(super) fn narrow(&mut self) {
        for [held, seen] in &mut self.words {
            *held = *seen;
            *seen = 0;
        }
    }
}

/// Appends `from + i` for each set bit `i` of `word`, ascending.
fn push_bits(word: u64, from: u32, out: &mut Vec<u32>) {
    let mut rest = word;
    while rest != 0 {
        out.push(from + rest.trailing_zeros());
        rest &= rest - 1;
    }
}

/// The offset of word `i`'s first bit: below the bitmap's range, which fits a `u32`.
fn word_start(i: usize) -> u32 {
    u32::try_from(i * 64).unwrap_or(u32::MAX)
}

fn word_of(off: u32) -> usize {
    usize::try_from(off >> 6).unwrap_or(usize::MAX)
}

fn bit_of(off: u32) -> u64 {
    1 << (off & 63)
}
