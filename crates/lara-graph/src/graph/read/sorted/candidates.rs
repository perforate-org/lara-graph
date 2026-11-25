//! The candidate targets of a k-way intersection: an open-addressing hash table of node indexes,
//! each stamped with the last round that kept it, so a round marks the candidates a bucket holds
//! without moving or rebuilding anything.

use alloc::vec::Vec;

/// A slot no key occupies. No node's index is `u32::MAX`: a node stores its index's complement,
/// which is never zero.
const EMPTY: u32 = u32::MAX;

pub(super) struct Candidates {
    /// Each slot's key and the last round that kept it, side by side so a lookup reads one place.
    slots: Vec<[u32; 2]>,
    mask: usize,
    /// Bits a key's product drops to land in the table: 32 less the table's log2.
    shift: u32,
    /// The round under way: candidates stamped with the one before it are alive.
    round: u32,
    alive: usize,
    /// Candidates alive when this round started.
    before: usize,
}

impl Candidates {
    /// Room for `n` keys at most half full.
    pub(super) fn with_room(n: usize) -> Self {
        let len = (2 * n).next_power_of_two().max(16);
        Self {
            slots: alloc::vec![[EMPTY, 0]; len],
            mask: len - 1,
            shift: 32 - len.trailing_zeros().min(32),
            round: 0,
            alive: 0,
            before: 0,
        }
    }

    /// Adds `key` in round 0; a repeat changes nothing.
    #[inline]
    pub(super) fn insert(&mut self, key: u32) {
        let at = self.slot_of(key);
        if let Some(slot) = self.slots.get_mut(at)
            && slot[0] == EMPTY
        {
            slot[0] = key;
            self.alive += 1;
        }
    }

    /// Starts the next round: no candidate is kept until [`Self::keep`] marks it.
    pub(super) fn next_round(&mut self) {
        self.round += 1;
        self.before = self.alive;
        self.alive = 0;
    }

    /// Keeps `key` into this round if it was alive in the last one.
    #[inline]
    pub(super) fn keep(&mut self, key: u32) {
        let at = self.slot_of(key);
        let round = self.round;
        if let Some(slot) = self.slots.get_mut(at)
            && slot[0] == key
            && slot[1] + 1 == round
        {
            slot[1] = round;
            self.alive += 1;
        }
    }

    /// The slot holding `key`, or the empty one where it would go: linear probing from its home.
    /// The table is at most half full, so an empty slot ends every probe.
    #[inline]
    fn slot_of(&self, key: u32) -> usize {
        let mut at = self.home(key);
        loop {
            match self.slots.get(at) {
                Some(&[k, _]) if k != key && k != EMPTY => at = (at + 1) & self.mask,
                Some(_) | None => return at,
            }
        }
    }

    /// The candidates alive before this round, to filter one by one.
    pub(super) fn previous(&self) -> Vec<u32> {
        self.with_stamp(self.round.wrapping_sub(1))
    }

    /// Candidates alive before this round.
    pub(super) const fn previous_count(&self) -> usize {
        self.before
    }

    /// Candidates alive now.
    pub(super) const fn alive(&self) -> usize {
        self.alive
    }

    /// The candidates alive now, in no order.
    pub(super) fn survivors(&self) -> Vec<u32> {
        self.with_stamp(self.round)
    }

    fn with_stamp(&self, round: u32) -> Vec<u32> {
        self.slots
            .iter()
            .filter(|&&[k, s]| k != EMPTY && s == round)
            .map(|&[k, _]| k)
            .collect()
    }

    /// Fibonacci hashing: the top bits of the key times the golden ratio, which every bit of the
    /// key reaches.
    #[inline]
    fn home(&self, key: u32) -> usize {
        let top = key
            .wrapping_mul(0x9E37_79B9)
            .checked_shr(self.shift)
            .unwrap_or(0);
        usize::try_from(top).unwrap_or(0) & self.mask
    }
}
