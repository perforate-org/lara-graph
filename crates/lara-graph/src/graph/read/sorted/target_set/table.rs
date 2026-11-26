//! A target set as an open-addressing hash table, for targets too sparse for a bitmap.

use alloc::vec::Vec;

/// A slot no key occupies. No node has the index `u32::MAX`: the last proxy's is one less.
const EMPTY: u32 = u32::MAX;

/// Linear probing, at most half full: each slot's key and the last query that matched it. Stamps
/// only grow, so the keys stamped `alive` or later are the set: narrowing it to a query's matches
/// raises `alive` to that query.
pub(super) struct Table {
    slots: Vec<[u32; 2]>,
    mask: usize,
    shift: u32,
    query: u32,
    alive: u32,
}

impl Table {
    /// The table of `targets`, which may repeat, in `slots`' memory, and how many distinct
    /// targets it holds.
    pub(super) fn of(mut slots: Vec<[u32; 2]>, targets: &[u32]) -> (Self, usize) {
        let size = (2 * targets.len()).next_power_of_two().max(16);
        slots.clear();
        slots.resize(size, [EMPTY, 0]);
        let mask = size - 1;
        let shift = 32 - size.trailing_zeros().min(32);
        let mut len = 0;
        for &t in targets {
            let at = slot_of(&slots, mask, shift, t);
            if let Some(slot) = slots.get_mut(at)
                && slot[0] == EMPTY
            {
                slot[0] = t;
                len += 1;
            }
        }
        let table = Self {
            slots,
            mask,
            shift,
            query: 0,
            alive: 0,
        };
        (table, len)
    }

    /// The memory the table took, for a later refill.
    pub(super) fn into_slots(self) -> Vec<[u32; 2]> {
        self.slots
    }

    /// Whether the table holds `t`.
    pub(super) fn contains(&self, t: u32) -> bool {
        self.slots
            .get(slot_of(&self.slots, self.mask, self.shift, t))
            .is_some_and(|s| s[0] == t && s[1] >= self.alive)
    }

    /// Appends the targets, in no order, to `out`.
    pub(super) fn push_keys(&self, out: &mut Vec<u32>) {
        out.extend(
            self.slots
                .iter()
                .filter(|s| s[0] != EMPTY && s[1] >= self.alive)
                .map(|s| s[0]),
        );
    }

    /// Starts a query: a new stamp, so nothing is matched yet.
    pub(super) fn begin(&mut self) {
        if self.query == u32::MAX {
            // Out of stamps: the set's keys restart at 1, the others at 0.
            for slot in &mut self.slots {
                slot[1] = u32::from(slot[1] >= self.alive);
            }
            self.alive = 1;
            self.query = 1;
        }
        self.query += 1;
    }

    /// Stamps `t` if the table holds it and this query has not; whether it was stamped now.
    #[inline]
    pub(super) fn hit(&mut self, t: u32) -> bool {
        let at = slot_of(&self.slots, self.mask, self.shift, t);
        match self.slots.get_mut(at) {
            Some(slot) if slot[0] == t && slot[1] >= self.alive && slot[1] != self.query => {
                slot[1] = self.query;
                true
            }
            _ => false,
        }
    }

    /// Keeps only the targets this query stamped.
    pub(super) fn narrow(&mut self) {
        self.alive = self.query;
    }

    /// Moves the query stamp to `query`, to test its wrapping.
    #[cfg(test)]
    pub(super) fn set_query(&mut self, query: u32) {
        self.query = query;
    }
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
