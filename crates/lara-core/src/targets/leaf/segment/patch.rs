//! Changing a segment in its own slot: lowering or raising one posting's count, or adding a
//! posting, without re-coding the rest of the segment.

use super::{Entry, Read, SEGMENT_MAX, count_bits};
use crate::corrupt::corrupt;
use crate::targets::bits::{Reader, Writer};
use crate::targets::tree::Change;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::Range;

impl Entry {
    /// Raises the count of the posting at `key` by one in its own slot, without re-coding the
    /// segment: a posting it holds gets its count bits rewritten; a new one is coded after the
    /// postings before it, the next posting's gap is coded again, and the bits after that move
    /// along unchanged. `None` when that does not fit: a count or block past its width, a full
    /// segment, a slot too small, or a new first posting (the table and the parent list the old
    /// one).
    pub(in crate::targets::leaf) fn raise(
        &self,
        read: Read<'_>,
        key: (u32, u32),
    ) -> Option<Patched> {
        let code = self.padded(read);
        let spot = self.spot_of(&code, key)?;
        let total = self.total.checked_add(1)?;
        match spot {
            Spot::Held { count_at, count } => self.bumped(code, total, (count_at, count)),
            Spot::Before { at, previous, next } => {
                self.inserted(code, total, key, (at, previous, next))
            }
        }
    }

    /// The patch that rewrites the count at bit `count_at` one higher.
    fn bumped(
        &self,
        mut code: Vec<u8>,
        total: u32,
        (count_at, count): (usize, u32),
    ) -> Option<Patched> {
        let count = count.checked_add(1)?;
        if count_bits(count) > self.c {
            return None;
        }
        set_bits(&mut code, count_at, self.c, u64::from(count));
        let end = (count_at + usize::try_from(self.c).ok()?).div_ceil(8);
        Some(Patched {
            code,
            changed: count_at / 8..end,
            entry: Self { total, ..*self },
        })
    }

    /// The patch that codes a posting for `key` at bit `at`, after one to `previous`, then the
    /// gap of the posting to `next`'s target, and moves the bits after that gap along.
    fn inserted(
        &self,
        mut code: Vec<u8>,
        total: u32,
        key: (u32, u32),
        (at, previous, next): (usize, u32, Option<(u32, usize)>),
    ) -> Option<Patched> {
        if self.n >= SEGMENT_MAX || count_bits(key.1) > self.w {
            return None;
        }
        let byte = at / 8;
        let lead = u32::try_from(at % 8).ok()?;
        let mut out = Writer::with_capacity(self.len() - byte + 8);
        // The bits of `at`'s byte before it stay as they are.
        out.fixed(u32::from(*code.get(byte)?), lead);
        out.rice(key.0 - previous, self.k);
        out.fixed(1, self.c);
        out.fixed(key.1, self.w);
        if let Some((target, gap_end)) = next {
            out.rice(target - key.0, self.k);
            copy_bits(&mut out, &code, gap_end..self.bits);
        }
        let bits = byte * 8 + out.bits();
        if bits.div_ceil(8) > self.cap {
            return None;
        }
        code.truncate(byte);
        code.extend_from_slice(&out.into_bytes());
        Some(Patched {
            changed: byte..code.len(),
            code,
            entry: Self {
                total,
                bits,
                n: self.n + 1,
                ..*self
            },
        })
    }

    /// Where `key` is in `code`: the posting that holds it, or where a new posting for it goes.
    /// `None` for a key before the first posting.
    fn spot_of(&self, code: &[u8], key: (u32, u32)) -> Option<Spot> {
        let mut reader = Reader::new(code);
        let mut target = self.first.0;
        let mut previous = None;
        for i in 0..self.n {
            let start = reader.position();
            if i > 0 {
                target += reader.rice(self.k);
            }
            let gap_end = reader.position();
            let count = reader.fixed(self.c);
            let block = reader.fixed(self.w);
            match (target, block).cmp(&key) {
                Ordering::Less => previous = Some(target),
                Ordering::Equal => {
                    return Some(Spot::Held {
                        count_at: gap_end,
                        count,
                    });
                }
                Ordering::Greater => {
                    return Some(Spot::Before {
                        at: start,
                        previous: previous?,
                        next: Some((target, gap_end)),
                    });
                }
            }
        }
        Some(Spot::Before {
            at: self.bits,
            previous: previous?,
            next: None,
        })
    }

    /// Lowers the counts of postings it holds by `changes` (sorted, all decrements) in place, or
    /// `None` when a change names a posting it lacks. A lower count always fits its field; one
    /// that reaches zero stays until the segment is rewritten.
    #[inline]
    pub(in crate::targets::leaf) fn lower(
        &self,
        read: Read<'_>,
        changes: &[Change],
    ) -> Option<Patched> {
        let mut code = self.padded(read);
        let mut found: Vec<(usize, u32)> = Vec::with_capacity(changes.len());
        let mut rest = changes.iter().peekable();
        self.scan(&code, |p, at| {
            if rest.next_if(|c| c.key == p.key()).is_some() {
                found.push((at, p.count));
            }
            rest.peek().is_none()
        });
        if found.len() != changes.len() {
            return None;
        }
        let mut total = i64::from(self.total);
        let (mut low, mut high) = (usize::MAX, 0);
        for (&(at, count), c) in found.iter().zip(changes) {
            let next = u64::try_from(i64::from(count) + c.delta)
                .unwrap_or_else(|_| corrupt("an index count below zero"));
            set_bits(&mut code, at, self.c, next);
            total += c.delta;
            low = low.min(at / 8);
            high = high.max((at + usize::try_from(self.c).unwrap_or(32)).div_ceil(8));
        }
        let total = u32::try_from(total).ok()?;
        Some(Patched {
            code,
            changed: low..high,
            entry: Self { total, ..*self },
        })
    }
}

/// Appends the bits `range` of `code` to `out`.
fn copy_bits(out: &mut Writer, code: &[u8], range: Range<usize>) {
    let mut rest = Reader::at(code, range.start);
    let mut left = range.len();
    while left > 0 {
        let width = left.min(32);
        let bits = u32::try_from(width).unwrap_or(32);
        out.fixed(rest.fixed(bits), bits);
        left -= width;
    }
}

/// Where a key is in a segment's code.
enum Spot {
    /// In the posting whose count field starts at bit `count_at`.
    Held { count_at: usize, count: u32 },
    /// Not there: a posting for it goes at bit `at`, after a posting to `previous`, and before
    /// the one to `next`'s target, whose gap ends at its bit (none at the end).
    Before {
        at: usize,
        previous: u32,
        next: Option<(u32, usize)>,
    },
}

/// A segment changed in its own slot.
pub(in crate::targets::leaf) struct Patched {
    /// The slot's code.
    code: Vec<u8>,
    /// The bytes of `code` that changed.
    changed: Range<usize>,
    /// The entry with its new total and length.
    pub(in crate::targets::leaf) entry: Entry,
}

impl Patched {
    /// The changed bytes of the code, and where in the leaf they go.
    pub(in crate::targets::leaf) fn changed(&self) -> (usize, &[u8]) {
        (
            self.entry.offset + self.changed.start,
            self.code.get(self.changed.clone()).unwrap_or(&[]),
        )
    }
}

/// Writes the low `width` bits of `value` at bit `at` of `bytes` (least significant first).
fn set_bits(bytes: &mut [u8], at: usize, width: u32, value: u64) {
    for i in 0..usize::try_from(width).unwrap_or(0) {
        let bit = at + i;
        if let Some(byte) = bytes.get_mut(bit / 8) {
            let mask = 1u8 << (bit % 8);
            if value >> i & 1 == 1 {
                *byte |= mask;
            } else {
                *byte &= !mask;
            }
        }
    }
}
