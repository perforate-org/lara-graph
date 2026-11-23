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
            Spot::Held { count_at, count } => {
                self.recounted(code, total, (count_at, count.checked_add(1)?))
            }
            Spot::Before { at, previous, next } => {
                self.inserted(code, total, key, (at, previous, next))
            }
        }
    }

    /// The patch that rewrites the count at bit `count_at` as `count`, or `None` when it does
    /// not fit the count width.
    fn recounted(
        &self,
        mut code: Vec<u8>,
        total: u32,
        (count_at, count): (usize, u32),
    ) -> Option<Patched> {
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
        code: Vec<u8>,
        total: u32,
        key: (u32, u32),
        (at, previous, next): (usize, u32, Option<(u32, usize)>),
    ) -> Option<Patched> {
        if self.n >= SEGMENT_MAX || count_bits(key.1) > self.w {
            return None;
        }
        let rest = next.map_or(self.bits, |(_, gap_end)| gap_end);
        let (code, bits, byte) = self.spliced(code, at, rest, |out| {
            out.rice(key.0 - previous, self.k);
            out.fixed(1, self.c);
            out.fixed(key.1, self.w);
            if let Some((target, _)) = next {
                out.rice(target - key.0, self.k);
            }
        })?;
        if bits.div_ceil(8) > self.cap {
            return None;
        }
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

    /// Lowers the count of the posting at `key` by one in its own slot. A posting whose count
    /// reaches zero leaves the segment: the next posting's gap is coded again (or it becomes the
    /// first), and the bits after it move along. `None` when the segment lacks the posting.
    pub(in crate::targets::leaf) fn lower_one(
        &self,
        read: Read<'_>,
        key: (u32, u32),
    ) -> Option<Lowered> {
        let code = self.padded(read);
        let held = self.held(&code, key)?;
        let total = self.total.checked_sub(1)?;
        if held.count > 1 {
            let patched = self.recounted(code, total, (held.count_at, held.count - 1))?;
            return Some(Lowered::Patched(patched));
        }
        let Some(next) = held.next else {
            return match held.previous {
                // The last posting goes: the code ends where it started.
                Some(_) => Some(Lowered::Patched(self.dropped(
                    code,
                    total,
                    (held.start, self.bits),
                    None,
                )?)),
                None => Some(Lowered::Emptied),
            };
        };
        let patched = if let Some(previous) = held.previous {
            let gap = (next.target - previous, self.k);
            self.dropped(code, total, (held.start, next.gap_end), Some(gap))?
        } else {
            // The first posting goes: the next one, without its gap, starts the code.
            let mut patched = self.dropped(code, total, (0, next.gap_end), None)?;
            patched.entry.first = (next.target, next.block);
            patched
        };
        Some(Lowered::Patched(patched))
    }

    /// The patch that removes the bits `at..rest` of a posting with count one, writing `gap` (a
    /// Rice code of the next posting's new gap) in their place.
    fn dropped(
        &self,
        code: Vec<u8>,
        total: u32,
        (at, rest): (usize, usize),
        gap: Option<(u32, u32)>,
    ) -> Option<Patched> {
        let (code, bits, byte) = self.spliced(code, at, rest, |out| {
            if let Some((value, k)) = gap {
                out.rice(value, k);
            }
        })?;
        Some(Patched {
            changed: byte..code.len(),
            code,
            entry: Self {
                total,
                bits,
                n: self.n - 1,
                ..*self
            },
        })
    }

    /// `code` with its bits from `at` on replaced by what `middle` writes, followed by the bits
    /// from `rest` on, unchanged. Returns the code, its length in bits, and the first byte that
    /// changed.
    fn spliced(
        &self,
        mut code: Vec<u8>,
        at: usize,
        rest: usize,
        middle: impl FnOnce(&mut Writer),
    ) -> Option<(Vec<u8>, usize, usize)> {
        let byte = at / 8;
        let lead = u32::try_from(at % 8).ok()?;
        let mut out = Writer::with_capacity(self.len() - byte + 8);
        // The bits of `at`'s byte before it stay as they are.
        out.fixed(u32::from(*code.get(byte)?), lead);
        middle(&mut out);
        copy_bits(&mut out, &code, rest..self.bits);
        let bits = byte * 8 + out.bits();
        code.truncate(byte);
        code.extend_from_slice(&out.into_bytes());
        Some((code, bits, byte))
    }

    /// The posting at `key` in `code`: where it starts, its count, and its neighbours.
    fn held(&self, code: &[u8], key: (u32, u32)) -> Option<Held> {
        let mut reader = Reader::new(code);
        let mut target = self.first.0;
        let mut previous = None;
        for i in 0..self.n {
            let start = reader.position();
            if i > 0 {
                target += reader.rice(self.k);
            }
            let count_at = reader.position();
            let count = reader.fixed(self.c);
            let block = reader.fixed(self.w);
            match (target, block).cmp(&key) {
                Ordering::Less => previous = Some(target),
                Ordering::Greater => return None,
                Ordering::Equal => {
                    let next = (i + 1 < self.n).then(|| {
                        let target = target + reader.rice(self.k);
                        let gap_end = reader.position();
                        reader.fixed(self.c);
                        let block = reader.fixed(self.w);
                        Next {
                            target,
                            block,
                            gap_end,
                        }
                    });
                    return Some(Held {
                        start,
                        count_at,
                        count,
                        previous,
                        next,
                    });
                }
            }
        }
        None
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

/// A posting found in a segment's code.
struct Held {
    /// Where it starts (its gap, or its count for the first posting).
    start: usize,
    count_at: usize,
    count: u32,
    /// The target of the posting before it.
    previous: Option<u32>,
    next: Option<Next>,
}

/// The posting after a held one.
struct Next {
    target: u32,
    block: u32,
    /// Where its gap ends: its count starts.
    gap_end: usize,
}

/// What lowering one posting's count does to its segment.
pub(in crate::targets::leaf) enum Lowered {
    Patched(Patched),
    /// Its only posting reached zero: the segment would be empty.
    Emptied,
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
