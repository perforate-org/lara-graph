//! Segments: runs of sorted postings coded together, and the table entries that describe them.

use super::{ENTRY_BYTES, Read};
use crate::corrupt::corrupt;
use crate::targets::Posting;
use crate::targets::bits::{PADDING, Reader, Writer, rice_len};
use crate::targets::tree::Change;
use alloc::vec::Vec;

/// Postings a segment is cut to when it is rebuilt.
const SEGMENT_TARGET: usize = 16;
/// Postings a segment may grow to before it is cut.
const SEGMENT_MAX: usize = 2 * SEGMENT_TARGET;
const MAX_K: u32 = 24;

/// A coded segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Coded {
    /// The first posting's key.
    pub(super) first: (u32, u32),
    pub(super) total: u32,
    pub(super) k: u32,
    pub(super) w: u32,
    /// Bits of each count: a count changes in place while it fits.
    pub(super) c: u32,
    pub(super) n: usize,
    pub(super) bytes: Vec<u8>,
}

/// Bits a count field takes for counts up to `max`.
fn count_bits(max: u32) -> u32 {
    (32 - max.leading_zeros()).max(1)
}

impl Coded {
    /// Codes sorted postings (at most [`SEGMENT_MAX`]): each target gap after the first in
    /// Rice(`k`), each count in `c` bits, each block in `w` bits.
    fn encode(postings: &[Posting]) -> Self {
        let Some(first) = postings.first().map(Posting::key) else {
            corrupt("an empty index segment")
        };
        let max_block = postings.iter().map(|p| p.block).max().unwrap_or(0);
        let w = (32 - max_block.leading_zeros()).max(1);
        let k = best_k(postings);
        let c = count_bits(postings.iter().map(|p| p.count).max().unwrap_or(1));
        let mut out = Writer::with_capacity(postings.len() * 4);
        let mut previous = first.0;
        for (i, p) in postings.iter().enumerate() {
            if i > 0 {
                out.rice(p.target - previous, k);
            }
            previous = p.target;
            out.fixed(p.count, c);
            out.fixed(p.block, w);
        }
        Self {
            first,
            total: postings.iter().map(|p| p.count).sum(),
            k,
            w,
            c,
            n: postings.len(),
            bytes: out.into_bytes(),
        }
    }

    /// The segment a table entry describes, with the code read from its slot.
    pub(super) fn kept(e: &Entry, read: Read<'_>) -> Self {
        let mut bytes = alloc::vec![0u8; e.len];
        read(e.offset, &mut bytes);
        Self {
            first: e.first,
            total: e.total,
            k: e.k,
            w: e.w,
            c: e.c,
            n: e.n,
            bytes,
        }
    }
}

/// The Rice parameter for the target gaps of sorted `postings`: near log2 of their mean, the best
/// of it and its neighbours.
fn best_k(postings: &[Posting]) -> u32 {
    let gaps = || {
        postings.windows(2).map(|pair| match pair {
            [a, b] => b.target - a.target,
            _ => 0,
        })
    };
    let bits = |k: u32| -> usize { gaps().map(|g| rice_len(g, k)).sum() };
    let span = match (postings.first(), postings.last()) {
        (Some(a), Some(b)) => u64::from(b.target - a.target),
        _ => 0,
    };
    let mean = span / u64::try_from(postings.len().saturating_sub(1).max(1)).unwrap_or(1);
    let guess = mean.max(1).ilog2();
    (guess.saturating_sub(1)..=(guess + 1).min(MAX_K))
        .min_by_key(|&k| bits(k))
        .unwrap_or(0)
}

/// A segment's table entry: its first posting's key, its total count, its slot, the length of
/// its code, and its coding parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Entry {
    pub(super) first: (u32, u32),
    pub(super) total: u32,
    pub(super) offset: usize,
    pub(super) cap: usize,
    len: usize,
    k: u32,
    w: u32,
    c: u32,
    n: usize,
}

/// Bit widths of the packed fields after an entry's total: offset, capacity, length (12 bits
/// each, as a leaf is at most 4096 bytes), `k` (5), `w` (6), the count width less one (5) and `n`
/// (7).
const PACKED: [u32; 7] = [12, 12, 12, 5, 6, 5, 7];

impl Entry {
    /// The entry of `c` placed in the slot `(offset, cap)`.
    pub(super) fn placed(c: &Coded, (offset, cap): (usize, usize)) -> Self {
        Self {
            first: c.first,
            total: c.total,
            offset,
            cap,
            len: c.bytes.len(),
            k: c.k,
            w: c.w,
            c: c.c,
            n: c.n,
        }
    }

    pub(super) const fn slot(&self) -> (usize, usize) {
        (self.offset, self.cap)
    }

    pub(super) fn to_bytes(self) -> [u8; ENTRY_BYTES] {
        let wide = |v: usize| u64::try_from(v).unwrap_or(u64::MAX);
        let fields = [
            wide(self.offset),
            wide(self.cap),
            wide(self.len),
            u64::from(self.k),
            u64::from(self.w),
            u64::from(self.c.saturating_sub(1)),
            wide(self.n),
        ];
        let mut packed = 0u64;
        let mut shift = 0;
        for (value, width) in fields.into_iter().zip(PACKED) {
            if value >> width != 0 {
                corrupt("an index entry field out of range");
            }
            packed |= value << shift;
            shift += width;
        }
        let [t0, t1, t2, t3] = self.first.0.to_le_bytes();
        let [b0, b1, b2, b3] = self.first.1.to_le_bytes();
        let [c0, c1, c2, c3] = self.total.to_le_bytes();
        let [p0, p1, p2, p3, p4, p5, p6, p7] = packed.to_le_bytes();
        [
            t0, t1, t2, t3, b0, b1, b2, b3, c0, c1, c2, c3, p0, p1, p2, p3, p4, p5, p6, p7,
        ]
    }

    pub(super) fn from_bytes(bytes: [u8; ENTRY_BYTES]) -> Self {
        let [
            t0,
            t1,
            t2,
            t3,
            b0,
            b1,
            b2,
            b3,
            c0,
            c1,
            c2,
            c3,
            p0,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
        ] = bytes;
        let mut packed = u64::from_le_bytes([p0, p1, p2, p3, p4, p5, p6, p7]);
        let mut field = |width: u32| {
            let value = packed & ((1 << width) - 1);
            packed >>= width;
            value
        };
        let [offset, cap, len, k, w, c, n] = PACKED.map(&mut field);
        let narrow = |v: u64| usize::try_from(v).unwrap_or(usize::MAX);
        let small = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        Self {
            first: (
                u32::from_le_bytes([t0, t1, t2, t3]),
                u32::from_le_bytes([b0, b1, b2, b3]),
            ),
            total: u32::from_le_bytes([c0, c1, c2, c3]),
            offset: narrow(offset),
            cap: narrow(cap),
            len: narrow(len),
            k: small(k),
            w: small(w),
            c: small(c) + 1,
            n: narrow(n),
        }
    }

    /// The slot's code, followed by zeros that let a reader load whole words.
    fn padded(&self, read: Read<'_>) -> Vec<u8> {
        let mut slot = alloc::vec![0u8; self.len + PADDING];
        if let Some(code) = slot.get_mut(..self.len) {
            read(self.offset, code);
        }
        slot
    }

    /// Calls `f` with each posting of `code` (from [`Self::padded`]) in order and the bit its
    /// count starts at, until it returns true.
    fn scan(&self, code: &[u8], mut f: impl FnMut(Posting, usize) -> bool) {
        let mut reader = Reader::new(code);
        let mut target = self.first.0;
        for i in 0..self.n {
            if i > 0 {
                target += reader.rice(self.k);
            }
            let at = reader.position();
            let count = reader.fixed(self.c);
            let block = reader.fixed(self.w);
            if f(
                Posting {
                    target,
                    block,
                    count,
                },
                at,
            ) {
                return;
            }
        }
    }

    /// Calls `f` with the segment's postings in order, until it returns true.
    pub(super) fn walk(&self, read: Read<'_>, mut f: impl FnMut(Posting) -> bool) {
        self.scan(&self.padded(read), |p, _| f(p));
    }

    /// Applies `changes` (sorted, all to postings it holds) by rewriting their counts in place:
    /// the code's changed bytes (from the slot's start) and the entry with its new total, or
    /// `None` when a change names a posting it lacks or a count outgrows its width. A count
    /// that reaches zero stays until the segment is rewritten.
    pub(super) fn patch(
        &self,
        read: Read<'_>,
        changes: &[Change],
    ) -> Option<(usize, Vec<u8>, Self)> {
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
        let limit = (1u64 << self.c) - 1;
        let mut total = i64::from(self.total);
        let (mut low, mut high) = (usize::MAX, 0);
        for (&(at, count), c) in found.iter().zip(changes) {
            let next = u64::try_from(i64::from(count) + c.delta)
                .ok()
                .filter(|&n| n <= limit)?;
            set_bits(&mut code, at, self.c, next);
            total += c.delta;
            low = low.min(at / 8);
            high = high.max((at + usize::try_from(self.c).unwrap_or(32)).div_ceil(8));
        }
        let bytes = code.get(low..high)?.to_vec();
        let total = u32::try_from(total).ok()?;
        Some((low, bytes, Self { total, ..*self }))
    }

    /// The segment's postings, read from its slot.
    pub(super) fn decode(&self, read: Read<'_>) -> Vec<Posting> {
        let mut out = Vec::with_capacity(self.n + 1);
        self.walk(read, |p| {
            out.push(p);
            false
        });
        out
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

/// Segments of [`SEGMENT_TARGET`] postings for `postings`, unless they fit one segment.
pub(super) fn rebuild(postings: &[Posting]) -> Vec<Coded> {
    let per = if postings.len() <= SEGMENT_MAX {
        SEGMENT_MAX
    } else {
        SEGMENT_TARGET
    };
    postings.chunks(per).map(Coded::encode).collect()
}

/// Postings with changes applied; a count that reaches zero drops.
pub(super) fn merge(postings: &[Posting], changes: &[Change]) -> Vec<Posting> {
    let mut out = Vec::with_capacity(postings.len() + changes.len());
    let mut changes = changes.iter().peekable();
    let mut push = |key: (u32, u32), count: i64| {
        let count = u32::try_from(count).unwrap_or_else(|_| corrupt("an index count below zero"));
        if count > 0 {
            out.push(Posting {
                target: key.0,
                block: key.1,
                count,
            });
        }
    };
    for p in postings {
        while let Some(c) = changes.next_if(|c| c.key < p.key()) {
            push(c.key, c.delta);
        }
        let delta = changes.next_if(|c| c.key == p.key()).map_or(0, |c| c.delta);
        push(p.key(), i64::from(p.count) + delta);
    }
    for c in changes {
        push(c.key, c.delta);
    }
    out
}
