//! Segments: runs of sorted postings coded together, and the table entries that describe them.

use super::{ENTRY_BYTES, Read};
use crate::bucket::target_index::Change;
use crate::bucket::target_index::Posting;
use crate::bucket::target_index::bits::{PADDING, Reader, Writer, rice_len};
use crate::store::corrupt::corrupt;
use alloc::vec::Vec;

mod patch;

pub(super) use patch::{Lowered, Patched, lower, lower_one, raise};

/// Postings a segment is cut to when it is rebuilt. Against 16, the index takes 16 % fewer pages
/// for distinct targets and 8 % fewer for random ones, for a removal from the leaf's side 7 %
/// dearer; 32 saved 17 % more space but made pairing and detaching a satellite 16-23 % dearer.
const SEGMENT_TARGET: usize = 24;
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
    /// Bits of each count: a count changes in place while it fits. None when every count is
    /// one.
    pub(super) c: u32,
    pub(super) head: Head,
    pub(super) n: usize,
    /// The length of the code in bits.
    pub(super) bits: usize,
    pub(super) bytes: Vec<u8>,
}

/// The first posting's count in a segment without count bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Head {
    /// As the count bits say: one when there are none.
    Counted,
    /// Zero: it stays as the others' block base until the segment is rewritten.
    Zero,
}

/// Bits a count field takes for counts up to `max`.
fn count_bits(max: u32) -> u32 {
    bits_of(max).max(1)
}

/// Bits that hold `value`: none for zero.
const fn bits_of(value: u32) -> u32 {
    32 - value.leading_zeros()
}

/// A block as a segment codes it: its offset from the segment's first block, zigzagged so that a
/// small offset either way is a small number. Blocks of targets inserted about together are near
/// each other, so these take a few bits where the blocks themselves take many.
fn block_code(block: u32, base: u32) -> u32 {
    let offset = i64::from(block) - i64::from(base);
    let zigzag = if offset >= 0 {
        2 * offset
    } else {
        -2 * offset - 1
    };
    u32::try_from(zigzag).unwrap_or_else(|_| corrupt("an index block offset"))
}

/// The block that `code` stands for in a segment whose first block is `base`.
fn block_of(code: u32, base: u32) -> u32 {
    let half = i64::from(code >> 1);
    let offset = if code & 1 == 0 { half } else { -half - 1 };
    u32::try_from(i64::from(base) + offset).unwrap_or_else(|_| corrupt("an index block"))
}

impl Coded {
    /// Codes sorted postings (at most [`SEGMENT_MAX`]): each target gap after the first in
    /// Rice(`k`), each count in `c` bits, each block's code ([`block_code`]) in `w` bits.
    fn encode(postings: &[Posting]) -> Self {
        let base = postings.first().map_or(0, |p| p.block);
        let widest = postings.iter().map(|p| block_code(p.block, base)).max();
        let w = bits_of(widest.unwrap_or(0));
        let k = best_k(postings);
        // Counts of one, the usual case, take no bits.
        let c = match postings.iter().map(|p| p.count).max().unwrap_or(1) {
            0 | 1 => 0,
            most => bits_of(most),
        };
        Self::code(postings, (k, w, c))
    }

    /// Codes sorted postings with the parameters given; each count and block must fit its width.
    #[inline]
    fn code(postings: &[Posting], (k, w, c): (u32, u32, u32)) -> Self {
        let Some(first) = postings.first().map(Posting::key) else {
            corrupt("an empty index segment")
        };
        let mut out = Writer::with_capacity(postings.len() * 4);
        let mut previous = first.0;
        for (i, p) in postings.iter().enumerate() {
            if i > 0 {
                out.rice(p.target - previous, k);
            }
            previous = p.target;
            out.fixed(p.count, c);
            out.fixed(block_code(p.block, first.1), w);
        }
        Self {
            first,
            total: postings.iter().map(|p| p.count).sum(),
            k,
            w,
            c,
            head: Head::Counted,
            n: postings.len(),
            bits: out.bits(),
            bytes: out.into_bytes(),
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
/// its code in bits, and its coding parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Entry {
    pub(super) first: (u32, u32),
    pub(super) total: u32,
    pub(super) offset: usize,
    pub(super) cap: usize,
    bits: usize,
    k: u32,
    w: u32,
    c: u32,
    head: Head,
    n: usize,
}

/// Bit widths of the packed fields after an entry's total: offset and capacity (12 bits each, as
/// a leaf is at most 4096 bytes), the code's length in bits (15), `k` (5), `w` (6), the count
/// width (6), `n` (7), and whether the first posting is at zero without count bits (1).
const PACKED: [u32; 8] = [12, 12, 15, 5, 6, 6, 7, 1];

impl Entry {
    /// The segment this entry describes, with the code read from its slot.
    pub(super) fn kept(&self, read: Read<'_>) -> Coded {
        let mut bytes = alloc::vec![0u8; self.len()];
        read(self.offset, &mut bytes);
        Coded {
            first: self.first,
            total: self.total,
            k: self.k,
            w: self.w,
            c: self.c,
            head: self.head,
            n: self.n,
            bits: self.bits,
            bytes,
        }
    }

    /// The entry of `c` placed in the slot `(offset, cap)`.
    pub(super) fn placed(c: &Coded, (offset, cap): (usize, usize)) -> Self {
        Self {
            first: c.first,
            total: c.total,
            offset,
            cap,
            bits: c.bits,
            k: c.k,
            w: c.w,
            c: c.c,
            head: c.head,
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
            wide(self.bits),
            u64::from(self.k),
            u64::from(self.w),
            u64::from(self.c),
            wide(self.n),
            u64::from(self.head == Head::Zero),
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
        let [offset, cap, bits, k, w, c, n, zero] = PACKED.map(&mut field);
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
            bits: narrow(bits),
            k: small(k),
            w: small(w),
            c: small(c),
            head: if zero == 1 { Head::Zero } else { Head::Counted },
            n: narrow(n),
        }
    }

    /// Bytes of the code.
    const fn len(&self) -> usize {
        self.bits.div_ceil(8)
    }

    /// The slot's code, followed by zeros that let a reader load whole words.
    fn padded(&self, read: Read<'_>) -> Vec<u8> {
        let mut slot = alloc::vec![0u8; self.len() + PADDING];
        if let Some(code) = slot.get_mut(..self.len()) {
            read(self.offset, code);
        }
        slot
    }

    /// Calls `f` with each posting of `code` (from [`Self::padded`]) in order and the bit its
    /// count starts at, until it returns true.
    #[inline]
    fn scan(&self, code: &[u8], mut f: impl FnMut(Posting, usize) -> bool) {
        let mut reader = Reader::new(code);
        let mut target = self.first.0;
        for i in 0..self.n {
            if i > 0 {
                target += reader.rice(self.k);
            }
            let at = reader.position();
            let count = self.count(i, reader.fixed(self.c));
            let block = block_of(reader.fixed(self.w), self.first.1);
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

    /// The count of posting `i`, whose count bits read `bits`: without count bits, one, or zero
    /// for a first posting marked so.
    #[inline]
    const fn count(&self, i: usize, bits: u32) -> u32 {
        match (self.c, self.head) {
            (0, Head::Zero) if i == 0 => 0,
            (0, _) => 1,
            _ => bits,
        }
    }

    /// Calls `f` with the segment's postings in order, until it returns true.
    #[inline]
    pub(super) fn walk(&self, read: Read<'_>, mut f: impl FnMut(Posting) -> bool) {
        self.scan(&self.padded(read), |p, _| f(p));
    }

    /// Checks that the recorded length is where the last posting ends (test checks).
    #[cfg(test)]
    pub(super) fn check_length(&self, read: Read<'_>) {
        let fields = usize::try_from(self.c + self.w).unwrap();
        let mut end = 0;
        self.scan(&self.padded(read), |_, at| {
            end = at + fields;
            false
        });
        assert_eq!(end, self.bits, "a segment's length in bits");
    }

    /// The segment's postings, read from its slot.
    #[inline]
    pub(super) fn decode(&self, read: Read<'_>) -> Vec<Posting> {
        let mut out = Vec::with_capacity(self.n + 1);
        self.walk(read, |p| {
            out.push(p);
            false
        });
        out
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
