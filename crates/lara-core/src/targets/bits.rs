//! Bit streams for packed index segments: Rice codes and fixed widths. Bits go least
//! significant first. Both ends work a 64-bit word at a time: the writer flushes whole bytes
//! from an accumulator, the reader loads eight bytes at its bit position, so a reader's input
//! must be followed by [`PADDING`] bytes.

use alloc::vec::Vec;

/// Bytes of zeros a reader needs past its input.
pub(super) const PADDING: usize = 8;

/// Appends bits.
pub(super) struct Writer {
    bytes: Vec<u8>,
    word: u64,
    used: u32,
}

impl Writer {
    pub(super) fn with_capacity(bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes),
            word: 0,
            used: 0,
        }
    }

    /// The low `width` bits of `value` (`width` at most 32).
    pub(super) fn fixed(&mut self, value: u32, width: u32) {
        let masked = u64::from(value) & ((1u64 << width) - 1);
        self.word |= masked << self.used;
        self.used += width;
        if self.used >= 32 {
            let [b0, b1, b2, b3, ..] = self.word.to_le_bytes();
            self.bytes.extend_from_slice(&[b0, b1, b2, b3]);
            self.word >>= 32;
            self.used -= 32;
        }
    }

    /// Rice code with parameter `k`: the quotient in unary (ones, then a zero), the remainder in
    /// `k` bits.
    pub(super) fn rice(&mut self, value: u32, k: u32) {
        let mut q = value >> k;
        while q >= 31 {
            self.fixed(u32::MAX >> 1, 31);
            q -= 31;
        }
        self.fixed((1 << q) - 1, q + 1);
        self.fixed(value, k);
    }

    /// The bytes written, the last one padded with zeros.
    pub(super) fn into_bytes(mut self) -> Vec<u8> {
        let tail = usize::try_from(self.used.div_ceil(8)).unwrap_or(0);
        self.bytes
            .extend_from_slice(self.word.to_le_bytes().get(..tail).unwrap_or_default());
        self.bytes
    }
}

/// Bits a Rice code of `value` takes.
pub(super) fn rice_len(value: u32, k: u32) -> usize {
    usize::try_from((value >> k) + 1 + k).unwrap_or(usize::MAX)
}

/// Reads bits written by [`Writer`] from input followed by [`PADDING`] bytes.
pub(super) struct Reader<'a> {
    bytes: &'a [u8],
    /// The next bit to read.
    at: usize,
}

impl<'a> Reader<'a> {
    pub(super) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    /// At least 57 bits from the current position.
    fn window(&self) -> u64 {
        let word = self
            .bytes
            .get(self.at / 8..)
            .and_then(|b| b.first_chunk::<8>())
            .map_or(0, |&b| u64::from_le_bytes(b));
        word >> (self.at % 8)
    }

    fn skip(&mut self, bits: u32) {
        self.at += usize::try_from(bits).unwrap_or(0);
    }

    /// `width` bits (at most 32).
    pub(super) fn fixed(&mut self, width: u32) -> u32 {
        let value = self.window() & ((1u64 << width) - 1);
        self.skip(width);
        u32::try_from(value).unwrap_or(0)
    }

    /// The next bit to read.
    pub(super) const fn position(&self) -> usize {
        self.at
    }

    pub(super) fn rice(&mut self, k: u32) -> u32 {
        let mut q = 0u32;
        loop {
            let ones = self.window().trailing_ones().min(32);
            q += ones;
            self.skip(ones);
            if ones < 32 {
                break;
            }
        }
        // The terminating zero.
        self.skip(1);
        q << k | self.fixed(k)
    }
}
