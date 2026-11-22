//! External boundary tags (Bonwick & Adams 2001): one fixed-size record per segment, kept
//! outside the slab and linked in address order, so coalescing finds both neighbours in O(1).
//!
//! This module is the only place that knows the wire layout; sentinel values exist only here.

use crate::units::{Slot, SlotRange, SpanLen};

/// One tag record. Records are recycled, never moved, so an id stays valid while its segment
/// lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TagId(u32);

impl TagId {
    pub(crate) const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub(crate) const fn get(self) -> u32 {
        self.0
    }
}

/// A contiguous run of slab slots, allocated or free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Segment {
    pub(super) start: Slot,
    pub(super) len: SpanLen,
    /// Address-order neighbours.
    pub(super) before: Option<TagId>,
    pub(super) after: Option<TagId>,
    pub(super) use_: SegmentUse,
}

/// Whether a segment is owned, and if free, its free-list links.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SegmentUse {
    Allocated,
    Free {
        prev: Option<TagId>,
        next: Option<TagId>,
    },
}

/// A decoded record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Record {
    Segment(Segment),
    /// An unused record waiting for reuse, linked to the next one.
    Recycled {
        next: Option<TagId>,
    },
}

impl Segment {
    pub(super) fn range(self) -> SlotRange {
        // A segment's end is checked when it is created (allocation advances a checked frontier,
        // merges join lengths with `SpanLen::join`).
        SlotRange::new(self.start, self.len.count()).unwrap_or_else(|_| corrupt("segment end"))
    }

    pub(super) fn is_free(self) -> bool {
        matches!(self.use_, SegmentUse::Free { .. })
    }
}

/// Bytes per record.
pub(super) const RECORD_BYTES: u64 = 32;

const NONE: u32 = u32::MAX;
const STATE_ALLOCATED: u32 = 1;
const STATE_FREE: u32 = 2;
const STATE_RECYCLED: u32 = 3;

/// Encodes an optional link; `u32::MAX` means none.
pub(super) fn encode_link(link: Option<TagId>) -> [u8; 4] {
    link.map_or(NONE, TagId::get).to_le_bytes()
}

pub(super) fn decode_link(bytes: [u8; 4]) -> Option<TagId> {
    let raw = u32::from_le_bytes(bytes);
    (raw != NONE).then_some(TagId(raw))
}

impl Record {
    // Layout: start u64 | len u32 | state u32 | before u32 | after u32 | prev u32 | next u32.
    pub(super) fn encode(self) -> [u8; 32] {
        let (start, len, state, before, after, prev, next) = match self {
            Self::Recycled { next } => (0, 0, STATE_RECYCLED, None, None, None, next),
            Self::Segment(s) => {
                let (state, prev, next) = match s.use_ {
                    SegmentUse::Allocated => (STATE_ALLOCATED, None, None),
                    SegmentUse::Free { prev, next } => (STATE_FREE, prev, next),
                };
                (
                    s.start.get(),
                    s.len.get(),
                    state,
                    s.before,
                    s.after,
                    prev,
                    next,
                )
            }
        };
        let bytes = start
            .to_le_bytes()
            .into_iter()
            .chain(len.to_le_bytes())
            .chain(state.to_le_bytes())
            .chain(encode_link(before))
            .chain(encode_link(after))
            .chain(encode_link(prev))
            .chain(encode_link(next));
        let mut out = [0u8; 32];
        for (dst, src) in out.iter_mut().zip(bytes) {
            *dst = src;
        }
        out
    }

    pub(super) fn decode(bytes: [u8; 32]) -> Self {
        let [s0, s1, s2, s3, s4, s5, s6, s7, rest @ ..] = bytes;
        let start = Slot::new(u64::from_le_bytes([s0, s1, s2, s3, s4, s5, s6, s7]));
        let [
            l0,
            l1,
            l2,
            l3,
            t0,
            t1,
            t2,
            t3,
            b0,
            b1,
            b2,
            b3,
            a0,
            a1,
            a2,
            a3,
            rest @ ..,
        ] = rest;
        let [p0, p1, p2, p3, n0, n1, n2, n3] = rest;
        let next = decode_link([n0, n1, n2, n3]);
        let use_ = match u32::from_le_bytes([t0, t1, t2, t3]) {
            STATE_RECYCLED => return Self::Recycled { next },
            STATE_ALLOCATED => SegmentUse::Allocated,
            STATE_FREE => SegmentUse::Free {
                prev: decode_link([p0, p1, p2, p3]),
                next,
            },
            _ => corrupt("tag state"),
        };
        let len = u32::from_le_bytes([l0, l1, l2, l3]);
        Self::Segment(Segment {
            start,
            len: SpanLen::new(len).unwrap_or_else(|| corrupt("zero-length segment")),
            before: decode_link([b0, b1, b2, b3]),
            after: decode_link([a0, a1, a2, a3]),
            use_,
        })
    }
}

/// The allocator's own metadata is inconsistent. Not recoverable: stop rather than corrupt the
/// slab further.
#[cold]
pub(super) fn corrupt(what: &str) -> ! {
    panic!("span allocator metadata is corrupt: {what}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip() {
        let segment = Segment {
            start: Slot::new(1 << 40),
            len: SpanLen::new(77).unwrap(),
            before: None,
            after: Some(TagId(3)),
            use_: SegmentUse::Free {
                prev: Some(TagId(9)),
                next: None,
            },
        };
        for record in [
            Record::Segment(segment),
            Record::Segment(Segment {
                use_: SegmentUse::Allocated,
                ..segment
            }),
            Record::Recycled {
                next: Some(TagId(0)),
            },
            Record::Recycled { next: None },
        ] {
            assert_eq!(Record::decode(record.encode()), record);
        }
    }

    #[test]
    #[should_panic(expected = "tag state")]
    fn unknown_state_is_corruption() {
        Record::decode([0; 32]);
    }
}
