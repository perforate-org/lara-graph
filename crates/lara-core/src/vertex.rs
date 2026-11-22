//! A vertex's record: where its bucket list is and whether it is being detached.

use crate::labels::{self, BucketList, LIST_REF_BYTES};
use crate::page_store::{ByteClass, EXTENT_BYTES};

/// Bytes per vertex record.
pub(crate) const RECORD_BYTES: usize = LIST_REF_BYTES;

/// The top bit of the list extent's class byte (its last) marks a detaching vertex: classes are
/// log2 values of at most 12. An empty list leaves the extent bytes zero, so the bit is free
/// there too.
const DETACHING: u8 = 0x80;
const FLAG_AT: usize = EXTENT_BYTES - 1;
const _: () = assert!(ByteClass::PAGE.log2() < DETACHING);

/// Whether a vertex takes new edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum State {
    Attached,
    /// Its edges are being removed step by step; no edge may be added to it.
    Detaching,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Record {
    pub(crate) list: Option<BucketList>,
    pub(crate) state: State,
}

impl Record {
    pub(crate) const NEW: Self = Self {
        list: None,
        state: State::Attached,
    };

    /// Layout: the list reference, with [`DETACHING`] set in its class byte.
    pub(crate) fn encode(self) -> [u8; RECORD_BYTES] {
        let mut bytes = labels::encode_ref(self.list);
        if let (State::Detaching, Some(flags)) = (self.state, bytes.get_mut(FLAG_AT)) {
            *flags |= DETACHING;
        }
        bytes
    }

    pub(crate) fn decode(mut bytes: [u8; RECORD_BYTES]) -> Self {
        let mut state = State::Attached;
        if let Some(flags) = bytes.get_mut(FLAG_AT).filter(|f| **f & DETACHING != 0) {
            *flags &= !DETACHING;
            state = State::Detaching;
        }
        Self {
            list: labels::decode_ref(bytes),
            state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip_with_their_state() {
        for state in [State::Attached, State::Detaching] {
            let record = Record { list: None, state };
            assert_eq!(Record::decode(record.encode()), record);
        }
    }
}
