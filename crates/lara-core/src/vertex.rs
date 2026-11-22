//! A vertex's record: where its bucket list is and where the vertex is in its life.

use crate::corrupt::corrupt;
use crate::id::VertexState;
use crate::labels::{self, BucketList, LIST_REF_BYTES};
use crate::page_store::{ByteClass, EXTENT_BYTES};

/// Bytes per vertex record.
pub(crate) const RECORD_BYTES: usize = LIST_REF_BYTES;

/// The top two bits of the list extent's class byte (its last) hold the state: classes are log2
/// values of at most 12. An empty list leaves the extent bytes zero, so the bits are free there
/// too.
const STATE_BITS: u8 = 0b1100_0000;
const DETACHING: u8 = 0b0100_0000;
const DELETED: u8 = 0b1000_0000;
const FLAG_AT: usize = EXTENT_BYTES - 1;
const _: () = assert!(ByteClass::PAGE.log2() & STATE_BITS == 0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Record {
    pub(crate) list: Option<BucketList>,
    pub(crate) state: VertexState,
}

impl Record {
    pub(crate) const NEW: Self = Self {
        list: None,
        state: VertexState::Live,
    };

    /// Layout: the list reference, with the state in its class byte's top bits.
    pub(crate) fn encode(self) -> [u8; RECORD_BYTES] {
        let mut bytes = labels::encode_ref(self.list);
        let bits = match self.state {
            VertexState::Live => 0,
            VertexState::Detaching => DETACHING,
            VertexState::Deleted => DELETED,
        };
        if let Some(flags) = bytes.get_mut(FLAG_AT) {
            *flags |= bits;
        }
        bytes
    }

    pub(crate) fn decode(mut bytes: [u8; RECORD_BYTES]) -> Self {
        let mut state = VertexState::Live;
        if let Some(flags) = bytes.get_mut(FLAG_AT) {
            state = match *flags & STATE_BITS {
                0 => VertexState::Live,
                DETACHING => VertexState::Detaching,
                DELETED => VertexState::Deleted,
                _ => corrupt("vertex state"),
            };
            *flags &= !STATE_BITS;
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
        for state in [
            VertexState::Live,
            VertexState::Detaching,
            VertexState::Deleted,
        ] {
            let record = Record { list: None, state };
            assert_eq!(Record::decode(record.encode()), record);
        }
    }
}
