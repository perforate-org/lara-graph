//! A node's record: where its label bucket list is and where the node is in its life.
//!
//! # Record layout
//!
//! An empty list leaves the extent and count bytes zero; the state bits are set either way.
//!
//! ```text
//! ---------------------------------------- <- Offset 0
//! Label bucket list extent        ↕ 5 bytes
//!   (the extent's two spare bits, bits 6-7 of its last byte, carry the
//!    state: 0 live, 1 detaching, 2 deleted)
//! ----------------------------------------
//! Entry count; 0xFF if tiered     ↕ 1 byte
//! ----------------------------------------
//! ```

use crate::id::NodeState;
use crate::node::label_bucket_list::{self, LABEL_BUCKET_LIST_REF_BYTES, LabelBucketList};
use crate::store::corrupt::corrupt;
use crate::store::page_store::{EXTENT_BYTES, EXTENT_SPARE_BITS};

/// Bytes per node record.
pub(crate) const RECORD_BYTES: usize = LABEL_BUCKET_LIST_REF_BYTES;

/// The top two bits of the label bucket list extent's class byte (its last) hold the state: classes
/// are log2 values of at most 12. An empty list leaves the extent bytes zero, so the bits are free
/// there too.
const STATE_BITS: u8 = 0b1100_0000;
const DETACHING: u8 = 0b0100_0000;
const DELETED: u8 = 0b1000_0000;
const FLAG_AT: usize = EXTENT_BYTES - 1;
const _: () = assert!(STATE_BITS == EXTENT_SPARE_BITS);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Record {
    pub(crate) list: Option<LabelBucketList>,
    pub(crate) state: NodeState,
}

impl Record {
    pub(crate) const NEW: Self = Self {
        list: None,
        state: NodeState::Live,
    };

    /// As the module's Record layout diagram shows.
    pub(crate) fn encode(self) -> [u8; RECORD_BYTES] {
        let mut bytes = label_bucket_list::encode_ref(self.list);
        let bits = match self.state {
            NodeState::Live => 0,
            NodeState::Detaching => DETACHING,
            NodeState::Deleted => DELETED,
        };
        if let Some(flags) = bytes.get_mut(FLAG_AT) {
            *flags |= bits;
        }
        bytes
    }

    pub(crate) fn decode(mut bytes: [u8; RECORD_BYTES]) -> Self {
        let mut state = NodeState::Live;
        if let Some(flags) = bytes.get_mut(FLAG_AT) {
            state = match *flags & STATE_BITS {
                0 => NodeState::Live,
                DETACHING => NodeState::Detaching,
                DELETED => NodeState::Deleted,
                _ => corrupt("node state"),
            };
            *flags &= !STATE_BITS;
        }
        Self {
            list: label_bucket_list::decode_ref(bytes),
            state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip_with_their_state() {
        for state in [NodeState::Live, NodeState::Detaching, NodeState::Deleted] {
            let record = Record { list: None, state };
            assert_eq!(Record::decode(record.encode()), record);
        }
    }
}

/// Proofs over every input, checked by Kani (`cargo kani -p lara-graph`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Any six bytes with a known state decode to a record that encodes back to itself.
    #[kani::proof]
    fn records_round_trip() {
        let bytes: [u8; RECORD_BYTES] = kani::any();
        let flags = bytes.get(FLAG_AT).copied().unwrap_or(0) & STATE_BITS;
        kani::assume(flags != STATE_BITS);
        let record = Record::decode(bytes);
        assert_eq!(Record::decode(record.encode()), record);
    }
}
