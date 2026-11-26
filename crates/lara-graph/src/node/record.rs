//! A node's record: where its label bucket list is and where the node is in its life.
//!
//! # Record layout
//!
//! An empty list leaves the extent and count bytes zero; the state bits are set either way.
//!
//! ```text
//! ---------------------------------------- <- Offset 0
//! Label bucket list extent        ↕ 6 bytes
//!   (the extent's two spare bits, bits 6-7 of its last byte, carry the
//!    state: 0 live, 1 deleting, 2 deleted)
//! ----------------------------------------
//! Entry count; 0xFF if tiered     ↕ 1 byte
//! ----------------------------------------
//! Reserved                        ↕ 1 byte
//! ----------------------------------------
//! ```

use crate::id::NodeState;
use crate::node::label_bucket_list::{self, LABEL_BUCKET_LIST_REF_BYTES, LabelBucketList};
use crate::store::corrupt::corrupt;
use crate::store::page_store::{EXTENT_BYTES, EXTENT_SPARE_BITS};

/// Bytes per node record: the list reference, then a reserved byte, so that a record is eight
/// bytes, a line of eight is 64, and a record decodes from one aligned word: with seven, reading
/// a node's buckets cost 2-4 % more (`degree_1024`, `has_edges_1024`, `node_state_1024`).
pub(crate) const RECORD_BYTES: usize = LABEL_BUCKET_LIST_REF_BYTES + 1;
const _: () = assert!(RECORD_BYTES == 8);

/// The top two bits of the label bucket list extent's last byte hold the state: the extent leaves
/// them spare ([`EXTENT_SPARE_BITS`]). An empty list leaves the extent bytes zero, so the bits are free
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
        let [r0, r1, r2, r3, r4, r5, r6] = label_bucket_list::encode_ref(self.list);
        let mut bytes = [r0, r1, r2, r3, r4, r5, r6, 0];
        let bits = match self.state {
            NodeState::Live => 0,
            NodeState::Deleting => DETACHING,
            NodeState::Deleted => DELETED,
        };
        if let Some(flags) = bytes.get_mut(FLAG_AT) {
            *flags |= bits;
        }
        bytes
    }

    pub(crate) fn decode(bytes: [u8; RECORD_BYTES]) -> Self {
        let state = Self::state_of(bytes);
        let [r0, r1, r2, r3, r4, r5, r6, _] = bytes;
        let mut list = [r0, r1, r2, r3, r4, r5, r6];
        if let Some(flags) = list.get_mut(FLAG_AT) {
            *flags &= !STATE_BITS;
        }
        Self {
            list: label_bucket_list::decode_ref(list),
            state,
        }
    }

    /// The state alone, from the state bits: a read of a node's state decodes no extent.
    pub(crate) fn state_of(bytes: [u8; RECORD_BYTES]) -> NodeState {
        match bytes.get(FLAG_AT).map_or(0, |flags| flags & STATE_BITS) {
            0 => NodeState::Live,
            DETACHING => NodeState::Deleting,
            DELETED => NodeState::Deleted,
            _ => corrupt("node state"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip_with_their_state() {
        for state in [NodeState::Live, NodeState::Deleting, NodeState::Deleted] {
            let record = Record { list: None, state };
            assert_eq!(Record::decode(record.encode()), record);
        }
    }
}

/// Proofs over every input, checked by Kani (`cargo kani -p lara-graph`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Any seven bytes with a known state decode to a record that encodes back to itself.
    #[kani::proof]
    fn records_round_trip() {
        let bytes: [u8; RECORD_BYTES] = kani::any();
        let flags = bytes.get(FLAG_AT).copied().unwrap_or(0) & STATE_BITS;
        kani::assume(flags != STATE_BITS);
        let record = Record::decode(bytes);
        assert_eq!(Record::decode(record.encode()), record);
    }
}
