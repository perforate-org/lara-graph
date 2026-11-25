//! Per-node structures over the bucket layer: a node's record, its label bucket list, its labels
//! and each label's nodes, and the set of nodes being detached.

pub(crate) mod detaching_set;
pub(crate) mod label_bucket_list;
pub(crate) mod label_postings;
pub(crate) mod node_labels;
pub(crate) mod record;
