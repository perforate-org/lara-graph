//! One label bucket's storage: its descriptor, its slots in a run or pages, its value plane, its
//! target index, the header that ties them, and compaction. Knows nothing of nodes or graphs.

pub(crate) mod compact;
pub(crate) mod descriptor;
pub(crate) mod label_bucket;
pub(crate) mod paged;
pub(crate) mod stored;
pub(crate) mod targets;
pub(crate) mod values;
