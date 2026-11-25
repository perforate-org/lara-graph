//! One label bucket's storage: its descriptor, its slots in a run or pages, its value plane, its
//! target index, the header that ties them, and compaction. Knows nothing of nodes or graphs.

pub(crate) mod compact;
pub(crate) mod descriptor;
pub(crate) mod label_bucket;
pub(crate) mod paged;
pub(crate) mod stored;
pub(crate) mod targets;
pub(crate) mod values;

use crate::id::Width;

/// Pages one half can take from the edge store: its slots, a header, the target index's change,
/// and for a label with a width, a run replacing a tiny bucket and the value's write.
pub(crate) fn pages_per_half(width: Option<Width>) -> u32 {
    label_bucket::PAGES_PER_INSERT
        + 1
        + targets::PAGES_PER_CHANGE
        + width.map_or(0, |_| 1 + values::PAGES_PER_WRITE)
}
