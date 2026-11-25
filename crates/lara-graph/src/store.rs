//! The bottom layer: bytes and pages. The memory seam, the page store every structure allocates
//! from, scoped allocation, read buffers and checked units.

pub(crate) mod corrupt;
pub(crate) mod memory;
pub(crate) mod page_store;
pub(crate) mod read_buf;
pub(crate) mod scale;
pub(crate) mod txn;
pub(crate) mod units;
