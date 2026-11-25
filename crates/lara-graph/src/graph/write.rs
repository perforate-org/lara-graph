//! Writes: inserting and removing edges, bulk load, edge values, detaching nodes and compaction.
//! Writes stand on the bucket layer only.

mod bulk;
mod compaction;
mod detach;
mod insert;
mod remove;
mod values;

pub use insert::{EdgeKind, NewEdge};
pub use values::EdgeValue;
