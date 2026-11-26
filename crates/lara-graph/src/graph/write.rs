//! Writes: adding nodes, inserting and removing edges, bulk load, edge values, deleting nodes and compaction.
//! Writes stand on the node and edge stores.

mod bulk;
mod compact;
mod detach;
mod insert;
mod nodes;
mod remove;
mod values;

pub use detach::DeletingSet;
