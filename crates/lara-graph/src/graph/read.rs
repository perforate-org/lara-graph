//! Reads: one node's edges, edges between two nodes, edge values, and `ReadBatch` over many
//! nodes. Reads stand on the bucket layer only.

mod batch;
mod between;
mod neighbors;
mod values;

pub use batch::ReadBatch;
