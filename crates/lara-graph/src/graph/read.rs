//! Reads: a node, one node's edges (also sorted by target), edges between two nodes, the targets two
//! nodes share, edge values, and `ReadBatch` over many nodes. Reads stand on the node and edge stores.

mod batch;
mod between;
mod neighbors;
mod nodes;
mod sorted;
mod values;

pub use batch::ReadBatch;
