//! The graph's bucket layer: one label bucket of one node at a time. Finding and publishing it,
//! its values, its target index, the width table, and pairing an edge's two halves. Reads and
//! writes stand on it; it stands on the storage layers below the graph.

mod halves;
mod index;
mod locate;
mod values;
mod widths;

pub(in crate::graph) use halves::Pair;
pub(in crate::graph) use index::{Batch, Between};
pub(in crate::graph) use locate::{Found, LastBucket};
pub(in crate::graph) use values::visit_joined;
pub(in crate::graph) use widths::{AT_WIDTHS, Widths};
