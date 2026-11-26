//! Compacting one of a node's buckets: the edge store's step, then the bucket's list entry
//! published, or removed once no position is left.

use crate::bucket::LabelBucketKey;
use crate::edge::Compacted;
use crate::error::Error;
use crate::graph::Graph;
use crate::graph::GraphMemories;
use crate::id::{Adjacency, NodeId};
use crate::node::label_bucket_list;
use crate::outcome::Progress;
use crate::outcome::{Compaction, Step};
use crate::store::corrupt::corrupt;
use alloc::vec::Vec;

impl<S: GraphMemories> Graph<S> {
    /// Reclaims removed edges' positions in `v`'s `adjacency` bucket, one bounded step at
    /// a time: at most one page of edges moves. Live edges keep their relative order, so the halves
    /// stay paired; each one that moves is reported. Once the slots need nothing, a paged bucket's
    /// target index merges sparse leaves, a pair per step, moving no edge. Call again while
    /// [`Compaction::progress`] is [`Progress::Remaining`].
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], or [`Error::OutOfMemory`] when a smaller run, moved values or the
    /// target index's changes cannot be allocated. On error the graph is unchanged.
    pub fn compact(&mut self, v: NodeId, adjacency: Adjacency) -> Result<Compaction, Error> {
        let (label, direction) = (adjacency.label(), adjacency.direction());
        let Some(found) = self.find_label_bucket(v, LabelBucketKey { label, direction })? else {
            return Ok(Step::new(Vec::new(), Progress::Done));
        };
        let (compacted, moves, progress) = self.edges.compact(found.bucket)?;
        match compacted {
            Compacted::Unchanged => {}
            Compacted::Kept(bucket) => {
                self.publish(found.entry, bucket);
            }
            Compacted::Emptied => {
                let record = self.record(v)?;
                let list = record
                    .list
                    .unwrap_or_else(|| corrupt("present bucket without a list"));
                let list =
                    label_bucket_list::remove(&mut self.label_bucket_lists, list, found.entry);
                self.set_label_bucket_list(v, record, list);
            }
        }
        Ok(Step::new(moves, progress))
    }
}
