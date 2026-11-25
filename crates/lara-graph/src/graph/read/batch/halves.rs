//! Batched reads over edge halves; consecutive halves in one bucket reuse its lookup.

use super::{Mode, ReadBatch, check_all};
use crate::bucket::descriptor::Descriptor;
use crate::graph::Error;
use crate::id::{EdgeRef, NodeId};
use crate::node::label_bucket_list::LabelBucketKey;
use crate::store::memory::Memory;
use alloc::vec::Vec;

impl<M: Memory> ReadBatch<'_, M> {
    /// Replaces `out` with the value of each live half's edge, as that half's bucket holds it;
    /// `None` for a hole, a position past the end or a missing bucket. Consecutive halves in the
    /// same bucket reuse one lookup.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    ///
    /// [`Graph::node_count`]: crate::Graph::node_count
    pub fn edge_values_each(
        &mut self,
        halves: &[EdgeRef],
        out: &mut Vec<Option<Vec<u8>>>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        check_all(graph, halves.iter().map(EdgeRef::node))?;
        out.clear();
        out.reserve(halves.len());
        for &half in halves {
            let node = half.node();
            let key = LabelBucketKey::of(half);
            let found = self
                .last_found
                .get_or_try(node, key, || graph.find_label_bucket(node, key))?;
            let Some(found) = found else {
                out.push(None);
                continue;
            };
            if graph
                .view()
                .target(&found.bucket.slots, half.position())
                .is_none()
            {
                out.push(None);
                continue;
            }
            out.push(Some(graph.read_value(
                &found,
                half.label(),
                half.position(),
            )));
        }
        Ok(())
    }

    /// Replaces `out` with the target at each half's position: `None` for a hole or a position
    /// past the end. Consecutive halves in the same bucket reuse one descriptor.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    ///
    /// [`Graph::node_count`]: crate::Graph::node_count
    pub fn edge_targets_each(
        &mut self,
        halves: &[EdgeRef],
        out: &mut Vec<Option<NodeId>>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        check_all(graph, halves.iter().map(EdgeRef::node))?;
        out.clear();
        out.reserve(halves.len());
        for &half in halves {
            let node = half.node();
            let key = LabelBucketKey::of(half);
            let descriptor = self.last_bucket.get(node, key).unwrap_or_else(|| {
                let descriptor = self.bucket(node, key);
                self.last_bucket.set(node, key, descriptor);
                descriptor
            });
            let view = graph.view();
            out.push(descriptor.and_then(|descriptor| view.target(&descriptor, half.position())));
        }
        Ok(())
    }

    fn bucket(&mut self, v: NodeId, key: LabelBucketKey) -> Option<Descriptor> {
        let view = self.graph.view();
        let stored = self.stored_at(&view, v, key, Mode::Sparse)?;
        Some(view.slots(stored))
    }

    /// Replaces `out` with the other half of each edge, or `None` when the half is not live.
    /// Consecutive halves in the same bucket reuse one bucket lookup.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    ///
    /// [`Graph::node_count`]: crate::Graph::node_count
    pub fn counterparts_each(
        &mut self,
        halves: &[EdgeRef],
        out: &mut Vec<Option<EdgeRef>>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        check_all(graph, halves.iter().map(EdgeRef::node))?;
        out.clear();
        out.reserve(halves.len());
        for &half in halves {
            let node = half.node();
            let key = LabelBucketKey::of(half);
            let own = self
                .last_found
                .get_or_try(node, key, || graph.find_label_bucket(node, key))?;
            let Some(own) = own else {
                out.push(None);
                continue;
            };
            let Some(other) = graph.view().target(&own.bucket.slots, half.position()) else {
                out.push(None);
                continue;
            };
            let key = LabelBucketKey::of(half).opposite();
            let theirs = self
                .last_theirs
                .get_or_else(other, key, || graph.present(other, key));
            out.push(Some(
                graph.pair_with_theirs(half, own, other, key, theirs).other,
            ));
        }
        Ok(())
    }
}
