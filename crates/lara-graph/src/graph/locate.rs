//! Finding one of a node's label buckets through its label bucket list, the edge's other half
//! with its bucket, publishing a bucket's next state in its list entry, and the one-bucket caches
//! batches keep of the last one found.

use crate::bucket::LabelBucket;
use crate::bucket::LabelBucketKey;
use crate::bucket::slots;
use crate::error::Error;
use crate::graph::Graph;
use crate::id::EdgeRef;
use crate::id::NodeId;
use crate::node::label_bucket_list::{self, EntryAt, Lookup};
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;

impl<M: Memory> Graph<M> {
    /// Publishes a bucket's new state: its header, and its list entry when it is plain or just
    /// changed form. Returns the bucket as published. Within reserved memory.
    pub(in crate::graph) fn publish(&mut self, entry: EntryAt, bucket: LabelBucket) -> LabelBucket {
        let had_header = bucket.header.is_some();
        let bucket = self.edges.seal(bucket);
        if !had_header || bucket.header.is_none() {
            label_bucket_list::set(&self.label_bucket_lists, entry, &bucket.stored());
        }
        bucket
    }

    /// The bucket a live half proves exists.
    pub(in crate::graph) fn present(&self, v: NodeId, key: LabelBucketKey) -> Found {
        self.find_label_bucket(v, key)
            .ok()
            .flatten()
            .unwrap_or_else(|| corrupt("an edge half without its bucket"))
    }

    pub(in crate::graph) fn find_label_bucket(
        &self,
        v: NodeId,
        key: LabelBucketKey,
    ) -> Result<Option<Found>, Error> {
        let list = self.label_bucket_list(v)?;
        Ok(
            match label_bucket_list::lookup(&self.label_bucket_lists.pages(), list, key) {
                Lookup::Present { entry, stored } => Some(Found {
                    entry,
                    bucket: LabelBucket::resolve(&self.edge_view().pages(), stored),
                }),
                Lookup::Absent { .. } => None,
            },
        )
    }

    /// `v`'s bucket `key` for a read: without the list entry's position, which only an update
    /// needs.
    pub(in crate::graph) fn read_label_bucket(
        &self,
        v: NodeId,
        key: LabelBucketKey,
    ) -> Result<Option<LabelBucket>, Error> {
        let list = self.label_bucket_list(v)?;
        Ok(
            label_bucket_list::lookup_stored(&self.label_bucket_lists.pages(), list, key)
                .map(|stored| LabelBucket::resolve(&self.edge_view().pages(), stored)),
        )
    }
}

/// The bucket last looked up for one `(node, key)`, so consecutive halves in that bucket reuse
/// the lookup.
#[derive(Clone, Copy)]
pub(in crate::graph) struct LastBucket<T>(Option<(NodeId, LabelBucketKey, T)>);

impl<T> Default for LastBucket<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T: Copy> LastBucket<T> {
    /// The bucket held for `(node, key)`, if that is the one held.
    #[inline]
    pub(in crate::graph) fn get(&self, node: NodeId, key: LabelBucketKey) -> Option<T> {
        match self.0 {
            Some((last_node, last_key, found)) if last_node == node && last_key == key => {
                Some(found)
            }
            _ => None,
        }
    }

    /// The bucket held for `(node, key)`, else `find`'s, which is then held.
    #[inline]
    pub(in crate::graph) fn get_or_try<E>(
        &mut self,
        node: NodeId,
        key: LabelBucketKey,
        find: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        if let Some(found) = self.get(node, key) {
            return Ok(found);
        }
        let found = find()?;
        self.set(node, key, found);
        Ok(found)
    }

    /// The bucket held for `(node, key)`, else `find`'s, which is then held.
    #[inline]
    pub(in crate::graph) fn get_or_else(
        &mut self,
        node: NodeId,
        key: LabelBucketKey,
        find: impl FnOnce() -> T,
    ) -> T {
        if let Some(found) = self.get(node, key) {
            return found;
        }
        let found = find();
        self.set(node, key, found);
        found
    }

    #[inline]
    pub(in crate::graph) fn set(&mut self, node: NodeId, key: LabelBucketKey, found: T) {
        self.0 = Some((node, key, found));
    }

    pub(in crate::graph) fn take(&mut self) -> Option<T> {
        self.0.take().map(|(_, _, found)| found)
    }

    /// Takes the held bucket when it is not `(node, key)`'s.
    pub(in crate::graph) fn take_other(&mut self, node: NodeId, key: LabelBucketKey) -> Option<T> {
        if self
            .0
            .is_some_and(|(last_node, last_key, _)| last_node != node || last_key != key)
        {
            self.take()
        } else {
            None
        }
    }
}

impl<M: Memory> Graph<M> {
    /// Both halves of the edge `half` belongs to, with their buckets; `None` when `half` is not
    /// live.
    pub(in crate::graph) fn pair(&self, half: EdgeRef) -> Result<Option<Pair>, Error> {
        let Some(own) = self.find_label_bucket(half.node(), LabelBucketKey::of(half))? else {
            return Ok(None);
        };
        Ok(self.pair_from(half, own))
    }

    /// [`Self::pair`] with `half`'s own bucket already found.
    pub(in crate::graph) fn pair_from(&self, half: EdgeRef, own: Found) -> Option<Pair> {
        let view = self.edge_view().pages();
        let other = slots::get(&own.bucket.slots, half.position(), &view)?;
        let key = LabelBucketKey::of(half).opposite();
        let theirs = self.present(other, key);
        Some(self.pair_with_theirs(half, own, other, key, theirs))
    }
}

/// A label bucket found in its node's label bucket list.
#[derive(Clone, Copy)]
pub(in crate::graph) struct Found {
    pub(in crate::graph) entry: EntryAt,
    pub(in crate::graph) bucket: LabelBucket,
}

/// Both halves of one edge: the given half's bucket, the other half, and its bucket.
pub(in crate::graph) struct Pair {
    pub(in crate::graph) own: Found,
    pub(in crate::graph) other: EdgeRef,
    pub(in crate::graph) theirs: Found,
}

impl<M: Memory> Graph<M> {
    /// [`Self::pair_from`] with the target and its bucket already found.
    pub(in crate::graph) fn pair_with_theirs(
        &self,
        half: EdgeRef,
        own: Found,
        other: NodeId,
        key: LabelBucketKey,
        theirs: Found,
    ) -> Pair {
        let other = self
            .edge_view()
            .other_half(half, &own.bucket, (other, key), &theirs.bucket);
        Pair { own, other, theirs }
    }
}
