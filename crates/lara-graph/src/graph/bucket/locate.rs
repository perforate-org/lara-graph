//! Finding one of a node's label buckets and publishing its next state.

use crate::bucket::stored::{HEADER_CLASS, LabelBucket};
use crate::graph::{Error, Graph, reserved};
use crate::id::NodeId;
use crate::node::label_bucket_list::{self, EntryAt, LabelBucketKey, Lookup};
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;

impl<M: Memory> Graph<M> {
    /// Publishes a bucket's new state: its header, and its list entry when it is plain or just
    /// changed form. Returns the bucket as published. Within reserved memory.
    pub(in crate::graph) fn publish(&mut self, entry: EntryAt, bucket: LabelBucket) -> LabelBucket {
        let had_header = bucket.header.is_some();
        let bucket = self.settle(bucket);
        bucket.write_header(&self.edges);
        if !had_header || bucket.header.is_none() {
            label_bucket_list::set(&self.label_bucket_lists, entry, &bucket.stored());
        }
        bucket
    }

    /// Gives a bucket with values or an index a header, and releases a header nothing needs any
    /// more. Within reserved memory.
    pub(in crate::graph) fn settle(&mut self, bucket: LabelBucket) -> LabelBucket {
        match (bucket.extended(), bucket.header) {
            (true, None) => LabelBucket {
                header: Some(reserved(self.edges.allocate(HEADER_CLASS))),
                ..bucket
            },
            (false, Some(header)) => {
                self.edges.release(header);
                LabelBucket {
                    header: None,
                    ..bucket
                }
            }
            (true, Some(_)) | (false, None) => bucket,
        }
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
                    bucket: LabelBucket::resolve(&self.edges.pages(), stored),
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
                .map(|stored| LabelBucket::resolve(&self.edges.pages(), stored)),
        )
    }
}

/// A label bucket found in its node's label bucket list.
#[derive(Clone, Copy)]
pub(in crate::graph) struct Found {
    pub(in crate::graph) entry: EntryAt,
    pub(in crate::graph) bucket: LabelBucket,
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
