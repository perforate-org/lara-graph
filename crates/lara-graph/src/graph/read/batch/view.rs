//! What a batch read sees of the graph's stores, as the operations batches need: a node's bucket list
//! looked up or walked, a bucket's descriptor or whole state from its list entry, its live edges
//! in order, the target at a position, and its edges with their values. Batches go through it
//! instead of the stores, so they cannot write or reach allocation metadata.

use crate::bucket::LabelBucket;
use crate::bucket::LabelBucketKey;
use crate::bucket::descriptor::Descriptor;
use crate::bucket::slots::{self, Span, Window};
use crate::bucket::stored::Stored;
use crate::edge::{Widths, visit_joined};
use crate::graph::Graph;
use crate::id::{Label, Neighbor, NodeId, Position};
use crate::input::Order;
use crate::node::label_bucket_list::{self, LabelBucketList, Listed};
use crate::store::memory::Memory;
use crate::store::page_store::{PageId, Pages};
use core::ops::ControlFlow;

/// A batch read's view of the graph: its label bucket lists, its edge pages and its width table.
pub(super) struct BatchView<'g, M: Memory> {
    lists: Pages<'g, M>,
    edges: Pages<'g, M>,
    widths: &'g Widths,
}

impl<M: Memory> Graph<M> {
    /// The view reads go through.
    #[inline]
    pub(super) fn view(&self) -> BatchView<'_, M> {
        BatchView {
            lists: self.label_bucket_lists.pages(),
            edges: self.edge_view().pages(),
            widths: &self.widths,
        }
    }
}

impl<M: Memory> BatchView<'_, M> {
    /// The list entry of `list`'s bucket `key`, if it has one.
    #[inline]
    pub(super) fn lookup(
        &self,
        list: Option<LabelBucketList>,
        key: LabelBucketKey,
    ) -> Option<Stored> {
        label_bucket_list::lookup_stored(&self.lists, list, key)
    }

    /// Every entry of `list`, in order, until `f` breaks.
    #[inline]
    pub(super) fn walk<B>(
        &self,
        list: LabelBucketList,
        f: &mut impl FnMut(LabelBucketKey, Listed) -> ControlFlow<B>,
    ) -> Option<B> {
        label_bucket_list::for_each(&self.lists, Some(list), f)
    }

    /// Fills `buf` with list page `page`: for a batch that keeps a page of lists.
    #[inline]
    pub(super) fn read_list_page(&self, page: PageId, buf: &mut [u8]) {
        self.lists.read_into(page.offset(), buf);
    }

    /// The descriptor a list entry stands for: in the entry, or in the header it points to.
    #[inline]
    pub(super) fn slots(&self, stored: Stored) -> Descriptor {
        stored.slots(&self.edges)
    }

    /// The whole bucket a list entry stands for, its values and index included.
    #[inline]
    pub(super) fn resolve(&self, stored: Stored) -> LabelBucket {
        LabelBucket::resolve(&self.edges, stored)
    }

    /// Calls `f` with each live edge of `slots` in `order`, until it breaks.
    #[inline]
    pub(super) fn visit<B>(
        &self,
        slots: &Descriptor,
        order: Order,
        f: impl FnMut(Neighbor) -> ControlFlow<B>,
    ) -> Option<B> {
        slots::visit(slots, &self.edges, Span::all(slots, order), Window::Page, f)
    }

    /// The live target at `at` of `slots`, if any.
    #[inline]
    pub(super) fn target(&self, slots: &Descriptor, at: Position) -> Option<NodeId> {
        slots::get(slots, at, &self.edges)
    }

    /// Calls `f` with each live edge of `bucket` (of `label`) and its value, in `order`, until it
    /// breaks.
    #[inline]
    pub(super) fn visit_with_values<B>(
        &self,
        bucket: &LabelBucket,
        label: Label,
        order: Order,
        f: &mut impl FnMut(Neighbor, &[u8]) -> ControlFlow<B>,
    ) -> Option<B> {
        visit_joined(&self.edges, bucket, self.widths.get(label), order, f)
    }
}
