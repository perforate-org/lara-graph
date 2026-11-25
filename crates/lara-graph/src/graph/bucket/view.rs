//! What a read sees of the graph's stores, as the operations reads need: a node's bucket list
//! looked up or walked, a bucket's descriptor or whole state from its list entry, its live edges
//! in order, the target at a position, and its edges with their values. Reads go through it
//! instead of the stores, so they cannot write or reach allocation metadata.

use crate::bucket::descriptor::Descriptor;
use crate::bucket::label_bucket::{self, Span, Window};
use crate::bucket::stored::{LabelBucket, Stored};
use crate::graph::Graph;
use crate::graph::bucket::{Widths, visit_joined};
use crate::id::{Edge, Label, NodeId, Order, Position};
use crate::node::label_bucket_list::{self, LabelBucketKey, LabelBucketList, Listed};
use crate::store::memory::Memory;
use crate::store::page_store::{PageId, Pages};
use core::ops::ControlFlow;

/// A read's view of the graph: its label bucket lists, its edge pages and its width table.
pub(in crate::graph) struct View<'g, M: Memory> {
    lists: Pages<'g, M>,
    edges: Pages<'g, M>,
    widths: &'g Widths,
}

impl<M: Memory> Graph<M> {
    /// The view reads go through.
    #[inline]
    pub(in crate::graph) fn view(&self) -> View<'_, M> {
        View {
            lists: self.label_bucket_lists.pages(),
            edges: self.edges.pages(),
            widths: &self.widths,
        }
    }
}

impl<M: Memory> View<'_, M> {
    /// The list entry of `list`'s bucket `key`, if it has one.
    #[inline]
    pub(in crate::graph) fn lookup(
        &self,
        list: Option<LabelBucketList>,
        key: LabelBucketKey,
    ) -> Option<Stored> {
        label_bucket_list::lookup_stored(&self.lists, list, key)
    }

    /// Every entry of `list`, in order, until `f` breaks.
    #[inline]
    pub(in crate::graph) fn walk<B>(
        &self,
        list: LabelBucketList,
        f: &mut impl FnMut(LabelBucketKey, Listed) -> ControlFlow<B>,
    ) -> Option<B> {
        label_bucket_list::for_each(&self.lists, Some(list), f)
    }

    /// Fills `buf` with list page `page`: for a batch that keeps a page of lists.
    #[inline]
    pub(in crate::graph) fn read_list_page(&self, page: PageId, buf: &mut [u8]) {
        self.lists.read_into(page.offset(), buf);
    }

    /// The descriptor a list entry stands for: in the entry, or in the header it points to.
    #[inline]
    pub(in crate::graph) fn slots(&self, stored: Stored) -> Descriptor {
        match stored {
            Stored::Plain(slots) => slots,
            Stored::Extended(header) => Descriptor::decode(self.edges.read_array(header.offset())),
        }
    }

    /// The whole bucket a list entry stands for, its values and index included.
    #[inline]
    pub(in crate::graph) fn resolve(&self, stored: Stored) -> LabelBucket {
        LabelBucket::resolve(&self.edges, stored)
    }

    /// Calls `f` with each live edge of `slots` in `order`, until it breaks.
    #[inline]
    pub(in crate::graph) fn visit<B>(
        &self,
        slots: &Descriptor,
        order: Order,
        f: impl FnMut(Edge) -> ControlFlow<B>,
    ) -> Option<B> {
        label_bucket::visit(slots, &self.edges, Span::all(slots, order), Window::Page, f)
    }

    /// The live target at `at` of `slots`, if any.
    #[inline]
    pub(in crate::graph) fn target(&self, slots: &Descriptor, at: Position) -> Option<NodeId> {
        label_bucket::get(slots, at, &self.edges)
    }

    /// Calls `f` with each live edge of `bucket` (of `label`) and its value, in `order`, until it
    /// breaks.
    #[inline]
    pub(in crate::graph) fn visit_with_values<B>(
        &self,
        bucket: &LabelBucket,
        label: Label,
        order: Order,
        f: &mut impl FnMut(Edge, &[u8]) -> ControlFlow<B>,
    ) -> Option<B> {
        visit_joined(&self.edges, bucket, self.widths.get(label), order, f)
    }
}
