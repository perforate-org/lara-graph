//! Reads beyond a whole bucket in position order: either order, windows and chosen positions.

use super::pair::{Found, key_of, opposite};
use super::{Error, Graph, RecordLines};
use crate::corrupt::corrupt;
use crate::descriptor::Descriptor;
use crate::id::{Cursor, Direction, Edge, EdgeRef, Label, NodeId, NodeState, Order, Position};
use crate::label_bucket::{self, Span, Window};
use crate::label_bucket_list::{self, LabelBucketKey, LabelBucketList};
use crate::memory::Memory;
use crate::node::Record;
use crate::page_store::{Extent, PAGE_BYTES, Pages};
use crate::stored::Stored;
use alloc::vec::Vec;
use core::ops::ControlFlow;

const LIST_PAGE_BYTES: usize = 4096;
const _: () = assert!(PAGE_BYTES == 4096);

/// A flat label bucket list's current page, kept for one batch. A page is loaded only once a run
/// of three consecutive lists lands in it, so a shuffled or scattered order never pays a 4 KiB
/// read for a single 19-byte list.
struct ListPageCache {
    page: Option<u32>,
    bytes: Vec<u8>,
    last_page: u32,
    run: u32,
}

impl ListPageCache {
    fn new() -> Self {
        Self {
            page: None,
            bytes: Vec::new(),
            last_page: u32::MAX,
            run: 0,
        }
    }

    /// The page slice at `extent`, loaded once a run of three lists lands in its page.
    #[inline]
    fn page<'a, M: Memory>(&'a mut self, pages: &Pages<'_, M>, extent: Extent) -> Option<&'a [u8]> {
        let page = extent.page().get();
        if page == self.last_page {
            self.run += 1;
        } else {
            self.last_page = page;
            self.run = 1;
        }
        if self.run < 3 {
            return None;
        }
        if self.page != Some(page) {
            self.bytes.resize(LIST_PAGE_BYTES, 0);
            pages.read_into(extent.page().offset(), &mut self.bytes);
            self.page = Some(page);
        }
        let at = usize::try_from(extent.offset().get() - extent.page().offset().get())
            .unwrap_or_else(|_| corrupt("list page offset"));
        self.bytes.get(at..)
    }

    #[inline]
    fn stored<M: Memory>(
        &mut self,
        pages: &Pages<'_, M>,
        list: LabelBucketList,
        key: LabelBucketKey,
        extent: Extent,
        count: usize,
    ) -> Option<Stored> {
        let page = extent.page().get();
        if page == self.last_page {
            self.run += 1;
        } else {
            self.last_page = page;
            self.run = 1;
        }
        if self.run >= 3 {
            if self.page != Some(page) {
                self.bytes.resize(LIST_PAGE_BYTES, 0);
                pages.read_into(extent.page().offset(), &mut self.bytes);
                self.page = Some(page);
            }
            let at = usize::try_from(extent.offset().get() - extent.page().offset().get())
                .unwrap_or_else(|_| corrupt("list page offset"));
            return label_bucket_list::lookup_stored_in(&self.bytes, at, count, key);
        }
        label_bucket_list::lookup_stored(pages, Some(list), key)
    }

    #[inline]
    fn for_each<M: Memory, B>(
        &mut self,
        pages: &Pages<'_, M>,
        list: LabelBucketList,
        f: &mut impl FnMut(LabelBucketKey, Stored) -> ControlFlow<B>,
    ) -> Option<B> {
        if let Some((extent, count)) = list.flat()
            && let Some(bytes) = self.page(pages, extent)
        {
            return label_bucket_list::for_each_bytes(bytes, 0, count, f);
        }
        label_bucket_list::for_each(pages, Some(list), f)
    }
}

/// A read-only batch over a list of nodes. The node-record line and flat-list page buffers
/// live as long as the batch, so several reads in one query share them. Drop it before mutating
/// the graph.
pub struct ReadBatch<'g, M: Memory> {
    graph: &'g Graph<M>,
    records: RecordLines,
    list_page: ListPageCache,
    last_bucket: Option<(NodeId, LabelBucketKey, Option<Descriptor>)>,
    last_found: Option<(NodeId, LabelBucketKey, Option<Found>)>,
    last_theirs: Option<(NodeId, LabelBucketKey, Found)>,
}

impl<'g, M: Memory> ReadBatch<'g, M> {
    fn new(graph: &'g Graph<M>) -> Self {
        Self {
            graph,
            records: RecordLines::default(),
            list_page: ListPageCache::new(),
            last_bucket: None,
            last_found: None,
            last_theirs: None,
        }
    }

    /// Calls `f` with `(node, edge)` for every live edge of each node's `(label, direction)`
    /// bucket, in `nodes` order and `order` within each bucket, until `f` breaks.
    ///
    /// The batch keeps an eight-record line of node records and, when a run of flat lists shares
    /// a 4 KiB page, that page. All nodes are checked before the first callback, so an unknown
    /// node leaves the callbacks untouched.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn visit_neighbors_each<B>(
        &mut self,
        nodes: &[NodeId],
        label: Label,
        direction: Direction,
        order: Order,
        f: impl FnMut(NodeId, Edge) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        visit_each(
            self.graph,
            nodes,
            LabelBucketKey { label, direction },
            order,
            &mut self.records,
            &mut self.list_page,
            f,
        )
    }

    /// Replaces `out` with the live-edge counts of each node's `(label, direction)` bucket, one
    /// entry per input node and 0 when the node has no bucket. Validates all nodes before it
    /// writes `out`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn degree_each(
        &mut self,
        nodes: &[NodeId],
        label: Label,
        direction: Direction,
        out: &mut Vec<u32>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        for &v in nodes {
            graph.check(v)?;
        }
        out.clear();
        out.reserve(nodes.len());
        let mode = mode(nodes);
        let key = LabelBucketKey { label, direction };
        let lists = graph.label_bucket_lists.pages();
        let view = graph.edges.pages();
        let records = &mut self.records;
        let list_page = &mut self.list_page;
        for &v in nodes {
            let record = record(graph, records, v, mode);
            let Some(list) = record.list else {
                out.push(0);
                continue;
            };
            let Some(stored) = stored(&lists, list_page, list, key, mode) else {
                out.push(0);
                continue;
            };
            out.push(slots(&view, stored).live());
        }
        Ok(())
    }

    /// Calls `f` with `(node, label)` for every bucket of each node on `direction`, in list
    /// order, until `f` breaks.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn labels_each<B>(
        &mut self,
        nodes: &[NodeId],
        direction: Direction,
        mut f: impl FnMut(NodeId, Label) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        for &v in nodes {
            graph.check(v)?;
        }
        let mode = mode(nodes);
        let records = &mut self.records;
        let list_page = &mut self.list_page;
        let lists = graph.label_bucket_lists.pages();
        for &v in nodes {
            let record = record(graph, records, v, mode);
            let Some(list) = record.list else { continue };
            if let Some(found) = for_each_list(&lists, list_page, list, mode, &mut |key, _| {
                if key.direction == direction {
                    f(v, key.label)
                } else {
                    ControlFlow::<B>::Continue(())
                }
            }) {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// Calls `f` with `(node, bool)` for each node: whether it has a live edge under any label,
    /// on any side. Within a node, stops at the first live bucket.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn has_edges_each<B>(
        &mut self,
        nodes: &[NodeId],
        mut f: impl FnMut(NodeId, bool) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        let graph = self.graph;
        for &v in nodes {
            graph.check(v)?;
        }
        let mode = mode(nodes);
        let records = &mut self.records;
        let list_page = &mut self.list_page;
        let lists = graph.label_bucket_lists.pages();
        let view = graph.edges.pages();
        for &v in nodes {
            let record = record(graph, records, v, mode);
            let has = if let Some(list) = record.list {
                for_each_list(&lists, list_page, list, mode, &mut |_, stored| {
                    if slots(&view, stored).live() > 0 {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::<()>::Continue(())
                    }
                })
                .is_some()
            } else {
                false
            };
            if let ControlFlow::Break(found) = f(v, has) {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// Replaces `out` with the value of each live half's edge, as that half's bucket holds it;
    /// `None` for a hole, a position past the end or a missing bucket. Consecutive halves in the
    /// same bucket reuse one lookup.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn edge_values_each(
        &mut self,
        halves: &[EdgeRef],
        out: &mut Vec<Option<Vec<u8>>>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        for half in halves {
            graph.check(half.node())?;
        }
        out.clear();
        out.reserve(halves.len());
        for &half in halves {
            let node = half.node();
            let key = key_of(half);
            let found = match self.last_found {
                Some((last_node, last_key, found)) if last_node == node && last_key == key => found,
                _ => {
                    let found = graph.find_label_bucket(node, key)?;
                    self.last_found = Some((node, key, found));
                    found
                }
            };
            let Some(found) = found else {
                out.push(None);
                continue;
            };
            let view = graph.edges.pages();
            if label_bucket::get(&found.bucket.slots, half.position(), &view).is_none() {
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
    pub fn edge_targets_each(
        &mut self,
        halves: &[EdgeRef],
        out: &mut Vec<Option<NodeId>>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        for half in halves {
            graph.check(half.node())?;
        }
        out.clear();
        out.reserve(halves.len());
        for &half in halves {
            let node = half.node();
            let key = key_of(half);
            let descriptor = match self.last_bucket {
                Some((last_node, last_key, descriptor)) if last_node == node && last_key == key => {
                    descriptor
                }
                _ => {
                    let descriptor = self.bucket(node, key);
                    self.last_bucket = Some((node, key, descriptor));
                    descriptor
                }
            };
            let view = graph.edges.pages();
            out.push(
                descriptor
                    .and_then(|descriptor| label_bucket::get(&descriptor, half.position(), &view)),
            );
        }
        Ok(())
    }

    fn bucket(&mut self, v: NodeId, key: LabelBucketKey) -> Option<Descriptor> {
        let record = record(self.graph, &mut self.records, v, Mode::Sparse);
        let list = record.list?;
        let lists = self.graph.label_bucket_lists.pages();
        let stored = stored(&lists, &mut self.list_page, list, key, Mode::Sparse)?;
        let view = self.graph.edges.pages();
        Some(slots(&view, stored))
    }

    /// Replaces `out` with the other half of each edge, or `None` when the half is not live.
    /// Consecutive halves in the same bucket reuse one bucket lookup.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn counterparts_each(
        &mut self,
        halves: &[EdgeRef],
        out: &mut Vec<Option<EdgeRef>>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        for half in halves {
            graph.check(half.node())?;
        }
        out.clear();
        out.reserve(halves.len());
        for &half in halves {
            let node = half.node();
            let key = key_of(half);
            let own = match self.last_found {
                Some((last_node, last_key, own)) if last_node == node && last_key == key => own,
                _ => {
                    let own = graph.find_label_bucket(node, key)?;
                    self.last_found = Some((node, key, own));
                    own
                }
            };
            let Some(own) = own else {
                out.push(None);
                continue;
            };
            let view = graph.edges.pages();
            let Some(other) = label_bucket::get(&own.bucket.slots, half.position(), &view) else {
                out.push(None);
                continue;
            };
            let key = LabelBucketKey {
                label: half.label(),
                direction: opposite(half.direction()),
            };
            let theirs = match self.last_theirs {
                Some((last_node, last_key, theirs)) if last_node == other && last_key == key => {
                    theirs
                }
                _ => {
                    let theirs = graph.present(other, key);
                    self.last_theirs = Some((other, key, theirs));
                    theirs
                }
            };
            out.push(Some(
                graph.pair_with_theirs(half, own, other, key, theirs).other,
            ));
        }
        Ok(())
    }

    /// Replaces `out` with the state of each input node. Validates all nodes before it writes
    /// `out`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`] for any node at or past [`Graph::node_count`].
    pub fn node_state_each(
        &mut self,
        nodes: &[NodeId],
        out: &mut Vec<NodeState>,
    ) -> Result<(), Error> {
        let graph = self.graph;
        for &v in nodes {
            graph.check(v)?;
        }
        out.clear();
        out.reserve(nodes.len());
        let mode = mode(nodes);
        let records = &mut self.records;
        for &v in nodes {
            out.push(record(graph, records, v, mode).state);
        }
        Ok(())
    }
}

#[inline]
fn visit_each<M: Memory, B>(
    graph: &Graph<M>,
    nodes: &[NodeId],
    key: LabelBucketKey,
    order: Order,
    records: &mut RecordLines,
    list_page: &mut ListPageCache,
    mut f: impl FnMut(NodeId, Edge) -> ControlFlow<B>,
) -> Result<Option<B>, Error> {
    for &v in nodes {
        graph.check(v)?;
    }
    let mode = mode(nodes);
    let lists = graph.label_bucket_lists.pages();
    let view = graph.edges.pages();
    for &v in nodes {
        let record = record(graph, records, v, mode);
        let Some(list) = record.list else { continue };
        let Some(stored) = stored(&lists, list_page, list, key, mode) else {
            continue;
        };
        let slots = slots(&view, stored);
        if let Some(found) =
            label_bucket::visit(&slots, &view, Span::all(&slots, order), Window::Page, |e| {
                f(v, e)
            })
        {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

/// Whether a batch of ids is dense enough for a 64-byte node-record line. A false positive still
/// reads the right line, only without a gain.
#[derive(Clone, Copy)]
enum Mode {
    Dense,
    Sparse,
}

fn dense(nodes: &[NodeId]) -> bool {
    nodes.len() >= 8
        && nodes
            .first()
            .map(|v| v.index())
            .zip(nodes.get(7).map(|v| v.index()))
            .is_some_and(|(first, eighth)| first.checked_add(7) == Some(eighth))
}

fn mode(nodes: &[NodeId]) -> Mode {
    if dense(nodes) {
        Mode::Dense
    } else {
        Mode::Sparse
    }
}

#[inline]
fn record<M: Memory>(graph: &Graph<M>, records: &mut RecordLines, v: NodeId, mode: Mode) -> Record {
    match mode {
        Mode::Dense => records.record(graph, v),
        Mode::Sparse => graph.record(v).unwrap_or_else(|_| corrupt("a known node")),
    }
}

#[inline]
fn stored<M: Memory>(
    lists: &Pages<'_, M>,
    list_page: &mut ListPageCache,
    list: LabelBucketList,
    key: LabelBucketKey,
    mode: Mode,
) -> Option<Stored> {
    match mode {
        Mode::Dense => match list.flat() {
            Some((extent, count)) => list_page.stored(lists, list, key, extent, count),
            None => label_bucket_list::lookup_stored(lists, Some(list), key),
        },
        Mode::Sparse => label_bucket_list::lookup_stored(lists, Some(list), key),
    }
}

#[inline]
fn for_each_list<M: Memory, B>(
    lists: &Pages<'_, M>,
    list_page: &mut ListPageCache,
    list: LabelBucketList,
    mode: Mode,
    f: &mut impl FnMut(LabelBucketKey, Stored) -> ControlFlow<B>,
) -> Option<B> {
    match mode {
        Mode::Dense => list_page.for_each(lists, list, f),
        Mode::Sparse => label_bucket_list::for_each(lists, Some(list), f),
    }
}

#[inline]
fn slots<M: Memory>(view: &Pages<'_, M>, stored: Stored) -> Descriptor {
    match stored {
        Stored::Plain(slots) => slots,
        Stored::Extended(header) => Descriptor::decode(view.read_array(header.offset())),
    }
}

impl<M: Memory> Graph<M> {
    /// A read-only batch that keeps its node-record and list-page buffers across calls. Drop it
    /// before mutating the graph.
    #[must_use]
    pub fn read_batch(&self) -> ReadBatch<'_, M> {
        ReadBatch::new(self)
    }

    /// Calls `f` with each live edge of `v`'s `(label, direction)` bucket in `order`, until it
    /// breaks; returns the break value. Reads like [`Self::neighbors_directed`] without collecting.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn visit_neighbors<B>(
        &self,
        v: NodeId,
        label: Label,
        direction: Direction,
        order: Order,
        f: impl FnMut(Edge) -> ControlFlow<B>,
    ) -> Result<Option<B>, Error> {
        Ok(self
            .bucket(v, LabelBucketKey { label, direction })?
            .and_then(|d| {
                label_bucket::visit(
                    &d,
                    &self.edges.pages(),
                    Span::all(&d, order),
                    Window::Page,
                    f,
                )
            }))
    }

    /// Calls `f` with each live edge in `cursor`'s window of `v`'s `(label, direction)` bucket, in
    /// its order, and returns the next window, or `None` when the bucket has no positions past this
    /// one. Holes count toward the span, so each call reads at most `span` slots.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn visit_window(
        &self,
        v: NodeId,
        label: Label,
        direction: Direction,
        cursor: Cursor,
        mut f: impl FnMut(Edge),
    ) -> Result<Option<Cursor>, Error> {
        let Some(d) = self.bucket(v, LabelBucketKey { label, direction })? else {
            return Ok(None);
        };
        let stored = d.stored();
        let span = cursor.span().get();
        let (window, next) = match cursor.order() {
            Order::Ascending => {
                let low = cursor.from().map_or(0, Position::index);
                let high = low.saturating_add(span).min(stored);
                (
                    Span::new(low, high, Order::Ascending),
                    (high < stored).then_some(high),
                )
            }
            Order::Descending => {
                // The window ends at `from` (included), clamped to the last position.
                let Some(last) = stored.checked_sub(1) else {
                    return Ok(None);
                };
                let top = cursor.from().map_or(last, |p| p.index().min(last));
                let low = (top + 1).saturating_sub(span);
                (
                    Span::new(low, top + 1, Order::Descending),
                    low.checked_sub(1),
                )
            }
        };
        label_bucket::visit(&d, &self.edges.pages(), window, Window::Page, |e| {
            f(e);
            ControlFlow::<()>::Continue(())
        });
        Ok(next.map(|at| Cursor::at(cursor.order(), Position::new(at), cursor.span())))
    }

    /// Replaces the contents of `out` with the targets at `positions` of `v`'s `(label, direction)`
    /// bucket, in the order given: `None` for a hole or a position past the end. Reads each
    /// touched run or page once.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`].
    pub fn targets_at(
        &self,
        v: NodeId,
        label: Label,
        direction: Direction,
        positions: &[Position],
        out: &mut Vec<Option<NodeId>>,
    ) -> Result<(), Error> {
        out.clear();
        let Some(d) = self.bucket(v, LabelBucketKey { label, direction })? else {
            out.resize(positions.len(), None);
            return Ok(());
        };
        // Positions already ascending and distinct, as a reader of windows passes them, go
        // straight through; others are sorted for the one pass and put back in their order.
        if positions.is_sorted_by(|a, b| a < b) {
            label_bucket::read_positions(&d, &self.edges.pages(), positions, |_, target| {
                out.push(target);
            });
            return Ok(());
        }
        let mut sorted = positions.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let mut found = Vec::with_capacity(sorted.len());
        label_bucket::read_positions(&d, &self.edges.pages(), &sorted, |_, target| {
            found.push(target);
        });
        out.extend(positions.iter().map(|p| {
            sorted
                .binary_search(p)
                .ok()
                .and_then(|i| found.get(i).copied().flatten())
        }));
        Ok(())
    }
}
