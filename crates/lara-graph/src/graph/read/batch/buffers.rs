//! A batch's node-record and flat-list page reads: dense batches read through the shared
//! buffers, sparse ones directly.

use super::view::BatchView;
use crate::bucket::LabelBucketKey;
use crate::bucket::stored::Stored;
use crate::graph::Graph;
use crate::id::{NodeId, NodeState};
use crate::node::label_bucket_list::{self, LabelBucketList, Listed};
use crate::node::labels::CellLines;
use crate::node::labels::set::LabelSet;
use crate::node::record::{RECORD_BYTES, Record};
use crate::node::store::RecordLines;
use crate::store::corrupt::corrupt;
use crate::store::memory::Memory;
use crate::store::page_store::{Extent, PAGE_BYTES};
use alloc::vec::Vec;
use core::ops::ControlFlow;

const LIST_PAGE_BYTES: usize = 4096;
const _: () = assert!(PAGE_BYTES == 4096);

/// A flat label bucket list's current page, kept for one batch. A page is loaded only once a run
/// of three consecutive lists lands in it, so a shuffled or scattered order never pays a 4 KiB
/// read for a single 19-byte list.
pub(super) struct ListPageCache {
    page: Option<u32>,
    bytes: Vec<u8>,
    last_page: u32,
    run: u32,
}

impl ListPageCache {
    pub(super) fn new() -> Self {
        Self {
            page: None,
            bytes: Vec::new(),
            last_page: u32::MAX,
            run: 0,
        }
    }

    /// The page slice at `extent`, loaded once a run of three lists lands in its page. Always
    /// inlined: left a call, it cost a batch of label reads 6 %.
    #[expect(clippy::inline_always, reason = "measured: see the doc")]
    #[inline(always)]
    fn page<'a, M: Memory>(
        &'a mut self,
        view: &BatchView<'_, M>,
        extent: Extent,
    ) -> Option<&'a [u8]> {
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
            view.read_list_page(extent.page(), &mut self.bytes);
            self.page = Some(page);
        }
        let at = usize::try_from(extent.offset().get() - extent.page().offset().get())
            .unwrap_or_else(|_| corrupt("list page offset"));
        self.bytes.get(at..)
    }

    #[inline]
    fn stored<M: Memory>(
        &mut self,
        view: &BatchView<'_, M>,
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
                view.read_list_page(extent.page(), &mut self.bytes);
                self.page = Some(page);
            }
            let at = usize::try_from(extent.offset().get() - extent.page().offset().get())
                .unwrap_or_else(|_| corrupt("list page offset"));
            return label_bucket_list::lookup_stored_in(&self.bytes, at, count, key);
        }
        view.lookup(Some(list), key)
    }

    #[inline]
    fn for_each<M: Memory, B>(
        &mut self,
        view: &BatchView<'_, M>,
        list: LabelBucketList,
        f: &mut impl FnMut(LabelBucketKey, Listed) -> ControlFlow<B>,
    ) -> Option<B> {
        if let Some((extent, count)) = list.flat()
            && let Some(bytes) = self.page(view, extent)
        {
            return label_bucket_list::for_each_bytes(bytes, 0, count, f);
        }
        view.walk(list, f)
    }
}

/// Whether a batch of ids is dense enough for a 64-byte node-record line. A false positive still
/// reads the right line, only without a gain.
#[derive(Clone, Copy)]
pub(super) enum Mode {
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

pub(super) fn mode(nodes: &[NodeId]) -> Mode {
    if dense(nodes) {
        Mode::Dense
    } else {
        Mode::Sparse
    }
}

/// `v`'s record, through the batch's line in `Dense` mode. Always inlined: left a call, it cost
/// a batch of node states 15 %.
#[expect(clippy::inline_always, reason = "measured: see the doc")]
#[inline(always)]
pub(super) fn record<M: Memory>(
    graph: &Graph<M>,
    records: &mut RecordLines,
    v: NodeId,
    mode: Mode,
) -> Record {
    Record::decode(record_bytes(graph, records, v, mode))
}

/// `v`'s state alone, as [`record`] reads the record: no extent decoded.
#[expect(clippy::inline_always, reason = "as record")]
#[inline(always)]
pub(super) fn state<M: Memory>(
    graph: &Graph<M>,
    records: &mut RecordLines,
    v: NodeId,
    mode: Mode,
) -> NodeState {
    Record::state_of(record_bytes(graph, records, v, mode))
}

#[expect(clippy::inline_always, reason = "as record")]
#[inline(always)]
fn record_bytes<M: Memory>(
    graph: &Graph<M>,
    records: &mut RecordLines,
    v: NodeId,
    mode: Mode,
) -> [u8; RECORD_BYTES] {
    match mode {
        Mode::Dense => graph.line_record_bytes(records, v),
        Mode::Sparse => graph.record_bytes(v),
    }
}

#[inline]
pub(super) fn stored<M: Memory>(
    view: &BatchView<'_, M>,
    list_page: &mut ListPageCache,
    list: LabelBucketList,
    key: LabelBucketKey,
    mode: Mode,
) -> Option<Stored> {
    match mode {
        Mode::Dense => match list.flat() {
            Some((extent, count)) => list_page.stored(view, list, key, extent, count),
            None => view.lookup(Some(list), key),
        },
        Mode::Sparse => view.lookup(Some(list), key),
    }
}

#[inline]
pub(super) fn for_each_list<M: Memory, B>(
    view: &BatchView<'_, M>,
    list_page: &mut ListPageCache,
    list: LabelBucketList,
    mode: Mode,
    f: &mut impl FnMut(LabelBucketKey, Listed) -> ControlFlow<B>,
) -> Option<B> {
    match mode {
        Mode::Dense => list_page.for_each(view, list, f),
        Mode::Sparse => view.walk(list, f),
    }
}

/// `v`'s label cell, through the batch's line in `Dense` mode.
#[inline]
pub(super) fn label_cell<M: Memory>(
    graph: &Graph<M>,
    lines: &mut CellLines,
    v: NodeId,
    mode: Mode,
) -> LabelSet {
    match mode {
        Mode::Dense => graph.label_cell_in(lines, v),
        Mode::Sparse => graph.label_cell(v),
    }
}
