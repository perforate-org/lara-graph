//! A bucket's value plane: one fixed-width value per position, beside the slots.
//!
//! Values are addressed by position, like slots: removing an edge writes no value, refilling a
//! hole writes one, and only compaction moves values (one cell per moved edge). A position no
//! value was written to reads as zeros, so declaring a width on a label with edges costs nothing.
//!
//! A plane is **empty**, a **run** (one extent of up to a page, covering its first cells), or a
//! **table**: a radix tree of node pages (1024 page ids each, `u32::MAX` for none) over value
//! pages of `4096 / width` cells. Missing nodes and pages read as zeros, so a write anywhere
//! allocates at most the path to it: [`PAGES_PER_WRITE`] pages.
//!
//! # Plane root
//!
//! ```text
//! ---------------------------------------- <- Empty
//! Kind (0)                        ↕ 1 byte
//! ----------------------------------------
//! Reserved space                  ↕ 5 bytes
//! ---------------------------------------- <- Run: cells 0, 1, … of `width` bytes
//! Kind (1)                        ↕ 1 byte
//! ----------------------------------------
//! Extent                          ↕ 5 bytes
//! ---------------------------------------- <- Table
//! Kind (2)                        ↕ 1 byte
//! ----------------------------------------
//! Root node page id               ↕ 4 bytes
//! ----------------------------------------
//! Depth (levels of nodes)         ↕ 1 byte
//! ----------------------------------------
//! ```
//!
//! # Table pages
//!
//! Value page `i` covers positions from `i × ⌊4096 / width⌋`.
//!
//! ```text
//! ---------------------------------------- <- Node page
//! Child page id 0                 ↕ 4 bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Child page id 1023              ↕ 4 bytes
//! ---------------------------------------- <- Value page
//! Cell 0                          ↕ width bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Cell ⌊4096 / width⌋ - 1         ↕ width bytes
//! ----------------------------------------
//! Unused space
//! ```

use crate::id::Width;
use crate::store::corrupt::corrupt;
use crate::store::memory::{Memory, WriteError};
use crate::store::page_store::{
    ByteClass, EXTENT_BYTES, Extent, PAGE_BYTES, PageId, PageStore, Pages,
};
use crate::store::read_buf::ReadBuf;
use crate::store::units::ByteOffset;
use alloc::vec;
use alloc::vec::Vec;

mod table;

use table::{MAX_DEPTH, NODE_ENTRIES, data_page, grow_table, release_node, table_with};

/// Pages one [`write`](fn@write) can take. The worst case turns a run into a table and writes far away: a
/// root, value page 0 for the run's cells, three roots as the table deepens to four levels, the
/// three nodes below them and the value page.
pub(crate) const PAGES_PER_WRITE: u32 = 9;
/// Pages one compaction step's value moves can take. Moved edges land on at most a page of
/// positions (1024), contiguous: that many cells' value pages, plus nodes along their paths
/// (three levels, at most two partly covered nodes per level beyond the shared ones).
pub(crate) fn pages_per_moves(width: Width) -> u32 {
    let pages = crate::bucket::descriptor::PAGE_SLOTS.div_ceil(cells_per_page(width)) + 1;
    pages + 3 * (pages / NODE_ENTRIES + 2)
}

/// Bytes of a plane's root.
pub(crate) const ROOT_BYTES: usize = 6;

const KIND_EMPTY: u8 = 0;
const KIND_RUN: u8 = 1;
const KIND_TABLE: u8 = 2;

/// Where a bucket's values are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Plane {
    /// No value written: every position reads as zeros.
    Empty,
    /// Cells `0..extent.len() / width` in one extent.
    Run(Extent),
    /// A radix tree `depth` levels of nodes deep.
    Table { root: PageId, depth: u8 },
    /// Cells `0..cells` beside a run's slots, from byte `at` of `page`: owned by the run, which
    /// grows, moves and frees them with its slots. Never stored in a header. As small as the other
    /// forms, so that a bucket carrying a plane is no larger for it.
    Beside { page: PageId, at: u16, cells: u16 },
}

impl Plane {
    /// Cells `0..cells` beside a run's slots, from `start` on `page`.
    pub(crate) fn beside(page: PageId, start: ByteOffset, cells: u16) -> Self {
        let at = u16::try_from(start.get() - page.offset().get())
            .unwrap_or_else(|_| corrupt("cells beyond their page"));
        Self::Beside { page, at, cells }
    }

    /// As the module's Plane root diagram shows.
    pub(crate) fn encode(self) -> [u8; ROOT_BYTES] {
        match self {
            Self::Empty => [KIND_EMPTY, 0, 0, 0, 0, 0],
            Self::Run(extent) => {
                let [e0, e1, e2, e3, e4] = extent.encode();
                [KIND_RUN, e0, e1, e2, e3, e4]
            }
            Self::Table { root, depth } => {
                let [r0, r1, r2, r3] = root.get().to_le_bytes();
                [KIND_TABLE, r0, r1, r2, r3, depth]
            }
            Self::Beside { .. } => corrupt("values beside slots in a header"),
        }
    }

    pub(crate) fn decode(bytes: [u8; ROOT_BYTES]) -> Self {
        let [kind, b1, b2, b3, b4, b5] = bytes;
        match kind {
            KIND_EMPTY => Self::Empty,
            KIND_RUN => Self::Run(Extent::decode([b1, b2, b3, b4, b5])),
            KIND_TABLE if (1..=MAX_DEPTH).contains(&b5) => Self::Table {
                root: PageId::new(u32::from_le_bytes([b1, b2, b3, b4])),
                depth: b5,
            },
            _ => corrupt("value plane"),
        }
    }
}

/// Cells per value page.
pub(crate) fn cells_per_page(width: Width) -> u32 {
    PAGE_BYTES / u32::from(width.get())
}

fn width_len(width: Width) -> usize {
    usize::from(width.get())
}

/// Where the cell of `position` is, if its storage exists.
fn cell_at<M: Memory>(
    view: &Pages<'_, M>,
    plane: Plane,
    width: Width,
    position: u32,
) -> Option<ByteOffset> {
    let w = u64::from(width.get());
    match plane {
        Plane::Empty => None,
        Plane::Run(extent) => (u64::from(position) * w + w <= u64::from(extent.len()))
            .then(|| ByteOffset::new(extent.offset().get() + u64::from(position) * w)),
        Plane::Table { root, depth } => {
            let cpp = cells_per_page(width);
            let page = data_page(view, root, depth, u64::from(position / cpp))?;
            Some(ByteOffset::new(
                page.offset().get() + u64::from(position % cpp) * w,
            ))
        }
        Plane::Beside { page, at, cells } => (position < u32::from(cells)).then(|| {
            ByteOffset::new(page.offset().get() + u64::from(at) + u64::from(position) * w)
        }),
    }
}

/// Fills `out` (one width long) with the value at `position`, zeros if none was written.
pub(crate) fn read<M: Memory>(
    view: &Pages<'_, M>,
    plane: Plane,
    width: Width,
    position: u32,
    out: &mut [u8],
) {
    match cell_at(view, plane, width, position) {
        Some(at) => view.read_into(at, out),
        None => out.fill(0),
    }
}

/// Calls `f` with the values of positions `low..high`, a value page (or the run) at a time, as
/// `(first position, bytes)`; missing storage comes as zeros. Lowest first, or highest first
/// when `descending` (each chunk's bytes stay in position order).
pub(crate) fn visit<M: Memory, B>(
    view: &Pages<'_, M>,
    plane: Plane,
    width: Width,
    (low, high, descending): (u32, u32, bool),
    mut f: impl FnMut(u32, &[u8]) -> core::ops::ControlFlow<B>,
) -> Option<B> {
    let cpp = cells_per_page(width);
    let w = width_len(width);
    let (first_page, end_page) = (low / cpp, high.div_ceil(cpp));
    let mut buf = ReadBuf::new();
    for step in 0..end_page.saturating_sub(first_page) {
        let page = if descending {
            end_page - 1 - step
        } else {
            first_page + step
        };
        let (a, b) = ((page * cpp).max(low), ((page + 1) * cpp).min(high));
        if a >= b {
            continue;
        }
        let chunk = buf.take(usize::try_from(b - a).unwrap_or(0) * w);
        // The cells with storage: a run may cover only the front of its chunk.
        let cell = cell_at(view, plane, width, a);
        let stored = stored_cells(plane, width, cell, (a, b));
        let (head, tail) = chunk.split_at_mut(usize::try_from(stored).unwrap_or(0) * w);
        if let Some(at) = cell.filter(|_| !head.is_empty()) {
            view.read_into(at, head);
        }
        tail.fill(0);
        if let core::ops::ControlFlow::Break(done) = f(a, chunk) {
            return Some(done);
        }
    }
    None
}

/// How many of positions `a..b`, all on one value page (or in the run), have storage, given the
/// first one's `cell`: none when it has none, and a run may cover only the front.
fn stored_cells(plane: Plane, width: Width, cell: Option<ByteOffset>, (a, b): (u32, u32)) -> u32 {
    match (plane, cell) {
        (_, None) | (Plane::Empty, _) => 0,
        (Plane::Run(extent), Some(_)) => (extent.len() / u32::from(width.get())).saturating_sub(a),
        (Plane::Beside { cells, .. }, Some(_)) => u32::from(cells).saturating_sub(a),
        (Plane::Table { .. }, Some(_)) => b - a,
    }
    .min(b - a)
}

/// Writes `value` (one width long) at `position`, allocating at most [`PAGES_PER_WRITE`] pages,
/// and returns the plane's new root. Callers reserve those pages first; then it cannot fail.
pub(crate) fn write<M: Memory>(
    store: &mut PageStore<M>,
    plane: Plane,
    width: Width,
    position: u32,
    value: &[u8],
) -> Result<Plane, WriteError> {
    if let Some(at) = cell_at(&store.pages(), plane, width, position) {
        store.write(at, value);
        return Ok(plane);
    }
    if let Plane::Beside { .. } = plane {
        let at = cell_at(&store.pages(), plane, width, position)
            .unwrap_or_else(|| corrupt("a value past a run's slots"));
        store.write(at, value);
        return Ok(plane);
    }
    let cpp = cells_per_page(width);
    let next = match plane {
        Plane::Beside { .. } => plane,
        Plane::Empty if position < cpp => grow_run(store, None, width, position)?,
        Plane::Run(extent) if position < cpp => grow_run(store, Some(extent), width, position)?,
        Plane::Empty => table_with(store, None, u64::from(position / cpp))?,
        Plane::Run(extent) => {
            let next = table_with(store, Some(extent), u64::from(position / cpp))?;
            store.release(extent);
            next
        }
        Plane::Table { root, depth } => {
            grow_table(store, (root, depth), u64::from(position / cpp))?
        }
    };
    let at = cell_at(&store.pages(), next, width, position)
        .unwrap_or_else(|| corrupt("a value cell just made"));
    store.write(at, value);
    Ok(next)
}

/// A run covering `position`: the old run's cells copied, the rest zero.
fn grow_run<M: Memory>(
    store: &mut PageStore<M>,
    old: Option<Extent>,
    width: Width,
    position: u32,
) -> Result<Plane, WriteError> {
    let bytes = (position + 1) * u32::from(width.get());
    let class = ByteClass::for_bytes(bytes).unwrap_or_else(|| corrupt("value run beyond a page"));
    let extent = store.allocate(class)?;
    let mut content = vec![0u8; usize::try_from(extent.len()).unwrap_or(0)];
    if let Some(old) = old {
        let len = usize::try_from(old.len()).unwrap_or(0);
        if let Some(head) = content.get_mut(..len) {
            store.pages().read_into(old.offset(), head);
        }
    }
    store.write(extent.offset(), &content);
    if let Some(old) = old {
        store.release(old);
    }
    Ok(Plane::Run(extent))
}

pub(super) fn page_len() -> usize {
    usize::try_from(PAGE_BYTES).unwrap_or(0)
}

/// Writes `cells`, whole values of `width` for the positions from `first`, all in one value page
/// (or the run), and returns the plane's new root: one allocation path and one write. Callers
/// reserve [`PAGES_PER_WRITE`] pages first.
pub(crate) fn write_cells<M: Memory>(
    store: &mut PageStore<M>,
    plane: Plane,
    width: Width,
    first: u32,
    cells: &[u8],
) -> Result<Plane, WriteError> {
    let w = width_len(width);
    let count = u32::try_from(cells.len() / w).unwrap_or(0);
    let Some(last) = count.checked_sub(1) else {
        return Ok(plane);
    };
    let tail = cells.get(cells.len() - w..).unwrap_or(&[]);
    // The last cell's write makes the storage of the whole page; then one write fills it.
    let plane = write(store, plane, width, first + last, tail)?;
    let at = cell_at(&store.pages(), plane, width, first)
        .unwrap_or_else(|| corrupt("a value page just made"));
    store.write(at, cells);
    Ok(plane)
}

/// The values of positions `low..high`, back to back: zeros where none was written.
pub(crate) fn read_cells<M: Memory>(
    view: &Pages<'_, M>,
    plane: Plane,
    width: Width,
    (low, high): (u32, u32),
) -> Vec<u8> {
    let mut out = Vec::with_capacity(usize::try_from(high - low).unwrap_or(0) * width_len(width));
    visit(view, plane, width, (low, high, false), |_, bytes| {
        out.extend_from_slice(bytes);
        core::ops::ControlFlow::<()>::Continue(())
    });
    out
}

/// Copies the value at `from` to `to`; a zero value onto missing storage writes nothing.
pub(crate) fn copy<M: Memory>(
    store: &mut PageStore<M>,
    plane: Plane,
    width: Width,
    (from, to): (u32, u32),
) -> Result<Plane, WriteError> {
    let mut value = vec![0u8; width_len(width)];
    read(&store.pages(), plane, width, from, &mut value);
    if cell_at(&store.pages(), plane, width, to).is_none() && value.iter().all(|&b| b == 0) {
        return Ok(plane);
    }
    write(store, plane, width, to, &value)
}

/// Releases the storage of positions `low..high` that no longer hold edges: every value page
/// wholly inside the range, nodes left empty, and the whole plane when nothing is left. Values
/// there read as zeros afterwards. A run is released only when the range covers it.
pub(crate) fn release_range<M: Memory>(
    store: &mut PageStore<M>,
    plane: Plane,
    width: Width,
    (low, high): (u32, u32),
) -> Plane {
    match plane {
        Plane::Run(extent) if low == 0 && high >= extent.len() / u32::from(width.get()) => {
            store.release(extent);
            Plane::Empty
        }
        Plane::Empty | Plane::Run(_) | Plane::Beside { .. } => plane,
        Plane::Table { root, depth } => {
            release_table_range(store, (root, depth), width, (low, high))
        }
    }
}

/// [`release_range`] for a table: the pages wholly inside the range.
fn release_table_range<M: Memory>(
    store: &mut PageStore<M>,
    (root, depth): (PageId, u8),
    width: Width,
    (low, high): (u32, u32),
) -> Plane {
    let cpp = cells_per_page(width);
    let first = u64::from(low.div_ceil(cpp));
    // The last page is whole only if the range reaches its end.
    let end = if high == u32::MAX {
        table::reach(depth)
    } else {
        u64::from(high / cpp)
    };
    let table = Plane::Table { root, depth };
    if first >= end || !table::release_pages(store, (root, depth, 0), (first, end)) {
        return table;
    }
    store.release(Extent::whole(root));
    Plane::Empty
}

/// Releases every page and extent of a plane.
pub(crate) fn release<M: Memory>(store: &mut PageStore<M>, plane: Plane) {
    match plane {
        Plane::Empty | Plane::Beside { .. } => {}
        Plane::Run(extent) => store.release(extent),
        Plane::Table { root, depth } => release_node(store, root, depth),
    }
}

/// A plane holding `cells` (whole values, at most a page of them) from position 0.
pub(crate) fn run_of<M: Memory>(
    store: &mut PageStore<M>,
    cells: &[u8],
) -> Result<Plane, WriteError> {
    if cells.is_empty() {
        return Ok(Plane::Empty);
    }
    let len = u32::try_from(cells.len()).unwrap_or(u32::MAX);
    let class = ByteClass::for_bytes(len).unwrap_or_else(|| corrupt("value run beyond a page"));
    let extent = store.allocate(class)?;
    // The run covers its whole extent: cells past the given ones read as zeros.
    let mut content = vec![0u8; usize::try_from(extent.len()).unwrap_or(0)];
    if let Some(head) = content.get_mut(..cells.len()) {
        head.copy_from_slice(cells);
    }
    store.write(extent.offset(), &content);
    Ok(Plane::Run(extent))
}

/// Every extent a plane owns (test invariant checks).
#[cfg(test)]
pub(crate) fn extents<M: Memory>(view: &Pages<'_, M>, plane: Plane) -> Vec<Extent> {
    use table::NONE;
    fn walk<M: Memory>(view: &Pages<'_, M>, node: PageId, level: u8, out: &mut Vec<Extent>) {
        out.push(Extent::whole(node));
        if level == 0 {
            return;
        }
        let bytes = view.read(node.offset(), page_len());
        for &raw in bytes.as_chunks::<4>().0 {
            let raw = u32::from_le_bytes(raw);
            if raw != NONE {
                walk(view, PageId::new(raw), level - 1, out);
            }
        }
    }
    let mut out = Vec::new();
    match plane {
        Plane::Empty | Plane::Beside { .. } => {}
        Plane::Run(extent) => out.push(extent),
        Plane::Table { root, depth } => walk(view, root, depth, &mut out),
    }
    out
}

const _: () = assert!(EXTENT_BYTES + 1 == ROOT_BYTES);

#[cfg(test)]
mod tests;

/// Proofs over every input, checked by Kani (`cargo kani -p lara-graph`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Any six bytes of a known kind decode to a plane root that encodes back to itself.
    #[kani::proof]
    fn plane_roots_round_trip() {
        let bytes: [u8; ROOT_BYTES] = kani::any();
        let [kind, .., depth] = bytes;
        kani::assume(
            kind == KIND_EMPTY
                || kind == KIND_RUN
                || (kind == KIND_TABLE && (1..=MAX_DEPTH).contains(&depth)),
        );
        let plane = Plane::decode(bytes);
        assert_eq!(Plane::decode(plane.encode()), plane);
    }
}
