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

use crate::corrupt::corrupt;
use crate::id::Width;
use crate::memory::{Memory, WriteError};
use crate::page_store::{ByteClass, EXTENT_BYTES, Extent, PAGE_BYTES, PageId, PageStore, Pages};
use crate::units::ByteOffset;
use alloc::vec;
use alloc::vec::Vec;

mod table;

use table::{MAX_DEPTH, NODE_ENTRIES, data_page, grow_table, release_node, table_with};

/// Pages one [`write`] can take. The worst case turns a run into a table and writes far away: a
/// root, value page 0 for the run's cells, three roots as the table deepens to four levels, the
/// three nodes below them and the value page.
pub(crate) const PAGES_PER_WRITE: u32 = 9;
/// Pages one compaction step's value moves can take. Moved edges land on at most a page of
/// positions (1024), contiguous: that many cells' value pages, plus nodes along their paths
/// (three levels, at most two partly covered nodes per level beyond the shared ones).
pub(crate) fn pages_per_moves(width: Width) -> u32 {
    let pages = crate::descriptor::PAGE_SLOTS.div_ceil(cells_per_page(width)) + 1;
    pages + 3 * (pages / NODE_ENTRIES + 2)
}

/// Bytes of a plane's root.
pub(crate) const ROOT_BYTES: usize = 8;

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
}

impl Plane {
    /// Layout: kind | extent (run) or root page id u32 and depth u8 (table) | padding.
    pub(crate) fn encode(self) -> [u8; ROOT_BYTES] {
        match self {
            Self::Empty => [KIND_EMPTY, 0, 0, 0, 0, 0, 0, 0],
            Self::Run(extent) => {
                let [e0, e1, e2, e3, e4, e5, e6] = extent.encode();
                [KIND_RUN, e0, e1, e2, e3, e4, e5, e6]
            }
            Self::Table { root, depth } => {
                let [r0, r1, r2, r3] = root.get().to_le_bytes();
                [KIND_TABLE, r0, r1, r2, r3, depth, 0, 0]
            }
        }
    }

    pub(crate) fn decode(bytes: [u8; ROOT_BYTES]) -> Self {
        let [kind, b1, b2, b3, b4, b5, b6, b7] = bytes;
        match kind {
            KIND_EMPTY => Self::Empty,
            KIND_RUN => Self::Run(Extent::decode([b1, b2, b3, b4, b5, b6, b7])),
            KIND_TABLE if (1..=MAX_DEPTH).contains(&b5) => Self::Table {
                root: PageId::new(u32::from_le_bytes([b1, b2, b3, b4])),
                depth: b5,
            },
            _ => corrupt("value plane"),
        }
    }
}

/// Cells per value page.
fn cells_per_page(width: Width) -> u32 {
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
    let mut chunks: Vec<(u32, u32)> = (low / cpp..high.div_ceil(cpp))
        .map(|k| ((k * cpp).max(low), ((k + 1) * cpp).min(high)))
        .filter(|(a, b)| a < b)
        .collect();
    if descending {
        chunks.reverse();
    }
    let mut buf = Vec::new();
    for (a, b) in chunks {
        let len = usize::try_from(b - a).unwrap_or(0) * w;
        if buf.len() < len {
            buf.resize(len, 0);
        }
        let chunk = buf.get_mut(..len).unwrap_or(&mut []);
        // The cells with storage: a run may cover only the front of its chunk.
        let cell = cell_at(view, plane, width, a);
        let stored = match (plane, cell) {
            (_, None) | (Plane::Empty, _) => 0,
            (Plane::Run(extent), Some(_)) => {
                (extent.len() / u32::from(width.get())).saturating_sub(a)
            }
            (Plane::Table { .. }, Some(_)) => b - a,
        }
        .min(b - a);
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
    let cpp = cells_per_page(width);
    let next = match plane {
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
        Plane::Empty | Plane::Run(_) => plane,
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
        Plane::Empty => {}
        Plane::Run(extent) => store.release(extent),
        Plane::Table { root, depth } => release_node(store, root, depth),
    }
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
        Plane::Empty => {}
        Plane::Run(extent) => out.push(extent),
        Plane::Table { root, depth } => walk(view, root, depth, &mut out),
    }
    out
}

const _: () = assert!(EXTENT_BYTES + 1 == ROOT_BYTES);

#[cfg(test)]
mod tests;
