//! Merging sparse neighbouring leaves, so that an index whose postings were mostly removed gives
//! its pages back. Compaction does it a pair at a time once the bucket's slots need nothing.

use super::Root;
use super::node::{Child, FANOUT, children, counts, node_len, page_at, read_into, write_node};
use crate::memory::Memory;
use crate::page_store::{Extent, PageId, PageStore, Pages};
use crate::targets::{Posting, leaf};
use alloc::vec::Vec;

/// Neighbouring leaves whose counts sum to this or less are merged. A count is a leaf's live
/// edges, so after merging every pair of neighbours indexes more than this many edges: an index
/// page per at least half of it. A leaf holds 700 to 1200 postings with a count each, so the
/// merged postings fit one. Tests use leaves of a quarter page.
const MERGE_AT: u32 = crate::scale::Scale::pick(64, 512);

/// What a merge step did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Merged {
    /// No pair is sparse enough.
    Nothing,
    /// Two leaves became one; the tree's root is this now.
    Into(Root),
}

/// Merges one sparse pair of neighbours, going down from the root: at each node, the pair of
/// children with the smallest counts merges when it is sparse enough (leaves whose counts sum to
/// [`MERGE_AT`] or less, or nodes whose children fit one node); otherwise the walk goes into the
/// child with the fewest edges per child of its own. A root left with one child then gives way
/// to it. Bounded: it reads one node per level with its children's lengths, and the pair; it
/// rewrites one of the pair and its parent, and only releases pages.
pub(crate) fn merge_step<M: Memory>(store: &mut PageStore<M>, root: Root) -> Merged {
    let (mut node, mut level) = (root.page, root.depth);
    while level > 0 {
        if merge_pair(store, node, level) {
            return Merged::Into(collapsed(store, root));
        }
        let next = (level > 1)
            .then(|| sparsest_child(&store.pages(), node))
            .flatten();
        let Some(next) = next else {
            break;
        };
        node = next;
        level -= 1;
    }
    Merged::Nothing
}

/// Merges the pair of `node`'s children with the smallest counts when it is sparse enough;
/// whether it did. `level` is the node's: 1 for a parent of leaves.
fn merge_pair<M: Memory>(store: &mut PageStore<M>, node: PageId, level: u8) -> bool {
    let view = store.pages();
    let sums = counts(&view, node, node_len(&view, node));
    let Some((i, sum)) = sparsest_pair(&sums) else {
        return false;
    };
    let mut kids = children(&view, node);
    let (Some(&a), Some(&b)) = (kids.get(i), kids.get(i + 1)) else {
        return false;
    };
    let merged = if level == 1 {
        sum <= MERGE_AT && merge_leaves(store, a, b)
    } else {
        merge_nodes(store, a, b)
    };
    if merged {
        if let Some(kid) = kids.get_mut(i) {
            *kid = Child { count: sum, ..a };
        }
        kids.remove(i + 1);
        write_node(store, node, &kids);
        store.release(Extent::whole(b.page));
    }
    merged
}

/// The neighbouring pair with the smallest summed count, as its first index and sum.
fn sparsest_pair(sums: &[u32]) -> Option<(usize, u32)> {
    sums.windows(2)
        .enumerate()
        .filter_map(|(i, w)| match w {
            [a, b] => Some((i, a + b)),
            _ => None,
        })
        .min_by_key(|&(_, sum)| sum)
}

/// The internal child of `node` with the fewest edges per child of its own, among those with two
/// or more: where a merge below is most likely.
fn sparsest_child<M: Memory>(view: &Pages<'_, M>, node: PageId) -> Option<PageId> {
    children(view, node)
        .into_iter()
        .map(|c| (c, node_len(view, c.page)))
        .filter(|&(_, len)| len >= 2)
        .min_by(|(a, a_len), (b, b_len)| {
            let per = |c: &Child, len: usize| u64::from(c.count) * u64::try_from(len).unwrap_or(0);
            // a.count / a_len against b.count / b_len, without dividing.
            per(a, *b_len).cmp(&per(b, *a_len))
        })
        .map(|(c, _)| c.page)
}

/// Writes the postings of leaves `a` and `b` into `a`'s page, when they fit one leaf. With no
/// counts left, `a` stays as it is: a leaf is never empty.
fn merge_leaves<M: Memory>(store: &PageStore<M>, a: Child, b: Child) -> bool {
    let view = store.pages();
    let postings: Vec<Posting> = [a, b]
        .iter()
        .flat_map(|c| {
            leaf::counted(&|offset, buf: &mut [u8]| read_into(&view, c.page, offset, buf))
        })
        .collect();
    if postings.is_empty() {
        return true;
    }
    let images = leaf::packed(&postings);
    let [image] = images.as_slice() else {
        return false;
    };
    store.write(a.page.offset(), &image.bytes);
    true
}

/// Writes the children of nodes `a` and `b` into `a`'s page, when together they fit one node.
fn merge_nodes<M: Memory>(store: &PageStore<M>, a: Child, b: Child) -> bool {
    let view = store.pages();
    if node_len(&view, a.page) + node_len(&view, b.page) > FANOUT {
        return false;
    }
    let mut kids = children(&view, a.page);
    kids.extend(children(&view, b.page));
    write_node(store, a.page, &kids);
    true
}

/// The root, after a root with a single child gives way to it.
fn collapsed<M: Memory>(store: &mut PageStore<M>, mut root: Root) -> Root {
    while root.depth > 0 && node_len(&store.pages(), root.page) == 1 {
        let child = page_at(&store.pages(), root.page, 0);
        store.release(Extent::whole(root.page));
        root = Root {
            page: child,
            depth: root.depth - 1,
        };
    }
    root
}
