//! The structural invariants every graph test checks.

use super::*;

/// Every structural invariant: sorted bucket keys, consistent counts, per-page tombstone counts,
/// paired halves, and exactly the extents the buckets and lists own are allocated, each once.
pub(super) fn check<M: Memory>(g: &Graph<M>) {
    let mut seen = Seen::default();
    for i in 0..g.vertex_count() {
        check_vertex(g, i, &mut seen);
    }
    if let Some(extent) = g.widths_extent() {
        assert!(seen.lists.insert(key(extent)), "width table shared");
    }
    check_pairs(&seen.halves);
    let allocated: BTreeSet<_> = g.edges.allocated().into_iter().map(key).collect();
    assert_eq!(allocated, seen.edges, "edge extents: none leaked or shared");
    let allocated: BTreeSet<_> = g.lists.allocated().into_iter().map(key).collect();
    assert_eq!(allocated, seen.lists, "list extents: none leaked or shared");
}

/// What [`check`] has seen so far.
#[derive(Default)]
struct Seen {
    edges: BTreeSet<(u32, u64, u8)>,
    lists: BTreeSet<(u32, u64, u8)>,
    /// (owner, label, side, other end) → live halves.
    halves: BTreeMap<(u32, Label, Side, u32), usize>,
}

fn check_vertex<M: Memory>(g: &Graph<M>, i: u32, seen: &mut Seen) {
    let list = g.list(v(i)).unwrap();
    if let Some(list) = list {
        labels::check_shape(&g.lists.pages(), list);
        for extent in labels::extents(&g.lists.pages(), list) {
            assert!(seen.lists.insert(key(extent)), "list extent shared");
        }
    }
    let buckets = labels::buckets(&g.lists.pages(), list);
    assert!(
        buckets.windows(2).all(|w| w[0].0 < w[1].0),
        "bucket keys sorted and unique"
    );
    for (bucket_key, stored) in buckets {
        let bucket = Bucket::resolve(&g.edges.pages(), stored);
        let state = g.record(v(i)).unwrap().state;
        check_values(g, &bucket, state, &mut seen.edges);
        for target in check_bucket(g, &bucket.slots, &mut seen.edges) {
            let entry = (i, bucket_key.label, bucket_key.side, target);
            *seen.halves.entry(entry).or_default() += 1;
        }
    }
}

/// A bucket has a header exactly when it has values or a target index; a bucket with values is
/// never tiny; a paged bucket's index holds exactly the live edges of its complete blocks (and
/// is empty when there are none), unless it was dropped while its vertex detaches; the header,
/// value plane and index are owned once.
fn check_values<M: Memory>(
    g: &Graph<M>,
    bucket: &Bucket,
    state: VertexState,
    owned: &mut BTreeSet<(u32, u64, u8)>,
) {
    let view = g.edges.pages();
    assert_eq!(
        bucket.header.is_some(),
        bucket.extended(),
        "header iff values or index"
    );
    if bucket.values.is_some() {
        assert!(
            !matches!(bucket.slots, Descriptor::Tiny(_)),
            "a valued bucket is tiny"
        );
    }
    if bucket.index == Index::Dropped {
        assert_eq!(
            state,
            VertexState::Detaching,
            "an index dropped outside a detach"
        );
        assert!(
            matches!(bucket.slots, Descriptor::Paged(_)),
            "a dropped index on a small bucket"
        );
    }
    // What the index must cover, worked out here rather than through the code under test: a
    // paged bucket's complete blocks, unless its index was dropped.
    let end = match (&bucket.slots, bucket.index) {
        (Descriptor::Paged(_), Index::Empty | Index::Tree(_)) => {
            bucket.slots.stored() / targets::BLOCK * targets::BLOCK
        }
        (Descriptor::Paged(_), Index::Dropped) | (Descriptor::Tiny(_) | Descriptor::Run(_), _) => 0,
    };
    let mut expected: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut live = Vec::new();
    bucket::read(&bucket.slots, &view, &mut live);
    for e in live.iter().filter(|e| e.position().index() < end) {
        let key = (e.target().index(), e.position().index() / targets::BLOCK);
        *expected.entry(key).or_default() += 1;
    }
    let postings: BTreeMap<(u32, u32), u32> = targets::postings(&view, bucket.index.tree())
        .into_iter()
        .map(|p| (p.key(), p.count))
        .collect();
    assert_eq!(postings, expected, "target index matches the slots");
    // Removals leave zero counts in place, so a tree may outlive its last indexed edge.
    if !expected.is_empty() {
        assert!(
            matches!(bucket.index, Index::Tree(_)),
            "indexed live edges without an index tree"
        );
    }
    let plane = bucket
        .values
        .map_or_else(Vec::new, |cells| values::extents(&view, cells.plane));
    let index = targets::extents(&view, bucket.index.tree());
    for extent in bucket.header.into_iter().chain(plane).chain(index) {
        assert!(
            owned.insert(key(extent)),
            "header, value or index extent shared"
        );
    }
}

/// Checks one bucket; returns its live targets.
fn check_bucket<M: Memory>(
    g: &Graph<M>,
    d: &Descriptor,
    owned: &mut BTreeSet<(u32, u64, u8)>,
) -> Vec<u32> {
    let view = g.edges.pages();
    let mut out = Vec::new();
    bucket::read(d, &view, &mut out);
    assert_eq!(u32::try_from(out.len()).unwrap(), d.live(), "live count");
    assert!(d.live() <= d.stored());
    match d {
        Descriptor::Tiny(_) => assert!(d.stored() <= u32::try_from(TINY_SLOTS).unwrap()),
        Descriptor::Run(run) => assert!(u32::from(run.stored) <= run.capacity()),
        Descriptor::Paged(paged) => check_page_tombstones(&view, paged),
    }
    for extent in bucket::owned_extents(d, &view) {
        assert!(owned.insert(key(extent)), "edge extent shared");
    }
    out.iter().map(|e| e.target().index()).collect()
}

/// Each data page's recorded tombstone count matches its slots.
fn check_page_tombstones<M: Memory>(view: &Pages<'_, M>, paged: &Paged) {
    assert!(paged.stored > PAGE_SLOTS);
    for (number, leaf) in (0u32..).zip(paged::leaves(view, paged)) {
        let count = (paged.stored - number * PAGE_SLOTS).min(PAGE_SLOTS);
        let dead = match leaf.page {
            Some(page) => {
                let bytes = view.read(page.offset(), usize::try_from(count * 4).unwrap());
                Slot::decode_all(&bytes)
                    .filter(|s| *s == Slot::Tombstone)
                    .count()
            }
            None => usize::try_from(count).unwrap(),
        };
        assert_eq!(
            usize::from(leaf.tombstones),
            dead,
            "page {number} tombstones"
        );
    }
}

/// Every bucket's live entries to a vertex are matched by as many entries back.
fn check_pairs(halves: &BTreeMap<(u32, Label, Side, u32), usize>) {
    for (&(owner, label, side, other), &count) in halves {
        let paired = halves.get(&(other, label, opposite(side), owner));
        assert_eq!(
            paired,
            Some(&count),
            "halves of {owner} {label:?} {side:?} → {other}"
        );
    }
}

pub(super) fn key(e: Extent) -> (u32, u64, u8) {
    (e.page().get(), e.offset().get(), e.class().log2())
}
