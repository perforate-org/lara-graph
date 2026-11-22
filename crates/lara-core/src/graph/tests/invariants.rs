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
        check_values(g, &bucket, &mut seen.edges);
        for target in check_bucket(g, &bucket.slots, &mut seen.edges) {
            let entry = (i, bucket_key.label, bucket_key.side, target);
            *seen.halves.entry(entry).or_default() += 1;
        }
    }
}

/// A bucket with values is never tiny, and owns its header and value plane.
fn check_values<M: Memory>(g: &Graph<M>, bucket: &Bucket, owned: &mut BTreeSet<(u32, u64, u8)>) {
    let Some(cells) = bucket.values else {
        return;
    };
    assert!(
        !matches!(bucket.slots, Descriptor::Tiny(_)),
        "a valued bucket is tiny"
    );
    let plane = values::extents(&g.edges.pages(), cells.plane);
    for extent in core::iter::once(cells.header).chain(plane) {
        assert!(owned.insert(key(extent)), "header or value extent shared");
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
