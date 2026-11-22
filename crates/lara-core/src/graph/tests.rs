use super::*;
use crate::descriptor::{PAGE_SLOTS, Paged, Slot, TINY_SLOTS};
use crate::id::Position;
use crate::memory::{PAGE_SIZE, VecMemory};
use crate::page_store::{ByteClass, Extent, Pages};
use crate::paged;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use core::num::NonZeroU32;
use core::ops::ControlFlow;

/// The backing memories of a test graph; graphs borrow them so they can be reopened.
#[derive(Default)]
struct Backing {
    memories: [VecMemory; 5],
}

impl Backing {
    fn memories(&self) -> Memories<&VecMemory> {
        let [
            vertices,
            edge_pages,
            edge_directory,
            list_pages,
            list_directory,
        ] = &self.memories;
        Memories::new(
            vertices,
            edge_pages,
            edge_directory,
            list_pages,
            list_directory,
        )
    }

    fn create(&self, n: u32) -> Graph<&VecMemory> {
        let mut g = Graph::create(self.memories()).unwrap();
        for _ in 0..n {
            g.add_vertex().unwrap();
        }
        g
    }

    fn open(&self) -> Graph<&VecMemory> {
        Graph::open(self.memories()).unwrap()
    }
}

fn v(i: u32) -> VertexId {
    VertexId::new(i)
}

fn l(i: u16) -> Label {
    Label::new(i)
}

fn p(i: u32) -> Position {
    Position::new(i)
}

/// The live edges of `x`'s `(label, side)` bucket as `(position, target)`.
fn edges_on<M: Memory>(g: &Graph<M>, x: VertexId, label: Label, side: Side) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    g.neighbors(x, label, side, &mut out).unwrap();
    out.into_iter()
        .map(|e| (e.position().index(), e.target().index()))
        .collect()
}

/// The live out-edges of `x` under `label`.
fn edges<M: Memory>(g: &Graph<M>, x: VertexId, label: Label) -> Vec<(u32, u32)> {
    edges_on(g, x, label, Side::Out)
}

fn out_ref(x: VertexId, label: Label, at: Position) -> EdgeRef {
    EdgeRef::new(x, label, Side::Out, at)
}

/// Inserts `from -[label]-> to` and returns the out half's position.
fn insert<M: Memory>(g: &mut Graph<M>, from: VertexId, label: Label, to: VertexId) -> Position {
    g.insert_directed(from, label, to)
        .unwrap()
        .source()
        .position()
}

/// Removes the directed edge whose out half is at `at`; returns its target if it was live.
fn remove<M: Memory>(g: &mut Graph<M>, from: VertexId, label: Label, at: Position) -> Option<u32> {
    match g.remove_edge(out_ref(from, label, at)).unwrap() {
        Removal::Removed(halves) => Some(halves.target().vertex().index()),
        Removal::Absent => None,
    }
}

fn out_bucket<M: Memory>(g: &Graph<M>, x: VertexId, label: Label) -> Option<Descriptor> {
    g.bucket(
        x,
        BucketKey {
            label,
            side: Side::Out,
        },
    )
    .unwrap()
}

/// Every structural invariant: sorted bucket keys, consistent counts, per-page tombstone counts,
/// paired halves, and exactly the extents the buckets and lists own are allocated, each once.
fn check<M: Memory>(g: &Graph<M>) {
    let mut edge_extents = BTreeSet::new();
    let mut list_extents = BTreeSet::new();
    // (owner, label, side, other end) → live halves.
    let mut halves: BTreeMap<(u32, Label, Side, u32), usize> = BTreeMap::new();
    for i in 0..g.vertex_count() {
        let list = g.list(v(i)).unwrap();
        if let Some(list) = list {
            assert!(
                list_extents.insert(key(labels::extent(list))),
                "list shared"
            );
        }
        let buckets = labels::buckets(&g.lists.pages(), list);
        assert!(
            buckets.windows(2).all(|w| w[0].0 < w[1].0),
            "bucket keys sorted and unique"
        );
        for (key, d) in buckets {
            for target in check_bucket(g, &d, &mut edge_extents) {
                *halves.entry((i, key.label, key.side, target)).or_default() += 1;
            }
        }
    }
    for (&(owner, label, side, other), &count) in &halves {
        let paired = halves.get(&(other, label, opposite(side), owner));
        assert_eq!(
            paired,
            Some(&count),
            "halves of {owner} {label:?} {side:?} → {other}"
        );
    }
    let allocated: BTreeSet<_> = g.edges.allocated().into_iter().map(key).collect();
    assert_eq!(
        allocated, edge_extents,
        "edge extents: none leaked or shared"
    );
    let allocated: BTreeSet<_> = g.lists.allocated().into_iter().map(key).collect();
    assert_eq!(
        allocated, list_extents,
        "list extents: none leaked or shared"
    );
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

fn key(e: Extent) -> (u32, u64, u8) {
    (e.page().get(), e.offset().get(), e.class().log2())
}

#[test]
fn a_new_graph_is_empty_and_rejects_unknown_vertices() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    assert_eq!(g.vertex_count(), 2);
    assert_eq!(g.degree(v(1), l(0), Side::Out), Ok(0));
    assert_eq!(
        g.insert_directed(v(0), l(0), v(2)),
        Err(Error::UnknownVertex(v(2)))
    );
    assert_eq!(
        g.insert_undirected(v(5), l(0), v(0)),
        Err(Error::UnknownVertex(v(5)))
    );
    assert_eq!(
        g.remove_edge(out_ref(v(0), l(0), p(0))),
        Ok(Removal::Absent)
    );
    assert_eq!(g.counterpart(out_ref(v(0), l(0), p(0))), Ok(None));
    check(&g);
}

#[test]
fn a_directed_edge_is_stored_at_both_ends() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let e = g.insert_directed(v(0), l(4), v(2)).unwrap();
    assert_eq!(e.source(), EdgeRef::new(v(0), l(4), Side::Out, p(0)));
    assert_eq!(e.target(), EdgeRef::new(v(2), l(4), Side::In, p(0)));
    assert_eq!(edges_on(&g, v(2), l(4), Side::In), [(0, 0)]);
    assert_eq!(g.edge(e.target()), Ok(Some(v(0))));
    assert_eq!(g.counterpart(e.source()), Ok(Some(e.target())));
    assert_eq!(g.counterpart(e.target()), Ok(Some(e.source())));
    let mut labels = Vec::new();
    g.labels(v(2), Side::In, &mut labels).unwrap();
    assert_eq!(labels, [l(4)]);
    g.labels(v(2), Side::Out, &mut labels).unwrap();
    assert_eq!(labels, Vec::<Label>::new());
    check(&g);
    assert_eq!(g.remove_edge(e.target()), Ok(Removal::Removed(e)));
    assert_eq!(g.degree(v(0), l(4), Side::Out), Ok(0));
    assert_eq!(g.degree(v(2), l(4), Side::In), Ok(0));
    check(&g);
}

#[test]
fn an_undirected_edge_is_ordered_by_vertex_and_a_self_loop_has_one_half() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let e = g.insert_undirected(v(2), l(0), v(1)).unwrap();
    assert_eq!(e.source(), EdgeRef::new(v(1), l(0), Side::Undirected, p(0)));
    assert_eq!(e.target(), EdgeRef::new(v(2), l(0), Side::Undirected, p(0)));
    assert_eq!(g.counterpart(e.target()), Ok(Some(e.source())));
    let s = g.insert_undirected(v(1), l(0), v(1)).unwrap();
    assert_eq!(s.source(), s.target());
    assert_eq!(s.source().position(), p(1));
    assert_eq!(g.counterpart(s.source()), Ok(Some(s.source())));
    assert_eq!(edges_on(&g, v(1), l(0), Side::Undirected), [(0, 2), (1, 1)]);
    check(&g);
    assert_eq!(g.remove_edge(s.target()), Ok(Removal::Removed(s)));
    assert_eq!(g.remove_edge(e.target()), Ok(Removal::Removed(e)));
    assert_eq!(g.degree(v(1), l(0), Side::Undirected), Ok(0));
    check(&g);
}

#[test]
fn a_directed_self_loop_has_an_out_and_an_in_half() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    let first = g.insert_directed(v(0), l(0), v(0)).unwrap();
    let second = g.insert_directed(v(0), l(0), v(0)).unwrap();
    assert_eq!(second.source(), EdgeRef::new(v(0), l(0), Side::Out, p(1)));
    assert_eq!(second.target(), EdgeRef::new(v(0), l(0), Side::In, p(1)));
    assert_eq!(g.remove_edge(first.target()), Ok(Removal::Removed(first)));
    assert_eq!(g.counterpart(second.target()), Ok(Some(second.source())));
    check(&g);
}

#[test]
fn parallel_edges_pair_by_rank() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let (a, b) = (v(0), v(1));
    // a→b at Out 0, 2, 3; other traffic interleaves the In side at b.
    let first = g.insert_directed(a, l(0), b).unwrap();
    g.insert_directed(a, l(0), v(2)).unwrap();
    g.insert_directed(v(2), l(0), b).unwrap();
    let second = g.insert_directed(a, l(0), b).unwrap();
    let third = g.insert_directed(a, l(0), b).unwrap();
    assert_eq!(second.target().position(), p(2));
    g.remove_edge(first.source()).unwrap();
    // A new a→b must not take hole 0 on either side: it would rank before the survivors.
    let fourth = g.insert_directed(a, l(0), b).unwrap();
    assert_eq!(fourth.source().position(), p(4));
    assert_eq!(fourth.target().position(), p(4));
    for e in [second, third, fourth] {
        assert_eq!(g.counterpart(e.source()), Ok(Some(e.target())));
        assert_eq!(g.counterpart(e.target()), Ok(Some(e.source())));
    }
    check(&g);
}

#[test]
fn visiting_neighbors_sees_live_edges_in_order_and_stops_on_break() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    for target in [1, 2, 1, 2] {
        insert(&mut g, v(0), l(0), v(target));
    }
    remove(&mut g, v(0), l(0), p(1));
    let mut seen = Vec::new();
    let none = g.visit_neighbors(v(0), l(0), Side::Out, |e| {
        seen.push((e.position().index(), e.target().index()));
        ControlFlow::<()>::Continue(())
    });
    assert_eq!(none, Ok(None));
    assert_eq!(seen, edges(&g, v(0), l(0)));
    let second_to_2 = g.visit_neighbors(v(0), l(0), Side::Out, |e| {
        if e.target() == v(2) {
            ControlFlow::Break(e.position())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(second_to_2, Ok(Some(p(3))));
    assert_eq!(
        g.visit_neighbors(v(1), l(5), Side::In, |_| ControlFlow::Break(())),
        Ok(None)
    );
}

#[test]
fn labels_are_separate_sorted_buckets() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    assert_eq!(insert(&mut g, v(0), l(7), v(1)), p(0));
    assert_eq!(insert(&mut g, v(0), l(2), v(2)), p(0));
    assert_eq!(insert(&mut g, v(0), l(7), v(2)), p(1));
    let mut labels = Vec::new();
    g.labels(v(0), Side::Out, &mut labels).unwrap();
    assert_eq!(labels, [l(2), l(7)]);
    assert_eq!(edges(&g, v(0), l(7)), [(0, 1), (1, 2)]);
    assert_eq!(edges(&g, v(0), l(2)), [(0, 2)]);
    assert_eq!(g.edge(out_ref(v(0), l(7), p(1))), Ok(Some(v(2))));
    check(&g);
}

#[test]
fn too_many_buckets_is_an_error_and_changes_nothing() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let max = u16::try_from(labels::MAX_BUCKETS).unwrap();
    for label in 0..max - 1 {
        insert(&mut g, v(0), l(label), v(1));
    }
    // A directed self-loop needs two new buckets at vertex 0; there is room for one.
    assert_eq!(
        g.insert_directed(v(0), l(max), v(0)),
        Err(Error::TooManyBuckets(v(0)))
    );
    // The in half's bucket is the full one: the out half must not land either.
    insert(&mut g, v(2), l(max), v(1));
    assert_eq!(
        g.insert_directed(v(0), l(max + 1), v(1)),
        Err(Error::TooManyBuckets(v(1)))
    );
    assert_eq!(g.degree(v(0), l(max), Side::Out), Ok(0));
    assert_eq!(g.degree(v(0), l(max + 1), Side::Out), Ok(0));
    insert(&mut g, v(0), l(max), v(2));
    check(&g);
}

#[test]
fn an_insert_that_runs_out_of_memory_changes_nothing() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    for _ in 0..PAGE_SLOTS {
        insert(&mut g, v(0), l(0), v(1));
    }
    let before = (edges(&g, v(0), l(0)), edges_on(&g, v(1), l(0), Side::In));
    // Leave room for three more pages: promoting both full runs to pages takes two each, so the
    // out half alone would fit and the in half would not.
    let memory = &backing.memories[1];
    memory.cap(memory.size());
    let capacity = u32::try_from(memory.size() * (PAGE_SIZE / 4096)).unwrap();
    let mut held = Vec::new();
    while capacity - g.edges.minted() > 3 {
        held.push(g.edges.allocate(ByteClass::PAGE).unwrap());
    }
    let error = g.insert_directed(v(0), l(0), v(1));
    assert!(matches!(error, Err(Error::OutOfMemory(_))), "{error:?}");
    assert_eq!(
        (edges(&g, v(0), l(0)), edges_on(&g, v(1), l(0), Side::In)),
        before
    );
    for extent in held {
        g.edges.release(extent);
    }
    check(&g);
}

#[test]
fn removal_keeps_positions_and_a_hole_is_reused_only_after_equal_targets() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let (t, x) = (v(1), v(2));
    // [a→t, b→x, c→t, d→x]
    for target in [t, x, t, x] {
        insert(&mut g, v(0), l(0), target);
    }
    assert_eq!(remove(&mut g, v(0), l(0), p(1)), Some(2));
    assert_eq!(remove(&mut g, v(0), l(0), p(1)), None);
    assert_eq!(
        edges(&g, v(0), l(0)),
        [(0, 1), (2, 1), (3, 2)],
        "positions unchanged"
    );
    // Hole 1 lies before c→t: a new →t must not take it.
    assert_eq!(insert(&mut g, v(0), l(0), t), p(4));
    assert_eq!(remove(&mut g, v(0), l(0), p(4)), Some(1));
    assert_eq!(
        insert(&mut g, v(0), l(0), v(0)),
        p(1),
        "no later edge to vertex 0: hole reused"
    );
    check(&g);
}

#[test]
fn a_bucket_moves_from_tiny_to_run_to_pages() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    let total = PAGE_SLOTS * 3 + 5;
    for i in 0..total {
        assert_eq!(insert(&mut g, v(0), l(0), v(i % 2)), p(i));
        if matches!(i + 1, 3 | 4 | 8 | 9 | 1024 | 1025 | 2048 | 2049) {
            check(&g);
        }
    }
    assert!(matches!(
        out_bucket(&g, v(0), l(0)),
        Some(Descriptor::Paged(_))
    ));
    assert_eq!(edges(&g, v(0), l(0)).len(), usize::try_from(total).unwrap());
    // Kill a whole page: the scan skips it, positions elsewhere stay.
    for i in PAGE_SLOTS..2 * PAGE_SLOTS {
        remove(&mut g, v(0), l(0), p(i));
    }
    let live = edges(&g, v(0), l(0));
    assert_eq!(live.len(), usize::try_from(total - PAGE_SLOTS).unwrap());
    assert!(
        live.iter()
            .all(|&(pos, _)| !(PAGE_SLOTS..2 * PAGE_SLOTS).contains(&pos))
    );
    check(&g);
}

#[test]
fn a_full_run_becomes_the_first_page_without_moving() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    for _ in 0..PAGE_SLOTS {
        insert(&mut g, v(0), l(0), v(1));
    }
    let Some(Descriptor::Run(run)) = out_bucket(&g, v(0), l(0)) else {
        panic!("a full run")
    };
    insert(&mut g, v(0), l(0), v(1));
    let Some(Descriptor::Paged(paged)) = out_bucket(&g, v(0), l(0)) else {
        panic!("paged")
    };
    assert_eq!(
        paged::leaves(&g.edges.pages(), &paged)[0].page,
        Some(run.extent.page())
    );
    check(&g);
}

#[test]
fn the_page_index_deepens_past_one_page_of_entries() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    let total = (paged::FANOUT + 2) * PAGE_SLOTS + 3;
    let mut most = 0;
    for _ in 0..total {
        let before = g.edges.minted();
        // An undirected self-loop: one half, one bucket insert.
        g.insert_undirected(v(0), l(0), v(0)).unwrap();
        most = most.max(g.edges.minted() - before);
    }
    assert!(
        most <= bucket::PAGES_PER_INSERT,
        "an insert took {most} pages"
    );
    check(&g);
    let positions: Vec<u32> = edges_on(&g, v(0), l(0), Side::Undirected)
        .into_iter()
        .map(|(pos, _)| pos)
        .collect();
    assert_eq!(positions, (0..total).collect::<Vec<_>>());
    for at in [0, PAGE_SLOTS * paged::FANOUT, total - 1] {
        let half = EdgeRef::new(v(0), l(0), Side::Undirected, p(at));
        assert_eq!(g.edge(half), Ok(Some(v(0))));
    }
    check(&backing.open());
}

#[test]
fn reopen_reads_the_same_graph() {
    let backing = Backing::default();
    let snapshot = |g: &Graph<&VecMemory>| {
        (0..4)
            .map(|x| {
                (0..3)
                    .flat_map(|y| [Side::Out, Side::In].map(|s| edges_on(g, v(x), l(y), s)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let before = {
        let mut g = backing.create(4);
        for i in 0..3_000 {
            insert(
                &mut g,
                v(i % 4),
                l(u16::try_from(i % 3).unwrap()),
                v((i * 7) % 4),
            );
        }
        remove(&mut g, v(1), l(1), p(3));
        snapshot(&g)
    };
    let mut g = backing.open();
    assert_eq!(snapshot(&g), before);
    check(&g);
    insert(&mut g, v(2), l(1), v(1));
    check(&g);
}

#[test]
fn open_and_create_reject_the_wrong_memories() {
    let empty = Backing::default();
    assert!(matches!(
        Graph::open(empty.memories()),
        Err(OpenError::NotAGraph)
    ));
    let used = Backing::default();
    used.memories[0].grow(1).unwrap();
    assert!(matches!(
        Graph::create(used.memories()),
        Err(CreateError::NotEmpty)
    ));
}

mod compaction;
mod detach;
mod model;
