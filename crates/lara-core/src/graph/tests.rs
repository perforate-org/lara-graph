use super::*;
use crate::descriptor::{PAGE_SLOTS, Paged, Slot, TINY_SLOTS};
use crate::id::{Cursor, Halves, Order, Placement, Position, Width};
use crate::memory::{PAGE_SIZE, VecMemory};
use crate::page_store::{ByteClass, Extent, Pages};
use crate::paged;
use crate::stored::{Bucket, Index};
use crate::targets;
use crate::values;
use crate::vertex_set::VertexSet;
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
    g.insert_directed(from, label, to, Placement::ReuseHoles, &[])
        .unwrap()
        .canonical()
        .position()
}

/// The report of removing the edge with `halves` and no value.
fn removed(halves: Halves) -> Removal {
    Removal::Removed(RemovedEdge::new(halves, Vec::new()))
}

/// Removes the directed edge whose out half is at `at`; returns its target if it was live.
fn remove<M: Memory>(g: &mut Graph<M>, from: VertexId, label: Label, at: Position) -> Option<u32> {
    match g.remove_edge(out_ref(from, label, at)).unwrap() {
        Removal::Removed(r) => Some(r.halves().other().vertex().index()),
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

#[test]
fn a_new_graph_is_empty_and_rejects_unknown_vertices() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    assert_eq!(g.vertex_count(), 2);
    assert_eq!(g.degree(v(1), l(0), Side::Out), Ok(0));
    assert_eq!(
        g.insert_directed(v(0), l(0), v(2), Placement::ReuseHoles, &[]),
        Err(Error::UnknownVertex(v(2)))
    );
    assert_eq!(
        g.insert_undirected(v(5), l(0), v(0), Placement::ReuseHoles, &[]),
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
    let e = g
        .insert_directed(v(0), l(4), v(2), Placement::ReuseHoles, &[])
        .unwrap();
    assert_eq!(e.canonical(), EdgeRef::new(v(0), l(4), Side::Out, p(0)));
    assert_eq!(e.other(), EdgeRef::new(v(2), l(4), Side::In, p(0)));
    assert_eq!(edges_on(&g, v(2), l(4), Side::In), [(0, 0)]);
    assert_eq!(g.edge(e.other()), Ok(Some(v(0))));
    assert_eq!(g.counterpart(e.canonical()), Ok(Some(e.other())));
    assert_eq!(g.counterpart(e.other()), Ok(Some(e.canonical())));
    let mut labels = Vec::new();
    g.labels(v(2), Side::In, &mut labels).unwrap();
    assert_eq!(labels, [l(4)]);
    g.labels(v(2), Side::Out, &mut labels).unwrap();
    assert_eq!(labels, Vec::<Label>::new());
    check(&g);
    assert_eq!(g.remove_edge(e.other()), Ok(removed(e)));
    assert_eq!(g.degree(v(0), l(4), Side::Out), Ok(0));
    assert_eq!(g.degree(v(2), l(4), Side::In), Ok(0));
    check(&g);
}

/// `has_edges` sees edges at either end, on any side and in buckets with values, and looks past
/// buckets that hold only holes.
#[test]
fn has_edges_sees_every_side_and_looks_past_buckets_of_holes() {
    let backing = Backing::default();
    let mut g = backing.create(6);
    g.declare_width(l(3), Width::new(2).unwrap()).unwrap();
    assert_eq!(g.has_edges(v(0)), Ok(false));
    let directed = g
        .insert_directed(v(0), l(1), v(1), Placement::ReuseHoles, &[])
        .unwrap();
    let undirected = g
        .insert_undirected(v(2), l(2), v(3), Placement::Append, &[])
        .unwrap();
    let valued = g
        .insert_directed(v(4), l(3), v(5), Placement::Append, &[1, 2])
        .unwrap();
    for x in 0..6 {
        assert_eq!(g.has_edges(v(x)), Ok(true), "vertex {x}");
    }
    for halves in [directed, undirected, valued] {
        g.remove_edge(halves.other()).unwrap();
    }
    let mut labels = Vec::new();
    g.labels(v(0), Side::Out, &mut labels).unwrap();
    assert_eq!(labels, [l(1)], "a bucket of holes stays listed");
    for x in 0..6 {
        assert_eq!(g.has_edges(v(x)), Ok(false), "vertex {x}");
    }
    assert_eq!(g.has_edges(v(9)), Err(Error::UnknownVertex(v(9))));
    check(&g);
}

/// Compacting a hub whose edges were mostly removed packs its pages, then merges its index's
/// sparse leaves: the index gives most of its pages back and still pairs every half.
#[test]
fn compacting_a_mostly_emptied_hub_shrinks_its_index() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    let hub = g.add_vertex().unwrap();
    let mut edges = Vec::new();
    for _ in 0..6000 {
        let leaf = g.add_vertex().unwrap();
        edges.push(
            g.insert_directed(hub, l(0), leaf, Placement::Append, &[])
                .unwrap(),
        );
    }
    let index_pages = |g: &Graph<_>| {
        let found = g
            .find_bucket(
                hub,
                BucketKey {
                    label: l(0),
                    side: Side::Out,
                },
            )
            .unwrap()
            .unwrap();
        targets::extents(&g.edges.pages(), found.bucket.index.tree()).len()
    };
    let before = index_pages(&g);
    for (i, halves) in edges.iter().enumerate() {
        if i % 16 != 0 {
            g.remove_edge(halves.other()).unwrap();
        }
    }
    while g.compact(hub, l(0), Side::Out).unwrap().progress() == Progress::Remaining {}
    let after = index_pages(&g);
    assert!(after * 2 < before, "{after} index pages of {before}");
    for halves in edges.iter().step_by(16) {
        let canonical = g.counterpart(halves.other()).unwrap().unwrap();
        assert_eq!(g.counterpart(canonical), Ok(Some(halves.other())));
    }
    check(&g);
}

#[test]
fn an_undirected_edges_canonical_half_is_at_the_larger_vertex_and_a_self_loop_has_one_half() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let e = g
        .insert_undirected(v(2), l(0), v(1), Placement::ReuseHoles, &[])
        .unwrap();
    // The canonical half is at the larger vertex.
    assert_eq!(
        e.canonical(),
        EdgeRef::new(v(2), l(0), Side::Undirected, p(0))
    );
    assert_eq!(e.other(), EdgeRef::new(v(1), l(0), Side::Undirected, p(0)));
    assert_eq!(g.counterpart(e.other()), Ok(Some(e.canonical())));
    let s = g
        .insert_undirected(v(1), l(0), v(1), Placement::ReuseHoles, &[])
        .unwrap();
    assert_eq!(s.canonical(), s.other());
    assert_eq!(s.canonical().position(), p(1));
    assert_eq!(g.counterpart(s.canonical()), Ok(Some(s.canonical())));
    assert_eq!(edges_on(&g, v(1), l(0), Side::Undirected), [(0, 2), (1, 1)]);
    check(&g);
    assert_eq!(g.remove_edge(s.other()), Ok(removed(s)));
    assert_eq!(g.remove_edge(e.other()), Ok(removed(e)));
    assert_eq!(g.degree(v(1), l(0), Side::Undirected), Ok(0));
    check(&g);
}

#[test]
fn a_directed_self_loop_has_an_out_and_an_in_half() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    let first = g
        .insert_directed(v(0), l(0), v(0), Placement::ReuseHoles, &[])
        .unwrap();
    let second = g
        .insert_directed(v(0), l(0), v(0), Placement::ReuseHoles, &[])
        .unwrap();
    assert_eq!(
        second.canonical(),
        EdgeRef::new(v(0), l(0), Side::Out, p(1))
    );
    assert_eq!(second.other(), EdgeRef::new(v(0), l(0), Side::In, p(1)));
    assert_eq!(g.remove_edge(first.other()), Ok(removed(first)));
    assert_eq!(g.counterpart(second.other()), Ok(Some(second.canonical())));
    check(&g);
}

#[test]
fn parallel_edges_pair_by_rank() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let (a, b) = (v(0), v(1));
    // a→b at Out 0, 2, 3; other traffic interleaves the In side at b.
    let first = g
        .insert_directed(a, l(0), b, Placement::ReuseHoles, &[])
        .unwrap();
    g.insert_directed(a, l(0), v(2), Placement::ReuseHoles, &[])
        .unwrap();
    g.insert_directed(v(2), l(0), b, Placement::ReuseHoles, &[])
        .unwrap();
    let second = g
        .insert_directed(a, l(0), b, Placement::ReuseHoles, &[])
        .unwrap();
    let third = g
        .insert_directed(a, l(0), b, Placement::ReuseHoles, &[])
        .unwrap();
    assert_eq!(second.other().position(), p(2));
    g.remove_edge(first.canonical()).unwrap();
    // A new a→b must not take hole 0 on either side: it would rank before the survivors.
    let fourth = g
        .insert_directed(a, l(0), b, Placement::ReuseHoles, &[])
        .unwrap();
    assert_eq!(fourth.canonical().position(), p(4));
    assert_eq!(fourth.other().position(), p(4));
    for e in [second, third, fourth] {
        assert_eq!(g.counterpart(e.canonical()), Ok(Some(e.other())));
        assert_eq!(g.counterpart(e.other()), Ok(Some(e.canonical())));
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
    let none = g.visit_neighbors(v(0), l(0), Side::Out, Order::Ascending, |e| {
        seen.push((e.position().index(), e.target().index()));
        ControlFlow::<()>::Continue(())
    });
    assert_eq!(none, Ok(None));
    assert_eq!(seen, edges(&g, v(0), l(0)));
    let second_to_2 = g.visit_neighbors(v(0), l(0), Side::Out, Order::Ascending, |e| {
        if e.target() == v(2) {
            ControlFlow::Break(e.position())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(second_to_2, Ok(Some(p(3))));
    assert_eq!(
        g.visit_neighbors(v(1), l(5), Side::In, Order::Ascending, |_| {
            ControlFlow::Break(())
        }),
        Ok(None)
    );
}

#[test]
fn appending_never_fills_a_hole_in_any_bucket_form() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    for total in [3, 10, PAGE_SLOTS + 10] {
        let label = l(u16::try_from(total % 1000).unwrap());
        for _ in 0..total {
            insert(&mut g, v(0), label, v(1));
        }
        remove(&mut g, v(0), label, p(total - 2));
        // An edge to another vertex could safely take the hole; appending must not.
        let appended = g
            .insert_directed(v(0), label, v(2), Placement::Append, &[])
            .unwrap();
        assert_eq!(appended.canonical().position(), p(total), "{total} edges");
        // No live edge to vertex 0 follows the hole, so reusing it is safe.
        let reused = g
            .insert_directed(v(0), label, v(0), Placement::ReuseHoles, &[])
            .unwrap();
        assert_eq!(reused.canonical().position(), p(total - 2), "{total} edges");
    }
    check(&g);
}

#[test]
fn windows_count_holes_and_hand_back_the_next_cursor() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    for _ in 0..10 {
        insert(&mut g, v(0), l(0), v(1));
    }
    for at in [2, 3, 4] {
        remove(&mut g, v(0), l(0), p(at));
    }
    let span = NonZeroU32::new(4).unwrap();
    let window = |g: &Graph<&VecMemory>, cursor| {
        let mut seen = Vec::new();
        let next = g
            .visit_window(v(0), l(0), Side::Out, cursor, |e| {
                seen.push(e.position().index());
            })
            .unwrap();
        (seen, next)
    };
    // Ascending: positions 0..4 hold two live edges; holes count toward the span.
    let (seen, next) = window(&g, Cursor::first(Order::Ascending, span));
    assert_eq!(seen, [0, 1]);
    assert_eq!(next, Some(Cursor::at(Order::Ascending, p(4), span)));
    let (seen, next) = window(&g, Cursor::at(Order::Ascending, p(8), span));
    assert_eq!((seen, next), (vec![8, 9], None));
    // Descending starts at the last position; a start past the end is clamped.
    let (seen, next) = window(&g, Cursor::first(Order::Descending, span));
    assert_eq!(seen, [9, 8, 7, 6]);
    assert_eq!(next, Some(Cursor::at(Order::Descending, p(5), span)));
    let (seen, _) = window(&g, Cursor::at(Order::Descending, p(99), span));
    assert_eq!(seen, [9, 8, 7, 6]);
    let (seen, next) = window(&g, Cursor::at(Order::Descending, p(1), span));
    assert_eq!((seen, next), (vec![1, 0], None));
    // A bucket that does not exist has no window.
    let empty = g
        .visit_window(
            v(1),
            l(9),
            Side::In,
            Cursor::first(Order::Ascending, span),
            |_| {},
        )
        .unwrap();
    assert_eq!(empty, None);
}

#[test]
fn positional_reads_see_holes_released_pages_and_the_end() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    for _ in 0..3 * PAGE_SLOTS {
        insert(&mut g, v(0), l(0), v(1));
    }
    // Empty the middle page and compact once: it is released and its index entry left hollow.
    for at in PAGE_SLOTS..2 * PAGE_SLOTS {
        remove(&mut g, v(0), l(0), p(at));
    }
    g.compact(v(0), l(0), Side::Out).unwrap();
    remove(&mut g, v(0), l(0), p(5));
    let positions = [
        p(3 * PAGE_SLOTS),
        p(5),
        p(PAGE_SLOTS + 7),
        p(4),
        p(2 * PAGE_SLOTS),
    ];
    let mut out = Vec::new();
    g.targets_at(v(0), l(0), Side::Out, &positions, &mut out)
        .unwrap();
    assert_eq!(out, [None, None, None, Some(v(1)), Some(v(1))]);
    g.targets_at(v(1), l(3), Side::Out, &positions, &mut out)
        .unwrap();
    assert_eq!(out, [None; 5]);
}

#[test]
fn label_widths_are_declared_once_and_survive_a_reopen() {
    let backing = Backing::default();
    let w = |bytes| Width::new(bytes).unwrap();
    {
        let mut g = backing.create(1);
        assert_eq!(g.width(l(3)), None);
        g.declare_width(l(3), w(8)).unwrap();
        g.declare_width(l(1), w(4096)).unwrap();
        g.declare_width(l(3), w(8)).unwrap();
        assert_eq!(
            g.declare_width(l(3), w(16)),
            Err(Error::WidthConflict {
                label: l(3),
                declared: w(8)
            })
        );
        check(&g);
    }
    let g = backing.open();
    assert_eq!(
        (g.width(l(1)), g.width(l(3)), g.width(l(2))),
        (Some(w(4096)), Some(w(8)), None)
    );
    check(&g);
    assert_eq!(Width::new(0), None);
    assert_eq!(Width::new(4097), None);
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
    let mut g = backing.create(4);
    let room = |g: &Graph<&VecMemory>, x: u32, adding| {
        labels::has_room(&g.lists.pages(), g.list(v(x)).unwrap(), adding)
    };
    // Grow vertex 0 until its tiered list has room for one more bucket, not two.
    let mut label = 0;
    while room(&g, 0, 2) {
        insert(&mut g, v(0), l(label), v(2));
        label += 1;
    }
    assert!(room(&g, 0, 1));
    // A directed self-loop needs two new buckets at vertex 0.
    assert_eq!(
        g.insert_directed(v(0), l(label), v(0), Placement::ReuseHoles, &[]),
        Err(Error::TooManyBuckets(v(0)))
    );
    // Fill vertex 1; then an edge 0 → 1 fails on its in half and must not land at 0 either.
    let mut other = 0;
    while room(&g, 1, 1) {
        insert(&mut g, v(1), l(other), v(3));
        other += 1;
    }
    assert_eq!(
        g.insert_directed(v(0), l(label), v(1), Placement::ReuseHoles, &[]),
        Err(Error::TooManyBuckets(v(1)))
    );
    assert_eq!(g.degree(v(0), l(label), Side::Out), Ok(0));
    check(&g);
    insert(&mut g, v(0), l(label), v(2));
    check(&g);
    check(&backing.open());
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
    let error = g.insert_directed(v(0), l(0), v(1), Placement::ReuseHoles, &[]);
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
        g.insert_undirected(v(0), l(0), v(0), Placement::ReuseHoles, &[])
            .unwrap();
        most = most.max(g.edges.minted() - before);
    }
    let bound = insert::pages_per_half(None);
    assert!(
        most <= bound,
        "an insert took {most} pages, more than {bound}"
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
mod invariants;

use invariants::check;
mod detach;
mod edge_values;
mod model;
