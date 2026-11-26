//! Single edges: inserting and removing directed, undirected and self-loop edges, pairing
//! parallel edges, reusing holes, label buckets and failed inserts that change nothing.

use super::*;

#[test]
fn a_new_graph_is_empty_and_rejects_unknown_nodes() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    assert_eq!(g.node_count(), 2);
    assert_eq!(g.degree(v(1), l(0).outgoing()), Ok(0));
    assert_eq!(
        g.add_edge(Edge::directed(v(0), l(0), v(2))),
        Err(Error::UnknownNode(v(2)))
    );
    assert_eq!(
        g.add_edge(Edge::undirected(v(5), l(0), v(0))),
        Err(Error::UnknownNode(v(5)))
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
    let e = g.add_edge(Edge::directed(v(0), l(4), v(2))).unwrap();
    assert_eq!(e.canonical(), EdgeRef::new(v(0), l(4).outgoing(), p(0)));
    assert_eq!(e.other(), EdgeRef::new(v(2), l(4).incoming(), p(0)));
    assert_eq!(edges_on(&g, v(2), l(4), Direction::Incoming), [(0, 0)]);
    assert_eq!(g.other_end(e.other()), Ok(Some(v(0))));
    assert_eq!(g.counterpart(e.canonical()), Ok(Some(e.other())));
    assert_eq!(g.counterpart(e.other()), Ok(Some(e.canonical())));
    let mut labels = Vec::new();
    g.edge_labels(v(2), Direction::Incoming, &mut labels)
        .unwrap();
    assert_eq!(labels, [l(4)]);
    g.edge_labels(v(2), Direction::Outgoing, &mut labels)
        .unwrap();
    assert_eq!(labels, Vec::<EdgeLabel>::new());
    check(&g);
    assert_eq!(removed(g.remove_edge(e.other())), Some(e));
    assert_eq!(g.degree(v(0), l(4).outgoing()), Ok(0));
    assert_eq!(g.degree(v(2), l(4).incoming()), Ok(0));
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
    let directed = g.add_edge(Edge::directed(v(0), l(1), v(1))).unwrap();
    let undirected = g
        .add_edge_with(Edge::undirected(v(2), l(2), v(3)), Placement::Append)
        .unwrap();
    let valued = g
        .add_edge_with(
            Edge::directed(v(4), l(3), v(5)).value(&[1, 2]),
            Placement::Append,
        )
        .unwrap();
    for x in 0..6 {
        assert_eq!(g.has_edges(v(x)), Ok(true), "node {x}");
    }
    for halves in [directed, undirected, valued] {
        g.remove_edge(halves.other()).unwrap();
    }
    let mut labels = Vec::new();
    g.edge_labels(v(0), Direction::Outgoing, &mut labels)
        .unwrap();
    assert_eq!(labels, [l(1)], "a bucket of holes stays listed");
    for x in 0..6 {
        assert_eq!(g.has_edges(v(x)), Ok(false), "node {x}");
    }
    assert_eq!(g.has_edges(v(9)), Err(Error::UnknownNode(v(9))));
    check(&g);
}

#[test]
fn an_undirected_edges_canonical_half_is_at_the_larger_node_and_a_self_loop_has_one_half() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let e = g.add_edge(Edge::undirected(v(2), l(0), v(1))).unwrap();
    // The canonical half is at the larger node.
    assert_eq!(e.canonical(), EdgeRef::new(v(2), l(0).undirected(), p(0)));
    assert_eq!(e.other(), EdgeRef::new(v(1), l(0).undirected(), p(0)));
    assert_eq!(g.counterpart(e.other()), Ok(Some(e.canonical())));
    let s = g.add_edge(Edge::undirected(v(1), l(0), v(1))).unwrap();
    assert_eq!(s.canonical(), s.other());
    assert_eq!(s.canonical().position(), p(1));
    assert_eq!(g.counterpart(s.canonical()), Ok(Some(s.canonical())));
    assert_eq!(
        edges_on(&g, v(1), l(0), Direction::Undirected),
        [(0, 2), (1, 1)]
    );
    check(&g);
    assert_eq!(removed(g.remove_edge(s.other())), Some(s));
    assert_eq!(removed(g.remove_edge(e.other())), Some(e));
    assert_eq!(g.degree(v(1), l(0).undirected()), Ok(0));
    check(&g);
}

#[test]
fn a_directed_self_loop_has_an_out_and_an_in_half() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    let first = g.add_edge(Edge::directed(v(0), l(0), v(0))).unwrap();
    let second = g.add_edge(Edge::directed(v(0), l(0), v(0))).unwrap();
    assert_eq!(
        second.canonical(),
        EdgeRef::new(v(0), l(0).outgoing(), p(1))
    );
    assert_eq!(second.other(), EdgeRef::new(v(0), l(0).incoming(), p(1)));
    assert_eq!(removed(g.remove_edge(first.other())), Some(first));
    assert_eq!(g.counterpart(second.other()), Ok(Some(second.canonical())));
    check(&g);
}

#[test]
fn parallel_edges_pair_by_rank() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let (a, b) = (v(0), v(1));
    // a→b at Out 0, 2, 3; other traffic interleaves the In side at b.
    let first = g.add_edge(Edge::directed(a, l(0), b)).unwrap();
    g.add_edge(Edge::directed(a, l(0), v(2))).unwrap();
    g.add_edge(Edge::directed(v(2), l(0), b)).unwrap();
    let second = g.add_edge(Edge::directed(a, l(0), b)).unwrap();
    let third = g.add_edge(Edge::directed(a, l(0), b)).unwrap();
    assert_eq!(second.other().position(), p(2));
    g.remove_edge(first.canonical()).unwrap();
    // A new a→b must not take hole 0 on either side: it would rank before the survivors.
    let fourth = g.add_edge(Edge::directed(a, l(0), b)).unwrap();
    assert_eq!(fourth.canonical().position(), p(4));
    assert_eq!(fourth.other().position(), p(4));
    for e in [second, third, fourth] {
        assert_eq!(g.counterpart(e.canonical()), Ok(Some(e.other())));
        assert_eq!(g.counterpart(e.other()), Ok(Some(e.canonical())));
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
        "no later edge to node 0: hole reused"
    );
    check(&g);
}

#[test]
fn labels_are_separate_sorted_buckets() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    assert_eq!(insert(&mut g, v(0), l(7), v(1)), p(0));
    assert_eq!(insert(&mut g, v(0), l(2), v(2)), p(0));
    assert_eq!(insert(&mut g, v(0), l(7), v(2)), p(1));
    let mut labels = Vec::new();
    g.edge_labels(v(0), Direction::Outgoing, &mut labels)
        .unwrap();
    assert_eq!(labels, [l(2), l(7)]);
    assert_eq!(edges(&g, v(0), l(7)), [(0, 1), (1, 2)]);
    assert_eq!(edges(&g, v(0), l(2)), [(0, 2)]);
    assert_eq!(g.other_end(out_ref(v(0), l(7), p(1))), Ok(Some(v(2))));
    check(&g);
}

#[test]
fn too_many_buckets_is_an_error_and_changes_nothing() {
    let backing = Backing::default();
    let mut g = backing.create(4);
    let room = |g: &Graph<VecMemory>, x: u32, adding| {
        label_bucket_list::has_room(
            &g.edges.store().pages(),
            g.label_bucket_list(v(x)).unwrap(),
            adding,
        )
    };
    // Grow node 0 until its tiered list has room for one more bucket, not two.
    let mut label = 0;
    while room(&g, 0, 2) {
        insert(&mut g, v(0), l(label), v(2));
        label += 1;
    }
    assert!(room(&g, 0, 1));
    // A directed self-loop needs two new buckets at node 0.
    assert_eq!(
        g.add_edge(Edge::directed(v(0), l(label), v(0))),
        Err(Error::TooManyLabelBuckets(v(0)))
    );
    // Fill node 1; then an edge 0 → 1 fails on its in half and must not land at 0 either.
    let mut other = 0;
    while room(&g, 1, 1) {
        insert(&mut g, v(1), l(other), v(3));
        other += 1;
    }
    assert_eq!(
        g.add_edge(Edge::directed(v(0), l(label), v(1))),
        Err(Error::TooManyLabelBuckets(v(1)))
    );
    assert_eq!(g.degree(v(0), l(label).outgoing()), Ok(0));
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
    let before = (
        edges(&g, v(0), l(0)),
        edges_on(&g, v(1), l(0), Direction::Incoming),
    );
    // Leave room for three more pages: promoting both full runs to pages takes two each, so the
    // out half alone would fit and the in half would not.
    let memory = &backing.memories[1];
    memory.cap(memory.size());
    let capacity = u32::try_from(memory.size() * (PAGE_SIZE / 4096)).unwrap();
    let mut held = Vec::new();
    while capacity - g.edges.store().minted() > 3 {
        held.push(g.edges.store_mut().allocate(ByteClass::PAGE).unwrap());
    }
    let error = g.add_edge(Edge::directed(v(0), l(0), v(1)));
    assert!(matches!(error, Err(Error::OutOfMemory(_))), "{error:?}");
    assert_eq!(
        (
            edges(&g, v(0), l(0)),
            edges_on(&g, v(1), l(0), Direction::Incoming)
        ),
        before
    );
    for extent in held {
        g.edges.store_mut().release(extent);
    }
    check(&g);
}

/// At full size a node holds a bucket on each side of 32,768 directed labels. Added in key order,
/// they leave every leaf of its tiered list half full, the fewest buckets a leaf holds.
#[test]
#[ignore = "full size: run with RUSTFLAGS=\"--cfg lara_full_size\" and --release --ignored"]
fn a_node_holds_both_sides_of_every_15_bit_label() {
    assert_eq!(
        crate::store::scale::Scale::BUILD,
        crate::store::scale::Scale::Full,
        "build with RUSTFLAGS=\"--cfg lara_full_size\""
    );
    let mut g = Graph::<VecMemory>::format(Memories::from_fn(|_| VecMemory::new()));
    let (a, b) = (g.add_node(&[]).unwrap(), g.add_node(&[]).unwrap());
    let edges: Vec<Edge<'_>> = (0..1u16 << 15)
        .flat_map(|label| {
            [(a, b), (b, a)].map(|(from, to)| Edge::directed(from, EdgeLabel::new(label), to))
        })
        .collect();
    g.add_edges(&edges).unwrap();
    let mut out = Vec::new();
    g.edge_labels(a, Direction::Outgoing, &mut out).unwrap();
    assert_eq!(out.len(), 1 << 15);
    g.edge_labels(a, Direction::Incoming, &mut out).unwrap();
    assert_eq!(out.len(), 1 << 15);
}

#[test]
fn edges_between_two_nodes_are_removed_together() {
    let backing = Backing::default();
    let mut g = backing.create(4);
    for to in [1, 2, 1, 3, 1] {
        insert(&mut g, v(0), l(0), v(to));
    }
    g.add_edge(Edge::directed(v(1), l(0), v(0))).unwrap();
    g.add_edge(Edge::undirected(v(0), l(1), v(1))).unwrap();
    assert_eq!(g.remove_edges_between(v(0), l(0).outgoing(), v(1)), Ok(3));
    assert_eq!(
        edges(&g, v(0), l(0)),
        [(1, 2), (3, 3)],
        "the rest keep their positions"
    );
    assert_eq!(g.remove_edges_between(v(0), l(0).outgoing(), v(1)), Ok(0));
    assert_eq!(
        edges(&g, v(1), l(0)),
        [(0, 0)],
        "the reverse edge is another bucket's"
    );
    assert_eq!(g.remove_edges_between(v(1), l(1).undirected(), v(0)), Ok(1));
    assert_eq!(
        g.remove_edges_between(v(0), l(0).outgoing(), v(9)),
        Err(Error::UnknownNode(v(9)))
    );
    check(&g);
}
