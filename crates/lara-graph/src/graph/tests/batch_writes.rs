//! Batched writes (`add_nodes`, `add_edges`, `remove_edges`, `bulk_load`) against the same
//! writes one by one, and their checks before writing.

use super::*;

#[test]
fn remove_edges_matches_one_by_one() {
    let mut batch_graph = Backing::default().create(6);
    let mut loop_graph = Backing::default().create(6);
    for g in [&mut batch_graph, &mut loop_graph] {
        insert(g, v(0), l(0), v(1));
        insert(g, v(0), l(0), v(1));
        insert(g, v(0), l(0), v(2));
        insert(g, v(0), l(0), v(3));
        insert(g, v(0), l(1), v(4));
    }
    let halves = [
        out_ref(v(0), l(0), p(0)),
        out_ref(v(0), l(0), p(2)),
        out_ref(v(0), l(0), p(1)),
        out_ref(v(0), l(1), p(0)),
    ];
    let batch = batch_graph.remove_edges(&halves).unwrap();
    let one_by_one: Vec<Removal> = halves
        .iter()
        .map(|&half| loop_graph.remove_edge(half).unwrap())
        .collect();
    assert_eq!(batch, one_by_one);
    assert_eq!(batch_graph.degree(v(0), (l(0), Direction::Outgoing)), Ok(1));
    assert_eq!(batch_graph.degree(v(0), (l(1), Direction::Outgoing)), Ok(0));
    check(&batch_graph);
}

#[test]
fn remove_edges_checks_all_nodes_before_writing() {
    let backing = Backing::default();
    let mut g = backing.create(4);
    insert(&mut g, v(0), l(0), v(1));
    insert(&mut g, v(0), l(0), v(2));
    let halves = [out_ref(v(0), l(0), p(0)), out_ref(v(9), l(0), p(0))];
    let error = g.remove_edges(&halves).unwrap_err();
    assert_eq!(error.index(), 1);
    assert_eq!(*error.error(), Error::UnknownNode(v(9)));
    assert_eq!(g.degree(v(0), (l(0), Direction::Outgoing)), Ok(2));
    assert_eq!(g.degree(v(1), (l(0), Direction::Incoming)), Ok(1));
    check(&g);
}

#[test]
fn remove_edges_reports_an_already_removed_edge_as_absent() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    insert(&mut g, v(0), l(0), v(1));
    let half = out_ref(v(0), l(0), p(0));
    let first = g.remove_edges(&[half]).unwrap();
    let second = g.remove_edges(&[half]).unwrap();
    assert!(matches!(first.as_slice(), [Removal::Removed(_)]));
    assert_eq!(second, [Removal::Absent]);
    check(&g);
}

#[test]
fn add_nodes_adds_consecutive_ids() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    let first = g.add_nodes(NonZeroU32::new(3).unwrap()).unwrap();
    assert_eq!(first, v(1));
    assert_eq!(g.node_count(), 4);
    for node in 1..=3 {
        assert_eq!(g.node_state(v(node)).unwrap(), NodeState::Live);
    }
    insert(&mut g, v(3), l(0), v(1));
    assert_eq!(g.degree(v(3), (l(0), Direction::Outgoing)).unwrap(), 1);
}

#[test]
fn add_nodes_rejects_an_id_overflow_without_adding() {
    let backing = Backing::default();
    let mut g = backing.create(0);
    assert_eq!(
        g.add_nodes(NonZeroU32::MAX),
        Err(Error::AddressSpaceExhausted)
    );
    assert_eq!(g.node_count(), 0);
}

#[test]
fn add_edges_inserts_each_directed_edge() {
    let backing = Backing::default();
    let mut g = backing.create(5);
    let specs = [
        NewEdge::directed(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
        NewEdge::directed(v(0), l(0), v(2), Placement::ReuseHoles, &[]),
        NewEdge::directed(v(0), l(1), v(3), Placement::ReuseHoles, &[]),
    ];
    let halves = g.add_edges(&specs).unwrap();
    assert_eq!(halves.len(), 3);
    assert_eq!(g.edge_target(halves[0].canonical()).unwrap(), Some(v(1)));
    assert_eq!(g.degree(v(0), (l(0), Direction::Outgoing)).unwrap(), 2);
    assert_eq!(g.degree(v(0), (l(1), Direction::Outgoing)).unwrap(), 1);
    assert_eq!(g.degree(v(1), (l(0), Direction::Incoming)).unwrap(), 1);
}

#[test]
fn add_edges_reuses_a_repeated_target() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let specs = [
        NewEdge::directed(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
        NewEdge::directed(v(0), l(0), v(2), Placement::ReuseHoles, &[]),
        NewEdge::directed(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
    ];
    let halves = g.add_edges(&specs).unwrap();
    assert_eq!(halves.len(), 3);
    assert_eq!(g.degree(v(0), (l(0), Direction::Outgoing)).unwrap(), 3);
    assert_eq!(g.degree(v(1), (l(0), Direction::Incoming)).unwrap(), 2);
    assert_eq!(g.degree(v(2), (l(0), Direction::Incoming)).unwrap(), 1);
}

#[test]
fn add_edges_carries_directed_undirected_and_self_loops() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let specs = [
        NewEdge::directed(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
        NewEdge::undirected(v(0), l(1), v(2), Placement::ReuseHoles, &[]),
        NewEdge::directed(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
        NewEdge::undirected(v(2), l(2), v(2), Placement::ReuseHoles, &[]),
    ];
    let halves = g.add_edges(&specs).unwrap();
    assert_eq!(halves.len(), 4);
    assert_eq!(halves[3].canonical(), halves[3].other());
    assert_eq!(g.degree(v(0), (l(0), Direction::Outgoing)).unwrap(), 2);
    assert_eq!(g.degree(v(0), (l(1), Direction::Undirected)).unwrap(), 1);
    assert_eq!(g.degree(v(2), (l(1), Direction::Undirected)).unwrap(), 1);
    assert_eq!(g.degree(v(2), (l(2), Direction::Undirected)).unwrap(), 1);
}

#[test]
fn add_edges_caches_an_undirected_hub() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let specs = [
        NewEdge::undirected(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
        NewEdge::undirected(v(0), l(0), v(2), Placement::ReuseHoles, &[]),
        NewEdge::undirected(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
    ];
    let halves = g.add_edges(&specs).unwrap();
    assert_eq!(halves.len(), 3);
    assert_eq!(g.degree(v(0), (l(0), Direction::Undirected)).unwrap(), 3);
    assert_eq!(g.degree(v(1), (l(0), Direction::Undirected)).unwrap(), 2);
    assert_eq!(g.degree(v(2), (l(0), Direction::Undirected)).unwrap(), 1);
}

#[test]
fn add_edges_invalidates_a_target_that_gains_a_bucket() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    let specs = [
        NewEdge::directed(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
        NewEdge::directed(v(2), l(1), v(1), Placement::ReuseHoles, &[]),
        NewEdge::directed(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
    ];
    let halves = g.add_edges(&specs).unwrap();
    assert_eq!(halves.len(), 3);
    assert_eq!(g.degree(v(1), (l(0), Direction::Incoming)).unwrap(), 2);
    assert_eq!(g.degree(v(1), (l(1), Direction::Incoming)).unwrap(), 1);
}

#[test]
fn add_edges_reuses_paged_holes_from_a_cursor() {
    let backing = Backing::default();
    let mut g = backing.create(1537);
    for target in 1..=1024 {
        insert(&mut g, v(0), l(0), v(target));
    }
    for position in (0..1024).step_by(2) {
        g.remove_edge(EdgeRef::new(
            v(0),
            l(0),
            Direction::Outgoing,
            Position::new(position),
        ))
        .unwrap();
    }
    let specs: Vec<NewEdge<'_>> = (1025..=1536)
        .map(|target| NewEdge::directed(v(0), l(0), v(target), Placement::ReuseHoles, &[]))
        .collect();
    g.add_edges(&specs).unwrap();
    assert_eq!(g.degree(v(0), (l(0), Direction::Outgoing)).unwrap(), 1024);
    for target in 1..=1024 {
        let want = u32::from(target % 2 == 0);
        assert_eq!(
            g.degree(v(target), (l(0), Direction::Incoming)).unwrap(),
            want
        );
    }
    for target in 1025..=1536 {
        assert_eq!(g.degree(v(target), (l(0), Direction::Incoming)).unwrap(), 1);
    }
}

#[test]
fn add_edges_checks_all_endpoints_before_writing() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    let specs = [
        NewEdge::directed(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
        NewEdge::directed(v(0), l(0), v(9), Placement::ReuseHoles, &[]),
    ];
    let error = g.add_edges(&specs).unwrap_err();
    assert_eq!(error.index(), 1);
    assert_eq!(*error.error(), Error::UnknownNode(v(9)));
    assert_eq!(g.degree(v(0), (l(0), Direction::Outgoing)).unwrap(), 0);
    assert_eq!(g.degree(v(1), (l(0), Direction::Incoming)).unwrap(), 0);
}

#[test]
fn add_edges_checks_value_lengths_before_writing() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    g.declare_width(l(0), Width::new(2).unwrap()).unwrap();
    let specs = [
        NewEdge::directed(v(0), l(0), v(1), Placement::ReuseHoles, &[1, 2]),
        NewEdge::directed(v(0), l(0), v(1), Placement::ReuseHoles, &[1]),
    ];
    let error = g.add_edges(&specs).unwrap_err();
    assert_eq!(error.index(), 1);
    assert!(matches!(error.error(), Error::WrongValueLength { .. }));
    assert_eq!(g.degree(v(0), (l(0), Direction::Outgoing)).unwrap(), 0);
}

#[test]
fn add_edges_large_batch_indexes_every_block() {
    let backing = Backing::default();
    let mut g = backing.create(2049);
    let specs: Vec<NewEdge<'_>> = (1..=2048)
        .map(|target| NewEdge::directed(v(0), l(0), v(target), Placement::Append, &[]))
        .collect();
    g.add_edges(&specs).unwrap();
    check(&g);
}

#[test]
fn add_edges_keeps_a_cache_per_from_node() {
    let backing = Backing::default();
    let mut g = backing.create(4);
    let specs = [
        NewEdge::directed(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
        NewEdge::directed(v(2), l(0), v(3), Placement::ReuseHoles, &[]),
    ];
    g.add_edges(&specs).unwrap();
    assert_eq!(g.degree(v(0), (l(0), Direction::Outgoing)).unwrap(), 1);
    assert_eq!(g.degree(v(2), (l(0), Direction::Outgoing)).unwrap(), 1);
    assert_eq!(g.degree(v(1), (l(0), Direction::Incoming)).unwrap(), 1);
    assert_eq!(g.degree(v(3), (l(0), Direction::Incoming)).unwrap(), 1);
    check(&g);
}

#[test]
fn bulk_load_matches_add_edges() {
    let edges: Vec<NewEdge<'_>> = (1..=1500)
        .map(|target| NewEdge::directed(v(0), l(0), v(target), Placement::Append, &[]))
        .chain([
            NewEdge::undirected(v(1), l(1), v(2), Placement::Append, &[]),
            NewEdge::directed(v(3), l(1), v(3), Placement::Append, &[]),
            NewEdge::undirected(v(4), l(1), v(4), Placement::Append, &[]),
        ])
        .collect();
    let mut one = Backing::default().create(1502);
    let mut bulk = Backing::default().create(1502);
    one.add_edges(&edges).unwrap();
    bulk.bulk_load(&edges).unwrap();
    check(&one);
    check(&bulk);
    for node in 0..1502 {
        for label in 0..2 {
            for direction in [
                Direction::Outgoing,
                Direction::Incoming,
                Direction::Undirected,
            ] {
                assert_eq!(
                    edges_on(&one, v(node), l(label), direction),
                    edges_on(&bulk, v(node), l(label), direction),
                    "node {node} label {label} {direction:?}"
                );
            }
        }
    }
}

#[test]
fn bulk_load_refuses_a_graph_with_edges() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    insert(&mut g, v(0), l(0), v(1));
    let error = g
        .bulk_load(&[NewEdge::directed(v(0), l(0), v(1), Placement::Append, &[])])
        .unwrap_err();
    assert_eq!(error, Error::NotEmpty);
}

#[test]
fn bulk_load_writes_values() {
    let values: Vec<Vec<u8>> = (0..1500u32)
        .map(|i| vec![u8::try_from(i % 251).unwrap_or(0); 4])
        .collect();
    let edges: Vec<NewEdge<'_>> = (1..=1500)
        .map(|target| {
            NewEdge::directed(
                v(0),
                l(0),
                v(target),
                Placement::Append,
                &values[usize::try_from(target - 1).unwrap_or(0)],
            )
        })
        .collect();
    let mut one = Backing::default().create(1501);
    let mut bulk = Backing::default().create(1501);
    one.declare_width(l(0), Width::new(4).unwrap()).unwrap();
    bulk.declare_width(l(0), Width::new(4).unwrap()).unwrap();
    let one_halves = one.add_edges(&edges).unwrap();
    let bulk_halves = bulk.bulk_load(&edges).unwrap();
    check(&one);
    check(&bulk);
    for (a, b) in one_halves.iter().zip(bulk_halves.iter()) {
        assert_eq!(a.canonical().node(), b.canonical().node(), "canonical node");
        assert_eq!(
            a.canonical().position(),
            b.canonical().position(),
            "canonical position"
        );
        assert_eq!(
            one.edge_value(a.canonical()).unwrap(),
            bulk.edge_value(b.canonical()).unwrap(),
            "value"
        );
    }
}
