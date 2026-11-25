//! Proxies: ids apart from local nodes, a memory written only from the first one, the local half
//! as the canonical one, no labels, no edge between two proxies, and deletion through detach.

use super::*;

fn proxies_memory(backing: &Backing) -> &VecMemory {
    &backing.memories[usize::from(MemoryRole::Proxies.index())]
}

fn directed(from: NodeId, to: NodeId) -> NewEdge<'static> {
    NewEdge::directed(from, l(1), to, Placement::ReuseHoles, &[])
}

#[test]
fn the_proxies_memory_stays_empty_until_the_first_proxy() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    g.add_edge(directed(v(0), v(1))).unwrap();
    assert_eq!(proxies_memory(&backing).size(), 0);
    assert_eq!(g.proxy_count(), 0);

    let first = g.add_proxy().unwrap();
    assert_eq!(first, NodeId::proxy(0));
    assert!(first.is_proxy() && !v(2).is_proxy());
    assert_eq!(first.proxy_ordinal(), Some(0));
    assert_eq!(v(2).proxy_ordinal(), None);
    assert!(proxies_memory(&backing).size() > 0);
    let next = g.add_proxies(NonZeroU32::new(4).unwrap()).unwrap();
    assert_eq!(next, NodeId::proxy(1));
    assert_eq!((g.node_count(), g.proxy_count()), (3, 5));
    g.add_edge(directed(v(0), NodeId::proxy(4))).unwrap();
    check(&g);

    let reopened = backing.open();
    assert_eq!((reopened.node_count(), reopened.proxy_count()), (3, 5));
    assert_eq!(
        reopened.degree(NodeId::proxy(4), (l(1), Direction::Incoming)),
        Ok(1)
    );
    check(&reopened);
    drop((g, reopened));
    let fresh = Graph::new(backing.memories());
    assert_eq!(
        (fresh.node_count(), fresh.proxy_count()),
        (0, 0),
        "new formats the proxies too"
    );
}

#[test]
fn an_unknown_proxy_is_an_unknown_node() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    g.add_proxy().unwrap();
    let missing = NodeId::proxy(1);
    assert_eq!(g.node_state(missing), Err(Error::UnknownNode(missing)));
    assert_eq!(
        g.add_edge(directed(v(0), missing)),
        Err(Error::UnknownNode(missing))
    );
}

#[test]
fn the_local_half_is_canonical_whichever_way_the_edge_points() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    let proxy = g.add_proxy().unwrap();
    let to_proxy = g.add_edge(directed(v(0), proxy)).unwrap();
    assert_eq!(
        (
            to_proxy.canonical().node(),
            to_proxy.canonical().direction()
        ),
        (v(0), Direction::Outgoing)
    );
    let from_proxy = g.add_edge(directed(proxy, v(1))).unwrap();
    assert_eq!(
        (
            from_proxy.canonical().node(),
            from_proxy.canonical().direction()
        ),
        (v(1), Direction::Incoming),
        "an edge from a proxy is keyed at its local target"
    );
    let undirected = g
        .add_edge(NewEdge::undirected(
            proxy,
            l(1),
            v(1),
            Placement::ReuseHoles,
            &[],
        ))
        .unwrap();
    assert_eq!(
        undirected.canonical().node(),
        v(1),
        "the local end, though the proxy's id is larger"
    );
    for halves in [to_proxy, from_proxy, undirected] {
        assert_eq!(g.counterpart(halves.other()), Ok(Some(halves.canonical())));
        assert_eq!(
            Halves::of(halves.other(), halves.canonical()),
            halves,
            "either half names the same canonical one"
        );
    }
    let mut batch = vec![directed(proxy, v(0)), directed(v(1), proxy)];
    batch.push(NewEdge::undirected(
        v(0),
        l(1),
        proxy,
        Placement::ReuseHoles,
        &[],
    ));
    for halves in g.add_edges(&batch).unwrap() {
        assert!(
            !halves.canonical().node().is_proxy(),
            "add_edges keys the local half too"
        );
    }
    let Ok(Removal::Removed(removed)) = g.remove_edge(from_proxy.other()) else {
        panic!("the edge is removed")
    };
    assert_eq!(removed.halves(), from_proxy);
    check(&g);
}

#[test]
fn proxies_have_no_labels_and_no_edges_between_them() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    let a = g.add_proxy().unwrap();
    let b = g.add_proxy().unwrap();
    assert_eq!(g.set_node_labels(a, &[l(3)]), Err(Error::Proxy(a)));
    assert_eq!(g.has_node_label(a, l(3)), Ok(false));
    let mut labels = vec![l(9)];
    g.node_labels(a, &mut labels).unwrap();
    assert_eq!(labels, []);
    let mut has = Vec::new();
    g.read_batch()
        .has_node_label_each(&[v(0), a, b], l(3), &mut has)
        .unwrap();
    assert_eq!(has, [false, false, false]);
    assert_eq!(g.add_edge(directed(a, b)), Err(Error::Proxy(a)));
    assert_eq!(g.add_edge(directed(a, a)), Err(Error::Proxy(a)));
    assert_eq!(g.has_edges(a), Ok(false), "a refused edge leaves nothing");
    check(&g);
}

#[test]
fn bulk_load_places_no_proxy_and_sees_their_edges() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    let proxy = g.add_proxy().unwrap();
    assert_eq!(
        g.bulk_load(&[directed(v(0), v(1)), directed(v(0), proxy)]),
        Err(Error::Proxy(proxy))
    );
    g.add_edge(directed(v(0), proxy)).unwrap();
    g.remove_edge(EdgeRef::new(v(0), l(1), Direction::Outgoing, p(0)))
        .unwrap();
    g.compact(v(0), (l(1), Direction::Outgoing)).unwrap();
    g.add_edge(directed(proxy, v(1))).unwrap();
    assert_eq!(
        g.bulk_load(&[directed(v(0), v(1))]),
        Err(Error::NotEmpty),
        "a proxy's edges count"
    );
}

#[test]
fn detaching_a_proxy_removes_every_local_edge_to_it() {
    let backing = Backing::default();
    let mut g = backing.create(40);
    let proxy = g.add_proxy().unwrap();
    for i in 0..40 {
        g.add_edge(directed(v(i), proxy)).unwrap();
        g.add_edge(directed(v(i), v((i + 1) % 40))).unwrap();
    }
    g.set_node_labels(v(3), &[l(7)]).unwrap();
    g.detach(proxy).unwrap();
    assert_eq!(g.node_state(proxy), Ok(NodeState::Detaching));
    let budget = NonZeroU32::new(16).unwrap();
    while g.detach_step(proxy, budget).unwrap().progress() == Progress::Remaining {}
    assert_eq!(g.node_state(proxy), Ok(NodeState::Deleted));
    for i in 0..40 {
        assert_eq!(
            g.degree(v(i), (l(1), Direction::Outgoing)),
            Ok(1),
            "node {i}"
        );
    }
    assert_eq!(g.has_node_label(v(3), l(7)), Ok(true), "local labels stay");
    let mut states = Vec::new();
    g.read_batch()
        .node_state_each(&[v(0), proxy, v(1)], &mut states)
        .unwrap();
    assert_eq!(
        states,
        [NodeState::Live, NodeState::Deleted, NodeState::Live]
    );
    check(&g);
}

#[test]
fn local_ids_stop_below_the_proxies() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    let past = NonZeroU32::new(NodeId::FIRST_PROXY).unwrap();
    assert_eq!(g.add_nodes(past), Err(Error::AddressSpaceExhausted));
    assert_eq!(g.node_count(), 1);
    g.add_proxy().unwrap();
    let past = NonZeroU32::new(NodeId::PROXIES).unwrap();
    assert_eq!(g.add_proxies(past), Err(Error::AddressSpaceExhausted));
    assert_eq!(g.proxy_count(), 1);
}
