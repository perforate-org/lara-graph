//! Detaching a node step by step.

use super::*;

fn budget(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap()
}

/// Detaches `x` with steps of `n` edges, checking the graph after each; returns every edge the
/// steps reported.
fn detach_fully<M: Memory>(g: &mut Graph<M>, x: NodeId, n: u32) -> Vec<Halves> {
    g.detach(x).unwrap();
    let mut removed = Vec::new();
    for _ in 0..100_000 {
        let step = g.detach_step(x, budget(n)).unwrap();
        assert!(step.changes().len() <= usize::try_from(n).unwrap());
        removed.extend(step.changes().iter().map(RemovedEdge::halves));
        check(g);
        if step.progress() == Progress::Done {
            return removed;
        }
    }
    panic!("detach did not finish");
}

fn sorted(mut halves: Vec<Halves>) -> Vec<Halves> {
    halves.sort_by_key(|h| {
        let (s, t) = (h.canonical(), h.other());
        (
            s.node(),
            s.label(),
            s.direction(),
            s.position(),
            t.node(),
            t.position(),
        )
    });
    halves
}

#[test]
#[expect(clippy::too_many_lines, reason = "one detach covers every half")]
fn detaching_removes_every_edge_and_reports_its_true_halves() {
    let backing = Backing::default();
    let mut g = backing.create(5);
    let x = v(0);
    let mut incident = Vec::new();
    // Parallel edges x→1 interleaved with 4→1, so the in halves at 1 sit at other positions.
    for _ in 0..3 {
        incident.push(
            g.add_edge(
                EdgeKind::Directed,
                x,
                l(0),
                v(1),
                Placement::ReuseHoles,
                &[],
            )
            .unwrap(),
        );
        g.add_edge(
            EdgeKind::Directed,
            v(4),
            l(0),
            v(1),
            Placement::ReuseHoles,
            &[],
        )
        .unwrap();
    }
    let gone = incident.remove(0);
    g.remove_edge(gone.canonical()).unwrap();
    incident.push(
        g.add_edge(
            EdgeKind::Directed,
            v(3),
            l(0),
            x,
            Placement::ReuseHoles,
            &[],
        )
        .unwrap(),
    );
    incident.push(
        g.add_edge(EdgeKind::Directed, x, l(1), x, Placement::ReuseHoles, &[])
            .unwrap(),
    );
    incident.push(
        g.add_edge(EdgeKind::Directed, x, l(1), x, Placement::ReuseHoles, &[])
            .unwrap(),
    );
    incident.push(
        g.add_edge(
            EdgeKind::Undirected,
            v(2),
            l(1),
            x,
            Placement::ReuseHoles,
            &[],
        )
        .unwrap(),
    );
    incident.push(
        g.add_edge(EdgeKind::Undirected, x, l(2), x, Placement::ReuseHoles, &[])
            .unwrap(),
    );
    let other = g
        .add_edge(
            EdgeKind::Directed,
            v(1),
            l(0),
            v(2),
            Placement::ReuseHoles,
            &[],
        )
        .unwrap();
    g.detach(x).unwrap();
    for (from, to) in [(x, v(1)), (v(1), x)] {
        assert_eq!(
            g.add_edge(
                EdgeKind::Directed,
                from,
                l(0),
                to,
                Placement::ReuseHoles,
                &[]
            ),
            Err(Error::Detaching(x))
        );
    }
    assert_eq!(
        g.add_edge(EdgeKind::Undirected, x, l(9), x, Placement::ReuseHoles, &[]),
        Err(Error::Detaching(x))
    );
    let removed = detach_fully(&mut g, x, 1);
    assert_eq!(sorted(removed), sorted(incident));
    for direction in [
        Direction::Outgoing,
        Direction::Incoming,
        Direction::Undirected,
    ] {
        let mut labels = Vec::new();
        g.labels(x, direction, &mut labels).unwrap();
        assert_eq!(labels, Vec::<Label>::new(), "{direction:?}");
    }
    assert_eq!(g.counterpart(other.canonical()), Ok(Some(other.other())));
    assert_eq!(
        edges_on(&g, v(1), l(0), Direction::Incoming),
        [(1, 4), (3, 4), (5, 4)]
    );
    // Deleted for good: no new edges, no second detach, and the id is not reused.
    assert_eq!(g.node_state(x), Ok(NodeState::Deleted));
    assert_eq!(
        g.add_edge(
            EdgeKind::Directed,
            v(1),
            l(0),
            x,
            Placement::ReuseHoles,
            &[]
        ),
        Err(Error::Deleted(x))
    );
    assert_eq!(g.detach(x), Err(Error::Deleted(x)));
    assert_eq!(g.detach_step(x, budget(1)), Err(Error::NotDetaching(x)));
    assert_eq!(g.add_node(), Ok(v(5)));
    check(&g);
}

#[test]
fn a_step_needs_a_detaching_node_and_an_isolated_one_finishes_at_once() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    assert_eq!(
        g.detach_step(v(0), budget(1)),
        Err(Error::NotDetaching(v(0)))
    );
    assert_eq!(g.detach(v(2)), Err(Error::UnknownNode(v(2))));
    assert_eq!(g.node_state(v(0)), Ok(NodeState::Live));
    g.detach(v(0)).unwrap();
    g.detach(v(0)).unwrap();
    assert_eq!(g.node_state(v(0)), Ok(NodeState::Detaching));
    let step = g.detach_step(v(0), budget(1)).unwrap();
    assert_eq!((step.changes().len(), step.progress()), (0, Progress::Done));
    assert_eq!(g.node_state(v(0)), Ok(NodeState::Deleted));
    assert_eq!(
        g.detach_step(v(0), budget(1)),
        Err(Error::NotDetaching(v(0)))
    );
}

#[test]
fn a_hub_detaches_in_bounded_steps_across_reopens() {
    let backing = Backing::default();
    let mut g = backing.create(5);
    let total = 3 * PAGE_SLOTS + 5;
    for i in 0..total {
        insert(&mut g, v(0), l(0), v(1 + i % 4));
        if i % 7 == 0 {
            insert(&mut g, v(1 + i % 4), l(0), v(0));
        }
    }
    insert(&mut g, v(1), l(0), v(2));
    g.detach(v(0)).unwrap();
    let mut removed = 0;
    loop {
        let step = g.detach_step(v(0), budget(500)).unwrap();
        assert!(step.changes().len() <= 500);
        removed += step.changes().len();
        check(&g);
        if step.progress() == Progress::Done {
            break;
        }
        // The mark survives a reopen.
        g = backing.open();
        assert_eq!(
            g.add_edge(
                EdgeKind::Directed,
                v(1),
                l(0),
                v(0),
                Placement::ReuseHoles,
                &[]
            ),
            Err(Error::Detaching(v(0)))
        );
    }
    let expected = total + total.div_ceil(7);
    assert_eq!(removed, usize::try_from(expected).unwrap());
    // Positions stay: node 1's edges to the hub left holes before its edge to 2.
    let left: Vec<u32> = edges(&g, v(1), l(0)).into_iter().map(|(_, t)| t).collect();
    assert_eq!(left, [2]);
    for x in 1..5 {
        assert_eq!(
            g.degree(v(x), l(0), Direction::Incoming),
            Ok(u32::from(x == 2))
        );
    }
}

/// One sweep of a hub's in-bucket, in short calls, removes exactly the edges from detaching
/// nodes to the hub, parallel ones included, with values. Their own detaches then find those
/// edges gone and finish. A set naming a node that is not detaching changes nothing.
#[test]
fn sweeping_a_shared_bucket_removes_the_detaching_nodes_edges() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    g.declare_width(l(0), Width::new(2).unwrap()).unwrap();
    let hub = v(0);
    let sources: Vec<NodeId> = (0..1500).map(|_| g.add_node().unwrap()).collect();
    let mut edges = Vec::new();
    for (i, &s) in sources.iter().enumerate() {
        let value = u16::try_from(i).unwrap().to_le_bytes();
        for _ in 0..=i % 3 {
            edges.push((
                s,
                g.add_edge(EdgeKind::Directed, s, l(0), hub, Placement::Append, &value)
                    .unwrap(),
            ));
        }
    }
    // An edge out of the hub to a detaching node stays for that node's own detach.
    g.add_edge(
        EdgeKind::Directed,
        hub,
        l(0),
        sources[3],
        Placement::Append,
        &[9, 9],
    )
    .unwrap();
    let gone: Vec<NodeId> = sources.iter().copied().step_by(4).collect();
    let live = *sources.get(1).unwrap();
    for &d in &gone {
        g.detach(d).unwrap();
    }
    assert_eq!(
        g.detaching_set(gone.iter().copied().chain([live])),
        Err(Error::NotDetaching(live))
    );
    let detaching = g.detaching_set(gone.iter().copied()).unwrap();
    let mut removed = Vec::new();
    let mut from = Some(p(0));
    while let Some(at) = from {
        let sweep = g
            .sweep(
                hub,
                (l(0), Direction::Incoming),
                (at, budget(500)),
                &detaching,
            )
            .unwrap();
        removed.extend(sweep.changes().iter().cloned());
        from = sweep.next();
        check(&g);
    }
    let want: Vec<Halves> = edges
        .iter()
        .filter(|(s, _)| detaching.contains(*s))
        .map(|&(_, h)| h)
        .collect();
    assert_eq!(
        sorted(removed.iter().map(RemovedEdge::halves).collect()),
        sorted(want)
    );
    for r in &removed {
        let source = r.halves().canonical().node();
        let i = sources.iter().position(|&s| s == source).unwrap();
        assert_eq!(r.value(), u16::try_from(i).unwrap().to_le_bytes());
    }
    for &d in &gone {
        let rest = detach_fully(&mut g, d, 7);
        let out_of_hub = usize::from(d == sources[3]);
        assert_eq!(rest.len(), out_of_hub, "node {}", d.index());
        assert_eq!(g.node_state(d), Ok(NodeState::Deleted));
    }
    check(&g);
}
