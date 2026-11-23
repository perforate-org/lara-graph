//! Detaching a vertex step by step.

use super::*;

fn budget(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap()
}

/// Detaches `x` with steps of `n` edges, checking the graph after each; returns every edge the
/// steps reported.
fn detach_fully<M: Memory>(g: &mut Graph<M>, x: VertexId, n: u32) -> Vec<Halves> {
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
            s.vertex(),
            s.label(),
            s.side(),
            s.position(),
            t.vertex(),
            t.position(),
        )
    });
    halves
}

#[test]
fn detaching_removes_every_edge_and_reports_its_true_halves() {
    let backing = Backing::default();
    let mut g = backing.create(5);
    let x = v(0);
    let mut incident = Vec::new();
    // Parallel edges x→1 interleaved with 4→1, so the in halves at 1 sit at other positions.
    for _ in 0..3 {
        incident.push(
            g.insert_directed(x, l(0), v(1), Placement::ReuseHoles, &[])
                .unwrap(),
        );
        g.insert_directed(v(4), l(0), v(1), Placement::ReuseHoles, &[])
            .unwrap();
    }
    let gone = incident.remove(0);
    g.remove_edge(gone.canonical()).unwrap();
    incident.push(
        g.insert_directed(v(3), l(0), x, Placement::ReuseHoles, &[])
            .unwrap(),
    );
    incident.push(
        g.insert_directed(x, l(1), x, Placement::ReuseHoles, &[])
            .unwrap(),
    );
    incident.push(
        g.insert_directed(x, l(1), x, Placement::ReuseHoles, &[])
            .unwrap(),
    );
    incident.push(
        g.insert_undirected(v(2), l(1), x, Placement::ReuseHoles, &[])
            .unwrap(),
    );
    incident.push(
        g.insert_undirected(x, l(2), x, Placement::ReuseHoles, &[])
            .unwrap(),
    );
    let other = g
        .insert_directed(v(1), l(0), v(2), Placement::ReuseHoles, &[])
        .unwrap();
    g.detach(x).unwrap();
    for (from, to) in [(x, v(1)), (v(1), x)] {
        assert_eq!(
            g.insert_directed(from, l(0), to, Placement::ReuseHoles, &[]),
            Err(Error::Detaching(x))
        );
    }
    assert_eq!(
        g.insert_undirected(x, l(9), x, Placement::ReuseHoles, &[]),
        Err(Error::Detaching(x))
    );
    let removed = detach_fully(&mut g, x, 1);
    assert_eq!(sorted(removed), sorted(incident));
    for side in [Side::Out, Side::In, Side::Undirected] {
        let mut labels = Vec::new();
        g.labels(x, side, &mut labels).unwrap();
        assert_eq!(labels, Vec::<Label>::new(), "{side:?}");
    }
    assert_eq!(g.counterpart(other.canonical()), Ok(Some(other.other())));
    assert_eq!(edges_on(&g, v(1), l(0), Side::In), [(1, 4), (3, 4), (5, 4)]);
    // Deleted for good: no new edges, no second detach, and the id is not reused.
    assert_eq!(g.vertex_state(x), Ok(VertexState::Deleted));
    assert_eq!(
        g.insert_directed(v(1), l(0), x, Placement::ReuseHoles, &[]),
        Err(Error::Deleted(x))
    );
    assert_eq!(g.detach(x), Err(Error::Deleted(x)));
    assert_eq!(g.detach_step(x, budget(1)), Err(Error::NotDetaching(x)));
    assert_eq!(g.add_vertex(), Ok(v(5)));
    check(&g);
}

#[test]
fn a_step_needs_a_detaching_vertex_and_an_isolated_one_finishes_at_once() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    assert_eq!(
        g.detach_step(v(0), budget(1)),
        Err(Error::NotDetaching(v(0)))
    );
    assert_eq!(g.detach(v(2)), Err(Error::UnknownVertex(v(2))));
    assert_eq!(g.vertex_state(v(0)), Ok(VertexState::Live));
    g.detach(v(0)).unwrap();
    g.detach(v(0)).unwrap();
    assert_eq!(g.vertex_state(v(0)), Ok(VertexState::Detaching));
    let step = g.detach_step(v(0), budget(1)).unwrap();
    assert_eq!((step.changes().len(), step.progress()), (0, Progress::Done));
    assert_eq!(g.vertex_state(v(0)), Ok(VertexState::Deleted));
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
            g.insert_directed(v(1), l(0), v(0), Placement::ReuseHoles, &[]),
            Err(Error::Detaching(v(0)))
        );
    }
    let expected = total + total.div_ceil(7);
    assert_eq!(removed, usize::try_from(expected).unwrap());
    // Positions stay: vertex 1's edges to the hub left holes before its edge to 2.
    let left: Vec<u32> = edges(&g, v(1), l(0)).into_iter().map(|(_, t)| t).collect();
    assert_eq!(left, [2]);
    for x in 1..5 {
        assert_eq!(g.degree(v(x), l(0), Side::In), Ok(u32::from(x == 2)));
    }
}

/// One sweep of a hub's in-bucket, in short calls, removes exactly the edges from detaching
/// vertices to the hub, parallel ones included, with values. Their own detaches then find those
/// edges gone and finish. A set naming a vertex that is not detaching changes nothing.
#[test]
fn sweeping_a_shared_bucket_removes_the_detaching_vertices_edges() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    g.declare_width(l(0), Width::new(2).unwrap()).unwrap();
    let hub = v(0);
    let sources: Vec<VertexId> = (0..1500).map(|_| g.add_vertex().unwrap()).collect();
    let mut edges = Vec::new();
    for (i, &s) in sources.iter().enumerate() {
        let value = u16::try_from(i).unwrap().to_le_bytes();
        for _ in 0..=i % 3 {
            edges.push((
                s,
                g.insert_directed(s, l(0), hub, Placement::Append, &value)
                    .unwrap(),
            ));
        }
    }
    // An edge out of the hub to a detaching vertex stays for that vertex's own detach.
    g.insert_directed(hub, l(0), sources[3], Placement::Append, &[9, 9])
        .unwrap();
    let gone: Vec<VertexId> = sources.iter().copied().step_by(4).collect();
    let detaching = VertexSet::new(gone.iter().copied());
    let live = *sources.get(1).unwrap();
    let with_live = VertexSet::new(gone.iter().copied().chain([live]));
    for &d in &gone {
        g.detach(d).unwrap();
    }
    assert_eq!(
        g.sweep(hub, (l(0), Side::In), (p(0), budget(100)), &with_live),
        Err(Error::NotDetaching(live))
    );
    let mut removed = Vec::new();
    let mut from = Some(p(0));
    while let Some(at) = from {
        let sweep = g
            .sweep(hub, (l(0), Side::In), (at, budget(500)), &detaching)
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
        let source = r.halves().canonical().vertex();
        let i = sources.iter().position(|&s| s == source).unwrap();
        assert_eq!(r.value(), u16::try_from(i).unwrap().to_le_bytes());
    }
    for &d in &gone {
        let rest = detach_fully(&mut g, d, 7);
        let out_of_hub = usize::from(d == sources[3]);
        assert_eq!(rest.len(), out_of_hub, "vertex {}", d.index());
        assert_eq!(g.vertex_state(d), Ok(VertexState::Deleted));
    }
    check(&g);
}
