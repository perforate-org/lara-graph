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
        removed.extend_from_slice(step.changes());
        check(g);
        if step.progress() == Progress::Done {
            return removed;
        }
    }
    panic!("detach did not finish");
}

fn sorted(mut halves: Vec<Halves>) -> Vec<Halves> {
    halves.sort_by_key(|h| {
        let (s, t) = (h.source(), h.target());
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
        incident.push(g.insert_directed(x, l(0), v(1)).unwrap());
        g.insert_directed(v(4), l(0), v(1)).unwrap();
    }
    let gone = incident.remove(0);
    g.remove_edge(gone.source()).unwrap();
    incident.push(g.insert_directed(v(3), l(0), x).unwrap());
    incident.push(g.insert_directed(x, l(1), x).unwrap());
    incident.push(g.insert_directed(x, l(1), x).unwrap());
    incident.push(g.insert_undirected(v(2), l(1), x).unwrap());
    incident.push(g.insert_undirected(x, l(2), x).unwrap());
    let other = g.insert_directed(v(1), l(0), v(2)).unwrap();
    g.detach(x).unwrap();
    for (from, to) in [(x, v(1)), (v(1), x)] {
        assert_eq!(g.insert_directed(from, l(0), to), Err(Error::Detaching(x)));
    }
    assert_eq!(g.insert_undirected(x, l(9), x), Err(Error::Detaching(x)));
    let removed = detach_fully(&mut g, x, 1);
    assert_eq!(sorted(removed), sorted(incident));
    for side in [Side::Out, Side::In, Side::Undirected] {
        let mut labels = Vec::new();
        g.labels(x, side, &mut labels).unwrap();
        assert_eq!(labels, Vec::<Label>::new(), "{side:?}");
    }
    assert_eq!(g.counterpart(other.source()), Ok(Some(other.target())));
    assert_eq!(edges_on(&g, v(1), l(0), Side::In), [(1, 4), (3, 4), (5, 4)]);
    // Detached: the vertex takes edges again.
    g.insert_directed(x, l(0), v(1)).unwrap();
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
    g.detach(v(0)).unwrap();
    let step = g.detach_step(v(0), budget(1)).unwrap();
    assert_eq!((step.changes().len(), step.progress()), (0, Progress::Done));
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
            g.insert_directed(v(1), l(0), v(0)),
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
