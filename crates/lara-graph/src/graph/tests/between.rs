//! Edges between two nodes against a brute-force filter of one end's bucket: small buckets, an
//! indexed hub with parallel edges, undirected edges, self-loops and holes.

use super::*;

/// The edges between `a` and `b` found by `visit_edges_between`, each named by its half at `a`,
/// in the visit's order.
fn between_at_a(
    g: &Graph<VecMemory>,
    a: NodeId,
    key: (Label, Direction),
    b: NodeId,
    order: Order,
) -> Vec<Position> {
    let mut out = Vec::new();
    g.visit_edges_between(a, key, b, order, |half| {
        let at_a = if half.node() == a && half.direction() == key.1 {
            half
        } else {
            g.counterpart(half).unwrap().unwrap()
        };
        assert_eq!(
            (at_a.node(), at_a.label(), at_a.direction()),
            (a, key.0, key.1)
        );
        out.push(at_a.position());
        ControlFlow::<()>::Continue(())
    })
    .unwrap();
    out
}

/// The same, from a filter of `a`'s whole bucket.
fn brute(
    g: &Graph<VecMemory>,
    a: NodeId,
    key: (Label, Direction),
    b: NodeId,
    order: Order,
) -> Vec<Position> {
    let mut out = Vec::new();
    g.visit_neighbors(a, key.0, key.1, order, |e| {
        if e.target() == b {
            out.push(e.position());
        }
        ControlFlow::<()>::Continue(())
    })
    .unwrap();
    out
}

#[test]
fn edges_between_two_nodes_match_a_filter_of_either_end() {
    let mut g = Graph::new(Memories::from_fn(|_| VecMemory::new()));
    let nodes: Vec<NodeId> = (0..40).map(|_| g.add_node().unwrap()).collect();
    let (hub, rest) = nodes.split_first().unwrap();
    let hub = *hub;
    let mut halves = Vec::new();
    // A hub over a page of edges (so indexed), with parallel edges to a few targets spread over
    // every block, and single edges to the rest.
    for i in 0..3000u32 {
        let t = rest[usize::try_from(i % 7).unwrap()];
        halves.push(
            g.add_edge(EdgeKind::Directed, hub, l(0), t, Placement::Append, &[])
                .unwrap(),
        );
    }
    for &t in &rest[7..] {
        halves.push(
            g.add_edge(EdgeKind::Directed, hub, l(0), t, Placement::Append, &[])
                .unwrap(),
        );
    }
    // Small buckets: parallel edges both ways, undirected edges and self-loops.
    for (i, w) in rest.windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        for _ in 0..=i % 3 {
            halves.push(
                g.add_edge(EdgeKind::Directed, a, l(0), b, Placement::Append, &[])
                    .unwrap(),
            );
            halves.push(
                g.add_edge(EdgeKind::Undirected, a, l(1), b, Placement::Append, &[])
                    .unwrap(),
            );
        }
        halves.push(
            g.add_edge(EdgeKind::Directed, b, l(0), a, Placement::Append, &[])
                .unwrap(),
        );
    }
    for &a in rest.iter().step_by(5) {
        halves.push(
            g.add_edge(EdgeKind::Directed, a, l(0), a, Placement::Append, &[])
                .unwrap(),
        );
        halves.push(
            g.add_edge(EdgeKind::Undirected, a, l(1), a, Placement::Append, &[])
                .unwrap(),
        );
        halves.push(
            g.add_edge(EdgeKind::Directed, hub, l(0), a, Placement::Append, &[])
                .unwrap(),
        );
    }
    // A second hub that takes in more edges than the first sends, 300 of them from the first:
    // between the two, the first hub's indexed bucket is the smaller end.
    let hub2 = rest[39 - 1];
    for i in 0..4500u32 {
        let from = if i % 15 == 0 {
            hub
        } else {
            rest[usize::try_from(i % 30).unwrap()]
        };
        halves.push(
            g.add_edge(EdgeKind::Directed, from, l(0), hub2, Placement::Append, &[])
                .unwrap(),
        );
    }
    // Holes: every fifth edge goes.
    for h in halves.iter().step_by(5) {
        g.remove_edge(h.canonical()).unwrap();
    }
    let keys = [
        (l(0), Direction::Outgoing),
        (l(0), Direction::Incoming),
        (l(1), Direction::Undirected),
        (l(2), Direction::Outgoing),
    ];
    for &a in &nodes {
        for &b in &nodes {
            for key in keys {
                for order in [Order::Ascending, Order::Descending] {
                    let want = brute(&g, a, key, b, order);
                    assert_eq!(
                        between_at_a(&g, a, key, b, order),
                        want,
                        "{a:?} {key:?} {b:?} {order:?}"
                    );
                    assert_eq!(g.contains_edge(a, key, b).unwrap(), !want.is_empty());
                }
            }
        }
    }
    assert_eq!(
        g.contains_edge(hub, (l(0), Direction::Outgoing), v(1000)),
        Err(Error::UnknownNode(v(1000)))
    );
}
