//! Reading a node's edges under every label at once, against a loop over its labels.

use super::model::Rng;
use super::*;
use crate::id::Neighbor;

const NODES: u32 = 64;
const LABELS: u16 = 6;

/// Edges of many labels in all three directions, a paged hub, and holes.
fn graph(backing: &Backing) -> Graph<VecMemory> {
    let mut g = backing.create(NODES);
    let mut rng = Rng(0xA11);
    let mut placed = Vec::new();
    for i in 0..1500 {
        let edge = NewEdge::directed(v(0), l(0), v(i % NODES), Placement::ReuseHoles, &[]);
        placed.push(g.add_edge(edge).unwrap());
    }
    for _ in 0..3000 {
        let from = if rng.below(3) == 0 {
            0
        } else {
            rng.below(NODES)
        };
        let (to, label) = (
            rng.below(NODES),
            l(u16::try_from(rng.below(u32::from(LABELS))).unwrap()),
        );
        let edge = if rng.below(4) == 0 {
            NewEdge::undirected(v(from), label, v(to), Placement::ReuseHoles, &[])
        } else {
            NewEdge::directed(v(from), label, v(to), Placement::ReuseHoles, &[])
        };
        placed.push(g.add_edge(edge).unwrap());
    }
    for halves in placed {
        if rng.below(5) == 0 {
            g.remove_edge(halves.canonical()).unwrap();
        }
    }
    g
}

const DIRECTIONS: [Direction; 3] = [
    Direction::Outgoing,
    Direction::Incoming,
    Direction::Undirected,
];

/// `x`'s edges on `direction` label by label, through one bucket read each.
fn by_label(
    g: &Graph<VecMemory>,
    x: u32,
    direction: Direction,
    order: Order,
) -> Vec<(Label, Neighbor)> {
    let mut labels = Vec::new();
    g.edge_labels(v(x), direction, &mut labels).unwrap();
    if order == Order::Descending {
        labels.reverse();
    }
    let mut out = Vec::new();
    for label in labels {
        g.visit_neighbors(v(x), (label, direction), order, |e| {
            out.push((label, e));
            ControlFlow::<()>::Continue(())
        })
        .unwrap();
    }
    out
}

#[test]
fn every_label_reads_as_a_loop_over_the_labels() {
    let backing = Backing::default();
    let g = graph(&backing);
    assert!(
        matches!(out_bucket(&g, v(0), l(0)), Some(Descriptor::Paged(_))),
        "the hub has a paged bucket"
    );
    for x in 0..NODES {
        for direction in DIRECTIONS {
            for order in [Order::Ascending, Order::Descending] {
                let mut got = Vec::new();
                let none = g
                    .visit_all_neighbors(v(x), direction, order, |label, e| {
                        got.push((label, e));
                        ControlFlow::<()>::Continue(())
                    })
                    .unwrap();
                assert_eq!(none, None);
                let want = by_label(&g, x, direction, order);
                assert_eq!(got, want, "node {x} {direction:?} {order:?}");
            }
        }
    }
}

#[test]
fn every_label_reads_stop_where_the_visitor_breaks() {
    let backing = Backing::default();
    let g = graph(&backing);
    let want = by_label(&g, 0, Direction::Outgoing, Order::Descending);
    let mut seen = 0;
    let found = g
        .visit_all_neighbors(v(0), Direction::Outgoing, Order::Descending, |label, e| {
            seen += 1;
            if seen == 100 {
                ControlFlow::Break((label, e))
            } else {
                ControlFlow::Continue(())
            }
        })
        .unwrap();
    assert_eq!(found, want.get(99).copied());
    assert!(
        g.visit_all_neighbors(v(NODES), Direction::Outgoing, Order::Ascending, |_, _| {
            ControlFlow::<()>::Continue(())
        })
        .is_err()
    );
}

#[test]
fn batched_every_label_reads_match_single_reads() {
    let backing = Backing::default();
    let g = graph(&backing);
    let nodes: Vec<NodeId> = [3, 0, 17, 3, 63, 40].into_iter().map(v).collect();
    for direction in DIRECTIONS {
        for order in [Order::Ascending, Order::Descending] {
            let want: Vec<(NodeId, Label, Neighbor)> = nodes
                .iter()
                .flat_map(|&x| {
                    by_label(&g, x.index(), direction, order)
                        .into_iter()
                        .map(move |(label, e)| (x, label, e))
                })
                .collect();
            let mut got = Vec::new();
            g.read_batch()
                .visit_all_neighbors_each(&nodes, direction, order, |x, label, e| {
                    got.push((x, label, e));
                    ControlFlow::<()>::Continue(())
                })
                .unwrap();
            assert_eq!(got, want, "{direction:?} {order:?}");
        }
    }
}
