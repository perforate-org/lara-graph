//! Reading one bucket: visiting its neighbors, windows over its positions and reads at given
//! positions. Reads through a batch, of the edges between two nodes, and of sorted and shared
//! targets have modules of their own.

use super::model::Rng;
use super::*;
use crate::id::Neighbor;

mod batch;
mod between;
mod sorted;

#[test]
fn visiting_neighbors_sees_live_edges_in_order_and_stops_on_break() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    for target in [1, 2, 1, 2] {
        insert(&mut g, v(0), l(0), v(target));
    }
    remove(&mut g, v(0), l(0), p(1));
    let mut seen = Vec::new();
    let none = g.visit_neighbors(v(0), l(0).outgoing(), Order::Ascending, |e| {
        seen.push((e.position().index(), e.node().index()));
        ControlFlow::<()>::Continue(())
    });
    assert_eq!(none, Ok(None));
    assert_eq!(seen, edges(&g, v(0), l(0)));
    let second_to_2 = g.visit_neighbors(v(0), l(0).outgoing(), Order::Ascending, |e| {
        if e.node() == v(2) {
            ControlFlow::Break(e.position())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(second_to_2, Ok(Some(p(3))));
    assert_eq!(
        g.visit_neighbors(v(1), l(5).incoming(), Order::Ascending, |_| {
            ControlFlow::Break(())
        }),
        Ok(None)
    );
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
    let window = |g: &Graph<VecMemory>, cursor| {
        let mut seen = Vec::new();
        let next = g
            .visit_window(v(0), l(0).outgoing(), cursor, |e| {
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
            l(9).incoming(),
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
    g.compact(v(0), l(0).outgoing()).unwrap();
    remove(&mut g, v(0), l(0), p(5));
    let positions = [
        p(3 * PAGE_SLOTS),
        p(5),
        p(PAGE_SLOTS + 7),
        p(4),
        p(2 * PAGE_SLOTS),
    ];
    let mut out = Vec::new();
    g.neighbors_at(v(0), l(0).outgoing(), &positions, &mut out)
        .unwrap();
    assert_eq!(out, [None, None, None, Some(v(1)), Some(v(1))]);
    g.neighbors_at(v(1), l(3).outgoing(), &positions, &mut out)
        .unwrap();
    assert_eq!(out, [None; 5]);
}

/// The position after skipping any number of live edges agrees with the live edges listed in
/// order, in tiny, run and paged buckets with holes, whole dead pages included, both ways.
#[test]
fn nth_position_skips_live_edges_in_order() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    let key = l(0).outgoing();
    let check_all = |g: &Graph<VecMemory>| {
        let live = edges(g, v(0), l(0));
        for order in [Order::Ascending, Order::Descending] {
            let listed: Vec<u32> = match order {
                Order::Ascending => live.iter().map(|&(at, _)| at).collect(),
                Order::Descending => live.iter().rev().map(|&(at, _)| at).collect(),
            };
            for skip in 0..=u32::try_from(listed.len()).unwrap() + 1 {
                let expected = listed.get(usize::try_from(skip).unwrap()).copied().map(p);
                assert_eq!(
                    g.nth_position(v(0), key, order, skip),
                    Ok(expected),
                    "{order:?} {skip}"
                );
            }
        }
    };
    for count in [1u32, 3, 900, 3000] {
        while edges(&g, v(0), l(0)).len() < usize::try_from(count).unwrap() {
            insert(&mut g, v(0), l(0), v(1));
        }
        check_all(&g);
    }
    // Holes everywhere, then a whole dead page.
    for at in (0..3000).step_by(7) {
        remove(&mut g, v(0), l(0), p(at));
    }
    check_all(&g);
    for at in 1024..2048 {
        remove(&mut g, v(0), l(0), p(at));
    }
    check_all(&g);
    // Compaction releases the dead page and leaves its index entry hollow.
    g.compact(v(0), key).unwrap();
    assert!(matches!(
        out_bucket(&g, v(0), l(0)),
        Some(Descriptor::Paged(_))
    ));
    check_all(&g);
    assert_eq!(g.nth_position(v(1), key, Order::Ascending, 0), Ok(None));
}

// --- every label at once: a node's edges under all its labels, against a loop over them ---

const LABELED_NODES: u32 = 64;
const LABELS: u16 = 6;

/// Edges of many labels in all three directions, a paged hub, and holes.
fn labeled_graph(backing: &Backing) -> Graph<VecMemory> {
    let mut g = backing.create(LABELED_NODES);
    let mut rng = Rng(0xA11);
    let mut placed = Vec::new();
    for i in 0..1500 {
        let edge = Edge::directed(v(0), l(0), v(i % LABELED_NODES));
        placed.push(g.add_edge(edge).unwrap());
    }
    for _ in 0..3000 {
        let from = if rng.below(3) == 0 {
            0
        } else {
            rng.below(LABELED_NODES)
        };
        let (to, label) = (
            rng.below(LABELED_NODES),
            l(u16::try_from(rng.below(u32::from(LABELS))).unwrap()),
        );
        let edge = if rng.below(4) == 0 {
            Edge::undirected(v(from), label, v(to))
        } else {
            Edge::directed(v(from), label, v(to))
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
) -> Vec<(EdgeLabel, Neighbor)> {
    let mut labels = Vec::new();
    g.edge_labels(v(x), direction, &mut labels).unwrap();
    if order == Order::Descending {
        labels.reverse();
    }
    let mut out = Vec::new();
    for label in labels {
        g.visit_neighbors(v(x), Adjacency::new(label, direction), order, |e| {
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
    let g = labeled_graph(&backing);
    assert!(
        matches!(out_bucket(&g, v(0), l(0)), Some(Descriptor::Paged(_))),
        "the hub has a paged bucket"
    );
    for x in 0..LABELED_NODES {
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
    let g = labeled_graph(&backing);
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
        g.visit_all_neighbors(
            v(LABELED_NODES),
            Direction::Outgoing,
            Order::Ascending,
            |_, _| { ControlFlow::<()>::Continue(()) }
        )
        .is_err()
    );
}

#[test]
fn batched_every_label_reads_match_single_reads() {
    let backing = Backing::default();
    let g = labeled_graph(&backing);
    let nodes: Vec<NodeId> = [3, 0, 17, 3, 63, 40].into_iter().map(v).collect();
    for direction in DIRECTIONS {
        for order in [Order::Ascending, Order::Descending] {
            let want: Vec<(NodeId, EdgeLabel, Neighbor)> = nodes
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
