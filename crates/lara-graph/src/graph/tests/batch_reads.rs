//! Reads of many nodes or halves through one `ReadBatch`, against the same reads one by one.

use super::*;

#[test]
fn visiting_many_neighbors_matches_one_by_one_and_stops() {
    let backing = Backing::default();
    let mut g = backing.create(4);
    insert(&mut g, v(0), l(0), v(1));
    insert(&mut g, v(0), l(0), v(2));
    insert(&mut g, v(1), l(0), v(2));
    insert(&mut g, v(2), l(0), v(0));
    let nodes = [v(0), v(3), v(1), v(2)];
    let mut batched = Vec::new();
    let mut batch = g.read_batch();
    let none = batch.visit_neighbors_each(
        &nodes,
        (l(0), Direction::Outgoing),
        Order::Ascending,
        |node, e| {
            batched.push((node.index(), e.position().index(), e.target().index()));
            ControlFlow::<()>::Continue(())
        },
    );
    assert_eq!(none, Ok(None));
    drop(batch);
    let mut one_by_one = Vec::new();
    for &node in &nodes {
        let mut out = Vec::new();
        g.neighbors(node, (l(0), Direction::Outgoing), &mut out)
            .unwrap();
        for e in out {
            one_by_one.push((node.index(), e.position().index(), e.target().index()));
        }
    }
    assert_eq!(batched, one_by_one);

    let mut seen = 0;
    let mut batch = g.read_batch();
    let stopped = batch.visit_neighbors_each(
        &nodes,
        (l(0), Direction::Outgoing),
        Order::Ascending,
        |_, _| {
            seen += 1;
            if seen == 2 {
                ControlFlow::Break(seen)
            } else {
                ControlFlow::Continue(())
            }
        },
    );
    assert_eq!(stopped, Ok(Some(2)));
    drop(batch);

    let mut called = false;
    let mut batch = g.read_batch();
    assert_eq!(
        batch.visit_neighbors_each(
            &[v(0), v(9)],
            (l(0), Direction::Outgoing),
            Order::Ascending,
            |_, _| {
                called = true;
                ControlFlow::<()>::Continue(())
            },
        ),
        Err(Error::UnknownNode(v(9)))
    );
    assert!(!called);
}

#[test]
fn visiting_many_reads_dense_ids_in_lines() {
    let backing = Backing::default();
    let mut g = backing.create(13);
    for target in 1..=12 {
        insert(&mut g, v(0), l(0), v(target));
        // A second bucket per satellite, so its list has two entries.
        insert(&mut g, v(target), l(0), v(0));
    }
    let nodes: Vec<NodeId> = (1..=12).map(v).collect();
    let mut seen = Vec::new();
    let mut batch = g.read_batch();
    batch
        .visit_neighbors_each(
            &nodes,
            (l(0), Direction::Incoming),
            Order::Ascending,
            |node, e| {
                seen.push((node.index(), e.target().index()));
                ControlFlow::<()>::Continue(())
            },
        )
        .unwrap();
    assert_eq!(seen.len(), 12);
    for (at, (node, target)) in seen.into_iter().enumerate() {
        assert_eq!(node, u32::try_from(at).unwrap() + 1);
        assert_eq!(target, 0);
    }
}

#[test]
fn read_batch_shares_buffers_across_reads() {
    let backing = Backing::default();
    let mut g = backing.create(13);
    for target in 1..=12 {
        insert(&mut g, v(0), l(0), v(target));
        insert(&mut g, v(target), l(0), v(0));
    }
    let nodes: Vec<NodeId> = (1..=12).map(v).collect();
    let mut batch = g.read_batch();
    let mut degrees = Vec::new();
    batch
        .degree_each(&nodes, (l(0), Direction::Incoming), &mut degrees)
        .unwrap();
    assert_eq!(degrees, vec![1; 12]);
    let mut states = Vec::new();
    batch.node_state_each(&nodes, &mut states).unwrap();
    assert_eq!(states, vec![NodeState::Live; 12]);
    let mut seen = 0;
    batch
        .visit_neighbors_each(
            &nodes,
            (l(0), Direction::Incoming),
            Order::Ascending,
            |_, _| {
                seen += 1;
                ControlFlow::<()>::Continue(())
            },
        )
        .unwrap();
    assert_eq!(seen, 12);
    let mut labels = Vec::new();
    batch
        .edge_labels_each(&nodes, Direction::Incoming, |node, label| {
            labels.push((node.index(), label.get()));
            ControlFlow::<()>::Continue(())
        })
        .unwrap();
    assert_eq!(labels, (1..=12).map(|v| (v, 0)).collect::<Vec<_>>());
    let mut edges = Vec::new();
    batch.has_edges_each(&nodes, &mut edges).unwrap();
    assert_eq!(edges, alloc::vec![true; 12]);
    let mut degrees = Vec::new();
    assert_eq!(
        batch.degree_each(&[v(0), v(99)], (l(0), Direction::Outgoing), &mut degrees),
        Err(Error::UnknownNode(v(99)))
    );
    assert_eq!(degrees, Vec::<u32>::new());
    drop(batch);
    // The batch's shared borrow has ended; the graph can be mutated again.
    g.add_node().unwrap();
}

#[test]
fn batch_edge_targets_match_one_by_one() {
    let backing = Backing::default();
    let mut g = backing.create(10);
    for target in 1..=8 {
        insert(&mut g, v(0), l(0), v(target));
    }
    let mut edges = Vec::new();
    g.neighbors(v(0), (l(0), Direction::Outgoing), &mut edges)
        .unwrap();
    let halves: Vec<EdgeRef> = edges
        .iter()
        .map(|edge| EdgeRef::new(v(0), l(0), Direction::Outgoing, edge.position()))
        .collect();
    let mut batch = g.read_batch();
    let mut targets = Vec::new();
    batch.edge_targets_each(&halves, &mut targets).unwrap();
    let expected: Vec<Option<NodeId>> = halves
        .iter()
        .map(|&half| g.edge_target(half).unwrap())
        .collect();
    assert_eq!(targets, expected);
    let mut untouched = vec![Some(v(1))];
    assert_eq!(
        batch.edge_targets_each(
            &[EdgeRef::new(
                v(99),
                l(0),
                Direction::Outgoing,
                Position::new(0)
            )],
            &mut untouched,
        ),
        Err(Error::UnknownNode(v(99)))
    );
    assert_eq!(untouched, vec![Some(v(1))]);
    drop(batch);
    g.remove_edge(halves[0]).unwrap();
    let mut batch = g.read_batch();
    let mut after = Vec::new();
    batch.edge_targets_each(&halves, &mut after).unwrap();
    let expected: Vec<Option<NodeId>> = halves
        .iter()
        .map(|&half| g.edge_target(half).unwrap())
        .collect();
    assert_eq!(after, expected);
    assert_eq!(after[0], None);
}

#[test]
fn batch_counterparts_match_one_by_one() {
    let backing = Backing::default();
    let mut g = backing.create(9);
    for target in 1..=8 {
        insert(&mut g, v(0), l(0), v(target));
    }
    let mut edges = Vec::new();
    g.neighbors(v(0), (l(0), Direction::Outgoing), &mut edges)
        .unwrap();
    let halves: Vec<EdgeRef> = edges
        .iter()
        .map(|edge| EdgeRef::new(v(0), l(0), Direction::Outgoing, edge.position()))
        .collect();
    let expected: Vec<Option<EdgeRef>> = halves
        .iter()
        .map(|half| g.counterpart(*half).unwrap())
        .collect();
    let mut out = Vec::new();
    let mut batch = g.read_batch();
    batch.counterparts_each(&halves, &mut out).unwrap();
    assert_eq!(out, expected);
    assert_eq!(
        batch.counterparts_each(
            &[EdgeRef::new(
                v(99),
                l(0),
                Direction::Outgoing,
                Position::new(0)
            )],
            &mut out
        ),
        Err(Error::UnknownNode(v(99)))
    );
    assert_eq!(out, expected);
}

#[test]
fn batch_counterparts_reuses_parallel_target_buckets() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    for _ in 0..8 {
        insert(&mut g, v(0), l(0), v(1));
    }
    let mut edges = Vec::new();
    g.neighbors(v(0), (l(0), Direction::Outgoing), &mut edges)
        .unwrap();
    let halves: Vec<EdgeRef> = edges
        .iter()
        .map(|edge| EdgeRef::new(v(0), l(0), Direction::Outgoing, edge.position()))
        .collect();
    let expected: Vec<Option<EdgeRef>> = halves
        .iter()
        .map(|half| g.counterpart(*half).unwrap())
        .collect();
    let mut out = Vec::new();
    let mut batch = g.read_batch();
    batch.counterparts_each(&halves, &mut out).unwrap();
    assert_eq!(out, expected);
}
