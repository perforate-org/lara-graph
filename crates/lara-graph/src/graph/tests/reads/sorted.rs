//! Sorted views and shared targets against a plain model: small buckets, paged ones with their
//! indexed blocks and unindexed tails, parallel edges and holes.

use super::*;

const NODES: u32 = 400;
const LABEL: Label = Label::new(3);

/// Two paged hubs, nodes of every small size, parallel edges, then a fifth of the edges removed.
fn graph(backing: &Backing) -> Graph<VecMemory> {
    let mut g = backing.create(NODES);
    let mut rng = Rng(0x5EED);
    let mut placed = Vec::new();
    for (from, degree) in [(0, 3000), (1, 2500)]
        .into_iter()
        .chain((2..NODES).map(|i| (i, i % 41)))
    {
        for _ in 0..degree {
            let to = v(rng.below(NODES));
            placed.push((v(from), insert(&mut g, v(from), LABEL, to)));
        }
    }
    for (from, at) in placed {
        if rng.below(5) == 0 {
            g.remove_edge(out_ref(from, LABEL, at)).unwrap();
        }
    }
    g
}

fn sorted_model(g: &Graph<VecMemory>, x: u32) -> Vec<u32> {
    let mut targets: Vec<u32> = edges(g, v(x), LABEL).into_iter().map(|(_, t)| t).collect();
    targets.sort_unstable();
    targets
}

fn sorted(g: &Graph<VecMemory>, x: u32) -> Vec<u32> {
    let mut out = Vec::new();
    g.sorted_targets(v(x), (LABEL, Direction::Outgoing), &mut out)
        .unwrap();
    out.into_iter().map(NodeId::index).collect()
}

#[test]
fn sorted_targets_list_every_live_edge_in_node_order() {
    let backing = Backing::default();
    let g = graph(&backing);
    for hub in [0, 1] {
        assert!(
            matches!(out_bucket(&g, v(hub), LABEL), Some(Descriptor::Paged(_))),
            "node {hub} is a paged hub"
        );
    }
    for x in 0..NODES {
        assert_eq!(sorted(&g, x), sorted_model(&g, x), "node {x}");
    }
    let mut out = vec![v(0)];
    g.sorted_targets(v(5), (l(9), Direction::Outgoing), &mut out)
        .unwrap();
    assert!(out.is_empty(), "a missing bucket lists nothing");
}

#[test]
fn common_targets_match_set_intersection_whichever_way_they_are_found() {
    let backing = Backing::default();
    let g = graph(&backing);
    // Hub and hub (merged), hub and small (looked up), small and small (merged), and a node
    // with no bucket.
    let pairs = [
        (0, 1),
        (1, 0),
        (0, 7),
        (40, 1),
        (0, 80),
        (9, 40),
        (39, 81),
        (41, 3),
    ];
    for (a, b) in pairs {
        let ours: BTreeSet<u32> = sorted_model(&g, a).into_iter().collect();
        let theirs: BTreeSet<u32> = sorted_model(&g, b).into_iter().collect();
        let model: Vec<u32> = ours.intersection(&theirs).copied().collect();
        let mut out = Vec::new();
        g.common_targets(v(a), v(b), (LABEL, Direction::Outgoing), &mut out)
            .unwrap();
        let got: Vec<u32> = out.into_iter().map(NodeId::index).collect();
        assert_eq!(got, model, "nodes {a} and {b}");
    }
    let mut out = Vec::new();
    assert!(
        g.common_targets(v(0), v(NODES), (LABEL, Direction::Outgoing), &mut out)
            .is_err()
    );
}

#[test]
fn common_targets_of_many_match_the_intersection_of_all() {
    let backing = Backing::default();
    let g = graph(&backing);
    let groups: [&[u32]; 7] = [
        &[],
        &[7],
        &[0, 1, 40],
        &[0, 7, 80],
        &[39, 81, 120, 160],
        &[0, 1, 2, 3],
        &[1, 0, 41, 82],
    ];
    for group in groups {
        let model: Vec<u32> = group
            .iter()
            .map(|&x| sorted_model(&g, x).into_iter().collect::<BTreeSet<u32>>())
            .reduce(|a, b| a.intersection(&b).copied().collect())
            .map_or_else(Vec::new, |set| set.into_iter().collect());
        let nodes: Vec<NodeId> = group.iter().map(|&x| v(x)).collect();
        let mut out = vec![v(0)];
        g.common_targets_of(&nodes, (LABEL, Direction::Outgoing), &mut out)
            .unwrap();
        let got: Vec<u32> = out.into_iter().map(NodeId::index).collect();
        assert_eq!(got, model, "nodes {group:?}");
    }
}

#[test]
fn a_target_set_intersects_with_every_bucket_and_stays_as_it_was() {
    let backing = Backing::default();
    let g = graph(&backing);
    // Hubs (bitmaps, and probed when the other side is small), small nodes whose few targets
    // spread wide (hash tables), and a node with no bucket.
    for a in [0, 1, 2, 3, 7, 40, 41, 81, 399] {
        let mut set = g.target_set(v(a), (LABEL, Direction::Outgoing)).unwrap();
        let ours: BTreeSet<u32> = sorted_model(&g, a).into_iter().collect();
        assert_eq!(set.len(), ours.len(), "node {a}");
        assert!(ours.iter().all(|&t| set.contains(v(t))), "node {a}");
        let mut out = vec![v(5)];
        for b in 0..NODES {
            let theirs: BTreeSet<u32> = sorted_model(&g, b).into_iter().collect();
            let model: Vec<u32> = ours.intersection(&theirs).copied().collect();
            g.common_with(
                &mut set,
                (v(b), None),
                (LABEL, Direction::Outgoing),
                &mut out,
            )
            .unwrap();
            let got: Vec<u32> = out.iter().map(|t| t.index()).collect();
            assert_eq!(got, model, "nodes {a} and {b}");
        }
        assert_eq!(set.len(), ours.len(), "node {a}: the set is unchanged");
        assert!(
            g.common_with(
                &mut set,
                (v(NODES), None),
                (LABEL, Direction::Outgoing),
                &mut out
            )
            .is_err()
        );
    }
}

#[test]
fn target_sets_hold_proxies_with_local_nodes() {
    let backing = Backing::default();
    let mut g = backing.create(4);
    let proxy = g.add_proxy().unwrap();
    for from in [0, 1] {
        for to in [v(2), proxy, v(3), proxy] {
            insert(&mut g, v(from), LABEL, to);
        }
    }
    insert(&mut g, v(2), LABEL, proxy);
    let mut set = g.target_set(v(0), (LABEL, Direction::Outgoing)).unwrap();
    assert_eq!(set.len(), 3);
    assert!(set.contains(proxy) && !set.contains(v(1)));
    let mut out = Vec::new();
    g.common_with(
        &mut set,
        (v(1), None),
        (LABEL, Direction::Outgoing),
        &mut out,
    )
    .unwrap();
    assert_eq!(out, [v(2), v(3), proxy]);
    g.common_with(
        &mut set,
        (v(2), Some(v(3))),
        (LABEL, Direction::Outgoing),
        &mut out,
    )
    .unwrap();
    assert_eq!(out, [proxy]);
    g.common_targets_of(&[v(0), v(1), v(2)], (LABEL, Direction::Outgoing), &mut out)
        .unwrap();
    assert_eq!(out, [proxy]);
}

#[test]
fn a_bound_keeps_only_the_shared_targets_past_it() {
    let backing = Backing::default();
    let g = graph(&backing);
    for a in [0, 1, 3, 40, 81] {
        let mut set = g.target_set(v(a), (LABEL, Direction::Outgoing)).unwrap();
        let ours: BTreeSet<u32> = sorted_model(&g, a).into_iter().collect();
        let mut out = Vec::new();
        for b in [0, 1, 7, 40, 80, 120, 399] {
            let theirs: BTreeSet<u32> = sorted_model(&g, b).into_iter().collect();
            for above in [0, 1, b, 63, 64, 200, 398, 399] {
                let model: Vec<u32> = ours
                    .intersection(&theirs)
                    .copied()
                    .filter(|&t| t > above)
                    .collect();
                g.common_with(
                    &mut set,
                    (v(b), Some(v(above))),
                    (LABEL, Direction::Outgoing),
                    &mut out,
                )
                .unwrap();
                let got: Vec<u32> = out.iter().map(|t| t.index()).collect();
                assert_eq!(got, model, "nodes {a} and {b} past {above}");
            }
        }
    }
}

#[test]
fn common_with_each_matches_one_node_at_a_time() {
    let backing = Backing::default();
    let g = graph(&backing);
    let nodes: Vec<NodeId> = (0..NODES).map(v).collect();
    let key = (LABEL, Direction::Outgoing);
    let mut batch = g.read_batch();
    for a in [0, 1, 3, 40, 81, 399] {
        let mut set = g.target_set(v(a), key).unwrap();
        let mut want = Vec::new();
        let mut out = Vec::new();
        for &w in &nodes {
            g.common_with(&mut set, (w, Some(w)), key, &mut out)
                .unwrap();
            if !out.is_empty() {
                want.push((w, out.clone()));
            }
        }
        let mut got = Vec::new();
        let none = batch
            .common_with_each(&mut set, &nodes, key, Some, |w, shared| {
                got.push((w, shared.to_vec()));
                ControlFlow::<()>::Continue(())
            })
            .unwrap();
        assert_eq!((none, got), (None, want), "node {a}");
    }
    let mut set = g.target_set(v(0), key).unwrap();
    let mut calls = 0;
    let first = batch
        .common_with_each(
            &mut set,
            &nodes,
            key,
            |_| None,
            |w, _| {
                calls += 1;
                ControlFlow::Break(w)
            },
        )
        .unwrap();
    assert_eq!(
        (first.is_some(), calls),
        (true, 1),
        "a break ends the batch"
    );
    let mut called = false;
    assert!(
        batch
            .common_with_each(
                &mut set,
                &[v(0), v(NODES)],
                key,
                |_| None,
                |_, _| {
                    called = true;
                    ControlFlow::<()>::Continue(())
                }
            )
            .is_err()
    );
    assert!(!called, "every node is checked before the first callback");
}
