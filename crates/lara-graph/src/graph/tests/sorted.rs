//! Sorted views and shared targets against a plain model: small buckets, paged ones with their
//! indexed blocks and unindexed tails, parallel edges and holes.

use super::model::Rng;
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
