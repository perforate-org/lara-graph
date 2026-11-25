//! A bucket's forms: tiny to run to pages, a deepening page index, appending in every form, and
//! compaction shrinking a hub's index.

use super::*;

#[test]
fn a_bucket_moves_from_tiny_to_run_to_pages() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    let total = PAGE_SLOTS * 3 + 5;
    for i in 0..total {
        assert_eq!(insert(&mut g, v(0), l(0), v(i % 2)), p(i));
        if matches!(i + 1, 3 | 4 | 8 | 9 | 1024 | 1025 | 2048 | 2049) {
            check(&g);
        }
    }
    assert!(matches!(
        out_bucket(&g, v(0), l(0)),
        Some(Descriptor::Paged(_))
    ));
    assert_eq!(edges(&g, v(0), l(0)).len(), usize::try_from(total).unwrap());
    // Kill a whole page: the scan skips it, positions elsewhere stay.
    for i in PAGE_SLOTS..2 * PAGE_SLOTS {
        remove(&mut g, v(0), l(0), p(i));
    }
    let live = edges(&g, v(0), l(0));
    assert_eq!(live.len(), usize::try_from(total - PAGE_SLOTS).unwrap());
    assert!(
        live.iter()
            .all(|&(pos, _)| !(PAGE_SLOTS..2 * PAGE_SLOTS).contains(&pos))
    );
    check(&g);
}

#[test]
fn a_full_run_becomes_the_first_page_without_moving() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    for _ in 0..PAGE_SLOTS {
        insert(&mut g, v(0), l(0), v(1));
    }
    let Some(Descriptor::Run(run)) = out_bucket(&g, v(0), l(0)) else {
        panic!("a full run")
    };
    insert(&mut g, v(0), l(0), v(1));
    let Some(Descriptor::Paged(paged)) = out_bucket(&g, v(0), l(0)) else {
        panic!("paged")
    };
    assert_eq!(
        paged::leaves(&g.edges.store().pages(), &paged)[0].page,
        Some(run.extent.page())
    );
    check(&g);
}

#[test]
fn the_page_index_deepens_past_one_page_of_entries() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    let total = (paged::FANOUT + 2) * PAGE_SLOTS + 3;
    let mut most = 0;
    for _ in 0..total {
        let before = g.edges.store().minted();
        // An undirected self-loop: one half, one bucket insert.
        g.add_edge(NewEdge::undirected(
            v(0),
            l(0),
            v(0),
            Placement::ReuseHoles,
            &[],
        ))
        .unwrap();
        most = most.max(g.edges.store().minted() - before);
    }
    let bound = pages_per_half(None);
    assert!(
        most <= bound,
        "an insert took {most} pages, more than {bound}"
    );
    check(&g);
    let positions: Vec<u32> = edges_on(&g, v(0), l(0), Direction::Undirected)
        .into_iter()
        .map(|(pos, _)| pos)
        .collect();
    assert_eq!(positions, (0..total).collect::<Vec<_>>());
    for at in [0, PAGE_SLOTS * paged::FANOUT, total - 1] {
        let half = EdgeRef::new(v(0), l(0), Direction::Undirected, p(at));
        assert_eq!(g.edge_target(half), Ok(Some(v(0))));
    }
    check(&backing.open());
}

#[test]
fn appending_never_fills_a_hole_in_any_bucket_form() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    for total in [3, 10, PAGE_SLOTS + 10] {
        let label = l(u16::try_from(total % 1000).unwrap());
        for _ in 0..total {
            insert(&mut g, v(0), label, v(1));
        }
        remove(&mut g, v(0), label, p(total - 2));
        // An edge to another node could safely take the hole; appending must not.
        let appended = g
            .add_edge(NewEdge::directed(v(0), label, v(2), Placement::Append, &[]))
            .unwrap();
        assert_eq!(appended.canonical().position(), p(total), "{total} edges");
        // No live edge to node 0 follows the hole, so reusing it is safe.
        let reused = g
            .add_edge(NewEdge::directed(
                v(0),
                label,
                v(0),
                Placement::ReuseHoles,
                &[],
            ))
            .unwrap();
        assert_eq!(reused.canonical().position(), p(total - 2), "{total} edges");
    }
    check(&g);
}

/// Compacting a hub whose edges were mostly removed packs its pages, then merges its index's
/// sparse leaves: the index gives most of its pages back and still pairs every half.
#[test]
fn compacting_a_mostly_emptied_hub_shrinks_its_index() {
    let backing = Backing::default();
    let mut g = backing.create(1);
    let hub = g.add_node().unwrap();
    let mut edges = Vec::new();
    for _ in 0..6000 {
        let leaf = g.add_node().unwrap();
        edges.push(
            g.add_edge(NewEdge::directed(hub, l(0), leaf, Placement::Append, &[]))
                .unwrap(),
        );
    }
    let index_pages = |g: &Graph<_>| {
        let found = g
            .find_label_bucket(
                hub,
                LabelBucketKey {
                    label: l(0),
                    direction: Direction::Outgoing,
                },
            )
            .unwrap()
            .unwrap();
        target_index::extents(&g.edges.store().pages(), found.bucket.index.tree()).len()
    };
    let before = index_pages(&g);
    for (i, halves) in edges.iter().enumerate() {
        if i % 16 != 0 {
            g.remove_edge(halves.other()).unwrap();
        }
    }
    while g
        .compact(hub, (l(0), Direction::Outgoing))
        .unwrap()
        .progress()
        == Progress::Remaining
    {}
    let after = index_pages(&g);
    assert!(after * 2 < before, "{after} index pages of {before}");
    for halves in edges.iter().step_by(16) {
        let canonical = g.counterpart(halves.other()).unwrap().unwrap();
        assert_eq!(g.counterpart(canonical), Ok(Some(halves.other())));
    }
    check(&g);
}
