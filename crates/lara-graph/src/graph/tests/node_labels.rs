//! Node labels: sets of any size beside the record, through replacement, bulk load, reopening
//! and deletion.

use super::*;

fn labels_of<M: Memory>(g: &Graph<M>, x: NodeId) -> Vec<u16> {
    let mut out = Vec::new();
    g.node_labels(x, &mut out).unwrap();
    out.into_iter().map(Label::get).collect()
}

fn set(raw: &[u16]) -> Vec<Label> {
    raw.iter().copied().map(l).collect()
}

/// Sets of zero to many labels, given unsorted with repeats, read back sorted and distinct; a
/// replacement frees a spilled set; every label not in a set reads absent.
#[test]
fn node_labels_hold_sets_of_any_size() {
    let backing = Backing::default();
    let mut g = backing.create(6);
    let sets: [&[u16]; 6] = [
        &[],
        &[7],
        &[9, 2, 9],
        &[5, 1, 3],
        &[u16::MAX, 0, 40_000, 12, 12, 7],
        &[300; 4],
    ];
    for (i, raw) in sets.iter().enumerate() {
        g.set_node_labels(v(u32::try_from(i).unwrap()), &set(raw))
            .unwrap();
    }
    check(&g);
    for (i, raw) in sets.iter().enumerate() {
        let x = v(u32::try_from(i).unwrap());
        let mut want: Vec<u16> = raw.to_vec();
        want.sort_unstable();
        want.dedup();
        assert_eq!(labels_of(&g, x), want, "node {i}");
        for probe in [0, 1, 2, 3, 5, 7, 9, 12, 300, 40_000, u16::MAX, 8] {
            assert_eq!(
                g.has_node_label(x, l(probe)).unwrap(),
                want.contains(&probe)
            );
        }
    }
    // Spilled to inline, inline to spilled, spilled to none: nothing leaks.
    g.set_node_labels(v(4), &set(&[1])).unwrap();
    g.set_node_labels(v(1), &set(&[1, 2, 3, 4])).unwrap();
    g.set_node_labels(v(3), &[]).unwrap();
    check(&g);
    assert_eq!(labels_of(&g, v(1)), [1, 2, 3, 4]);
    // Edges and new nodes leave labels alone, and labels survive a reopen.
    insert(&mut g, v(1), l(0), v(2));
    g.add_nodes(NonZeroU32::new(20).unwrap()).unwrap();
    drop(g);
    let g = backing.open();
    assert_eq!(labels_of(&g, v(1)), [1, 2, 3, 4]);
    assert_eq!(labels_of(&g, v(2)), [2, 9]);
    assert_eq!(labels_of(&g, v(25)), Vec::<u16>::new());
    check(&g);
}

#[test]
fn node_labels_check_their_node_and_their_count() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    assert_eq!(g.has_node_label(v(9), l(0)), Err(Error::UnknownNode(v(9))));
    assert_eq!(g.set_node_labels(v(9), &[]), Err(Error::UnknownNode(v(9))));
    let most: Vec<Label> = (0..2047).map(l).collect();
    g.set_node_labels(v(0), &most).unwrap();
    let too_many: Vec<Label> = (0..2048).map(l).collect();
    assert_eq!(
        g.set_node_labels(v(0), &too_many),
        Err(Error::TooManyNodeLabels(v(0)))
    );
    assert_eq!(labels_of(&g, v(0)).len(), 2047);
    assert!(g.has_node_label(v(0), l(2046)).unwrap());
    check(&g);
}

/// Bulk load writes records over lines of nodes and keeps their labels; a deleted node drops
/// its labels and takes no new ones.
#[test]
fn node_labels_survive_bulk_load_and_go_with_deletion() {
    let backing = Backing::default();
    let mut g = backing.create(10);
    for i in 0..10 {
        let raw: Vec<u16> = (0..=(i % 4))
            .map(|k| u16::try_from(i * 10 + k).unwrap())
            .collect();
        g.set_node_labels(v(i), &set(&raw)).unwrap();
    }
    let edges: Vec<NewEdge<'_>> = (0..9)
        .map(|i| NewEdge::directed(v(i), l(0), v(i + 1), Placement::Append, &[]))
        .collect();
    g.bulk_load(&edges).unwrap();
    for i in 0..10 {
        let want: Vec<u16> = (0..=(i % 4))
            .map(|k| u16::try_from(i * 10 + k).unwrap())
            .collect();
        assert_eq!(labels_of(&g, v(i)), want, "node {i}");
    }
    check(&g);
    g.detach(v(3)).unwrap();
    while g.detach_step(v(3), NonZeroU32::MIN).unwrap().progress() == Progress::Remaining {}
    check(&g);
    assert_eq!(labels_of(&g, v(3)), Vec::<u16>::new());
    assert_eq!(
        g.set_node_labels(v(3), &set(&[1])),
        Err(Error::Deleted(v(3)))
    );
}
