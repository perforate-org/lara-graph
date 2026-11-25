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

/// Listed and bitmap sets at their edges, up to every label, swapped for one another without
/// leaking pages.
#[test]
fn node_labels_take_any_set_of_labels() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    assert_eq!(g.has_node_label(v(9), l(0)), Err(Error::UnknownNode(v(9))));
    assert_eq!(g.set_node_labels(v(9), &[]), Err(Error::UnknownNode(v(9))));
    let every: Vec<Label> = (0..=u16::MAX).rev().map(l).collect();
    let evens: Vec<Label> = (0..=u16::MAX).step_by(2).map(l).collect();
    for set in [
        (0..2047).map(l).collect::<Vec<_>>(),
        (0..2048).map(l).collect(),
        every.clone(),
        evens.clone(),
        (100..103).map(l).collect(),
    ] {
        g.set_node_labels(v(0), &set).unwrap();
        check(&g);
        let mut want: Vec<u16> = set.iter().map(|x| x.get()).collect();
        want.sort_unstable();
        assert_eq!(labels_of(&g, v(0)), want);
        for probe in [0, 1, 2046, 2047, 32_767, 32_768, 65_534, u16::MAX] {
            assert_eq!(
                g.has_node_label(v(0), l(probe)).unwrap(),
                want.binary_search(&probe).is_ok()
            );
        }
    }
    g.set_node_labels(v(1), &every).unwrap();
    g.set_node_labels(v(1), &[]).unwrap();
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

/// Each label's nodes come back in id order across chunks of 65,536 ids, through a list that
/// becomes a bitmap and back to empty, resuming after any node.
#[test]
fn nodes_with_a_label_come_back_in_id_order() {
    let backing = Backing::default();
    let mut g = backing.create(0);
    g.add_nodes(NonZeroU32::new(70_000).unwrap()).unwrap();
    let nodes_with = |g: &Graph<VecMemory>, label: u16, after: Option<NodeId>| {
        let mut out = Vec::new();
        g.visit_nodes_with_label(l(label), after, |x| {
            out.push(x.index());
            ControlFlow::<()>::Continue(())
        });
        out
    };
    // Every third node gets label 1 (a bitmap in the first chunk, a list in the second), every
    // thousandth label 2 too; labelled from the top down, so lists take inserts at the front.
    let mut want1 = Vec::new();
    let mut want2 = Vec::new();
    for i in (0..70_000u32).rev().filter(|i| i % 3 == 0) {
        let labels = if i % 1000 == 0 {
            set(&[1, 2])
        } else {
            set(&[1])
        };
        g.set_node_labels(v(i), &labels).unwrap();
        want1.push(i);
        if i % 1000 == 0 {
            want2.push(i);
        }
    }
    want1.reverse();
    want2.reverse();
    check(&g);
    assert_eq!(nodes_with(&g, 1, None), want1);
    assert_eq!(nodes_with(&g, 2, None), want2);
    assert_eq!(
        g.node_label_count(l(1)),
        u32::try_from(want1.len()).unwrap()
    );
    assert_eq!(nodes_with(&g, 3, None), Vec::<u32>::new());
    let resumed: Vec<u32> = want1.iter().copied().filter(|&i| i > 65_535).collect();
    assert_eq!(nodes_with(&g, 1, Some(v(65_535))), resumed);
    // A break stops the walk at once.
    let mut seen = 0;
    let stopped = g.visit_nodes_with_label(l(1), None, |_| {
        seen += 1;
        if seen == 5 {
            ControlFlow::Break(seen)
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(stopped, Some(5));
    // Losing label 1 everywhere empties the bitmap and the list, and frees them.
    for i in want1.clone() {
        let keep = if i % 1000 == 0 { set(&[2]) } else { Vec::new() };
        g.set_node_labels(v(i), &keep).unwrap();
    }
    check(&g);
    assert_eq!(nodes_with(&g, 1, None), Vec::<u32>::new());
    assert_eq!(g.node_label_count(l(1)), 0);
    assert_eq!(nodes_with(&g, 2, None), want2);
}

/// A label set whose postings run out of memory part way changes nothing: the postings added
/// before the failure are taken back, and the node keeps its old set.
#[test]
fn node_labels_that_run_out_of_memory_change_nothing() {
    let backing = Backing::default();
    let mut g = backing.create(4);
    g.set_node_labels(v(0), &set(&[1, 2, 3])).unwrap();
    g.set_node_labels(v(1), &set(&[2])).unwrap();
    let memory = &backing.memories[usize::from(MemoryRole::LabelBucketListPages.index())];
    memory.cap(memory.size());
    let capacity = u32::try_from(memory.size() * (PAGE_SIZE / 4096)).unwrap();
    // Leave room for the new set and the first few postings only.
    let mut held = Vec::new();
    while capacity - g.label_bucket_lists.minted() > 26 {
        held.push(g.label_bucket_lists.allocate(ByteClass::PAGE).unwrap());
    }
    let many: Vec<Label> = (0..6000).map(l).collect();
    let error = g.set_node_labels(v(0), &many);
    assert!(matches!(error, Err(Error::OutOfMemory(_))), "{error:?}");
    assert_eq!(labels_of(&g, v(0)), [1, 2, 3]);
    assert_eq!(g.node_label_count(l(2)), 2);
    assert_eq!(g.node_label_count(l(0)), 0);
    for extent in held {
        g.label_bucket_lists.release(extent);
    }
    check(&g);
}

/// A batch reads each node's labels, and whether it has one, as single reads do: for dense runs
/// of ids (read eight cells at a time), sparse ids, the last partial line, and sets of any size.
#[test]
fn batched_label_reads_agree_with_single_reads() {
    let backing = Backing::default();
    let mut g = backing.create(37);
    for i in 0..37u32 {
        let raw: Vec<u16> = (0..i % 5 * 3)
            .map(|k| u16::try_from(k * 7 + i).unwrap())
            .collect();
        g.set_node_labels(v(i), &set(&raw)).unwrap();
    }
    let dense: Vec<NodeId> = (0..37).map(v).collect();
    let sparse: Vec<NodeId> = [36, 3, 17, 0, 29, 36].map(v).to_vec();
    for nodes in [dense, sparse] {
        let mut batch = g.read_batch();
        let mut listed = Vec::new();
        assert_eq!(
            batch.node_labels_each(&nodes, |x, label| {
                listed.push((x.index(), label.get()));
                ControlFlow::<()>::Continue(())
            }),
            Ok(None)
        );
        let single: Vec<(u32, u16)> = nodes
            .iter()
            .flat_map(|&x| {
                labels_of(&g, x)
                    .into_iter()
                    .map(move |label| (x.index(), label))
            })
            .collect();
        assert_eq!(listed, single);
        for label in [0, 7, 14, 21, 1000] {
            let mut has = Vec::new();
            batch
                .has_node_label_each(&nodes, l(label), &mut has)
                .unwrap();
            let single: Vec<bool> = nodes
                .iter()
                .map(|&x| g.has_node_label(x, l(label)).unwrap())
                .collect();
            assert_eq!(has, single, "label {label}");
        }
    }
    let mut batch = g.read_batch();
    let mut has = Vec::new();
    assert_eq!(
        batch.has_node_label_each(&[v(0), v(99)], l(0), &mut has),
        Err(Error::UnknownNode(v(99)))
    );
    assert_eq!(
        batch.node_labels_each(&[v(1), v(1)], |_, _| ControlFlow::Break(7)),
        Ok(Some(7))
    );
}
