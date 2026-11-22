//! Edge values: widths, writes, reads, removal reports and compaction.

use super::*;

fn w(bytes: u16) -> Width {
    Width::new(bytes).unwrap()
}

fn val(seed: u8, width: usize) -> Vec<u8> {
    (0..width)
        .map(|i| seed.wrapping_add(u8::try_from(i % 256).unwrap()))
        .collect()
}

/// Inserts `0 → 1` edges under `label` with values `val(i)`; returns their canonical halves.
fn valued_edges(g: &mut Graph<&VecMemory>, label: Label, n: u32, width: usize) -> Vec<EdgeRef> {
    (0..n)
        .map(|i| {
            let value = val(u8::try_from(i % 251).unwrap(), width);
            g.insert_directed(v(0), label, v(1), Placement::Append, &value)
                .unwrap()
                .canonical()
        })
        .collect()
}

#[test]
fn compaction_moves_values_with_their_edges() {
    for (n, width) in [(12, 8), (3 * PAGE_SLOTS, 24)] {
        let backing = Backing::default();
        let mut g = backing.create(2);
        g.declare_width(l(0), w(u16::try_from(width).unwrap()))
            .unwrap();
        let halves = valued_edges(&mut g, l(0), n, width);
        // Remove most edges, keeping every fifth, then compact to the end.
        for (i, half) in (0u32..).zip(&halves) {
            if i % 5 != 0 {
                g.remove_edge(*half).unwrap();
            }
        }
        while g.compact(v(0), l(0), Side::Out).unwrap().progress() == Progress::Remaining {}
        g.compact(v(1), l(0), Side::In).unwrap();
        let mut seen = Vec::new();
        g.visit_neighbors_with_values(v(0), (l(0), Side::Out), Order::Ascending, |_, value| {
            seen.push(value.to_vec());
            ControlFlow::<()>::Continue(())
        })
        .unwrap();
        let want: Vec<Vec<u8>> = (0..n)
            .filter(|i| i % 5 == 0)
            .map(|i| val(u8::try_from(i % 251).unwrap(), width))
            .collect();
        assert_eq!(seen, want, "{n} edges of {width} bytes");
        assert!(!matches!(
            out_bucket(&g, v(0), l(0)),
            Some(Descriptor::Tiny(_))
        ));
        check(&g);
    }
}

#[test]
fn a_values_first_scan_skips_dead_pages_but_reads_past_them() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    g.declare_width(l(0), w(4)).unwrap();
    let halves = valued_edges(&mut g, l(0), 3 * PAGE_SLOTS, 4);
    for half in halves
        .iter()
        .skip(usize::try_from(PAGE_SLOTS).unwrap())
        .take(usize::try_from(PAGE_SLOTS).unwrap())
    {
        g.remove_edge(*half).unwrap();
    }
    let mut seen = 0u32;
    let mut last = None;
    g.visit_values(v(0), (l(0), Side::Out), Order::Ascending, |at, _| {
        assert!(
            !(PAGE_SLOTS..2 * PAGE_SLOTS).contains(&at.index()),
            "dead page read"
        );
        seen += 1;
        last = Some(at);
        ControlFlow::<()>::Continue(())
    })
    .unwrap();
    assert_eq!((seen, last), (2 * PAGE_SLOTS, Some(p(3 * PAGE_SLOTS - 1))));
}

#[test]
fn values_have_the_labels_width_and_edges_before_it_read_as_zeros() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    let early = g
        .insert_directed(v(0), l(0), v(1), Placement::ReuseHoles, &[])
        .unwrap();
    assert_eq!(
        g.insert_directed(v(0), l(0), v(1), Placement::ReuseHoles, &[1]),
        Err(Error::WrongValueLength {
            label: l(0),
            expected: 0,
            actual: 1
        })
    );
    g.declare_width(l(0), w(2)).unwrap();
    assert_eq!(g.value(early.other()), Ok(Some(vec![0, 0])));
    assert_eq!(
        g.insert_directed(v(0), l(0), v(1), Placement::ReuseHoles, &[]),
        Err(Error::WrongValueLength {
            label: l(0),
            expected: 2,
            actual: 0
        })
    );
    // The early edge's tiny buckets get value planes when it is first given a value.
    assert_eq!(
        g.set_value(early.canonical(), &[7, 8]),
        Ok(Some(vec![0, 0]))
    );
    assert_eq!(g.value(early.other()), Ok(Some(vec![7, 8])));
    let late = g
        .insert_directed(v(0), l(0), v(1), Placement::ReuseHoles, &[9, 9])
        .unwrap();
    assert_eq!(
        g.remove_edge(late.other()),
        Ok(Removal::Removed(RemovedEdge::new(late, vec![9, 9])))
    );
    assert_eq!(g.value(late.canonical()), Ok(None));
    check(&g);
    check(&backing.open());
}

#[test]
fn an_insert_with_a_value_that_runs_out_of_memory_changes_nothing() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    g.declare_width(l(0), w(4096)).unwrap();
    valued_edges(&mut g, l(0), 3, 4096);
    let before = (
        edges(&g, v(0), l(0)),
        g.value(EdgeRef::new(v(0), l(0), Side::Out, p(2))),
    );
    let memory = &backing.memories[1];
    memory.cap(memory.size());
    let capacity = u32::try_from(memory.size() * (PAGE_SIZE / 4096)).unwrap();
    let mut held = Vec::new();
    while capacity - g.edges.minted() > 3 {
        held.push(g.edges.allocate(ByteClass::PAGE).unwrap());
    }
    let error = g.insert_directed(v(0), l(0), v(1), Placement::Append, &val(1, 4096));
    assert!(matches!(error, Err(Error::OutOfMemory(_))), "{error:?}");
    let after = (
        edges(&g, v(0), l(0)),
        g.value(EdgeRef::new(v(0), l(0), Side::Out, p(2))),
    );
    assert_eq!(after, before);
    for extent in held {
        g.edges.release(extent);
    }
    check(&g);
}

#[test]
fn compaction_releases_the_value_pages_it_empties() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    g.declare_width(l(0), w(64)).unwrap();
    // 64-byte values: 64 per value page, 16 value pages per slot page.
    let halves = valued_edges(&mut g, l(0), 3 * PAGE_SLOTS, 64);
    let pages = |g: &Graph<&VecMemory>| g.edges.allocated().len();
    // A dead middle page: compaction hollows it and its 16 value pages go with it.
    for half in
        &halves[usize::try_from(PAGE_SLOTS).unwrap()..usize::try_from(2 * PAGE_SLOTS).unwrap()]
    {
        g.remove_edge(*half).unwrap();
    }
    let before = pages(&g);
    g.compact(v(0), l(0), Side::Out).unwrap();
    assert!(
        before - pages(&g) >= 17,
        "slot page and value pages released"
    );
    check(&g);
    // Demoting to a run leaves positions past the new end: their value pages go too.
    for half in halves.iter().skip(1) {
        g.remove_edge(*half).unwrap();
    }
    let before = pages(&g);
    while g.compact(v(0), l(0), Side::Out).unwrap().progress() == Progress::Remaining {}
    assert!(before - pages(&g) >= 30, "{before} → {}", pages(&g));
    assert_eq!(g.value(halves[0]), Ok(Some(val(0, 64))));
    check(&g);
}
