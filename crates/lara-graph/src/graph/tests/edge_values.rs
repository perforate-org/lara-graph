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
fn valued_edges(g: &mut Graph<VecMemory>, label: Label, n: u32, width: usize) -> Vec<EdgeRef> {
    (0..n)
        .map(|i| {
            let value = val(u8::try_from(i % 251).unwrap(), width);
            g.add_edge(NewEdge::directed(
                v(0),
                label,
                v(1),
                Placement::Append,
                &value,
            ))
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
        while g
            .compact(v(0), (l(0), Direction::Outgoing))
            .unwrap()
            .progress()
            == Progress::Remaining
        {}
        g.compact(v(1), (l(0), Direction::Incoming)).unwrap();
        let mut seen = Vec::new();
        g.visit_neighbors_with_values(
            v(0),
            (l(0), Direction::Outgoing),
            Order::Ascending,
            |_, value| {
                seen.push(value.to_vec());
                ControlFlow::<()>::Continue(())
            },
        )
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
    g.visit_values(
        v(0),
        (l(0), Direction::Outgoing),
        Order::Ascending,
        |at, _| {
            assert!(
                !(PAGE_SLOTS..2 * PAGE_SLOTS).contains(&at.index()),
                "dead page read"
            );
            seen += 1;
            last = Some(at);
            ControlFlow::<()>::Continue(())
        },
    )
    .unwrap();
    assert_eq!((seen, last), (2 * PAGE_SLOTS, Some(p(3 * PAGE_SLOTS - 1))));
}

#[test]
fn values_have_the_labels_width_and_edges_before_it_read_as_zeros() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    let early = g
        .add_edge(NewEdge::directed(
            v(0),
            l(0),
            v(1),
            Placement::ReuseHoles,
            &[],
        ))
        .unwrap();
    assert_eq!(
        g.add_edge(NewEdge::directed(
            v(0),
            l(0),
            v(1),
            Placement::ReuseHoles,
            &[1]
        )),
        Err(Error::WrongValueLength {
            label: l(0),
            expected: 0,
            actual: 1
        })
    );
    g.declare_width(l(0), w(2)).unwrap();
    assert_eq!(g.edge_value(early.other()), Ok(Some(vec![0, 0])));
    assert_eq!(
        g.add_edge(NewEdge::directed(
            v(0),
            l(0),
            v(1),
            Placement::ReuseHoles,
            &[]
        )),
        Err(Error::WrongValueLength {
            label: l(0),
            expected: 2,
            actual: 0
        })
    );
    // The early edge's tiny buckets get value planes when it is first given a value.
    assert_eq!(
        g.set_edge_value(early.canonical(), &[7, 8]),
        Ok(Some(vec![0, 0]))
    );
    assert_eq!(g.edge_value(early.other()), Ok(Some(vec![7, 8])));
    let late = g
        .add_edge(NewEdge::directed(
            v(0),
            l(0),
            v(1),
            Placement::ReuseHoles,
            &[9, 9],
        ))
        .unwrap();
    let Ok(Removal::Removed(edge)) = g.remove_edge(late.other()) else {
        panic!("the edge is removed")
    };
    assert_eq!((edge.halves(), edge.value()), (late, &[9, 9][..]));
    assert_eq!(g.edge_value(late.canonical()), Ok(None));
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
        g.edge_value(EdgeRef::new(v(0), l(0), Direction::Outgoing, p(2))),
    );
    let memory = &backing.memories[1];
    memory.cap(memory.size());
    let capacity = u32::try_from(memory.size() * (PAGE_SIZE / 4096)).unwrap();
    let mut held = Vec::new();
    while capacity - g.edges.store().minted() > 3 {
        held.push(g.edges.store_mut().allocate(ByteClass::PAGE).unwrap());
    }
    let error = g.add_edge(NewEdge::directed(
        v(0),
        l(0),
        v(1),
        Placement::Append,
        &val(1, 4096),
    ));
    assert!(matches!(error, Err(Error::OutOfMemory(_))), "{error:?}");
    let after = (
        edges(&g, v(0), l(0)),
        g.edge_value(EdgeRef::new(v(0), l(0), Direction::Outgoing, p(2))),
    );
    assert_eq!(after, before);
    for extent in held {
        g.edges.store_mut().release(extent);
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
    let pages = |g: &Graph<VecMemory>| g.edges.store().allocated().len();
    // A dead middle page: compaction hollows it and its 16 value pages go with it.
    for half in
        &halves[usize::try_from(PAGE_SLOTS).unwrap()..usize::try_from(2 * PAGE_SLOTS).unwrap()]
    {
        g.remove_edge(*half).unwrap();
    }
    let before = pages(&g);
    g.compact(v(0), (l(0), Direction::Outgoing)).unwrap();
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
    while g
        .compact(v(0), (l(0), Direction::Outgoing))
        .unwrap()
        .progress()
        == Progress::Remaining
    {}
    assert!(before - pages(&g) >= 30, "{before} → {}", pages(&g));
    assert_eq!(g.edge_value(halves[0]), Ok(Some(val(0, 64))));
    check(&g);
}

#[test]
fn batch_edge_values_match_the_single_reads() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    g.declare_width(l(0), w(4)).unwrap();
    let halves = valued_edges(&mut g, l(0), 6, 4);
    g.remove_edge(halves[2]).unwrap();
    let expected: Vec<Option<Vec<u8>>> = halves
        .iter()
        .map(|half| g.edge_value(*half).unwrap())
        .collect();
    let mut batch = g.read_batch();
    let mut values = Vec::new();
    batch.edge_values_each(&halves, &mut values).unwrap();
    assert_eq!(values, expected);
    assert!(values[2].is_none());
}

#[test]
fn batch_value_visits_match_the_single_visits() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    g.declare_width(l(0), w(4)).unwrap();
    let halves = valued_edges(&mut g, l(0), 6, 4);
    g.remove_edge(halves[2]).unwrap();
    let nodes = [v(0), v(0)];
    let mut single = Vec::new();
    for _ in &nodes {
        g.visit_neighbors_with_values(
            v(0),
            (l(0), Direction::Outgoing),
            Order::Descending,
            |edge, value| {
                single.push((edge.position(), value.to_vec()));
                ControlFlow::<()>::Continue(())
            },
        )
        .unwrap();
    }
    let mut batch = Vec::new();
    let mut reader = g.read_batch();
    reader
        .visit_neighbors_with_values_each(
            &nodes,
            (l(0), Direction::Outgoing),
            Order::Descending,
            |_, edge, value| {
                batch.push((edge.position(), value.to_vec()));
                ControlFlow::<()>::Continue(())
            },
        )
        .unwrap();
    assert_eq!(batch, single);
}

#[test]
fn set_edge_values_matches_one_by_one() {
    let setup = || {
        let backing = Backing::default();
        let mut g = backing.create(2);
        g.declare_width(l(0), w(4)).unwrap();
        let halves = valued_edges(&mut g, l(0), 4, 4);
        (g, halves)
    };
    let (mut one, halves) = setup();
    let mut expected = Vec::new();
    for (i, &half) in halves.iter().enumerate() {
        let value = val(u8::try_from(i + 10).unwrap(), 4);
        expected.push(one.set_edge_value(half, &value).unwrap());
    }
    let (mut batch, halves) = setup();
    let new_values: Vec<Vec<u8>> = (0..4)
        .map(|i| val(u8::try_from(i + 10).unwrap(), 4))
        .collect();
    let writes: Vec<EdgeValue<'_>> = halves
        .iter()
        .zip(&new_values)
        .map(|(&half, value)| EdgeValue::new(half, value))
        .collect();
    let old = batch.set_edge_values(&writes).unwrap();
    assert_eq!(old, expected);
    for &half in &halves {
        assert_eq!(
            batch.edge_value(half).unwrap(),
            one.edge_value(half).unwrap()
        );
    }
}

#[test]
fn set_edge_values_checks_all_nodes_before_writing() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    g.declare_width(l(0), w(2)).unwrap();
    let half = valued_edges(&mut g, l(0), 1, 2)[0];
    g.set_edge_values(&[EdgeValue::new(half, &[9, 9])]).unwrap();
    let writes = [
        EdgeValue::new(half, &[8, 8]),
        EdgeValue::new(
            EdgeRef::new(v(9), l(0), Direction::Outgoing, Position::new(0)),
            &[8, 8],
        ),
    ];
    let error = g.set_edge_values(&writes).unwrap_err();
    assert_eq!(error.index(), 1);
    assert_eq!(*error.error(), Error::UnknownNode(v(9)));
    assert_eq!(g.edge_value(half).unwrap().as_deref(), Some(&[9, 9][..]));
}

#[test]
fn visit_values_each_matches_one_by_one() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    g.declare_width(l(0), w(2))
        .unwrap_or_else(|e| panic!("{e}"));
    for i in 0..4 {
        let value = val(i, 2);
        g.add_edge(NewEdge::directed(
            v(0),
            l(0),
            v(1),
            Placement::Append,
            &value,
        ))
        .unwrap();
    }
    for i in 4..6 {
        let value = val(i, 2);
        g.add_edge(NewEdge::directed(
            v(2),
            l(0),
            v(1),
            Placement::Append,
            &value,
        ))
        .unwrap();
    }
    let nodes = [v(0), v(2), v(1)];
    let mut batch = g.read_batch();
    let mut batched = Vec::new();
    batch
        .visit_values_each(
            &nodes,
            (l(0), Direction::Outgoing),
            Order::Descending,
            |node, position, value| {
                batched.push((node.index(), position.index(), value.to_vec()));
                ControlFlow::<()>::Continue(())
            },
        )
        .unwrap();
    let mut single = Vec::new();
    for &node in &nodes {
        g.visit_values(
            node,
            (l(0), Direction::Outgoing),
            Order::Descending,
            |position, value| {
                single.push((node.index(), position.index(), value.to_vec()));
                ControlFlow::<()>::Continue(())
            },
        )
        .unwrap();
    }
    assert_eq!(batched, single);
}

/// Any label can have a width: thousands of them, the highest included, survive a reopen and keep
/// their widths, and a declared width cannot change.
#[test]
fn every_label_can_declare_a_width() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    let labels: Vec<u16> = (0..3000u16).map(|i| i * 21).chain([u16::MAX]).collect();
    for &label in &labels {
        g.declare_width(l(label), w(label % 64 + 1)).unwrap();
    }
    assert_eq!(g.declare_width(l(21), w(22)), Ok(()));
    assert_eq!(
        g.declare_width(l(21), w(3)),
        Err(Error::WidthConflict {
            label: l(21),
            declared: w(22)
        })
    );
    check(&g);
    drop(g);
    let g = backing.open();
    for &label in &labels {
        assert_eq!(g.width(l(label)), Some(w(label % 64 + 1)));
    }
    assert_eq!(g.width(l(1)), None);
    check(&g);
}

// --- layout changes: change_width, stale buckets and migrate_values ---

const OUT: Direction = Direction::Outgoing;
const IN: Direction = Direction::Incoming;

/// The new value of `old`: its bytes reversed, then 0xEE up to `width`.
fn widened(old: &[u8], new: &mut [u8]) {
    for (to, from) in new
        .iter_mut()
        .zip(old.iter().rev().chain(core::iter::repeat(&0xEE)))
    {
        *to = *from;
    }
}

/// Migrates `x`'s bucket to the end, counting steps; checks the graph every `check_every`.
fn migrate_fully(g: &mut Graph<VecMemory>, x: NodeId, key: (Label, Direction)) -> u32 {
    let mut steps = 0;
    while g.migrate_values(x, key, widened).unwrap() == Progress::Remaining {
        steps += 1;
        assert!(steps < 10_000, "migration did not finish");
    }
    steps
}

#[test]
fn a_width_change_leaves_stored_values_in_their_version_until_migrated() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    g.declare_width(l(0), w(8)).unwrap();
    let halves = valued_edges(&mut g, l(0), 5, 8);
    assert_eq!(g.value_version(l(0)), Some(ValueVersion::FIRST));
    assert_eq!(g.change_width(l(0), w(4)), Ok(ValueVersion::new(1)));
    assert_eq!(g.width(l(0)), Some(w(4)));
    // Stored values read in the width and version they were written in.
    assert_eq!(g.edge_value(halves[2]), Ok(Some(val(2, 8))));
    assert_eq!(
        g.bucket_value_version(v(0), (l(0), OUT)),
        Ok(Some(ValueVersion::FIRST))
    );
    assert_eq!(g.needs_migration(v(0), (l(0), OUT)), Ok(true));
    // A bucket with no values yet is in the new layout.
    assert_eq!(g.bucket_value_version(v(2), (l(0), OUT)), Ok(None));
    let stale = Error::StaleValues {
        node: v(0),
        label: l(0),
        direction: OUT,
    };
    assert_eq!(
        g.add_edge(NewEdge::directed(
            v(0),
            l(0),
            v(2),
            Placement::Append,
            &[1; 4]
        )),
        Err(stale)
    );
    assert_eq!(g.set_edge_value(halves[0], &[1; 4]), Err(stale));
    // New buckets (here both of a self-loop's) take the new layout at once.
    let fresh = g
        .add_edge(NewEdge::directed(
            v(2),
            l(0),
            v(2),
            Placement::Append,
            &[7; 4],
        ))
        .unwrap();
    assert_eq!(g.edge_value(fresh.canonical()), Ok(Some(vec![7; 4])));
    assert_eq!(
        g.bucket_value_version(v(2), (l(0), OUT)),
        Ok(Some(ValueVersion::new(1)))
    );
    // Removing from a stale bucket reports the value's own version.
    let Ok(Removal::Removed(removed)) = g.remove_edge(halves[4]) else {
        panic!("the edge is removed")
    };
    assert_eq!(
        (removed.value(), removed.value_version()),
        (&val(4, 8)[..], ValueVersion::FIRST)
    );
    check(&g);
}

#[test]
fn small_buckets_migrate_and_then_take_writes() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    g.declare_width(l(0), w(8)).unwrap();
    let halves = valued_edges(&mut g, l(0), 6, 8);
    // The same width in a new layout: only the version tells the values apart.
    g.change_width(l(0), w(8)).unwrap();
    assert_eq!(g.needs_migration(v(1), (l(0), IN)), Ok(true));
    for (x, direction) in [(v(0), OUT), (v(1), IN)] {
        migrate_fully(&mut g, x, (l(0), direction));
        assert_eq!(g.needs_migration(x, (l(0), direction)), Ok(false));
        assert_eq!(
            g.bucket_value_version(x, (l(0), direction)),
            Ok(Some(ValueVersion::new(1)))
        );
        check(&g);
    }
    for (i, half) in (0u8..).zip(&halves) {
        let mut want = vec![0; 8];
        widened(&val(i, 8), &mut want);
        assert_eq!(g.edge_value(*half), Ok(Some(want)));
    }
    assert_eq!(
        g.migrate_values(v(0), (l(0), OUT), widened),
        Ok(Progress::Done)
    );
    g.add_edge(NewEdge::directed(
        v(0),
        l(0),
        v(1),
        Placement::Append,
        &[3; 8],
    ))
    .unwrap();
    g.set_edge_value(halves[0], &[9; 8]).unwrap();
    check(&g);
    check(&backing.open());
}

#[test]
fn a_hub_migrates_a_page_at_a_time_and_reads_old_values_until_the_end() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    g.declare_width(l(0), w(8)).unwrap();
    let n = 3 * PAGE_SLOTS;
    let halves = valued_edges(&mut g, l(0), n, 8);
    for half in halves.iter().step_by(7) {
        g.remove_edge(*half).unwrap();
    }
    g.change_width(l(0), w(24)).unwrap();
    let key = (l(0), OUT);
    // A few steps in: still the old values, and no writes.
    for _ in 0..3 {
        assert_eq!(
            g.migrate_values(v(0), key, widened),
            Ok(Progress::Remaining)
        );
    }
    check(&g);
    assert_eq!(g.edge_value(halves[1]), Ok(Some(val(1, 8))));
    assert_eq!(
        g.bucket_value_version(v(0), key),
        Ok(Some(ValueVersion::FIRST))
    );
    assert!(matches!(
        g.set_edge_value(halves[1], &[0; 24]),
        Err(Error::StaleValues { .. })
    ));
    // The record survives a reopen.
    let mut g = backing.open();
    check(&g);
    // Compaction starts it over: its first step only cancels.
    assert_eq!(
        g.compact(v(0), key).unwrap().progress(),
        Progress::Remaining,
        "a cancelling step"
    );
    check(&g);
    while g.compact(v(0), key).unwrap().progress() == Progress::Remaining {}
    let steps = migrate_fully(&mut g, v(0), key);
    assert!(steps > 3, "a hub takes several steps ({steps})");
    check(&g);
    let mut live = Vec::new();
    g.neighbors(v(0), key, &mut live).unwrap();
    let mut values = Vec::new();
    g.visit_neighbors_with_values(v(0), key, Order::Ascending, |e, value| {
        values.push((e.position(), value.to_vec()));
        ControlFlow::<()>::Continue(())
    })
    .unwrap();
    assert_eq!(values.len(), live.len());
    for (_, value) in &values {
        assert_eq!(value.len(), 24);
        assert_eq!(&value[8..], &[0xEE; 16]);
    }
    assert_eq!(
        g.bucket_value_version(v(0), key),
        Ok(Some(ValueVersion::new(1)))
    );
    migrate_fully(&mut g, v(1), (l(0), IN));
    g.set_edge_value(halves[1], &[5; 24]).unwrap();
    assert_eq!(g.edge_value(halves[1]), Ok(Some(vec![5; 24])));
    check(&g);
}

#[test]
fn a_second_change_restarts_a_migration_under_way() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    g.declare_width(l(0), w(8)).unwrap();
    let halves = valued_edges(&mut g, l(0), 2 * PAGE_SLOTS, 8);
    g.change_width(l(0), w(16)).unwrap();
    let key = (l(0), OUT);
    g.migrate_values(v(0), key, widened).unwrap();
    g.migrate_values(v(0), key, widened).unwrap();
    g.change_width(l(0), w(4)).unwrap();
    check(&g);
    migrate_fully(&mut g, v(0), key);
    assert_eq!(
        g.bucket_value_version(v(0), key),
        Ok(Some(ValueVersion::new(2)))
    );
    let mut want = vec![0; 4];
    widened(&val(5, 8), &mut want);
    assert_eq!(g.edge_value(halves[5]), Ok(Some(want)));
    check(&g);
}
