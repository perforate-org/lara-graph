//! Compaction of each bucket mode.

use super::*;

/// A bucket of `n` edges to node 0, with the positions in `holes` removed.
fn bucket_with_holes(
    backing: &Backing,
    n: u32,
    holes: impl IntoIterator<Item = u32>,
) -> Graph<VecMemory> {
    let mut g = backing.create(2);
    for _ in 0..n {
        insert(&mut g, v(0), l(0), v(1));
    }
    for at in holes {
        remove(&mut g, v(0), l(0), p(at));
    }
    g
}

#[test]
fn compacting_a_run_packs_it_and_reports_every_move() {
    let backing = Backing::default();
    let mut g = bucket_with_holes(&backing, 100, (0..100).filter(|i| i % 4 != 0));
    let c = g.compact(v(0), l(0), Direction::Outgoing).unwrap();
    assert_eq!(c.progress(), Progress::Done);
    assert_eq!(c.changes().len(), 24, "edges 4, 8, … 96 move; edge 0 stays");
    assert_eq!(
        edges(&g, v(0), l(0)),
        (0..25).map(|i| (i, 1)).collect::<Vec<_>>()
    );
    let Some(Descriptor::Run(run)) = out_bucket(&g, v(0), l(0)) else {
        panic!("a run")
    };
    assert_eq!(run.capacity(), 32, "shrunk to the smallest class");
    check(&g);
}

#[test]
fn a_compacted_run_becomes_tiny_and_an_empty_bucket_goes_away() {
    let backing = Backing::default();
    let mut g = bucket_with_holes(&backing, 10, [0, 1, 2, 3, 4, 5, 7]);
    g.compact(v(0), l(0), Direction::Outgoing).unwrap();
    assert!(matches!(
        out_bucket(&g, v(0), l(0)),
        Some(Descriptor::Tiny(_))
    ));
    for at in 0..3 {
        remove(&mut g, v(0), l(0), p(at));
    }
    insert(&mut g, v(0), l(5), v(1));
    g.compact(v(0), l(0), Direction::Outgoing).unwrap();
    let mut labels = Vec::new();
    g.labels(v(0), Direction::Outgoing, &mut labels).unwrap();
    assert_eq!(labels, [l(5)], "the empty bucket's label is gone");
    check(&g);
}

#[test]
fn a_dead_middle_page_is_released_without_moving_anything() {
    let backing = Backing::default();
    let mut g = bucket_with_holes(&backing, 3 * PAGE_SLOTS, PAGE_SLOTS..2 * PAGE_SLOTS);
    let before = edges(&g, v(0), l(0));
    let c = g.compact(v(0), l(0), Direction::Outgoing).unwrap();
    assert_eq!((c.changes().len(), c.progress()), (0, Progress::Remaining));
    assert_eq!(edges(&g, v(0), l(0)), before);
    check(&g);
    assert_eq!(
        g.compact(v(0), l(0), Direction::Outgoing)
            .unwrap()
            .progress(),
        Progress::Done
    );
}

#[test]
fn two_half_pages_merge_into_one() {
    let backing = Backing::default();
    // Pages 0 and 1 keep their even positions; page 2 (the last) is untouched.
    let mut g = bucket_with_holes(
        &backing,
        3 * PAGE_SLOTS,
        (0..2 * PAGE_SLOTS).filter(|i| i % 2 == 1),
    );
    let c = g.compact(v(0), l(0), Direction::Outgoing).unwrap();
    assert_eq!(
        c.changes().len(),
        usize::try_from(PAGE_SLOTS - 1).unwrap(),
        "all but position 0 move"
    );
    let live = edges(&g, v(0), l(0));
    assert_eq!(live.len(), usize::try_from(2 * PAGE_SLOTS).unwrap());
    assert!(
        live.iter()
            .take(usize::try_from(PAGE_SLOTS).unwrap())
            .enumerate()
            .all(|(i, &(at, _))| at == u32::try_from(i).unwrap())
    );
    check(&g);
}

#[test]
fn a_sparse_paged_bucket_is_demoted_to_a_run() {
    let backing = Backing::default();
    let mut g = bucket_with_holes(
        &backing,
        2 * PAGE_SLOTS + 1,
        (0..2 * PAGE_SLOTS).filter(|i| i % 8 != 0),
    );
    let c = g.compact(v(0), l(0), Direction::Outgoing).unwrap();
    assert_eq!(c.progress(), Progress::Done);
    assert!(matches!(
        out_bucket(&g, v(0), l(0)),
        Some(Descriptor::Run(_))
    ));
    assert_eq!(g.holes(v(0), l(0), Direction::Outgoing), Ok(0));
    assert_eq!(edges(&g, v(0), l(0)).len(), 257);
    check(&g);
}

#[test]
fn a_dead_last_page_is_kept_for_inserts() {
    let backing = Backing::default();
    // Three full pages; the last one is all holes, and too much stays live to demote.
    let mut g = bucket_with_holes(&backing, 3 * PAGE_SLOTS, 2 * PAGE_SLOTS..3 * PAGE_SLOTS);
    let c = g.compact(v(0), l(0), Direction::Outgoing).unwrap();
    assert_eq!(
        (c.changes().len(), c.progress()),
        (0, Progress::Done),
        "nothing to do"
    );
    assert_eq!(
        insert(&mut g, v(0), l(0), v(1)),
        p(2 * PAGE_SLOTS),
        "the last page's first hole"
    );
    check(&g);
}
