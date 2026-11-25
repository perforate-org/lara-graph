//! Reading one bucket: visiting its neighbors, windows over its positions and reads at given
//! positions.

use super::*;

#[test]
fn visiting_neighbors_sees_live_edges_in_order_and_stops_on_break() {
    let backing = Backing::default();
    let mut g = backing.create(3);
    for target in [1, 2, 1, 2] {
        insert(&mut g, v(0), l(0), v(target));
    }
    remove(&mut g, v(0), l(0), p(1));
    let mut seen = Vec::new();
    let none = g.visit_neighbors(v(0), (l(0), Direction::Outgoing), Order::Ascending, |e| {
        seen.push((e.position().index(), e.target().index()));
        ControlFlow::<()>::Continue(())
    });
    assert_eq!(none, Ok(None));
    assert_eq!(seen, edges(&g, v(0), l(0)));
    let second_to_2 = g.visit_neighbors(v(0), (l(0), Direction::Outgoing), Order::Ascending, |e| {
        if e.target() == v(2) {
            ControlFlow::Break(e.position())
        } else {
            ControlFlow::Continue(())
        }
    });
    assert_eq!(second_to_2, Ok(Some(p(3))));
    assert_eq!(
        g.visit_neighbors(v(1), (l(5), Direction::Incoming), Order::Ascending, |_| {
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
            .visit_window(v(0), (l(0), Direction::Outgoing), cursor, |e| {
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
            (l(9), Direction::Incoming),
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
    g.compact(v(0), (l(0), Direction::Outgoing)).unwrap();
    remove(&mut g, v(0), l(0), p(5));
    let positions = [
        p(3 * PAGE_SLOTS),
        p(5),
        p(PAGE_SLOTS + 7),
        p(4),
        p(2 * PAGE_SLOTS),
    ];
    let mut out = Vec::new();
    g.targets_at(v(0), (l(0), Direction::Outgoing), &positions, &mut out)
        .unwrap();
    assert_eq!(out, [None, None, None, Some(v(1)), Some(v(1))]);
    g.targets_at(v(1), (l(3), Direction::Outgoing), &positions, &mut out)
        .unwrap();
    assert_eq!(out, [None; 5]);
}

/// The position after skipping any number of live edges agrees with the live edges listed in
/// order, in tiny, run and paged buckets with holes, whole dead pages included, both ways.
#[test]
fn nth_position_skips_live_edges_in_order() {
    let backing = Backing::default();
    let mut g = backing.create(2);
    let key = (l(0), Direction::Outgoing);
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
