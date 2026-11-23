//! Random adds and removes on one list against a sorted map, through flat and tiered shapes.

use super::*;
use crate::descriptor::{Descriptor, Tiny};
use crate::memory::VecMemory;
use alloc::collections::BTreeSet;

fn key(i: u32) -> LabelBucketKey {
    let direction = match i % 3 {
        0 => Direction::Outgoing,
        1 => Direction::Incoming,
        _ => Direction::Undirected,
    };
    LabelBucketKey {
        label: Label::new(u16::try_from(i / 3).unwrap()),
        direction,
    }
}

/// A descriptor that names its key, so a misplaced one shows.
fn descriptor(i: u32) -> Stored {
    let tiny = (0..i % 3).fold(Tiny::EMPTY, |t, _| {
        t.push(crate::descriptor::Slot::Live(crate::id::NodeId::new(i)))
    });
    Stored::Plain(Descriptor::Tiny(tiny))
}

fn ext_key(e: Extent) -> (u32, u64, u8) {
    (e.page().get(), e.offset().get(), e.class().log2())
}

fn assert_matches(
    store: &PageStore<VecMemory>,
    list: Option<LabelBucketList>,
    model: &BTreeSet<u32>,
) {
    let view = store.pages();
    let expected: Vec<(LabelBucketKey, Stored)> =
        model.iter().map(|&i| (key(i), descriptor(i))).collect();
    assert_eq!(entries(&view, list), expected);
    if let Some(list) = list {
        check_shape(&view, list);
    }
    let owned: BTreeSet<_> = list
        .map_or_else(Vec::new, |l| extents(&view, l))
        .into_iter()
        .map(ext_key)
        .collect();
    let allocated: BTreeSet<_> = store.allocated().into_iter().map(ext_key).collect();
    assert_eq!(owned, allocated, "extents: none leaked or shared");
    for &i in model {
        let Lookup::Present { entry, stored: d } = lookup(&view, list, key(i)) else {
            panic!("{i} present")
        };
        assert_eq!(d, descriptor(i));
        if model.iter().next() == Some(&i) {
            let list = list.unwrap();
            assert_eq!(first(&view, list), (entry, key(i), d));
        }
    }
}

#[test]
fn random_adds_and_removes_match_a_sorted_map_through_both_shapes() {
    let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
    let mut list = None;
    let mut model = BTreeSet::new();
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut tiered_seen = false;
    for step in 0..4000u32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let i = u32::try_from(state % 64).unwrap();
        let grow = (step / 500) % 2 == 0;
        let view = store.pages();
        match lookup(&view, list, key(i)) {
            Lookup::Absent { gap } if grow || state.is_multiple_of(4) => {
                if !has_room(&view, list, 1) {
                    continue;
                }
                list = Some(add(&mut store, list, gap, key(i), &descriptor(i)).unwrap());
                model.insert(i);
            }
            Lookup::Present { entry, .. } if !grow || state.is_multiple_of(4) => {
                list = remove(&mut store, list.unwrap(), entry);
                model.remove(&i);
            }
            Lookup::Present { entry, .. } => {
                set(&store, entry, &descriptor(i));
            }
            Lookup::Absent { .. } => {}
        }
        tiered_seen |= matches!(list, Some(LabelBucketList(Shape::Tiered { .. })));
        assert_matches(&store, list, &model);
        assert_eq!(decode_ref(encode_ref(list)), list);
    }
    assert!(tiered_seen, "the list tiered");
}

#[test]
fn a_full_top_refuses_more_and_a_one_leaf_list_turns_flat() {
    let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
    let mut list = None;
    let mut added = 0u32;
    // Ascending keys always land in the last leaf, which splits in half when full.
    while has_room(&store.pages(), list, 1) {
        let Lookup::Absent { gap } = lookup(&store.pages(), list, key(added)) else {
            panic!("absent")
        };
        list = Some(add(&mut store, list, gap, key(added), &descriptor(added)).unwrap());
        added += 1;
    }
    assert!(
        usize::try_from(added).unwrap() > LEAF_ENTRIES * 2,
        "{added} buckets"
    );
    // Remove all but a few: the list turns flat again and owns exactly its extent.
    for i in 0..added - 2 {
        let Lookup::Present { entry, .. } = lookup(&store.pages(), list, key(i)) else {
            panic!("present")
        };
        list = remove(&mut store, list.unwrap(), entry);
    }
    assert!(matches!(list, Some(LabelBucketList(Shape::Flat { .. }))));
    let model: BTreeSet<u32> = (added - 2..added).collect();
    assert_matches(&store, list, &model);
}

#[test]
fn encoded_keys_rank_in_key_order() {
    let directions = [
        Direction::Outgoing,
        Direction::Incoming,
        Direction::Undirected,
    ];
    let keys: Vec<LabelBucketKey> = [0, 1, 255, 256, 0x7FFF, u16::MAX]
        .into_iter()
        .flat_map(|l| {
            directions.map(|direction| LabelBucketKey {
                label: Label::new(l),
                direction,
            })
        })
        .collect();
    for a in &keys {
        for b in &keys {
            assert_eq!(a.rank().cmp(&b.rank()), a.cmp(b), "{a:?} against {b:?}");
        }
    }
}
