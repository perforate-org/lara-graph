//! Changes written in place: single decrements and increments, and merges of sparse leaves.

use super::*;

/// Single decrements lower counts in place, down to zero, and take no pages; a batch after them
/// still applies over the zero counts they leave.
#[test]
fn single_decrements_lower_counts_in_place() {
    let mut store = PageStore::create(VecMemory::new(), VecMemory::new()).unwrap();
    let mut model: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut rng = Rng(0x5851_F42D_4C95_7F2D);
    let seed: Vec<Change> = (0..3000).map(|_| add_some(&mut model, &mut rng)).collect();
    let mut root = commit_planned(&mut store, None, &seed);
    assert!(depth(root) >= 2, "depth {}", depth(root));
    let pages = store.allocated();
    for round in 0..2000 {
        let i = usize::try_from(rng.below(u64::try_from(model.len()).unwrap())).unwrap();
        let (&key, count) = model.iter_mut().nth(i).unwrap();
        *count -= 1;
        if *count == 0 {
            model.remove(&key);
        }
        lower(&store, root.unwrap(), key);
        if round % 250 == 0 {
            check(&store, root, &model);
        }
    }
    check(&store, root, &model);
    assert_eq!(store.allocated(), pages, "no page taken or released");
    // Only a segment's first posting stays at zero: the others' blocks are coded against it.
    let zeros = zeros_per_segment(&store.pages(), root);
    assert!(
        zeros.iter().all(|&z| z <= 1),
        "zero counts leave their segments: {zeros:?}"
    );
    let more: Vec<Change> = (0..500).map(|_| add_some(&mut model, &mut rng)).collect();
    root = commit_planned(&mut store, root, &more);
    check(&store, root, &model);
}

/// Single increments go in place when the segment takes them (an existing posting or a new one
/// inside it) and otherwise through a plan; either way the tree matches the model, and the
/// in-place ones take no pages.
#[test]
fn single_increments_go_in_place_or_through_a_plan() {
    let mut store = PageStore::create(VecMemory::new(), VecMemory::new()).unwrap();
    let mut model: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut rng = Rng(0x2D35_8DCC_AA6C_78A5);
    let seed: Vec<Change> = (0..3000).map(|_| add_some(&mut model, &mut rng)).collect();
    let mut root = commit_planned(&mut store, None, &seed);
    let (mut in_place, mut planned) = (0, 0);
    for round in 0..3000 {
        let key = if rng.below(2) == 0 {
            let i = usize::try_from(rng.below(u64::try_from(model.len()).unwrap())).unwrap();
            *model.keys().nth(i).unwrap()
        } else {
            let target = u32::try_from(rng.below(5_000_000)).unwrap();
            (target, u32::try_from(rng.below(4000)).unwrap())
        };
        *model.entry(key).or_default() += 1;
        let pages = store.allocated();
        match raise(&store, root.unwrap(), key) {
            InPlace::Done => {
                in_place += 1;
                assert_eq!(store.allocated(), pages, "an in-place raise takes no page");
            }
            InPlace::Unfit => {
                planned += 1;
                root = commit_planned(&mut store, root, &[Change { key, delta: 1 }]);
            }
        }
        if round % 300 == 0 {
            check(&store, root, &model);
        }
    }
    check(&store, root, &model);
    assert!(
        in_place > 1000 && planned > 100,
        "{in_place} in place, {planned} planned"
    );
}

/// Lowering every posting to zero one unit at a time leaves each segment with its first posting
/// only, at zero, and the tree reads as empty.
#[test]
fn single_decrements_drain_postings_out_of_their_segments() {
    let mut store = PageStore::create(VecMemory::new(), VecMemory::new()).unwrap();
    let mut model: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut rng = Rng(0x6A09_E667_F3BC_C908);
    let seed: Vec<Change> = (0..2000).map(|_| add_some(&mut model, &mut rng)).collect();
    let root = commit_planned(&mut store, None, &seed).unwrap();
    // From the back, the front and the middle of segments alike.
    let mut keys: Vec<(u32, u32)> = model.keys().copied().collect();
    for i in 0..keys.len() {
        let j = usize::try_from(rng.below(u64::try_from(keys.len() - i).unwrap())).unwrap();
        keys.swap(i, i + j);
    }
    for (round, key) in keys.iter().enumerate() {
        for _ in 0..model.remove(key).unwrap() {
            lower(&store, root, *key);
        }
        if round % 400 == 0 {
            check(&store, Some(root), &model);
        }
    }
    check(&store, Some(root), &model);
    let zeros = zeros_per_segment(&store.pages(), Some(root));
    assert!(
        zeros.iter().all(|&z| z == 1),
        "each segment keeps its first posting: {zeros:?}"
    );
}

/// Once most postings are gone, merge steps fold sparse leaves together and let the root give
/// way, until no pair is sparse enough; the postings left read the same, and pages come back.
#[test]
fn merge_steps_fold_sparse_leaves_and_give_pages_back() {
    let mut store = PageStore::create(VecMemory::new(), VecMemory::new()).unwrap();
    let mut model: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut rng = Rng(0xBB67_AE85_84CA_A73B);
    // One edge per posting: a leaf's count is its postings.
    let seed: Vec<Change> = (0..3000)
        .map(|_| {
            let key = (
                u32::try_from(rng.below(5_000_000)).unwrap(),
                u32::try_from(rng.below(4000)).unwrap(),
            );
            *model.entry(key).or_default() += 1;
            Change { key, delta: 1 }
        })
        .collect();
    let mut root = commit_planned(&mut store, None, &seed).unwrap();
    let (pages, depth) = (store.allocated().len(), root.depth);
    assert!(depth >= 2, "depth {depth}");
    let keys: Vec<(u32, u32)> = model.keys().copied().collect();
    for (i, key) in keys.iter().enumerate() {
        if i % 30 != 0 {
            for _ in 0..model.remove(key).unwrap() {
                lower(&store, root, *key);
            }
        }
    }
    let mut steps = 0;
    while let Merged::Into(next) = merge_step(&mut store, root) {
        root = next;
        steps += 1;
        check(&store, Some(root), &model);
    }
    assert!(steps > 10, "{steps} merges");
    assert!(
        store.allocated().len() * 4 < pages,
        "{} of {pages} pages",
        store.allocated().len()
    );
    assert!(root.depth < depth, "depth {} from {depth}", root.depth);
}

/// Segments whose counts are all one carry no count bits. Single lowers and raises, new postings
/// and batches of decrements all keep such a tree equal to the model: a first posting at zero is
/// marked, a raise that needs count bits goes through a plan.
#[test]
fn segments_without_count_bits_take_every_change() {
    let mut store = PageStore::create(VecMemory::new(), VecMemory::new()).unwrap();
    let mut model: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut rng = Rng(0x3C6E_F372_FE94_F82B);
    let fresh = |rng: &mut Rng| {
        (
            u32::try_from(rng.below(5_000_000)).unwrap(),
            u32::try_from(rng.below(4000)).unwrap(),
        )
    };
    let seed: Vec<Change> = (0..3000)
        .map(|_| {
            let key = fresh(&mut rng);
            *model.entry(key).or_default() += 1;
            Change { key, delta: 1 }
        })
        .collect();
    let mut root = commit_planned(&mut store, None, &seed);
    for round in 0..2000 {
        let pick = |model: &BTreeMap<(u32, u32), u32>, rng: &mut Rng| {
            let i = usize::try_from(rng.below(u64::try_from(model.len()).unwrap())).unwrap();
            *model.keys().nth(i).unwrap()
        };
        match rng.below(4) {
            0 => {
                let key = pick(&model, &mut rng);
                let count = model.get_mut(&key).unwrap();
                *count -= 1;
                if *count == 0 {
                    model.remove(&key);
                }
                lower(&store, root.unwrap(), key);
            }
            1 | 2 => {
                let key = if rng.below(2) == 0 {
                    pick(&model, &mut rng)
                } else {
                    fresh(&mut rng)
                };
                *model.entry(key).or_default() += 1;
                if raise(&store, root.unwrap(), key) == InPlace::Unfit {
                    root = commit_planned(&mut store, root, &[Change { key, delta: 1 }]);
                }
            }
            _ => {
                let mut keys: Vec<(u32, u32)> = (0..4).map(|_| pick(&model, &mut rng)).collect();
                keys.sort_unstable();
                keys.dedup();
                let batch: Vec<Change> = keys
                    .iter()
                    .map(|&key| {
                        let count = model.get_mut(&key).unwrap();
                        *count -= 1;
                        if *count == 0 {
                            model.remove(&key);
                        }
                        Change { key, delta: -1 }
                    })
                    .collect();
                root = commit_planned(&mut store, root, &batch);
            }
        }
        if round % 300 == 0 {
            check(&store, root, &model);
        }
    }
    check(&store, root, &model);
}
