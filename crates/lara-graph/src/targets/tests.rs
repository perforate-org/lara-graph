//! Random batches of changes against a map of postings.

use super::*;
use crate::memory::VecMemory;
use crate::page_store::PageStore;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

fn check(store: &PageStore<VecMemory>, root: Option<Root>, model: &BTreeMap<(u32, u32), u32>) {
    let view = store.pages();
    let want: Vec<Posting> = model
        .iter()
        .map(|(&(target, block), &count)| Posting {
            target,
            block,
            count,
        })
        .collect();
    assert_eq!(postings(&view, root), want);
    let owned: BTreeSet<u32> = extents(&view, root)
        .into_iter()
        .map(|e| e.page().get())
        .collect();
    let allocated: BTreeSet<u32> = store
        .allocated()
        .into_iter()
        .map(|e| e.page().get())
        .collect();
    assert_eq!(owned, allocated, "pages: none leaked or shared");
    let Some(root) = root else { return };
    // Rank and selection at a sample of keys and ranks.
    let mut before = 0u32;
    for (i, (&key, &count)) in model.iter().enumerate() {
        if i % 97 == 0 {
            assert_eq!(
                count_before(&view, root, key),
                before,
                "count before {key:?}"
            );
            let probe = (key.0, key.1 + 1);
            assert_eq!(count_before(&view, root, probe), before + count);
            let (p, within) = select(&view, root, before + count - 1).unwrap();
            assert_eq!(
                (p.key(), within),
                (key, count - 1),
                "select {}",
                before + count - 1
            );
        }
        before += count;
    }
    assert_eq!(select(&view, root, before), None);
    let sample: BTreeSet<u32> = model.keys().step_by(97).map(|&(t, _)| t).collect();
    for target in sample {
        check_target(&view, root, model, target);
    }
}

/// `nth` and `rank` over `target`'s postings, against the model: every unit's posting, and the
/// rank at every block the target has a posting in and just past it.
fn check_target(
    view: &crate::page_store::Pages<'_, VecMemory>,
    root: Root,
    model: &BTreeMap<(u32, u32), u32>,
    target: u32,
) {
    let mut seen = 0u32;
    for (&(_, block), &count) in model.range((target, 0)..=(target, u32::MAX)) {
        assert_eq!(
            rank(view, root, target, block),
            seen,
            "rank of {target} at {block}"
        );
        for within in [0, count - 1] {
            let p = Posting {
                target,
                block,
                count,
            };
            assert_eq!(nth(view, root, target, seen + within), Nth::At(p, within));
        }
        seen += count;
        assert_eq!(rank(view, root, target, block + 1), seen);
    }
    assert_eq!(nth(view, root, target, seen), Nth::Past(seen));
    assert_eq!(rank(view, root, target, u32::MAX), seen);
    let last = model
        .range((target, 0)..=(target, u32::MAX))
        .rev()
        .find(|(_, count)| **count > 0)
        .map(|(&(_, block), _)| block);
    assert_eq!(
        last_block(view, root, target),
        last,
        "last block of {target}"
    );
    let mut blocks = Vec::new();
    visit_blocks(view, root, target, |b| blocks.push(b));
    let want: Vec<u32> = model
        .range((target, 0)..=(target, u32::MAX))
        .filter(|(_, count)| **count > 0)
        .map(|(&(_, block), _)| block)
        .collect();
    assert_eq!(blocks, want, "blocks of {target}");
}

/// The deepest root seen, to show the test reaches several levels.
fn depth(root: Option<Root>) -> u8 {
    root.map_or(0, |r| r.depth)
}

/// A change removing part of a random posting's count, made to the model too.
fn remove_some(model: &mut BTreeMap<(u32, u32), u32>, rng: &mut Rng) -> Change {
    let i = usize::try_from(rng.below(u64::try_from(model.len()).unwrap())).unwrap();
    let (&key, &count) = model.iter().nth(i).unwrap();
    let delta = u32::try_from(1 + rng.below(u64::from(count))).unwrap();
    let left = count - delta;
    if left == 0 {
        model.remove(&key);
    } else {
        model.insert(key, left);
    }
    Change {
        key,
        delta: -i64::from(delta),
    }
}

/// A change adding to a random posting (sometimes heavily), made to the model too.
fn add_some(model: &mut BTreeMap<(u32, u32), u32>, rng: &mut Rng) -> Change {
    let target = u32::try_from(rng.below(5_000_000)).unwrap();
    let block = u32::try_from(rng.below(4000)).unwrap();
    let heavy = rng.below(10) == 0;
    let delta = u32::try_from(1 + rng.below(if heavy { 500 } else { 1 })).unwrap();
    *model.entry((target, block)).or_default() += delta;
    Change {
        key: (target, block),
        delta: i64::from(delta),
    }
}

#[test]
fn random_batches_match_a_map_of_postings() {
    let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
    let mut root = None;
    let mut model: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut deepest = 0;
    for round in 0..300u32 {
        // Grow for 200 rounds, then shrink.
        let size = 1 + rng.below(if round % 50 == 0 { 1500 } else { 40 });
        let changes: Vec<Change> = (0..size)
            .map(|_| {
                let removing = round >= 200 && !model.is_empty() && rng.below(3) != 0;
                if removing {
                    remove_some(&mut model, &mut rng)
                } else {
                    add_some(&mut model, &mut rng)
                }
            })
            .collect();
        root = commit_planned(&mut store, root, &changes);
        deepest = deepest.max(depth(root));
        if round % 10 == 9 {
            check(&store, root, &model);
        }
    }
    assert!(deepest >= 2, "depth {deepest}");
    drain(&mut store, root, &model);
}

/// Plans `changes`, commits the plan, and checks that the commit took no more pages than the plan
/// counted.
fn commit_planned(
    store: &mut PageStore<VecMemory>,
    root: Option<Root>,
    changes: &[Change],
) -> Option<Root> {
    let before = store.allocated().len();
    let planned = plan(&store.pages(), root, changes);
    let pages = usize::try_from(planned.pages()).unwrap();
    let root = commit(store, planned).unwrap();
    assert!(
        store.allocated().len() <= before + pages,
        "a commit took more pages than its plan counted"
    );
    root
}

/// Drains every posting. One new posting goes in with the removals, so its leaf is rewritten and
/// drops the zero counts there, while the other leaves are lowered in place. Lowering that last
/// posting leaves a count of zero too, and releasing the tree frees every page.
fn drain(store: &mut PageStore<VecMemory>, root: Option<Root>, model: &BTreeMap<(u32, u32), u32>) {
    let last = (u32::MAX - 1, 0);
    let mut all: Vec<Change> = model
        .iter()
        .map(|(&key, &count)| Change {
            key,
            delta: -i64::from(count),
        })
        .collect();
    all.push(Change {
        key: last,
        delta: 1,
    });
    let root = commit_planned(store, root, &all);
    check(store, root, &BTreeMap::from([(last, 1)]));
    let root = commit_planned(
        store,
        root,
        &[Change {
            key: last,
            delta: -1,
        }],
    );
    check(store, root, &BTreeMap::new());
    if let Some(root) = root {
        release(store, root);
    }
    assert_eq!(store.allocated().len(), 0);
}

/// One batch changing every posting: each segment's first key comes with the keys of the segment
/// before it, and must still go to its own segment.
#[test]
fn a_batch_over_every_posting_keeps_each_change_in_its_segment() {
    let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
    let mut model: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let seed: Vec<Change> = (0..3000).map(|_| add_some(&mut model, &mut rng)).collect();
    let mut root = commit_planned(&mut store, None, &seed);
    check(&store, root, &model);
    for delta in [1i64, -1] {
        let all: Vec<Change> = model.keys().map(|&key| Change { key, delta }).collect();
        root = commit_planned(&mut store, root, &all);
        for count in model.values_mut() {
            *count = u32::try_from(i64::from(*count) + delta).unwrap();
        }
        check(&store, root, &model);
    }
}

/// Targets whose postings fill several leaves are found and ranked across them; so are targets
/// between and beyond the postings.
#[test]
fn nth_and_rank_follow_targets_across_leaves() {
    let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
    let mut model: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut rng = Rng(0x1405_7B7E_F767_814F);
    let mut changes: Vec<Change> = (0..2000).map(|_| add_some(&mut model, &mut rng)).collect();
    // Many postings that run past one leaf, and (1_000_000) more than a walk decodes, in one.
    for (target, blocks) in [
        (7, 1500),
        (1_000_000, 100),
        (2_500_000, 1500),
        (4_999_999, 1500),
    ] {
        for block in 0..blocks {
            let delta = 1 + u32::try_from(rng.below(3)).unwrap();
            *model.entry((target, block)).or_default() += delta;
            changes.push(Change {
                key: (target, block),
                delta: i64::from(delta),
            });
        }
    }
    let root = commit_planned(&mut store, None, &changes).unwrap();
    assert!(root.depth >= 2, "depth {}", root.depth);
    let view = store.pages();
    for target in [0, 7, 8, 1_000_000, 2_500_000, 4_999_999, u32::MAX] {
        check_target(&view, root, &model, target);
    }
    // The zero count single decrements leave is skipped.
    for _ in 0..model.remove(&(7, 0)).unwrap() {
        lower(&store, root, (7, 0));
    }
    check_target(&store.pages(), root, &model, 7);
}

mod in_place;
