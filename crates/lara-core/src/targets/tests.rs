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
    let mut store = PageStore::create(VecMemory::new(), VecMemory::new()).unwrap();
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
        root = apply_reserved(&mut store, root, &changes).unwrap();
        deepest = deepest.max(depth(root));
        if round % 10 == 9 {
            check(&store, root, &model);
        }
    }
    assert!(deepest >= 2, "depth {deepest}");
    drain(&mut store, root, &model);
}

/// Drains every posting: counts drop to zero in place, and releasing the tree frees every page.
fn drain(store: &mut PageStore<VecMemory>, root: Option<Root>, model: &BTreeMap<(u32, u32), u32>) {
    let all: Vec<Change> = model
        .iter()
        .map(|(&key, &count)| Change {
            key,
            delta: -i64::from(count),
        })
        .collect();
    let root = apply_reserved(store, root, &all).unwrap();
    check(store, root, &BTreeMap::new());
    if let Some(root) = root {
        release(store, root);
    }
    assert_eq!(store.allocated().len(), 0);
}
