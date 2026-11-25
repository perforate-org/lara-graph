//! Random writes and copies against a map of position to value, across every plane form.

use super::*;
use crate::store::memory::VecMemory;
use alloc::collections::{BTreeMap, BTreeSet};
use core::ops::ControlFlow;

fn width(bytes: u16) -> Width {
    Width::new(bytes).unwrap()
}

fn value(width: Width, seed: u32) -> Vec<u8> {
    (0..width.get())
        .map(|i| u8::try_from((u32::from(i) * 31 + seed * 7) % 251 + 1).unwrap())
        .collect()
}

fn ext_key(e: Extent) -> (u32, u64, u8) {
    (e.page().get(), e.offset().get(), e.class().id())
}

fn check(
    store: &PageStore<VecMemory>,
    plane: Plane,
    width: Width,
    model: &BTreeMap<u32, Vec<u8>>,
    end: u32,
) {
    let view = store.pages();
    let zeros = vec![0u8; usize::from(width.get())];
    let mut out = zeros.clone();
    for (&p, v) in model {
        read(&view, plane, width, p, &mut out);
        assert_eq!(&out, v, "position {p}");
    }
    // Visit in both orders and compare with the model, zeros elsewhere: the front, and a few
    // pages around the farthest position.
    let cpp = PAGE_BYTES / u32::from(width.get());
    let far = model.keys().next_back().copied().unwrap_or(0);
    for range in [(0, end.min(cpp * 4)), (far.saturating_sub(cpp + 3), end)] {
        check_visits(&view, plane, width, model, range);
    }
    let owned: BTreeSet<_> = extents(&view, plane).into_iter().map(ext_key).collect();
    let allocated: BTreeSet<_> = store.allocated().into_iter().map(ext_key).collect();
    assert_eq!(owned, allocated, "extents: none leaked or shared");
    assert_eq!(Plane::decode(plane.encode()), plane);
}

fn check_visits(
    view: &Pages<'_, VecMemory>,
    plane: Plane,
    width: Width,
    model: &BTreeMap<u32, Vec<u8>>,
    (low, end): (u32, u32),
) {
    let zeros = vec![0u8; usize::from(width.get())];
    for descending in [false, true] {
        let mut seen: Vec<(u32, Vec<u8>)> = Vec::new();
        visit(
            view,
            plane,
            width,
            (low, end, descending),
            |first, bytes| {
                for (i, cell) in bytes.chunks(usize::from(width.get())).enumerate() {
                    seen.push((first + u32::try_from(i).unwrap(), cell.to_vec()));
                }
                ControlFlow::<()>::Continue(())
            },
        );
        if descending {
            // Chunks come highest first; restore position order.
            seen.sort_by_key(|(p, _)| *p);
        }
        let want: Vec<(u32, Vec<u8>)> = (low..end)
            .map(|p| (p, model.get(&p).cloned().unwrap_or_else(|| zeros.clone())))
            .collect();
        assert_eq!(seen, want, "visit, descending {descending}");
    }
}

#[test]
fn random_writes_and_copies_match_a_map_through_run_and_table() {
    for bytes in [1, 8, 24, 1000, 4096] {
        let width = width(bytes);
        let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
        let mut plane = Plane::Empty;
        let mut model: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
        let mut state = 0x2545_F491_4F6C_DD1Du64 ^ u64::from(bytes);
        let cpp = PAGE_BYTES / u32::from(bytes);
        // Positions cluster in the run, then spread across a few pages and one far away.
        let spread = [cpp / 2 + 1, cpp * 3, cpp * 1100 + 5];
        let mut end = 0;
        for step in 0..600u32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bound = spread[usize::try_from(step / 200).unwrap()];
            let position = u32::try_from(state % u64::from(bound)).unwrap();
            end = end.max(position + 1);
            if step % 5 == 4 && !model.is_empty() {
                let from = *model
                    .keys()
                    .nth(usize::try_from(state % 7).unwrap() % model.len())
                    .unwrap();
                plane = copy(&mut store, plane, width, (from, position)).unwrap();
                let moved = model[&from].clone();
                model.insert(position, moved);
            } else {
                let v = value(width, step);
                plane = write(&mut store, plane, width, position, &v).unwrap();
                model.insert(position, v);
            }
            if step % 50 == 49 {
                check(&store, plane, width, &model, end);
            }
        }
        check(&store, plane, width, &model, end);
        release(&mut store, plane);
        assert_eq!(store.allocated().len(), 0, "release frees everything");
    }
}

#[test]
fn a_write_takes_at_most_its_reserved_pages() {
    // The far positions force the deepest tables; a fresh store mints every page it takes.
    let width = width(4096);
    for first in [None, Some(0u32)] {
        let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
        let mut plane = Plane::Empty;
        if let Some(p) = first {
            plane = write(&mut store, plane, width, p, &value(width, 1)).unwrap();
        }
        for position in [5, 1 << 20, u32::MAX - 1] {
            let before = store.minted();
            plane = write(&mut store, plane, width, position, &value(width, 2)).unwrap();
            assert!(
                store.minted() - before <= PAGES_PER_WRITE,
                "position {position}"
            );
        }
        let mut out = vec![0u8; 4096];
        read(&store.pages(), plane, width, u32::MAX - 1, &mut out);
        assert_eq!(out, value(width, 2));
    }
}

#[test]
fn copying_a_zero_onto_missing_storage_writes_nothing() {
    let width = width(8);
    let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
    let plane = copy(&mut store, Plane::Empty, width, (3, 9000)).unwrap();
    assert_eq!(plane, Plane::Empty);
    assert_eq!(store.allocated().len(), 0);
}

/// Whether position `p` lost its storage when `low..high` was released from a plane now `plane`:
/// everything when the plane is gone, nothing from a run, a table's pages wholly inside.
fn released(plane: Plane, cpp: u32, (low, high): (u32, u32), p: u32) -> bool {
    match plane {
        Plane::Empty => true,
        Plane::Run(_) | Plane::Beside { .. } => false,
        Plane::Table { .. } => {
            let start = p / cpp * cpp;
            start >= low && (high == u32::MAX || start.saturating_add(cpp) <= high)
        }
    }
}

/// xorshift64 positions below a bound.
struct Positions(u64);

impl Positions {
    fn below(&mut self, bound: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        u32::try_from(self.0 % u64::from(bound)).unwrap()
    }
}

#[test]
fn releasing_ranges_frees_whole_pages_and_keeps_the_rest() {
    for bytes in [8, 300, 4096] {
        let width = width(bytes);
        let cpp = PAGE_BYTES / u32::from(bytes);
        let mut store = PageStore::format(VecMemory::new(), VecMemory::new()).unwrap();
        let mut plane = Plane::Empty;
        let mut model: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
        let mut rng = Positions(0x9E37_79B9_7F4A_7C15u64 ^ u64::from(bytes));
        let bound = cpp * 3000;
        for round in 0..40u32 {
            for _ in 0..20 {
                let p = rng.below(bound);
                let v = value(width, p);
                plane = write(&mut store, plane, width, p, &v).unwrap();
                model.insert(p, v);
            }
            let (a, b) = (rng.below(bound), rng.below(bound));
            let range = if round % 7 == 6 {
                (0, u32::MAX)
            } else {
                (a.min(b), a.max(b))
            };
            plane = release_range(&mut store, plane, width, range);
            model.retain(|&p, _| !released(plane, cpp, range, p));
            let end = model.keys().next_back().map_or(1, |&p| p + 1);
            check(&store, plane, width, &model, end);
        }
        release(&mut store, plane);
        assert_eq!(store.allocated().len(), 0);
    }
}
