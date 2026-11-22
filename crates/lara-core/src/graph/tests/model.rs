//! Random operations against a model of logical edges, with reopens and compaction.
//!
//! Every slot of the model records which logical edge it is a half of, so pairing is checked
//! against identity: [`Graph::counterpart`] and [`Graph::remove_edge`] must find the other half
//! of the same edge, through holes, reused holes and compaction moves.

use super::*;

/// xorshift64*: deterministic, dependency-free randomness.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u32) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        u32::try_from(self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33).unwrap() % n
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[usize::try_from(self.below(u32::try_from(items.len()).unwrap())).unwrap()]
    }
}

/// A bucket: `vertex`'s `(label, side)`.
type Key = (u32, Label, Side);

/// One live slot: the other end and the logical edge's id.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Slot {
    other: u32,
    id: u64,
}

/// Per bucket, position to slot (`None` for a hole).
#[derive(Default)]
struct Model {
    buckets: BTreeMap<Key, Vec<Option<Slot>>>,
    next_id: u64,
    /// The vertex being detached, if any.
    detaching: Option<u32>,
}

/// The bucket the random test grows past one page.
const HUB: Key = (0, Label::new(0), Side::Out);

impl Model {
    fn half(key: Key, at: usize) -> EdgeRef {
        let (x, label, side) = key;
        EdgeRef::new(v(x), label, side, p(u32::try_from(at).unwrap()))
    }

    /// Records a half the graph placed; it must fill a hole or append, after every live slot
    /// to the same vertex.
    fn place(&mut self, half: EdgeRef, slot: Slot) {
        let key = (half.vertex().index(), half.label(), half.side());
        let bucket = self.buckets.entry(key).or_default();
        let i = usize::try_from(half.position().index()).unwrap();
        if i == bucket.len() {
            bucket.push(Some(slot));
        } else {
            assert_eq!(bucket[i], None, "an insert fills a hole or appends");
            bucket[i] = Some(slot);
        }
        let later_equal = bucket
            .iter()
            .skip(i + 1)
            .flatten()
            .any(|s| s.other == slot.other);
        assert!(!later_equal, "equal-target edges stay in insertion order");
    }

    /// Every half of edge `id`.
    fn halves_of(&self, id: u64) -> Vec<EdgeRef> {
        self.buckets
            .iter()
            .flat_map(|(&key, slots)| {
                slots
                    .iter()
                    .enumerate()
                    .filter(move |(_, s)| s.is_some_and(|s| s.id == id))
                    .map(move |(at, _)| Self::half(key, at))
            })
            .collect()
    }

    /// The other half of the edge `half` belongs to.
    fn counterpart(&self, half: EdgeRef) -> EdgeRef {
        let id = self.slot(half).unwrap().id;
        let halves = self.halves_of(id);
        match halves.as_slice() {
            [only] => *only,
            [a, b] => {
                if *a == half {
                    *b
                } else {
                    *a
                }
            }
            other => panic!("edge {id} has halves {other:?}"),
        }
    }

    fn slot(&self, half: EdgeRef) -> Option<Slot> {
        let key = (half.vertex().index(), half.label(), half.side());
        let at = usize::try_from(half.position().index()).unwrap();
        self.buckets.get(&key)?.get(at).copied().flatten()
    }

    fn clear(&mut self, half: EdgeRef) {
        let key = (half.vertex().index(), half.label(), half.side());
        let at = usize::try_from(half.position().index()).unwrap();
        self.buckets.get_mut(&key).unwrap()[at] = None;
    }

    /// A random live half, from the hub half the time.
    fn live_half(&self, rng: &mut Rng) -> Option<EdgeRef> {
        let live_in = |key: &Key| -> Vec<usize> {
            self.buckets.get(key).map_or_else(Vec::new, |slots| {
                (0..slots.len()).filter(|&i| slots[i].is_some()).collect()
            })
        };
        let hub = live_in(&HUB);
        if !hub.is_empty() && rng.below(2) == 0 {
            return Some(Self::half(HUB, rng.pick(&hub)));
        }
        let keys: Vec<Key> = self
            .buckets
            .iter()
            .filter(|(_, slots)| slots.iter().any(Option::is_some))
            .map(|(&k, _)| k)
            .collect();
        if keys.is_empty() {
            return None;
        }
        let key = rng.pick(&keys);
        Some(Self::half(key, rng.pick(&live_in(&key))))
    }
}

/// Whether a phase of the random test mostly grows or mostly shrinks the graph.
#[derive(Clone, Copy)]
enum Phase {
    Grow,
    Shrink,
}

const SIDES: [Side; 3] = [Side::Out, Side::In, Side::Undirected];

fn random_step<M: Memory>(g: &mut Graph<M>, model: &mut Model, rng: &mut Rng, n: u32) {
    let roll = rng.below(100);
    let phase = if model.next_id < 4_000 {
        Phase::Grow
    } else {
        Phase::Shrink
    };
    let (insert_percent, compact_percent) = match phase {
        Phase::Grow => (80, 2),
        Phase::Shrink => (35, 8),
    };
    if let Some(x) = model.detaching.filter(|_| rng.below(10) == 0) {
        model_detach_step(g, model, x, rng);
        return;
    }
    if model.detaching.is_none() && rng.below(400) == 0 {
        let x = rng.below(n);
        g.detach(v(x)).unwrap();
        model.detaching = Some(x);
        return;
    }
    let live = model.live_half(rng);
    match live {
        Some(half) if roll < compact_percent => {
            let key = (half.vertex().index(), half.label(), half.side());
            model_compact(g, model, key);
        }
        Some(half) if roll >= compact_percent + insert_percent => {
            if rng.below(4) == 0 {
                assert_eq!(g.counterpart(half), Ok(Some(model.counterpart(half))));
            } else {
                model_remove(g, model, half);
            }
        }
        Some(_) | None => model_insert(g, model, rng, n),
    }
}

fn model_insert<M: Memory>(g: &mut Graph<M>, model: &mut Model, rng: &mut Rng, n: u32) {
    // Half the inserts leave the hub, which crosses into pages.
    let (from, label) = if rng.below(2) == 0 {
        (0, HUB.1)
    } else {
        (rng.below(n), l(u16::try_from(rng.below(3)).unwrap()))
    };
    let to = rng.below(3);
    if let Some(x) = model.detaching.filter(|x| [from, to].contains(x)) {
        assert_eq!(
            g.insert_directed(v(from), label, v(to)),
            Err(Error::Detaching(v(x)))
        );
        return;
    }
    let id = model.next_id;
    model.next_id += 1;
    let halves = if label == HUB.1 || rng.below(4) != 0 {
        let halves = g.insert_directed(v(from), label, v(to)).unwrap();
        assert_eq!(
            halves.source(),
            Model::half((from, label, Side::Out), at(halves.source()))
        );
        assert_eq!(
            halves.target(),
            Model::half((to, label, Side::In), at(halves.target()))
        );
        halves
    } else {
        let halves = g.insert_undirected(v(from), label, v(to)).unwrap();
        let (low, high) = (from.min(to), from.max(to));
        assert_eq!(halves.source().vertex(), v(low));
        assert_eq!(halves.target().vertex(), v(high));
        assert_eq!(halves.source().side(), Side::Undirected);
        halves
    };
    let (source, target) = (halves.source(), halves.target());
    model.place(
        source,
        Slot {
            other: target.vertex().index(),
            id,
        },
    );
    if target != source {
        model.place(
            target,
            Slot {
                other: source.vertex().index(),
                id,
            },
        );
    }
}

/// One detach step: every reported edge must be one of the model's, with its true halves.
fn model_detach_step<M: Memory>(g: &mut Graph<M>, model: &mut Model, x: u32, rng: &mut Rng) {
    let budget = NonZeroU32::new(1 + rng.below(40)).unwrap();
    let step = g.detach_step(v(x), budget).unwrap();
    assert!(step.changes().len() <= usize::try_from(budget.get()).unwrap());
    for &halves in step.changes() {
        let source = halves.source();
        assert!(
            [source.vertex(), halves.target().vertex()].contains(&v(x)),
            "an edge of the detaching vertex"
        );
        assert_eq!(halves, order(source, model.counterpart(source)));
        model.clear(source);
        model.clear(halves.target());
    }
    if step.progress() == Progress::Done {
        let left = model
            .buckets
            .iter()
            .filter(|((owner, _, _), _)| *owner == x)
            .any(|(_, slots)| slots.iter().any(Option::is_some));
        assert!(!left, "a detached vertex has no edges");
        model.detaching = None;
    }
}

fn at(half: EdgeRef) -> usize {
    usize::try_from(half.position().index()).unwrap()
}

fn model_remove<M: Memory>(g: &mut Graph<M>, model: &mut Model, half: EdgeRef) {
    let other = model.counterpart(half);
    let expected = order(half, other);
    assert_eq!(g.remove_edge(half), Ok(Removal::Removed(expected)));
    model.clear(half);
    model.clear(other);
    assert_eq!(g.edge(half), Ok(None));
    assert_eq!(g.remove_edge(other), Ok(Removal::Absent));
}

/// One compaction step, applied to the model through its reported moves.
fn model_compact<M: Memory>(g: &mut Graph<M>, model: &mut Model, key: Key) {
    let (x, label, side) = key;
    let bucket = model.buckets.entry(key).or_default();
    let before: Vec<Slot> = bucket.iter().flatten().copied().collect();
    let compaction = g.compact(v(x), label, side).unwrap();
    let stored = g.degree(v(x), label, side).unwrap() + g.holes(v(x), label, side).unwrap();
    let mut after = vec![None; usize::try_from(stored).unwrap()];
    let moved: BTreeMap<usize, (usize, u32)> = compaction
        .changes()
        .iter()
        .map(|m| {
            (
                usize::try_from(m.from().index()).unwrap(),
                (usize::try_from(m.to().index()).unwrap(), m.target().index()),
            )
        })
        .collect();
    for (i, slot) in bucket.iter().enumerate() {
        let Some(slot) = *slot else { continue };
        let to = match moved.get(&i) {
            Some(&(to, t)) => {
                assert_eq!(t, slot.other, "a move reports its edge's target");
                to
            }
            None => i,
        };
        assert_eq!(after[to], None, "two edges at one position");
        after[to] = Some(slot);
    }
    *bucket = after;
    let now: Vec<Slot> = bucket.iter().flatten().copied().collect();
    assert_eq!(now, before, "compaction keeps live edges in order");
}

fn assert_matches_model<M: Memory>(g: &Graph<M>, model: &Model, n: u32) {
    for x in 0..n {
        for label in (0..3).map(l) {
            for side in SIDES {
                let expected: Vec<(u32, u32)> =
                    model
                        .buckets
                        .get(&(x, label, side))
                        .map_or_else(Vec::new, |slots| {
                            (0u32..)
                                .zip(slots)
                                .filter_map(|(i, s)| s.map(|s| (i, s.other)))
                                .collect()
                        });
                assert_eq!(
                    edges_on(g, v(x), label, side),
                    expected,
                    "vertex {x} {label:?} {side:?}"
                );
            }
        }
    }
}

/// Compacts a bucket until done; a bounded number of steps must get there.
fn compact_fully<M: Memory>(g: &mut Graph<M>, model: &mut Model, key: Key) {
    let (x, label, side) = key;
    for _ in 0..10_000 {
        let before = g.holes(v(x), label, side).unwrap();
        model_compact(g, model, key);
        if g.compact(v(x), label, side).unwrap().progress() == Progress::Done
            && g.holes(v(x), label, side).unwrap() <= before
        {
            return;
        }
    }
    panic!("compaction did not finish");
}

#[test]
fn random_operations_match_a_model_across_reopens() {
    const N: u32 = 12;
    for seed in 1u64..=3 {
        let backing = Backing::default();
        let mut g = backing.create(N);
        let mut model = Model::default();
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut peak = 0;
        for step in 1..=8_000 {
            random_step(&mut g, &mut model, &mut rng, N);
            peak = peak.max(model.buckets.get(&HUB).map_or(0, Vec::len));
            if step % 1_000 == 0 {
                check(&g);
                assert_matches_model(&g, &model, N);
                g = backing.open();
            }
        }
        compact_fully(&mut g, &mut model, HUB);
        check(&g);
        assert_matches_model(&g, &model, N);
        // Every live half still finds its own edge's other half.
        let halves: Vec<EdgeRef> = model
            .buckets
            .iter()
            .flat_map(|(&key, slots)| {
                (0..slots.len())
                    .filter(|&i| slots[i].is_some())
                    .map(move |i| Model::half(key, i))
            })
            .collect();
        for half in halves {
            assert_eq!(g.counterpart(half), Ok(Some(model.counterpart(half))));
        }
        assert!(
            peak > usize::try_from(PAGE_SLOTS).unwrap(),
            "the hub reached pages (peak {peak})"
        );
    }
}
