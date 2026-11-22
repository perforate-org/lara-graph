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
    /// Vertices detached for good.
    deleted: Vec<u32>,
    /// Declared widths.
    widths: BTreeMap<Label, u16>,
    /// The value each logical edge was last given; edges without one read as zeros.
    values: BTreeMap<u64, Vec<u8>>,
}

/// Label 0 (the hub's) carries 8-byte values from the start, so the value plane grows from a run
/// into a table; label 2 carries 300 bytes (13 per value page); label 1 gets 3 bytes once 1500
/// edges exist, so its earlier edges read as zeros.
const WIDTHS_AT_START: [(u16, u16); 2] = [(0, 8), (2, 300)];
const LATE_WIDTH: (u16, u16, u64) = (1, 3, 1500);

/// Odd labels append; even ones (the hub's among them) reuse holes.
fn placement_of(label: Label) -> Placement {
    if label.get() % 2 == 1 {
        Placement::Append
    } else {
        Placement::ReuseHoles
    }
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
        if placement_of(half.label()) == Placement::Append {
            assert_eq!(
                i,
                bucket.len(),
                "an append goes after every stored position"
            );
        }
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

    /// The value edge `id` of `label` reads as.
    fn expected(&self, id: u64, label: Label) -> Vec<u8> {
        let width = usize::from(self.widths.get(&label).copied().unwrap_or(0));
        self.values
            .get(&id)
            .cloned()
            .unwrap_or_else(|| vec![0; width])
    }

    /// A fresh value for `label`, empty without a width.
    fn fresh_value(&self, label: Label, rng: &mut Rng) -> Vec<u8> {
        let width = self.widths.get(&label).copied().unwrap_or(0);
        (0..width)
            .map(|_| u8::try_from(rng.below(256)).unwrap())
            .collect()
    }

    /// Why an insert touching `x` fails, if it does.
    fn blocked(&self, x: u32) -> Option<Error> {
        if self.detaching == Some(x) {
            Some(Error::Detaching(v(x)))
        } else if self.deleted.contains(&x) {
            Some(Error::Deleted(v(x)))
        } else {
            None
        }
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
    // Detach a vertex now and then; never an insert target (0..3), so the hub keeps growing.
    // Targets' in-buckets are detached in the targeted tests.
    let x = 3 + rng.below(n - 3);
    if model.detaching.is_none() && !model.deleted.contains(&x) && rng.below(400) == 0 {
        g.detach(v(x)).unwrap();
        model.detaching = Some(x);
        return;
    }
    let (late, width, after) = LATE_WIDTH;
    if model.next_id >= after && !model.widths.contains_key(&l(late)) {
        g.declare_width(l(late), Width::new(width).unwrap())
            .unwrap();
        model.widths.insert(l(late), width);
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
            } else if rng.below(3) == 0 {
                model_set_value(g, model, half, rng);
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
    if let Some(error) = [from, to].into_iter().find_map(|x| model.blocked(x)) {
        assert_eq!(
            g.insert_directed(
                v(from),
                label,
                v(to),
                placement_of(label),
                &model.fresh_value(label, &mut Rng(1))
            ),
            Err(error)
        );
        return;
    }
    let id = model.next_id;
    model.next_id += 1;
    let value = model.fresh_value(label, rng);
    if !value.is_empty() {
        model.values.insert(id, value.clone());
    }
    let halves = if label == HUB.1 || rng.below(4) != 0 {
        let halves = g
            .insert_directed(v(from), label, v(to), placement_of(label), &value)
            .unwrap();
        assert_eq!(
            halves.canonical(),
            Model::half((from, label, Side::Out), at(halves.canonical()))
        );
        assert_eq!(
            halves.other(),
            Model::half((to, label, Side::In), at(halves.other()))
        );
        halves
    } else {
        let halves = g
            .insert_undirected(v(from), label, v(to), placement_of(label), &value)
            .unwrap();
        let (low, high) = (from.min(to), from.max(to));
        assert_eq!(halves.canonical().vertex(), v(high));
        assert_eq!(halves.other().vertex(), v(low));
        assert_eq!(halves.canonical().side(), Side::Undirected);
        halves
    };
    let (source, target) = (halves.canonical(), halves.other());
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
    for r in step.changes() {
        let halves = r.halves();
        let source = halves.canonical();
        assert!(
            [source.vertex(), halves.other().vertex()].contains(&v(x)),
            "an edge of the detaching vertex"
        );
        assert_eq!(halves, order(source, model.counterpart(source)));
        let id = model.slot(source).unwrap().id;
        assert_eq!(
            r.value(),
            model.expected(id, source.label()),
            "detached value"
        );
        model.clear(source);
        model.clear(halves.other());
    }
    if step.progress() == Progress::Done {
        let left = model
            .buckets
            .iter()
            .filter(|((owner, _, _), _)| *owner == x)
            .any(|(_, slots)| slots.iter().any(Option::is_some));
        assert!(!left, "a detached vertex has no edges");
        assert_eq!(g.vertex_state(v(x)), Ok(VertexState::Deleted));
        model.detaching = None;
        model.deleted.push(x);
    }
}

fn at(half: EdgeRef) -> usize {
    usize::try_from(half.position().index()).unwrap()
}

fn model_remove<M: Memory>(g: &mut Graph<M>, model: &mut Model, half: EdgeRef) {
    let other = model.counterpart(half);
    let expected = order(half, other);
    let id = model.slot(half).unwrap().id;
    let value = model.expected(id, half.label());
    assert_eq!(
        g.remove_edge(half),
        Ok(Removal::Removed(RemovedEdge::new(expected, value)))
    );
    model.clear(half);
    model.clear(other);
    assert_eq!(g.edge(half), Ok(None));
    assert_eq!(g.remove_edge(other), Ok(Removal::Absent));
}

/// Gives the edge `half` belongs to a new value and checks the old one came back.
fn model_set_value<M: Memory>(g: &mut Graph<M>, model: &mut Model, half: EdgeRef, rng: &mut Rng) {
    let id = model.slot(half).unwrap().id;
    let value = model.fresh_value(half.label(), rng);
    let old = model.expected(id, half.label());
    assert_eq!(g.set_value(half, &value), Ok(Some(old)));
    if !value.is_empty() {
        model.values.insert(id, value);
    }
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
                assert_other_reads(g, (v(x), label, side), &expected);
                assert_values(g, model, (v(x), label, side));
            }
        }
    }
}

/// Descending visits, windows of 7 positions both ways, and positional reads agree with the
/// ascending scan `expected`.
fn assert_other_reads<M: Memory>(
    g: &Graph<M>,
    (x, label, side): (VertexId, Label, Side),
    expected: &[(u32, u32)],
) {
    let mut down = Vec::new();
    g.visit_neighbors(x, label, side, Order::Descending, |e| {
        down.push((e.position().index(), e.target().index()));
        ControlFlow::<()>::Continue(())
    })
    .unwrap();
    let mut reversed = expected.to_vec();
    reversed.reverse();
    assert_eq!(down, reversed, "descending");
    let span = NonZeroU32::new(7).unwrap();
    for (order, want) in [
        (Order::Ascending, expected),
        (Order::Descending, &reversed[..]),
    ] {
        let mut seen = Vec::new();
        let mut cursor = Some(Cursor::first(order, span));
        while let Some(at) = cursor {
            cursor = g
                .visit_window(x, label, side, at, |e| {
                    seen.push((e.position().index(), e.target().index()));
                })
                .unwrap();
        }
        assert_eq!(seen, want, "{order:?} windows");
    }
    let stored = g.degree(x, label, side).unwrap() + g.holes(x, label, side).unwrap();
    // Every position and two past the end, highest first, with a repeat.
    let positions: Vec<Position> = (0..stored + 2).rev().chain([0]).map(p).collect();
    let mut targets = Vec::new();
    g.targets_at(x, label, side, &positions, &mut targets)
        .unwrap();
    let live: BTreeMap<u32, u32> = expected.iter().copied().collect();
    let want: Vec<Option<VertexId>> = positions
        .iter()
        .map(|at| live.get(&at.index()).map(|&t| v(t)))
        .collect();
    assert_eq!(targets, want, "targets_at");
}

/// Every live edge reads its model value through the combined scan, positional reads and the
/// values-first scan.
fn assert_values<M: Memory>(
    g: &Graph<M>,
    model: &Model,
    (x, label, side): (VertexId, Label, Side),
) {
    let Some(slots) = model.buckets.get(&(x.index(), label, side)) else {
        return;
    };
    let live: BTreeMap<u32, Vec<u8>> = (0u32..)
        .zip(slots)
        .filter_map(|(at, s)| s.map(|s| (at, model.expected(s.id, label))))
        .collect();
    let mut combined = BTreeMap::new();
    g.visit_neighbors_with_values(x, (label, side), Order::Ascending, |e, value| {
        combined.insert(e.position().index(), value.to_vec());
        ControlFlow::<()>::Continue(())
    })
    .unwrap();
    assert_eq!(combined, live, "combined scan of {x:?} {label:?} {side:?}");
    let positions: Vec<Position> = live.keys().rev().map(|&at| p(at)).collect();
    let mut bytes = Vec::new();
    g.values_at(x, (label, side), &positions, &mut bytes)
        .unwrap();
    let want: Vec<u8> = live.values().rev().flatten().copied().collect();
    assert_eq!(bytes, want, "values_at of {x:?} {label:?} {side:?}");
    let mut first_phase = BTreeMap::new();
    g.visit_values(x, (label, side), Order::Descending, |at, value| {
        if live.contains_key(&at.index()) {
            first_phase.insert(at.index(), value.to_vec());
        }
        ControlFlow::<()>::Continue(())
    })
    .unwrap();
    if model.widths.contains_key(&label) {
        assert_eq!(
            first_phase, live,
            "values-first scan of {x:?} {label:?} {side:?}"
        );
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
        for (label, width) in WIDTHS_AT_START {
            g.declare_width(l(label), Width::new(width).unwrap())
                .unwrap();
            model.widths.insert(l(label), width);
        }
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
