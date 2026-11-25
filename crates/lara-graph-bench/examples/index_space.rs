//! Space of a hub target index, measured on synthetic hubs before building it.
//!
//! The index under study is a counted B+ tree of
//! `(target, block, count)` postings: `count` live edges to `target` among the `block`-th run of
//! `B` positions. Leaves are 4096-byte pages of delta-coded postings; internal nodes hold a
//! separator, a child page and a subtree count per child.
//!
//! For each hub shape and block size this prints bytes per edge of: the postings' encoding alone,
//! leaves bulk-loaded full, leaves at the ~70 % fill a B-tree settles at after splits, and the
//! whole tree. The 4-byte slots are the baseline. Run with `cargo run --release --example
//! index_space -p lara-graph-bench`.

extern crate alloc;

use alloc::collections::BTreeMap;

/// Edges per synthetic hub.
const EDGES: u32 = 1_000_000;
const PAGE: usize = 4096;
/// Leaf header: posting count u16 and the first posting's target in full.
const LEAF_HEADER: usize = 2 + 4;
/// Internal entry: separator (target u32, block u32), child page u32, subtree count u32.
const INTERNAL_ENTRY: usize = 16;
/// The fill a B-tree's leaves settle at under random inserts and splits.
const STEADY_FILL: f64 = 0.7;

/// splitmix64: deterministic randomness.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A hub's targets in insertion (position) order.
struct Hub {
    name: &'static str,
    targets: Vec<u32>,
}

/// Distinct targets whose ids grow with insertion order, other nodes interleaved (new leaves
/// of an old hub).
fn sequential(rng: &mut Rng) -> Hub {
    let mut id = 1_000u32;
    let targets = (0..EDGES)
        .map(|_| {
            id += 1 + u32::try_from(rng.below(19)).unwrap_or(0);
            id
        })
        .collect();
    Hub {
        name: "distinct, ids rising with time",
        targets,
    }
}

/// Distinct targets drawn uniformly from `universe` ids.
fn random(rng: &mut Rng, universe: u64, name: &'static str) -> Hub {
    let mut seen = std::collections::HashSet::new();
    let mut targets = Vec::new();
    while targets.len() < usize::try_from(EDGES).unwrap_or(0) {
        let t = u32::try_from(rng.below(universe)).unwrap_or(0);
        if seen.insert(t) {
            targets.push(t);
        }
    }
    Hub { name, targets }
}

/// Parallel edges: `distinct` targets with Zipf(1) multiplicities, interleaved at random.
fn zipf(rng: &mut Rng, distinct: u32) -> Hub {
    // Inverse-CDF sampling over harmonic weights.
    let weights: Vec<f64> = (1..=distinct).map(|k| 1.0 / f64::from(k)).collect();
    let total: f64 = weights.iter().sum();
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for w in &weights {
        acc += w / total;
        cumulative.push(acc);
    }
    let ids: Vec<u32> = (0..distinct)
        .map(|_| u32::try_from(rng.below(10_000_000)).unwrap_or(0))
        .collect();
    let targets = (0..EDGES)
        .map(|_| {
            // A probability from the top 32 random bits.
            let u = f64::from(u32::try_from(rng.next() >> 32).unwrap_or(u32::MAX))
                / f64::from(u32::MAX);
            let k = cumulative.partition_point(|&c| c < u).min(ids.len() - 1);
            ids.get(k).copied().unwrap_or(0)
        })
        .collect();
    Hub {
        name: "parallel: 50k targets, Zipf multiplicity, interleaved",
        targets,
    }
}

/// Parallel edges arriving in bursts: each target's edges consecutively, 1-200 at a time.
fn bursts(rng: &mut Rng) -> Hub {
    let mut targets = Vec::new();
    while targets.len() < usize::try_from(EDGES).unwrap_or(0) {
        let t = u32::try_from(rng.below(10_000_000)).unwrap_or(0);
        let run = 1 + rng.below(200);
        for _ in 0..run {
            targets.push(t);
        }
    }
    targets.truncate(usize::try_from(EDGES).unwrap_or(0));
    Hub {
        name: "parallel: bursts of 1-200 to one target",
        targets,
    }
}

/// Postings `(target, block) → count`, sorted.
fn postings(hub: &Hub, block: u32) -> Vec<(u32, u32, u32)> {
    let mut map: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for (position, &t) in (0u32..).zip(&hub.targets) {
        *map.entry((t, position / block)).or_default() += 1;
    }
    map.into_iter().map(|((t, b), c)| (t, b, c)).collect()
}

fn varint_len(mut v: u32) -> usize {
    let mut len = 1;
    while v >= 0x80 {
        v >>= 7;
        len += 1;
    }
    len
}

/// Bytes of one posting after `prev`: target gap (0 when the same), block (a gap when the target
/// repeats, else absolute), count.
fn posting_len(prev: Option<(u32, u32)>, (t, b, c): (u32, u32, u32)) -> usize {
    let (target_gap, block) = match prev {
        Some((pt, pb)) if pt == t => (0, b - pb),
        Some((pt, _)) => (t - pt, b),
        None => (0, b),
    };
    varint_len(target_gap) + varint_len(block) + varint_len(c)
}

/// Leaves of `postings` packed to `fill` of a page; returns their count.
fn leaves(postings: &[(u32, u32, u32)], fill: f64) -> usize {
    let budget = budget(PAGE - LEAF_HEADER, fill);
    let mut count = 0;
    let mut used = 0;
    let mut prev = None;
    for &p in postings {
        let len = posting_len(prev, p);
        if prev.is_none() || used + len > budget {
            // A new leaf: its first posting is coded against nothing (the header has its target).
            count += 1;
            used = posting_len(None, p);
        } else {
            used += len;
        }
        prev = Some((p.0, p.1));
    }
    count
}

/// Bit-packed leaves: per leaf, the first target in full, a Rice parameter `k` for target gaps,
/// and a fixed block width `w`. Each posting is its target gap (Rice, 0 for the same target), a
/// one-bit "count is 1" flag (else the count in Elias gamma), and the block in `w` bits (a gap
/// from the previous block when the target repeats would only help parallel edges).
/// Returns the number of leaves at `fill`.
fn packed_leaves(postings: &[(u32, u32, u32)], fill: f64) -> usize {
    // Leaf header: posting count u16, first target u32, Rice k u8, block width u8.
    const HEADER_BITS: usize = (2 + 4 + 1 + 1) * 8;
    let budget = budget(PAGE * 8 - HEADER_BITS, fill);
    let mut leaves = 0;
    let mut start = 0;
    while start < postings.len() {
        // Grow the leaf while its encoding fits; parameters are chosen for the leaf's contents.
        let mut end = start + 1;
        let mut step = 512;
        while step > 0 {
            let candidate = (end + step).min(postings.len());
            let fits = postings
                .get(start..candidate)
                .is_some_and(|leaf| packed_bits(leaf) <= budget);
            if candidate > end && fits {
                end = candidate;
            } else {
                step /= 2;
            }
        }
        leaves += 1;
        start = end;
    }
    leaves
}

/// Bits of `postings` as one packed leaf, with the best Rice parameter for its gaps.
fn packed_bits(postings: &[(u32, u32, u32)]) -> usize {
    let max_block = postings.iter().map(|p| p.1).max().unwrap_or(0);
    let w = usize::try_from(32 - max_block.leading_zeros())
        .unwrap_or(32)
        .max(1);
    let gaps: Vec<u32> = postings
        .windows(2)
        .map(|pair| match pair {
            [a, b] => b.0 - a.0,
            _ => 0,
        })
        .collect();
    let rice = |k: u32| -> usize {
        gaps.iter()
            .map(|&g| usize::try_from((g >> k) + 1 + k).unwrap_or(64))
            .sum()
    };
    let gap_bits = (0..24).map(rice).min().unwrap_or(0);
    let count_bits: usize = postings
        .iter()
        .map(|p| {
            if p.2 == 1 {
                1
            } else {
                1 + 2 * usize::try_from(32 - p.2.leading_zeros()).unwrap_or(32) - 1
            }
        })
        .sum();
    gap_bits + count_bits + w * postings.len()
}

/// Pages of the internal levels above `leaves` leaves.
fn internal_pages(mut children: usize) -> usize {
    let fanout = PAGE / INTERNAL_ENTRY;
    let mut pages = 0;
    while children > 1 {
        children = children.div_ceil(fanout);
        pages += children;
    }
    pages
}

fn per_edge(bytes: usize) -> f64 {
    as_f64(bytes) / f64::from(EDGES)
}

/// `n` as a float, for ratios in a report (the counts here stay far below 2^32).
fn as_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

/// `fill` of `space` units, rounded down.
fn budget(space: usize, fill: f64) -> usize {
    // A whole number of thousandths keeps the arithmetic in integers.
    let thousandths = (0..=1000u32)
        .rev()
        .find(|&t| f64::from(t) <= fill * 1000.0)
        .unwrap_or(0);
    space * usize::try_from(thousandths).unwrap_or(0) / 1000
}

fn main() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let hubs = [
        sequential(&mut rng),
        random(&mut rng, 10_000_000, "distinct, random among 10M ids"),
        random(&mut rng, 1_000_000_000, "distinct, random among 1G ids"),
        zipf(&mut rng, 50_000),
        bursts(&mut rng),
    ];
    println!("bytes per edge (slots alone: 4.00)\n");
    println!(
        "{:<52} {:>6} {:>8} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7}",
        "hub", "block", "postings", "vint", "70%", "tree", "packed", "70%", "81%"
    );
    for hub in &hubs {
        for block in [64, 128, 256, 1024] {
            let p = postings(hub, block);
            let encoded: usize = p
                .iter()
                .scan(None, |prev, &q| {
                    let len = posting_len(*prev, q);
                    *prev = Some((q.0, q.1));
                    Some(len)
                })
                .sum();
            let full = leaves(&p, 1.0) * PAGE;
            let steady_leaves = leaves(&p, STEADY_FILL);
            let tree = (steady_leaves + internal_pages(steady_leaves)) * PAGE;
            let per_posting = as_f64(p.len()) / f64::from(EDGES);
            let packed_full = packed_leaves(&p, 1.0);
            let packed_70 = packed_leaves(&p, STEADY_FILL);
            let packed_81 = packed_leaves(&p, 0.81);
            let with_tree = |n: usize| per_edge((n + internal_pages(n)) * PAGE);
            let _ = full;
            println!(
                "{:<52} {:>6} {:>8.3} | {:>7.2} {:>7.2} {:>7.2} | {:>7.2} {:>7.2} {:>7.2}",
                hub.name,
                block,
                per_posting,
                per_edge(encoded),
                per_edge(steady_leaves * PAGE),
                per_edge(tree),
                with_tree(packed_full),
                with_tree(packed_70),
                with_tree(packed_81),
            );
        }
    }
}
