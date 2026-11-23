//! Leaf-grouped vs per-vertex slab layout: a cost simulation over the real span allocator.
//!
//! Both layouts share everything except how slab space is grouped: edges are 1 slot, rows of
//! at most `TINY` edges live inline in the descriptor (no slab) until their leaf/row first needs
//! slab, rows reaching `PROMOTE` move to tree blocks outside the slab, and spans come from
//! `lara-core`'s `TagAllocator` (including its in-place `try_extend`).
//!
//! Leaf layouts: `k` consecutive vertices share one span. When a leaf first needs slab, all its
//! rows are packed with capacity `alloc_for(x)`. A full row is then handled inside the span by
//! one of three policies; only when that fails does the leaf grow in place (re-packing inside
//! the span) or relocate to a new span of `Σ alloc_for(x)`:
//! - **shift**: borrow half of the spare capacity of the nearest row that has enough, shifting
//!   only the rows in between;
//! - **weighted**: re-lay every row inside the span, giving each a share of the leaf's slack
//!   proportional to its target headroom `alloc_for(x) - x` (PMA-style in-window rebalance);
//! - **adaptive**: whichever of the two moves fewer edges for this insert (weighted on a tie).
//! - **Leaf (repack only)**: calibration only. Every full row re-packs the leaf into a new span,
//!   as `lara-graph`'s `rebalance_leaf` does with the overflow log disabled.
//! - **Per-vertex (best)**: each row owns a span from a geometric class ladder (min 8, factor
//!   `g`, capped at `PROMOTE`). A full row grows in place if it can, else relocates alone.
//!
//! Insert-only; directed edges in stream order. Usage:
//!   cargo run -p benches --release --example layout_compare -- [synthetic] [orkut <path>]

use benches::{coo_power_law_stream, coo_uniform_stream};
use lara_core::bench_internals::{Span, SpanAllocator};
use std::io::{BufRead, BufReader};

const TINY: u32 = 4;
const PROMOTE: u32 = 1024;
const MIN_CLASS: u32 = 8;

/// The old crate's capacity policy: `max(x, min(max(2x, 1), PROMOTE))`.
fn alloc_for(x: u32) -> u32 {
    x.max((2 * x).max(1).min(PROMOTE))
}

// ---------------------------------------------------------------------------------------------
// Statistics

#[derive(Default)]
struct Stats {
    inserts: u64,
    relocations: u64,
    extends: u64,
    shifts: u64,
    rebalances: u64,
    promotions: u64,
    moved: u64,
    /// Slots moved by each insert that moved anything.
    nonzero_moves: Vec<u32>,
    allocator_calls: u64,
}

impl Stats {
    fn record(&mut self, moved: u32) {
        self.inserts += 1;
        if moved > 0 {
            self.moved += u64::from(moved);
            self.nonzero_moves.push(moved);
        }
    }

    /// Slots moved by the insert at percentile `p` (0..=1) of all inserts.
    fn percentile(&self, sorted: &[u32], p: f64) -> u32 {
        let zeros = self.inserts - sorted.len() as u64;
        let rank = ((p * self.inserts as f64).ceil() as u64).max(1) - 1;
        if rank < zeros {
            0
        } else {
            sorted[(rank - zeros) as usize]
        }
    }
}

/// Where a row's edges live, for the scan model.
enum Place {
    Inline,
    Slab { start: u64, len: u32 },
    Tree,
}

trait Layout {
    fn name(&self) -> String;
    fn insert(&mut self, v: u32, alloc: &mut SpanAllocator, stats: &mut Stats);
    fn place(&self, v: u32) -> Place;
    fn vertices(&self) -> u32;
    /// Panics if the layout is inconsistent; returns the number of edges it holds.
    fn validate(&self) -> u64;
    /// Bytes of tree blocks (1024 edges each) for rows past the promotion threshold.
    fn tree_bytes(&self) -> u64;
}

// ---------------------------------------------------------------------------------------------
// Leaf layout

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Inline,
    Slab,
    Tree,
}

#[derive(Clone, Copy)]
struct LeafRow {
    x: u32,
    off: u32,
    cap: u32,
    mode: Mode,
}

impl LeafRow {
    /// Edges this row keeps in the slab.
    fn slab_x(self) -> u32 {
        if self.mode == Mode::Slab { self.x } else { 0 }
    }

    /// Capacity the row holds but does not use.
    fn spare(self) -> u32 {
        self.cap - self.slab_x()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LeafPolicy {
    Shift,
    Weighted,
    Adaptive,
    RepackOnly,
}

/// Whether a re-pack writes into the leaf's current span or a new one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    SameSpan,
    NewSpan,
}

struct Leaf {
    span: Option<Span>,
    rows: Vec<LeafRow>,
}

struct LeafLayout {
    k: u32,
    policy: LeafPolicy,
    leaves: Vec<Leaf>,
    n: u32,
}

impl LeafLayout {
    fn new(n: u32, k: u32, policy: LeafPolicy) -> Self {
        let row = LeafRow {
            x: 0,
            off: 0,
            cap: 0,
            mode: Mode::Inline,
        };
        let leaves = (0..n.div_ceil(k))
            .map(|_| Leaf {
                span: None,
                rows: vec![row; k as usize],
            })
            .collect();
        Self {
            k,
            policy,
            leaves,
            n,
        }
    }
}

/// Lays every non-tree row out in vertex order with capacity `alloc_for(x)`; inline rows enter
/// the slab. Returns the slab edges whose position changed, not counting the edge row `fresh`
/// is receiving (it is written after the move).
fn pack(rows: &mut [LeafRow], target: Target, fresh: usize) -> u32 {
    let mut off = 0;
    let mut moved = 0;
    for (j, row) in rows.iter_mut().enumerate() {
        if row.mode == Mode::Tree {
            row.cap = 0;
            row.off = off;
            continue;
        }
        let relocated = row.mode == Mode::Inline || target == Target::NewSpan || row.off != off;
        if relocated {
            moved += row.x - u32::from(j == fresh);
        }
        row.mode = Mode::Slab;
        row.off = off;
        row.cap = alloc_for(row.x);
        off += row.cap;
    }
    moved
}

/// Gives row `i` at least `need` more capacity from the nearest row with enough spare, shifting
/// the rows in between. Returns the slab edges moved, or `None` when no row can lend.
fn borrow(rows: &mut [LeafRow], i: usize, need: u32) -> Option<u32> {
    let lender = (1..rows.len()).find_map(|d| {
        let left = i.checked_sub(d).filter(|&j| rows[j].spare() >= need);
        let right = Some(i + d).filter(|&j| j < rows.len() && rows[j].spare() >= need);
        left.or(right)
    })?;
    let give = need.max(rows[lender].spare().div_ceil(2));
    let mut moved = 0;
    if lender > i {
        // Rows i+1..=lender move right; the lender's end stays put.
        for row in &mut rows[i + 1..=lender] {
            row.off += give;
            moved += row.slab_x();
        }
    } else {
        // Rows lender+1..=i move left; row i's end stays put.
        for row in &mut rows[lender + 1..=i] {
            row.off -= give;
            moved += row.slab_x();
        }
    }
    rows[lender].cap -= give;
    rows[i].cap += give;
    Some(moved)
}

/// Re-lays every row inside the current span, sharing the slack in proportion to each slab
/// row's target headroom. Returns the slab edges moved, or `None` when the live edges do not fit.
fn rebalance(rows: &mut [LeafRow], total: u32, fresh: usize) -> Option<u32> {
    let live: u32 = rows.iter().map(|r| r.slab_x()).sum();
    let slack = total.checked_sub(live)?;
    let weight = |r: &LeafRow| {
        if r.mode == Mode::Slab {
            alloc_for(r.x) - r.x
        } else {
            0
        }
    };
    let total_weight: u64 = rows.iter().map(|r| u64::from(weight(r))).sum();
    let mut given = 0;
    let mut off = 0;
    let mut moved = 0;
    let last_slab = rows.iter().rposition(|r| r.mode == Mode::Slab);
    for (j, row) in rows.iter_mut().enumerate() {
        let mut share = if total_weight == 0 {
            0
        } else {
            (u64::from(slack) * u64::from(weight(row)) / total_weight) as u32
        };
        if Some(j) == last_slab {
            share = slack - given; // the rounding remainder
        }
        given += share;
        if row.off != off {
            moved += row.slab_x() - u32::from(j == fresh);
        }
        row.off = off;
        row.cap = row.slab_x() + share;
        off += row.cap;
    }
    Some(moved)
}

/// Handles a full row inside the leaf's current span. Returns the slab edges moved, or `None`
/// when the span must grow.
fn rebalance_in_span(
    rows: &mut Vec<LeafRow>,
    total: u32,
    i: usize,
    need: u32,
    policy: LeafPolicy,
) -> Option<(u32, LeafPolicy)> {
    match policy {
        LeafPolicy::RepackOnly => None,
        LeafPolicy::Shift => borrow(rows, i, need).map(|m| (m, LeafPolicy::Shift)),
        LeafPolicy::Weighted => rebalance(rows, total, i).map(|m| (m, LeafPolicy::Weighted)),
        LeafPolicy::Adaptive => {
            let mut shifted = rows.clone();
            let mut weighted = rows.clone();
            let by_shift = borrow(&mut shifted, i, need);
            let by_weight = rebalance(&mut weighted, total, i);
            match (by_shift, by_weight) {
                (Some(a), Some(b)) if a < b => {
                    *rows = shifted;
                    Some((a, LeafPolicy::Shift))
                }
                (_, Some(b)) => {
                    *rows = weighted;
                    Some((b, LeafPolicy::Weighted))
                }
                (Some(a), None) => {
                    *rows = shifted;
                    Some((a, LeafPolicy::Shift))
                }
                (None, None) => None,
            }
        }
    }
}

/// Grows the leaf so every row has `alloc_for(x)`: in place when the policy and the allocator
/// allow, else into a new span. Returns the slab edges moved.
fn grow(
    leaf: &mut Leaf,
    fresh: usize,
    alloc: &mut SpanAllocator,
    stats: &mut Stats,
    policy: LeafPolicy,
) -> u32 {
    let total: u32 = leaf
        .rows
        .iter()
        .filter(|r| r.mode != Mode::Tree)
        .map(|r| alloc_for(r.x))
        .sum();
    stats.allocator_calls += 1;
    let old = match (leaf.span.take(), policy) {
        (Some(span), LeafPolicy::Shift | LeafPolicy::Weighted | LeafPolicy::Adaptive) => {
            match alloc.try_extend(span, total) {
                Ok(span) => {
                    leaf.span = Some(span);
                    stats.extends += 1;
                    return pack(&mut leaf.rows, Target::SameSpan, fresh);
                }
                Err(span) => Some(span),
            }
        }
        (Some(span), LeafPolicy::RepackOnly) => Some(span),
        (None, _) => None,
    };
    leaf.span = alloc.allocate(total);
    if let Some(old) = old {
        alloc.release(old);
        stats.allocator_calls += 1;
    }
    stats.relocations += 1;
    pack(&mut leaf.rows, Target::NewSpan, fresh)
}

impl Layout for LeafLayout {
    fn tree_bytes(&self) -> u64 {
        let rows = self.leaves.iter().flat_map(|leaf| leaf.rows.iter());
        rows.filter(|r| r.mode == Mode::Tree)
            .map(|r| u64::from(r.x.div_ceil(PROMOTE)) * 4096)
            .sum()
    }

    fn name(&self) -> String {
        let policy = match self.policy {
            LeafPolicy::Shift => "shift",
            LeafPolicy::Weighted => "weighted",
            LeafPolicy::Adaptive => "adaptive",
            LeafPolicy::RepackOnly => "repack only, calibration",
        };
        format!("leaf k={} ({policy})", self.k)
    }

    fn vertices(&self) -> u32 {
        self.n
    }

    fn insert(&mut self, v: u32, alloc: &mut SpanAllocator, stats: &mut Stats) {
        let policy = self.policy;
        let leaf = &mut self.leaves[(v / self.k) as usize];
        let i = (v % self.k) as usize;
        let row = leaf.rows[i];
        let new_x = row.x + 1;
        let fits = match row.mode {
            Mode::Tree => true,
            Mode::Inline => new_x <= TINY,
            Mode::Slab => new_x <= row.cap && new_x < PROMOTE,
        };
        leaf.rows[i].x = new_x;
        if fits {
            return stats.record(0);
        }
        if row.mode == Mode::Slab && new_x >= PROMOTE {
            // The slab capacity stays with the row as spare for its neighbours.
            stats.promotions += 1;
            leaf.rows[i].mode = Mode::Tree;
            return stats.record(row.x);
        }
        if row.mode == Mode::Slab {
            let total = leaf.span.as_ref().map_or(0, Span::len);
            if let Some((moved, used)) =
                rebalance_in_span(&mut leaf.rows, total, i, new_x - row.cap, policy)
            {
                match used {
                    LeafPolicy::Shift => stats.shifts += 1,
                    _ => stats.rebalances += 1,
                }
                return stats.record(moved);
            }
        }
        let moved = grow(leaf, i, alloc, stats, policy);
        stats.record(moved);
    }

    fn place(&self, v: u32) -> Place {
        let leaf = &self.leaves[(v / self.k) as usize];
        let row = leaf.rows[(v % self.k) as usize];
        match row.mode {
            Mode::Inline => Place::Inline,
            Mode::Tree => Place::Tree,
            Mode::Slab => {
                let base = leaf.span.as_ref().map_or(0, Span::start);
                Place::Slab {
                    start: base + u64::from(row.off),
                    len: row.x,
                }
            }
        }
    }

    fn validate(&self) -> u64 {
        let mut edges = 0;
        for leaf in &self.leaves {
            let total = leaf.span.as_ref().map_or(0, Span::len);
            let mut end = 0;
            for row in &leaf.rows {
                edges += u64::from(row.x);
                if row.mode != Mode::Inline {
                    assert!(row.off >= end, "rows overlap or are out of order");
                    assert!(row.slab_x() <= row.cap, "row exceeds its capacity");
                    end = row.off + row.cap;
                }
            }
            assert!(end <= total, "rows exceed the leaf span");
        }
        edges
    }
}

// ---------------------------------------------------------------------------------------------
// Per-vertex layout

enum VMode {
    Inline,
    Slab(Span),
    Tree,
}

struct VertexRow {
    x: u32,
    mode: VMode,
}

struct VertexLayout {
    growth: f64,
    rows: Vec<VertexRow>,
}

impl VertexLayout {
    fn new(n: u32, growth: f64) -> Self {
        let rows = (0..n)
            .map(|_| VertexRow {
                x: 0,
                mode: VMode::Inline,
            })
            .collect();
        Self { growth, rows }
    }
}

/// The smallest class on the ladder (min 8, factor `growth`, capped at `PROMOTE`) that holds `x`.
fn class(growth: f64, x: u32) -> u32 {
    let mut c = MIN_CLASS;
    while c < x {
        c = ((f64::from(c) * growth).ceil() as u32)
            .max(c + 1)
            .min(PROMOTE);
    }
    c
}

impl Layout for VertexLayout {
    fn tree_bytes(&self) -> u64 {
        let tree = self.rows.iter().filter(|r| matches!(r.mode, VMode::Tree));
        tree.map(|r| u64::from(r.x.div_ceil(PROMOTE)) * 4096).sum()
    }

    fn name(&self) -> String {
        format!("per-vertex g={} (best)", self.growth)
    }

    fn vertices(&self) -> u32 {
        self.rows.len() as u32
    }

    fn insert(&mut self, v: u32, alloc: &mut SpanAllocator, stats: &mut Stats) {
        let growth = self.growth;
        let row = &mut self.rows[v as usize];
        let old_x = row.x;
        let new_x = old_x + 1;
        let fits = match &row.mode {
            VMode::Tree => true,
            VMode::Inline => new_x <= TINY,
            VMode::Slab(span) => new_x <= span.len() && new_x < PROMOTE,
        };
        row.x = new_x;
        if fits {
            return stats.record(0);
        }
        let old = std::mem::replace(&mut row.mode, VMode::Tree);
        if new_x >= PROMOTE {
            if let VMode::Slab(span) = old {
                alloc.release(span);
                stats.allocator_calls += 1;
            }
            stats.promotions += 1;
            return stats.record(old_x);
        }
        let target = class(growth, new_x);
        stats.allocator_calls += 1;
        let old = match old {
            VMode::Slab(span) => match alloc.try_extend(span, target) {
                Ok(span) => {
                    stats.extends += 1;
                    row.mode = VMode::Slab(span);
                    return stats.record(0);
                }
                Err(span) => Some(span),
            },
            VMode::Inline | VMode::Tree => None,
        };
        let span = alloc.allocate(target).expect("allocate row span");
        if let Some(old) = old {
            alloc.release(old);
            stats.allocator_calls += 1;
        }
        stats.relocations += 1;
        row.mode = VMode::Slab(span);
        stats.record(old_x);
    }

    fn place(&self, v: u32) -> Place {
        let row = &self.rows[v as usize];
        match &row.mode {
            VMode::Inline => Place::Inline,
            VMode::Tree => Place::Tree,
            VMode::Slab(span) => Place::Slab {
                start: span.start(),
                len: row.x,
            },
        }
    }

    fn validate(&self) -> u64 {
        let mut edges = 0;
        for row in &self.rows {
            edges += u64::from(row.x);
            if let VMode::Slab(span) = &row.mode {
                assert!(row.x <= span.len(), "row exceeds its span");
            }
        }
        edges
    }
}

// ---------------------------------------------------------------------------------------------
// Scan model and report

/// Reads every slab row in vertex order. Consecutive rows become one read when the next starts
/// at most `gap` slots after the current read ends (the gap is read too). Tree rows are laid out
/// identically in both layouts, so only slab reads are compared.
fn scan(layout: &dyn Layout, gap: u64) -> (u64, u64) {
    let (mut calls, mut slots) = (0u64, 0u64);
    let mut current: Option<(u64, u64)> = None;
    for v in 0..layout.vertices() {
        let Place::Slab { start, len } = layout.place(v) else {
            continue;
        };
        if len == 0 {
            continue;
        }
        let end = start + u64::from(len);
        match current {
            Some((s, e)) if start >= e && start - e <= gap => current = Some((s, end)),
            _ => {
                if let Some((s, e)) = current {
                    slots += e - s;
                }
                calls += 1;
                current = Some((start, end));
            }
        }
    }
    if let Some((s, e)) = current {
        slots += e - s;
    }
    (calls, slots)
}

fn live_slab(layout: &dyn Layout) -> u64 {
    (0..layout.vertices())
        .map(|v| match layout.place(v) {
            Place::Slab { len, .. } => u64::from(len),
            Place::Inline | Place::Tree => 0,
        })
        .sum()
}

fn run(workload: &str, layout: &mut dyn Layout, edges: &[(u32, u32)]) {
    let mut alloc = SpanAllocator::new();
    let mut stats = Stats::default();
    for &(u, _) in edges {
        layout.insert(u, &mut alloc, &mut stats);
    }
    assert_eq!(
        layout.validate(),
        edges.len() as u64,
        "every inserted edge is held"
    );
    let mut moves = std::mem::take(&mut stats.nonzero_moves);
    moves.sort_unstable();
    let live = live_slab(layout).max(1);
    let (calls0, read0) = scan(layout, 0);
    let (calls256, read256) = scan(layout, 256);
    println!(
        "| {workload} | {} | {} | {} | {} | {} | {} | {:.3} | {} | {} | {} | {:.2} | {:.2} | {:.3} | {} | {:.2} | {} | {:.2} |",
        layout.name(),
        stats.relocations,
        stats.extends,
        stats.shifts,
        stats.rebalances,
        stats.promotions,
        stats.moved as f64 / stats.inserts as f64,
        stats.percentile(&moves, 0.999),
        moves.last().copied().unwrap_or(0),
        stats.allocator_calls,
        alloc.frontier() as f64 / live as f64,
        // Slots are 4 bytes; tags are the allocator's per-segment records.
        (alloc.frontier() * 4 + alloc.tag_bytes()) as f64 / (live * 4) as f64,
        (alloc.frontier() * 4 + alloc.tag_bytes() + layout.tree_bytes()) as f64
            / (edges.len() as u64 * 4) as f64,
        calls0,
        read0 as f64 / live as f64,
        calls256,
        read256 as f64 / live as f64,
    );
}

fn header() {
    println!(
        "| workload | layout | relocations | in-place grows | local shifts | in-leaf rebalances | promotions | moved/insert | p99.9 moved | max moved | allocator calls | frontier/live | (frontier+tags)/live | (frontier+tags+trees)/(edges x 4B) | scan calls (gap 0) | read/live | scan calls (gap 256) | read/live |"
    );
    println!("|{}", "---|".repeat(18));
}

fn layouts(n: u32) -> Vec<Box<dyn Layout>> {
    vec![
        Box::new(LeafLayout::new(n, 31, LeafPolicy::RepackOnly)),
        Box::new(LeafLayout::new(n, 16, LeafPolicy::Shift)),
        Box::new(LeafLayout::new(n, 16, LeafPolicy::Weighted)),
        Box::new(LeafLayout::new(n, 16, LeafPolicy::Adaptive)),
        Box::new(LeafLayout::new(n, 31, LeafPolicy::Shift)),
        Box::new(LeafLayout::new(n, 31, LeafPolicy::Weighted)),
        Box::new(LeafLayout::new(n, 31, LeafPolicy::Adaptive)),
        Box::new(VertexLayout::new(n, 2.0)),
        Box::new(VertexLayout::new(n, 1.5)),
    ]
}

fn load_input(path: &str) -> (u32, Vec<(u32, u32)>) {
    let file = std::fs::File::open(path).expect("open input");
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().expect("header").expect("read header");
    let n: u32 = header
        .split_whitespace()
        .next()
        .expect("n")
        .parse()
        .expect("parse n");
    let edges = lines
        .map(|line| {
            let line = line.expect("read line");
            let mut it = line.split_whitespace();
            let u: u32 = it.next().expect("u").parse().expect("parse u");
            let v: u32 = it.next().expect("v").parse().expect("parse v");
            (u, v)
        })
        .collect();
    (n, edges)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    header();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "synthetic" => {
                let (n, m) = (1_000_000, 10_000_000);
                let power = coo_power_law_stream(n, m, 99, 1.0);
                for mut layout in layouts(n) {
                    run("power-law 1M/10M", layout.as_mut(), &power);
                }
                let uniform = coo_uniform_stream(n, m, 7);
                for mut layout in layouts(n) {
                    run("uniform 1M/10M", layout.as_mut(), &uniform);
                }
                i += 1;
            }
            "orkut" => {
                let (n, edges) = load_input(&args[i + 1]);
                for mut layout in layouts(n) {
                    run("orkut-10M", layout.as_mut(), &edges);
                }
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
}
