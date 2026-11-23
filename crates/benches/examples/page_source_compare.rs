//! Where should 1024-slot pages of high-degree buckets come from? A cost simulation.
//!
//! Buckets follow `lara-core`: up to `TINY` edges inline, runs of power-of-two classes from 8 to
//! 1024 slots (grown in place when possible, else relocated), and past 1024 edges a paged bucket:
//! full 1024-slot pages listed by an index span in the slab. Removal drops one edge (the bucket
//! keeps no gaps), releases a page when it empties, and turns one remaining page back into a run.
//!
//! - **Shared**: pages come from the same span allocator as runs. A full run becomes page 0 and
//!   page 0 becomes a run again without copying. Index entries hold a start slot, a tombstone
//!   count and a tag id (14 bytes, 4 slots).
//! - **Pool**: pages come from a dedicated fixed-size pool in its own memory. Promotion copies the
//!   run into a pool page; demotion copies the page into a new run. Index entries hold a page id
//!   and a tombstone count (6 bytes, 2 slots). Pool memory never returns to the slab.
//!
//! Span-allocator memory calls are counted on the real allocator. Pool calls are estimated: a
//! take is 2 reads + 2 writes (head, link, head, header), a give 1 read + 2 writes.
//!
//! Usage: cargo run -p benches --release --example page_source_compare -- [synthetic] [orkut <path>]

use benches::{coo_power_law_stream, coo_uniform_stream};
use lara_core::bench_internals::{Span, SpanAllocator};
use std::io::{BufRead, BufReader};

const TINY: u32 = 5;
const PAGE: u32 = 1024;
const MIN_RUN: u32 = 8;
const MIN_INDEX: u32 = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    Shared,
    Pool,
}

impl Source {
    fn entry_slots(self) -> u32 {
        match self {
            Source::Shared => 4,
            Source::Pool => 2,
        }
    }
}

enum Page {
    Span(Span),
    Pooled(u32),
}

enum Row {
    Tiny(u32),
    Run {
        span: Span,
        x: u32,
    },
    Paged {
        pages: Vec<Page>,
        index: Span,
        x: u32,
    },
}

#[derive(Default)]
struct Pool {
    minted: u32,
    free: Vec<u32>,
    reads: u64,
    writes: u64,
}

impl Pool {
    fn take(&mut self) -> u32 {
        self.reads += 2;
        self.writes += 2;
        self.free.pop().unwrap_or_else(|| {
            self.minted += 1;
            self.minted - 1
        })
    }

    fn give(&mut self, id: u32) {
        self.reads += 1;
        self.writes += 2;
        self.free.push(id);
    }
}

#[derive(Default)]
struct Stats {
    moved: u64,
    index_moved: u64,
    promotions: u64,
    demotions: u64,
    peak_footprint: u64,
}

struct Sim {
    source: Source,
    rows: Vec<Row>,
    alloc: SpanAllocator,
    pool: Pool,
    stats: Stats,
    /// Allocator reads made only to measure the footprint, excluded from the counts.
    probe_reads: u64,
}

impl Sim {
    fn new(n: u32, source: Source) -> Self {
        Self {
            source,
            rows: (0..n).map(|_| Row::Tiny(0)).collect(),
            alloc: SpanAllocator::new(),
            pool: Pool::default(),
            stats: Stats::default(),
            probe_reads: 0,
        }
    }

    fn take_page(&mut self) -> Page {
        match self.source {
            Source::Shared => Page::Span(self.alloc.allocate(PAGE).expect("page")),
            Source::Pool => Page::Pooled(self.pool.take()),
        }
    }

    fn give_page(&mut self, page: Page) {
        match page {
            Page::Span(span) => self.alloc.release(span),
            Page::Pooled(id) => self.pool.give(id),
        }
    }

    fn index(&mut self, entries: u32) -> Span {
        self.alloc
            .allocate(entries * self.source.entry_slots())
            .expect("index")
    }

    fn insert(&mut self, v: u32) {
        let row = std::mem::replace(&mut self.rows[v as usize], Row::Tiny(0));
        let next = match row {
            Row::Tiny(x) if x < TINY => Row::Tiny(x + 1),
            Row::Tiny(x) => {
                self.stats.moved += u64::from(x);
                Row::Run {
                    span: self.alloc.allocate(MIN_RUN).expect("run"),
                    x: x + 1,
                }
            }
            Row::Run { span, x } if x < span.len() => Row::Run { span, x: x + 1 },
            Row::Run { span, x } if x < PAGE => {
                let target = (x + 1).next_power_of_two().clamp(MIN_RUN, PAGE);
                let span = match self.alloc.try_extend(span, target) {
                    Ok(span) => span,
                    Err(old) => {
                        let span = self.alloc.allocate(target).expect("run");
                        self.alloc.release(old);
                        self.stats.moved += u64::from(x);
                        span
                    }
                };
                Row::Run { span, x: x + 1 }
            }
            Row::Run { span, x } => {
                self.stats.promotions += 1;
                let first = match self.source {
                    Source::Shared => Page::Span(span),
                    Source::Pool => {
                        self.alloc.release(span);
                        self.stats.moved += u64::from(x);
                        Page::Pooled(self.pool.take())
                    }
                };
                let second = self.take_page();
                Row::Paged {
                    pages: vec![first, second],
                    index: self.index(MIN_INDEX),
                    x: x + 1,
                }
            }
            Row::Paged {
                mut pages,
                mut index,
                x,
            } => {
                if x % PAGE == 0 {
                    let capacity = index.len() / self.source.entry_slots();
                    if pages.len() as u32 == capacity {
                        let grown = self.index(capacity * 2);
                        self.alloc.release(std::mem::replace(&mut index, grown));
                        self.stats.index_moved += u64::from(capacity * self.source.entry_slots());
                    }
                    let page = self.take_page();
                    pages.push(page);
                }
                Row::Paged {
                    pages,
                    index,
                    x: x + 1,
                }
            }
        };
        self.rows[v as usize] = next;
    }

    /// Removes one edge of `v`, if it has any.
    fn remove(&mut self, v: u32) {
        let row = std::mem::replace(&mut self.rows[v as usize], Row::Tiny(0));
        let next = match row {
            Row::Tiny(x) => Row::Tiny(x.saturating_sub(1)),
            Row::Run { span, x } => Row::Run {
                span,
                x: x.saturating_sub(1),
            },
            Row::Paged {
                mut pages,
                index,
                x,
            } => {
                let x = x - 1;
                if x % PAGE == 0 {
                    let page = pages.pop().expect("last page");
                    self.give_page(page);
                }
                if x == PAGE {
                    self.stats.demotions += 1;
                    self.alloc.release(index);
                    let page = pages.pop().expect("first page");
                    let span = match page {
                        Page::Span(span) => span,
                        Page::Pooled(id) => {
                            self.pool.give(id);
                            self.stats.moved += u64::from(PAGE);
                            self.alloc.allocate(PAGE).expect("run")
                        }
                    };
                    Row::Run { span, x }
                } else {
                    Row::Paged { pages, index, x }
                }
            }
        };
        self.rows[v as usize] = next;
    }

    fn degree(&self, v: u32) -> u32 {
        match &self.rows[v as usize] {
            Row::Tiny(x) | Row::Run { x, .. } | Row::Paged { x, .. } => *x,
        }
    }

    /// Slots in use by any memory: the slab up to its frontier plus every pool page ever minted.
    fn footprint(&mut self) -> u64 {
        self.probe_reads += 1;
        self.alloc.frontier() + u64::from(self.pool.minted) * u64::from(PAGE)
    }

    fn sample(&mut self) {
        let now = self.footprint();
        self.stats.peak_footprint = self.stats.peak_footprint.max(now);
    }

    /// Edges held outside descriptors (runs and pages).
    fn stored_edges(&self) -> u64 {
        self.rows
            .iter()
            .map(|row| match row {
                Row::Tiny(_) => 0,
                Row::Run { x, .. } | Row::Paged { x, .. } => u64::from(*x),
            })
            .sum()
    }

    fn report(&mut self, workload: &str, ops: u64) {
        let footprint = self.footprint();
        let stored = self.stored_edges().max(1);
        let calls = self.alloc.memory_ops();
        let reads = calls.reads() - self.probe_reads;
        let writes = calls.writes();
        let total = reads + writes + self.pool.reads + self.pool.writes;
        println!(
            "| {workload} | {} | {:.3} | {:.3} | {} | {} | {} | {:.3} | {:.3} | {} | {} |",
            match self.source {
                Source::Shared => "shared",
                Source::Pool => "pool",
            },
            footprint as f64 / stored as f64,
            self.stats.peak_footprint as f64 / stored as f64,
            reads,
            writes,
            self.pool.reads + self.pool.writes,
            total as f64 / ops as f64,
            (self.stats.moved + self.stats.index_moved) as f64 / ops as f64,
            self.stats.promotions,
            self.stats.demotions,
        );
    }
}

/// Inserts every edge's source, sampling the footprint as it goes.
fn insert_all(sim: &mut Sim, edges: &[(u32, u32)]) {
    for (i, &(u, _)) in edges.iter().enumerate() {
        sim.insert(u);
        if i % 50_000 == 0 {
            sim.sample();
        }
    }
    sim.sample();
}

fn run_inserts(workload: &str, n: u32, edges: &[(u32, u32)]) {
    for source in [Source::Shared, Source::Pool] {
        let mut sim = Sim::new(n, source);
        insert_all(&mut sim, edges);
        sim.report(workload, edges.len() as u64);
    }
}

/// Grow with `first`, cut every bucket above a page to 10 % of its degree, then grow with
/// `second`: freed page space can only serve the second phase's runs if it is shared.
fn run_churn(workload: &str, n: u32, first: &[(u32, u32)], second: &[(u32, u32)]) {
    for source in [Source::Shared, Source::Pool] {
        let mut sim = Sim::new(n, source);
        insert_all(&mut sim, first);
        let mut removed = 0u64;
        for v in 0..n {
            let d = sim.degree(v);
            if d > PAGE {
                for _ in 0..d - d / 10 {
                    sim.remove(v);
                    removed += 1;
                }
            }
        }
        sim.sample();
        insert_all(&mut sim, second);
        sim.report(workload, first.len() as u64 + removed + second.len() as u64);
    }
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
    println!(
        "| workload | pages | final footprint / stored | peak footprint / stored | allocator reads | allocator writes | pool calls (est.) | memory calls / op | slots moved / op | promotions | demotions |"
    );
    println!("|{}", "---|".repeat(11));
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "synthetic" => {
                let (n, m) = (1_000_000, 10_000_000);
                run_inserts("uniform 1M/10M", n, &coo_uniform_stream(n, m, 7));
                let power = coo_power_law_stream(n, m, 99, 1.0);
                run_inserts("power-law 1M/10M", n, &power);
                // The second phase's hubs are other vertices: ranks shifted by n/2.
                let shifted: Vec<_> = coo_power_law_stream(n, m / 2, 123, 1.0)
                    .into_iter()
                    .map(|(u, v)| ((u + n / 2) % n, v))
                    .collect();
                run_churn("power-law churn", n, &power, &shifted);
                i += 1;
            }
            "orkut" => {
                let (n, edges) = load_input(&args[i + 1]);
                run_inserts("orkut-10M", n, &edges);
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
}
