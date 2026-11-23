//! Storage designs for per-bucket adjacency, compared with every byte of metadata counted.
//!
//! Buckets are the same in both designs: up to `T` edges inline in the descriptor, then a run of a
//! power-of-two class from 8 to 1024 slots, then past 1024 edges a paged bucket of 1024-slot pages
//! listed by an index run. A full 1024 run becomes page 0 and back without copying. Removal drops
//! one edge (no gaps), releases a page when it empties and demotes to a run at one page.
//!
//! - **A, TLSF + chunks**: runs below `CHUNK_LIMIT` slots live in 1024-slot chunks carved from
//!   spans of the real `lara-core` TLSF allocator (one bit per run); larger runs and pages are TLSF
//!   spans, and runs try to grow in place. Metadata: a 32-byte tag per span, a 32-byte record per
//!   chunk. Index entries: start + tombstone count + tag id, 14 bytes.
//! - **B, page store**: everything is a 1024-slot page from one fixed-size pool; runs below 1024
//!   slots live in chunk pages. Metadata: a 32-byte directory record per page. Index entries:
//!   page id + tombstone count, 6 bytes.
//!
//! Memory calls: TLSF calls are counted on the real allocator; chunk and pool calls are estimates
//! (chunk slot take or give 1 read + 1 write; page take 2 reads + 2 writes, give 1 read + 2
//! writes).
//!
//! Usage: cargo run -p benches --release --example storage_compare -- [synthetic] [orkut <path>]

use benches::{coo_power_law_stream, coo_uniform_stream};
use lara_core::bench_internals::{Span, SpanAllocator};
use std::io::{BufRead, BufReader};

const PAGE: u32 = 1024;
const MIN_RUN: u32 = 8;
const MIN_INDEX: u32 = 4;
/// Design A: runs of at least this many slots come straight from TLSF.
const CHUNK_LIMIT: u32 = 256;
const CHUNK_RECORD_BYTES: u64 = 32;
const DIRECTORY_RECORD_BYTES: u64 = 32;

fn run_class(slots: u32) -> u32 {
    slots.next_power_of_two().clamp(MIN_RUN, PAGE)
}

// ---------------------------------------------------------------------------------------------
// Chunks: shared by both designs. A chunk is one 1024-slot unit split into runs of one class.

struct Chunk<U> {
    unit: U,
    class: u32,
    free_slots: Vec<u32>,
}

/// Chunks of every class over some source of 1024-slot units.
struct Chunks<U> {
    chunks: Vec<Option<Chunk<U>>>,
    recycled: Vec<usize>,
    /// Per class (log2), chunks with a free slot.
    partial: Vec<Vec<usize>>,
    live: usize,
    peak: usize,
    calls: u64,
}

impl<U> Chunks<U> {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            recycled: Vec::new(),
            partial: (0..11).map(|_| Vec::new()).collect(),
            live: 0,
            peak: 0,
            calls: 0,
        }
    }

    /// A slot of `class`, taking a new unit from `fresh` when no chunk of the class has room.
    fn take(&mut self, class: u32, fresh: impl FnOnce() -> U) -> (usize, u32) {
        self.calls += 2;
        let lane = class.trailing_zeros() as usize;
        let id = match self.partial[lane].last() {
            Some(&id) => id,
            None => {
                let chunk = Chunk {
                    unit: fresh(),
                    class,
                    free_slots: (0..PAGE / class).rev().collect(),
                };
                let id = match self.recycled.pop() {
                    Some(id) => {
                        self.chunks[id] = Some(chunk);
                        id
                    }
                    None => {
                        self.chunks.push(Some(chunk));
                        self.chunks.len() - 1
                    }
                };
                self.live += 1;
                self.peak = self.peak.max(self.live);
                self.partial[lane].push(id);
                id
            }
        };
        let chunk = self.chunks[id].as_mut().expect("live chunk");
        let slot = chunk.free_slots.pop().expect("free slot");
        if chunk.free_slots.is_empty() {
            self.partial[lane].pop();
        }
        (id, slot)
    }

    /// Returns a slot; hands back the unit when its chunk empties.
    fn give(&mut self, id: usize, slot: u32) -> Option<U> {
        self.calls += 2;
        let chunk = self.chunks[id].as_mut().expect("live chunk");
        let lane = chunk.class.trailing_zeros() as usize;
        let was_full = chunk.free_slots.is_empty();
        chunk.free_slots.push(slot);
        if chunk.free_slots.len() as u32 == PAGE / chunk.class {
            if !was_full {
                let at = self.partial[lane]
                    .iter()
                    .position(|&c| c == id)
                    .expect("partial chunk");
                self.partial[lane].swap_remove(at);
            }
            self.live -= 1;
            self.recycled.push(id);
            return self.chunks[id].take().map(|c| c.unit);
        }
        if was_full {
            self.partial[lane].push(id);
        }
        None
    }
}

// ---------------------------------------------------------------------------------------------
// Stores

trait Store {
    type Run;
    type Page;
    const INDEX_ENTRY_BYTES: u32;
    fn name(&self) -> &'static str;
    fn alloc_run(&mut self, class: u32) -> Self::Run;
    fn free_run(&mut self, run: Self::Run);
    /// Grows a run in place to `class`, or hands it back.
    fn try_grow(&mut self, run: Self::Run, class: u32) -> Result<Self::Run, Self::Run>;
    /// A full 1024 run as a page, and back: no copy in either design.
    fn run_to_page(&mut self, run: Self::Run) -> Self::Page;
    fn page_to_run(&mut self, page: Self::Page) -> Self::Run;
    fn alloc_page(&mut self) -> Self::Page;
    fn free_page(&mut self, page: Self::Page);
    /// Slab, pages and allocator metadata, in bytes. Memory never shrinks: high-water marks.
    fn memory_bytes(&mut self) -> u64;
    fn memory_calls(&self) -> u64;
}

enum ARun {
    Chunked { chunk: usize, slot: u32 },
    Span(Span),
}

struct TlsfChunks {
    alloc: SpanAllocator,
    chunks: Chunks<Span>,
    probe_reads: u64,
}

impl Store for TlsfChunks {
    type Run = ARun;
    type Page = Span;
    const INDEX_ENTRY_BYTES: u32 = 14;

    fn name(&self) -> &'static str {
        "A tlsf+chunks"
    }

    fn alloc_run(&mut self, class: u32) -> ARun {
        if class >= CHUNK_LIMIT {
            return ARun::Span(self.alloc.allocate(class).expect("span"));
        }
        let alloc = &mut self.alloc;
        let (chunk, slot) = self
            .chunks
            .take(class, || alloc.allocate(PAGE).expect("chunk"));
        ARun::Chunked { chunk, slot }
    }

    fn free_run(&mut self, run: ARun) {
        match run {
            ARun::Span(span) => self.alloc.release(span),
            ARun::Chunked { chunk, slot } => {
                if let Some(span) = self.chunks.give(chunk, slot) {
                    self.alloc.release(span);
                }
            }
        }
    }

    fn try_grow(&mut self, run: ARun, class: u32) -> Result<ARun, ARun> {
        match run {
            ARun::Span(span) if class >= CHUNK_LIMIT => self
                .alloc
                .try_extend(span, class)
                .map(ARun::Span)
                .map_err(ARun::Span),
            other => Err(other),
        }
    }

    fn run_to_page(&mut self, run: ARun) -> Span {
        match run {
            ARun::Span(span) => span,
            ARun::Chunked { .. } => unreachable!("a 1024 run is a span"),
        }
    }

    fn page_to_run(&mut self, page: Span) -> ARun {
        ARun::Span(page)
    }

    fn alloc_page(&mut self) -> Span {
        self.alloc.allocate(PAGE).expect("page")
    }

    fn free_page(&mut self, page: Span) {
        self.alloc.release(page);
    }

    fn memory_bytes(&mut self) -> u64 {
        self.probe_reads += 1;
        self.alloc.frontier() * 4
            + self.alloc.tag_bytes()
            + self.chunks.peak as u64 * CHUNK_RECORD_BYTES
    }

    fn memory_calls(&self) -> u64 {
        let ops = self.alloc.memory_ops();
        ops.reads() - self.probe_reads + ops.writes() + self.chunks.calls
    }
}

enum BRun {
    Chunked { chunk: usize, slot: u32 },
    Page(u32),
}

#[derive(Default)]
struct Pool {
    minted: u32,
    free: Vec<u32>,
    calls: u64,
}

impl Pool {
    fn take(&mut self) -> u32 {
        self.calls += 4;
        self.free.pop().unwrap_or_else(|| {
            self.minted += 1;
            self.minted - 1
        })
    }

    fn give(&mut self, id: u32) {
        self.calls += 3;
        self.free.push(id);
    }
}

struct PageStore {
    pool: Pool,
    chunks: Chunks<u32>,
}

impl Store for PageStore {
    type Run = BRun;
    type Page = u32;
    const INDEX_ENTRY_BYTES: u32 = 6;

    fn name(&self) -> &'static str {
        "B page store"
    }

    fn alloc_run(&mut self, class: u32) -> BRun {
        if class == PAGE {
            return BRun::Page(self.pool.take());
        }
        let pool = &mut self.pool;
        let (chunk, slot) = self.chunks.take(class, || pool.take());
        BRun::Chunked { chunk, slot }
    }

    fn free_run(&mut self, run: BRun) {
        match run {
            BRun::Page(page) => self.pool.give(page),
            BRun::Chunked { chunk, slot } => {
                if let Some(page) = self.chunks.give(chunk, slot) {
                    self.pool.give(page);
                }
            }
        }
    }

    fn try_grow(&mut self, run: BRun, _class: u32) -> Result<BRun, BRun> {
        Err(run)
    }

    fn run_to_page(&mut self, run: BRun) -> u32 {
        match run {
            BRun::Page(page) => page,
            BRun::Chunked { .. } => unreachable!("a 1024 run is a page"),
        }
    }

    fn page_to_run(&mut self, page: u32) -> BRun {
        BRun::Page(page)
    }

    fn alloc_page(&mut self) -> u32 {
        self.pool.take()
    }

    fn free_page(&mut self, page: u32) {
        self.pool.give(page);
    }

    fn memory_bytes(&mut self) -> u64 {
        u64::from(self.pool.minted) * (u64::from(PAGE) * 4 + DIRECTORY_RECORD_BYTES)
    }

    fn memory_calls(&self) -> u64 {
        self.pool.calls + self.chunks.calls
    }
}

// ---------------------------------------------------------------------------------------------
// Buckets

enum Row<S: Store> {
    Inline(u32),
    Run {
        run: S::Run,
        class: u32,
        x: u32,
    },
    /// `index` lists `pages` directly while they fit in one page of entries; past that, the
    /// entries live in `index_pages` and `index` lists those (a second level).
    Paged {
        pages: Vec<S::Page>,
        index: S::Run,
        index_class: u32,
        index_pages: Vec<S::Page>,
        x: u32,
    },
}

struct Sim<S: Store> {
    store: S,
    tiny: u32,
    rows: Vec<Row<S>>,
    moved: u64,
}

impl<S: Store> Sim<S> {
    fn new(store: S, n: u32, tiny: u32) -> Self {
        Self {
            store,
            tiny,
            rows: (0..n).map(|_| Row::Inline(0)).collect(),
            moved: 0,
        }
    }

    fn index_class(entries: u32) -> u32 {
        run_class((entries * S::INDEX_ENTRY_BYTES).div_ceil(4))
    }

    fn index_capacity(class: u32) -> u32 {
        class * 4 / S::INDEX_ENTRY_BYTES
    }

    fn insert(&mut self, v: u32) {
        let row = std::mem::replace(&mut self.rows[v as usize], Row::Inline(0));
        let next = match row {
            Row::Inline(x) if x < self.tiny => Row::Inline(x + 1),
            Row::Inline(x) => {
                self.moved += u64::from(x);
                let class = run_class(x + 1);
                Row::Run {
                    run: self.store.alloc_run(class),
                    class,
                    x: x + 1,
                }
            }
            Row::Run { run, class, x } if x < class => Row::Run {
                run,
                class,
                x: x + 1,
            },
            Row::Run { run, class, x } if class < PAGE => {
                let grown = class * 2;
                let run = match self.store.try_grow(run, grown) {
                    Ok(run) => run,
                    Err(old) => {
                        let run = self.store.alloc_run(grown);
                        self.store.free_run(old);
                        self.moved += u64::from(x);
                        run
                    }
                };
                Row::Run {
                    run,
                    class: grown,
                    x: x + 1,
                }
            }
            Row::Run { run, x, .. } => {
                let first = self.store.run_to_page(run);
                let second = self.store.alloc_page();
                let index_class = Self::index_class(MIN_INDEX);
                let index = self.store.alloc_run(index_class);
                Row::Paged {
                    pages: vec![first, second],
                    index,
                    index_class,
                    index_pages: Vec::new(),
                    x: x + 1,
                }
            }
            Row::Paged {
                mut pages,
                mut index,
                mut index_class,
                mut index_pages,
                x,
            } => {
                if x % PAGE == 0 {
                    let per_page = PAGE * 4 / S::INDEX_ENTRY_BYTES;
                    let n = pages.len() as u32;
                    if index_pages.is_empty() && n == Self::index_capacity(index_class) {
                        if index_class < PAGE {
                            // Grow the one-level index: copy its entries.
                            let grown = index_class * 2;
                            let new_index = self.store.alloc_run(grown);
                            self.store
                                .free_run(std::mem::replace(&mut index, new_index));
                            self.moved += u64::from((n * S::INDEX_ENTRY_BYTES).div_ceil(4));
                            index_class = grown;
                        } else {
                            // A full one-page index becomes the first second-level page as is.
                            let full = std::mem::replace(&mut index, self.store.alloc_run(MIN_RUN));
                            let page = self.store.run_to_page(full);
                            index_pages.push(page);
                            index_class = MIN_RUN;
                        }
                    }
                    if !index_pages.is_empty()
                        && n % per_page == 0
                        && n / per_page == index_pages.len() as u32
                    {
                        let page = self.store.alloc_page();
                        index_pages.push(page);
                        // The top level lists index pages by id; grow it when full.
                        if index_pages.len() as u32 > Self::index_capacity(index_class) {
                            let grown = (index_class * 2).min(PAGE);
                            let new_index = self.store.alloc_run(grown);
                            self.store
                                .free_run(std::mem::replace(&mut index, new_index));
                            self.moved += u64::from(
                                (index_pages.len() as u32 * S::INDEX_ENTRY_BYTES).div_ceil(4),
                            );
                            index_class = grown;
                        }
                    }
                    let page = self.store.alloc_page();
                    pages.push(page);
                }
                Row::Paged {
                    pages,
                    index,
                    index_class,
                    index_pages,
                    x: x + 1,
                }
            }
        };
        self.rows[v as usize] = next;
    }

    fn remove(&mut self, v: u32) {
        let row = std::mem::replace(&mut self.rows[v as usize], Row::Inline(0));
        let next = match row {
            Row::Inline(x) => Row::Inline(x.saturating_sub(1)),
            Row::Run { run, class, x } => Row::Run {
                run,
                class,
                x: x.saturating_sub(1),
            },
            Row::Paged {
                mut pages,
                index,
                index_class,
                index_pages,
                x,
            } => {
                let x = x - 1;
                if x % PAGE == 0 {
                    let page = pages.pop().expect("last page");
                    self.store.free_page(page);
                }
                if x == PAGE {
                    assert!(
                        index_pages.is_empty(),
                        "churn shrinks hubs within one index level"
                    );
                    self.store.free_run(index);
                    let page = pages.pop().expect("first page");
                    Row::Run {
                        run: self.store.page_to_run(page),
                        class: PAGE,
                        x,
                    }
                } else {
                    Row::Paged {
                        pages,
                        index,
                        index_class,
                        index_pages,
                        x,
                    }
                }
            }
        };
        self.rows[v as usize] = next;
    }

    fn degree(&self, v: u32) -> u32 {
        match &self.rows[v as usize] {
            Row::Inline(x) | Row::Run { x, .. } | Row::Paged { x, .. } => *x,
        }
    }

    fn report(&mut self, workload: &str, edges: u64, ops: u64) {
        // Descriptor size by inline capacity: 16 B holds 3 edges, 24 B 5, 32 B 7.
        let descriptor_bytes: u64 = match self.tiny {
            0..=3 => 16,
            4..=5 => 24,
            _ => 32,
        };
        let data = edges * 4;
        let memory = self.store.memory_bytes();
        let descriptors = self.rows.len() as u64 * descriptor_bytes;
        println!(
            "| {workload} | {} | {} ({descriptor_bytes} B) | {:.3} | {:.3} | {:.3} | {:.3} |",
            self.store.name(),
            self.tiny,
            memory as f64 / data as f64,
            (memory + descriptors) as f64 / data as f64,
            self.store.memory_calls() as f64 / ops as f64,
            self.moved as f64 / ops as f64,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Workloads

fn run_inserts<S: Store>(make: impl Fn() -> S, workload: &str, n: u32, edges: &[(u32, u32)]) {
    for tiny in [3, 5, 7] {
        let mut sim = Sim::new(make(), n, tiny);
        for &(u, _) in edges {
            sim.insert(u);
        }
        sim.report(workload, edges.len() as u64, edges.len() as u64);
    }
}

fn run_churn<S: Store>(make: impl Fn() -> S, n: u32, first: &[(u32, u32)], second: &[(u32, u32)]) {
    for tiny in [3, 5, 7] {
        let mut sim = Sim::new(make(), n, tiny);
        for &(u, _) in first {
            sim.insert(u);
        }
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
        for &(u, _) in second {
            sim.insert(u);
        }
        let live = first.len() as u64 + second.len() as u64 - removed;
        sim.report(
            "power-law churn",
            live,
            first.len() as u64 + removed + second.len() as u64,
        );
    }
}

fn tlsf() -> TlsfChunks {
    TlsfChunks {
        alloc: SpanAllocator::new(),
        chunks: Chunks::new(),
        probe_reads: 0,
    }
}

fn pages() -> PageStore {
    PageStore {
        pool: Pool::default(),
        chunks: Chunks::new(),
    }
}

fn both_inserts(workload: &str, n: u32, edges: &[(u32, u32)]) {
    run_inserts(tlsf, workload, n, edges);
    run_inserts(pages, workload, n, edges);
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
        "| workload | design | inline edges (descriptor) | memory / data | memory incl. descriptors / data | memory calls / op | slots moved / op |"
    );
    println!("|{}", "---|".repeat(7));
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "synthetic" => {
                let (n, m) = (1_000_000, 10_000_000);
                both_inserts("uniform 1M/10M", n, &coo_uniform_stream(n, m, 7));
                let power = coo_power_law_stream(n, m, 99, 1.0);
                both_inserts("power-law 1M/10M", n, &power);
                let shifted: Vec<_> = coo_power_law_stream(n, m / 2, 123, 1.0)
                    .into_iter()
                    .map(|(u, v)| ((u + n / 2) % n, v))
                    .collect();
                run_churn(tlsf, n, &power, &shifted);
                run_churn(pages, n, &power, &shifted);
                i += 1;
            }
            "orkut" => {
                let (n, edges) = load_input(&args[i + 1]);
                both_inserts("orkut-10M", n, &edges);
                i += 2;
            }
            other => panic!("unknown argument {other}"),
        }
    }
}
