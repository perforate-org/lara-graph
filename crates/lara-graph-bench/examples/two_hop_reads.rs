//! Memory traffic of the two-hop traversal (see `lara_graph_bench::two_hop`), layout by layout:
//! read calls, bytes, and distinct 4 KiB pages touched, per memory. Counts are the same on any
//! host, so they explain the canbench instruction counts without running on the IC. Run with
//! `cargo run --release --example two_hop_reads -p lara-graph-bench`.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::rc::Rc;
use core::cell::{Cell, RefCell};
use lara_graph::{GrowFailed, Memories, Memory, MemoryRole, VecMemory};
use lara_graph_bench::two_hop::{self, Layout};

/// Bytes per OS page, the unit the IC tracks stable memory access in.
const OS_PAGE: u64 = 4096;

/// A heap memory that counts what is read from it.
#[derive(Default)]
struct Counts {
    memory: VecMemory,
    calls: Cell<u64>,
    bytes: Cell<u64>,
    pages: RefCell<BTreeSet<u64>>,
}

impl Counts {
    fn reset(&self) {
        self.calls.set(0);
        self.bytes.set(0);
        self.pages.borrow_mut().clear();
    }
}

/// A shared handle to [`Counts`], so the report can read them while the graph holds the memory.
#[derive(Clone, Default)]
struct Probe(Rc<Counts>);

impl Memory for Probe {
    fn size(&self) -> u64 {
        self.0.memory.size()
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        self.0.memory.grow(pages)
    }

    fn read(&self, offset: u64, buf: &mut [u8]) {
        let len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
        let counts = &self.0;
        counts.calls.set(counts.calls.get() + 1);
        counts.bytes.set(counts.bytes.get() + len);
        if len > 0 {
            let mut pages = counts.pages.borrow_mut();
            pages.extend(offset / OS_PAGE..=(offset + len - 1) / OS_PAGE);
        }
        counts.memory.read(offset, buf);
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        self.0.memory.write(offset, buf);
    }
}

fn main() {
    println!(
        "{:<10} {:<26} {:>7} {:>9} {:>8}",
        "layout", "memory", "reads", "bytes", "4K pages"
    );
    for layout in Layout::ALL {
        let probes: Vec<(MemoryRole, Probe)> = MemoryRole::ALL
            .iter()
            .map(|&r| (r, Probe::default()))
            .collect();
        let memories = Memories::from_fn(|role| {
            probes
                .iter()
                .find(|(r, _)| *r == role)
                .map_or_else(|| panic!("no probe for {role:?}"), |(_, p)| p.clone())
        });
        let mut g = lara_graph::Graph::<Probe>::format(memories);
        let hub = two_hop::build(&mut g, layout);
        for (_, p) in &probes {
            p.0.reset();
        }
        let seen = two_hop::visit(&g, hub);
        assert_eq!(seen, u64::from(two_hop::SATELLITES * two_hop::FANOUT));
        let mut total = (0, 0, 0);
        for (role, p) in &probes {
            let (calls, bytes, pages) =
                (p.0.calls.get(), p.0.bytes.get(), p.0.pages.borrow().len());
            total = (total.0 + calls, total.1 + bytes, total.2 + pages);
            println!(
                "{:<10} {:<26} {calls:>7} {bytes:>9} {pages:>8}",
                layout.name(),
                format!("{role:?}")
            );
        }
        println!(
            "{:<10} {:<26} {:>7} {:>9} {:>8}\n",
            layout.name(),
            "total",
            total.0,
            total.1,
            total.2
        );
    }
}
