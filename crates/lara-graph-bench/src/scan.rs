//! Reading a hub's bucket: whole scans collected or visited, in either order, in windows, and at
//! scattered positions.

use crate::{LABEL, heap_graph, seed_hub, stable_graph};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use core::num::NonZeroU32;
use core::ops::ControlFlow;
use lara_graph::{Cursor, Direction, Graph, Memory, Order, Position};

/// 51 full scans of a 1024-edge hub bucket (ic-stable-lara `bench_l_s2_h_scn_d_1024`).
pub(crate) fn scan_hub<M: Memory>(mut g: Graph<M>, degree: u32) -> BenchResult {
    let (hub, _) = seed_hub(&mut g, degree);
    let mut out = Vec::new();
    bench_fn(|| {
        for _ in 0..51 {
            g.neighbors(hub, (LABEL, Direction::Outgoing), &mut out)
                .unwrap_or_else(|e| panic!("scan: {e}"));
            black_box(out.len());
        }
    })
}

/// The same scans through the visitor, counting edges without collecting them.
pub(crate) fn visit_hub<M: Memory>(g: Graph<M>, degree: u32) -> BenchResult {
    visit_hub_in(g, degree, Order::Ascending)
}

fn visit_hub_in<M: Memory>(mut g: Graph<M>, degree: u32, order: Order) -> BenchResult {
    let (hub, _) = seed_hub(&mut g, degree);
    bench_fn(|| {
        for _ in 0..51 {
            let mut count = 0u32;
            g.visit_neighbors(hub, (LABEL, Direction::Outgoing), order, |_| {
                count += 1;
                ControlFlow::<()>::Continue(())
            })
            .unwrap_or_else(|e| panic!("visit: {e}"));
            black_box(count);
        }
    })
}

#[bench(raw)]
fn hub_visit_1024() -> BenchResult {
    visit_hub(heap_graph(), 1024)
}

#[bench(raw)]
fn hub_visit_65536() -> BenchResult {
    visit_hub(heap_graph(), 65536)
}

#[bench(raw)]
fn hub_visit_descending_65536() -> BenchResult {
    visit_hub_in(heap_graph(), 65536, Order::Descending)
}

/// A whole 65536-edge hub read in windows of 256 positions, newest first.
#[bench(raw)]
fn hub_windows_descending_65536() -> BenchResult {
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, 65536);
    let span = NonZeroU32::new(256).unwrap_or(NonZeroU32::MIN);
    bench_fn(|| {
        let mut cursor = Some(Cursor::first(Order::Descending, span));
        let mut count = 0u32;
        while let Some(at) = cursor {
            cursor = g
                .visit_window(hub, (LABEL, Direction::Outgoing), at, |_| count += 1)
                .unwrap_or_else(|e| panic!("window: {e}"));
        }
        black_box(count);
    })
}

/// 256 scattered positions of a 65536-edge hub read in one call.
#[bench(raw)]
fn hub_targets_at_256_of_65536() -> BenchResult {
    let mut g = heap_graph();
    let (hub, _) = seed_hub(&mut g, 65536);
    let positions: Vec<Position> = (0..256u32)
        .map(|i| Position::new(i * 251 % 65536))
        .collect();
    let mut out = Vec::new();
    bench_fn(|| {
        g.targets_at(hub, (LABEL, Direction::Outgoing), &positions, &mut out)
            .unwrap_or_else(|e| panic!("targets: {e}"));
        black_box(out.len());
    })
}

#[bench(raw)]
fn hub_scan_1024() -> BenchResult {
    scan_hub(heap_graph(), 1024)
}

#[bench(raw)]
fn hub_scan_1024_stable() -> BenchResult {
    scan_hub(stable_graph(), 1024)
}

#[bench(raw)]
fn hub_scan_65536() -> BenchResult {
    scan_hub(heap_graph(), 65536)
}
