//! Opening a graph again on its memories, as a canister's upgrade does: the cost every
//! `post_upgrade` pays before its first call.

use crate::{Bench, graph, nodes};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use lara_graph::{EdgeLabel, Width};

/// A graph of 1024 nodes with `labels` given 8-byte values, opened again.
fn open_with_widths(labels: &[EdgeLabel]) -> BenchResult {
    let mut g: Bench = graph();
    nodes(&mut g, 1024);
    let width = Width::new(8).unwrap_or_else(|| panic!("width"));
    for &label in labels {
        g.declare_width(label, width)
            .unwrap_or_else(|e| panic!("width: {e}"));
    }
    let memories = g.into_memories();
    bench_fn(|| {
        black_box(Bench::load(memories));
    })
}

/// No label has a width: the width table is empty.
#[bench(raw)]
fn open_1024() -> BenchResult {
    open_with_widths(&[])
}

/// One low label has a width: the table is a run on one page.
#[bench(raw)]
fn open_width_low_label_1024() -> BenchResult {
    open_with_widths(&[EdgeLabel::new(2)])
}

/// A low and a high label have widths: the table is a tree with two data pages of the 64 the
/// label range spans.
#[bench(raw)]
fn open_width_high_label_1024() -> BenchResult {
    open_with_widths(&[EdgeLabel::new(2), EdgeLabel::new(60_000)])
}
