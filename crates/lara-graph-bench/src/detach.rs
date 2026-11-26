//! Deleting nodes: a hub whole or in steps, a satellite of a hub, and many of a hub's
//! neighbours one by one or after a sweep of the bucket they share.

use crate::{Bench, LABEL, directed, graph, node, nodes, seed_hub, seed_parallel};
use canbench_rs::{BenchResult, bench, bench_fn};
use core::hint::black_box;
use core::num::NonZeroU32;
use lara_graph::{DeletingSet, Graph, Memory, NodeId, Position, Progress};

fn budget(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap_or(NonZeroU32::MIN)
}

/// Detaches `v`'s remaining edges in steps of `step` until it is deleted.
fn detach_rest<M: Memory>(g: &mut Graph<M>, v: NodeId, step: u32) {
    while g
        .delete_step(v, budget(step))
        .unwrap_or_else(|e| panic!("step: {e}"))
        .progress()
        == Progress::Remaining
    {}
}

/// A hub with 1024 parallel edges to one target: every step re-enters the same opposite bucket.
#[bench(raw)]
fn detach_parallel_1024() -> BenchResult {
    let mut g = graph();
    let (hub, _) = seed_parallel(&mut g, 1024);
    bench_fn(|| {
        g.begin_delete(hub)
            .unwrap_or_else(|e| panic!("detach: {e}"));
        black_box(
            g.delete_step(hub, budget(u32::MAX))
                .unwrap_or_else(|e| panic!("detach step: {e}")),
        )
    })
}

/// Detaches a hub of `degree` leaves in steps of `step` edges (ic-stable-lara
/// `bench_l_s2_det_hub_<degree>` and its stepped variant).
fn detach_hub(degree: u32, step: u32) -> BenchResult {
    let mut g = graph();
    let (hub, _) = seed_hub(&mut g, degree);
    bench_fn(|| {
        g.begin_delete(hub)
            .unwrap_or_else(|e| panic!("detach: {e}"));
        detach_rest(&mut g, hub, step);
    })
}

#[bench(raw)]
fn detach_hub_1024() -> BenchResult {
    detach_hub(1024, u32::MAX)
}

#[bench(raw)]
fn detach_hub_4096() -> BenchResult {
    detach_hub(4096, u32::MAX)
}

#[bench(raw)]
fn detach_hub_4096_step_64() -> BenchResult {
    detach_hub(4096, 64)
}

#[bench(raw)]
fn detach_hub_65536() -> BenchResult {
    detach_hub(65536, u32::MAX)
}

/// Detaches a satellite whose one edge is the last into a hub of in-degree `degree`: the other
/// half is found by scanning the hub's in bucket (ic-stable-lara `bench_l_s2_det_sat_<degree>`).
fn detach_satellite(degree: u32) -> BenchResult {
    let mut g = graph();
    let hub = node(&mut g);
    for source in nodes(&mut g, degree) {
        directed(&mut g, source, hub);
    }
    let satellite = node(&mut g);
    directed(&mut g, satellite, hub);
    bench_fn(|| {
        g.begin_delete(satellite)
            .unwrap_or_else(|e| panic!("detach: {e}"));
        black_box(g.delete_step(satellite, budget(u32::MAX)))
    })
}

#[bench(raw)]
fn detach_satellite_1024() -> BenchResult {
    detach_satellite(1024)
}

#[bench(raw)]
fn detach_satellite_4096() -> BenchResult {
    detach_satellite(4096)
}

#[bench(raw)]
fn detach_satellite_65536() -> BenchResult {
    detach_satellite(65536)
}

/// How a bench detaches many nodes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Deleting {
    /// Each with its own detach steps.
    OneByOne,
    /// A sweep of the bucket they share first.
    Swept,
}

/// Detaches `n` of a hub's 65536 in-neighbours, spread evenly.
fn delete_many(n: u32, how: Deleting) -> BenchResult {
    let mut g = graph();
    let hub = node(&mut g);
    let sources = nodes(&mut g, 65536);
    for &s in &sources {
        directed(&mut g, s, hub);
    }
    let step = usize::try_from(65536 / n).unwrap_or(1);
    let gone: Vec<NodeId> = sources.iter().step_by(step).copied().collect();
    bench_fn(|| {
        for &d in &gone {
            g.begin_delete(d).unwrap_or_else(|e| panic!("detach: {e}"));
        }
        if how == Deleting::Swept {
            let detaching = g
                .deleting_set(gone.iter().copied())
                .unwrap_or_else(|e| panic!("detaching set: {e}"));
            sweep_all(&mut g, hub, &detaching);
        }
        for &d in &gone {
            detach_rest(&mut g, d, u32::MAX);
        }
    })
}

/// Sweeps the hub's in-bucket of edges from `detaching`, to its end.
fn sweep_all(g: &mut Bench, hub: NodeId, detaching: &DeletingSet) {
    let mut from = Some(Position::new(0));
    while let Some(at) = from {
        let sweep = g
            .sweep(hub, LABEL.incoming(), at, budget(4096), detaching)
            .unwrap_or_else(|e| panic!("sweep: {e}"));
        from = sweep.next();
    }
}

#[bench(raw)]
fn detach_1024_of_hub_65536() -> BenchResult {
    delete_many(1024, Deleting::OneByOne)
}

#[bench(raw)]
fn detach_1024_of_hub_65536_swept() -> BenchResult {
    delete_many(1024, Deleting::Swept)
}

#[bench(raw)]
fn detach_4096_of_hub_65536() -> BenchResult {
    delete_many(4096, Deleting::OneByOne)
}

#[bench(raw)]
fn detach_4096_of_hub_65536_swept() -> BenchResult {
    delete_many(4096, Deleting::Swept)
}
