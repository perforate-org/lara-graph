//! Native comparison bench target (harness = false).
//!
//! `--test` runs the 85 time cases once each (fixtures + independent verifier,
//! no timing) as a runtime smoke. Otherwise it registers the Criterion groups
//! (`--list` prints the frozen ids without building any fixture or graph).
//! Product code is not touched.
//!
//! Timing contract: all fixture/model/graph construction happens lazily inside
//! the `bench_with_input` callback, so `--list` and unrelated filters do no
//! heavy work. Mutations (`I`, `U`, `SET`) use
//! `iter_batched_ref(.., BatchSize::PerIteration)` with the input prepared
//! outside the timed routine. Read scans (`S1`, `S2`, `S3`, `CSR/SCAN`) are hot
//! repeated scans over a fixed, pre-verified graph via `b.iter`, with
//! `black_box` on the graph input. Not cold first-read latency.

use benches::compare::{
    AnyGraph, Model, Nodes, PgArm, TimeCase, WORKLOADS, Workload, apply_updates, check_dgap,
    check_lara, check_rows, insert_stream, model_for, pg_arm, scan_sum, stream, stream_fingerprint,
    time_cases, update_k, update_pairs,
};
use benches::{DgapAdapter, LaraAdapter, LaraConfig, StaticCsr};
use criterion::{BenchmarkId, Criterion, Throughput};
use lara_graph::traits::DynamicCsr;
use std::hint::black_box;

type Fixture = (Vec<(u32, u32)>, Vec<(u32, u32)>, Model);

fn workload(name: &str) -> Workload {
    *WORKLOADS
        .iter()
        .find(|w| w.name == name)
        .expect("known workload")
}

fn selected() -> Vec<Workload> {
    match std::env::var("LARA_CMP_CASE") {
        Ok(name) if !name.is_empty() => vec![workload(&name)],
        _ => WORKLOADS.to_vec(),
    }
}

fn data(w: &Workload) -> Fixture {
    let s = stream(w);
    let pairs = update_pairs(&s, update_k(w.m));
    let model = model_for(&s, w.n);
    (s, pairs, model)
}

// ---------------------------------------------------------------------------
// Runtime smoke (--test): execute the 85 cases once, verify, print one row each.
// ---------------------------------------------------------------------------

fn smoke() {
    let cases = time_cases();
    assert_eq!(cases.len(), 85, "frozen time case table");
    for case in &cases {
        let TimeCase {
            workload: wname,
            arm,
            phase,
        } = *case;
        let w = workload(wname);
        let s = stream(&w);
        let fp = stream_fingerprint(&s);
        let k = update_k(w.m);
        let pairs = update_pairs(&s, k);
        let mut model = model_for(&s, w.n);
        let tag = format!("{wname}/{arm}/{phase}");
        let (apply, settle, scan) = match phase {
            "I" | "S1" => (false, false, phase == "S1"),
            "U" => (true, false, false),
            "S2" => (true, false, true),
            "SET" => (true, true, false),
            "S3" => (true, true, true),
            "SCAN" => (false, false, true),
            other => panic!("unknown phase {other}"),
        };
        if apply {
            for &(src, dst) in &pairs {
                assert!(model.remove(src, dst), "{tag}: model delete");
                model.insert(src, dst);
            }
        }
        if arm == "CSR" {
            let a = StaticCsr::new(w.n, &s);
            check_rows(&a, &model, Nodes::Exact, &tag);
            let checksum = scan_sum(&a, w.n);
            println!(
                "CASE,{wname},{arm},{phase},m={},req_nodes={},act_nodes={},fp={fp},checksum={checksum}",
                w.m,
                w.n,
                a.num_nodes()
            );
            continue;
        }
        // Same shared paths as the timed bodies: insert_stream / apply_updates.
        let mut a = AnyGraph::new(arm, &w, &s, if arm == "PGP" { w.m } else { 0 });
        a.insert_all(&s);
        if apply {
            a.apply(&pairs);
        }
        if settle {
            a.settle_lara();
        }
        a.check(&model, &tag);
        let checksum = if scan { a.scan_sum(w.n) } else { 0 };
        println!(
            "CASE,{wname},{arm},{phase},m={},req_nodes={},act_nodes={},fp={fp},checksum={checksum}",
            w.m,
            w.n,
            a.num_nodes()
        );
    }
    println!("SMOKE_OK,time_cases=85");
}

// ---------------------------------------------------------------------------
// Criterion groups.
// ---------------------------------------------------------------------------

fn inserted_lara(w: &Workload, s: &[(u32, u32)]) -> LaraAdapter {
    let mut a = LaraAdapter::new(w.n, LaraConfig::default()).unwrap();
    insert_stream(&mut a, s);
    a
}
fn inserted_pg(w: &Workload, s: &[(u32, u32)], pre: u64) -> PgArm {
    let mut a = pg_arm(w.n, pre);
    insert_stream(&mut a, s);
    a
}
fn inserted_dgap(w: &Workload, s: &[(u32, u32)]) -> DgapAdapter {
    let mut a = DgapAdapter::new(w.n).unwrap();
    insert_stream(&mut a, s);
    a
}

/// Registers a hot repeated-scan benchmark. `prepare` builds the fixed graph and
/// its independent model lazily, only when the benchmark actually runs (never
/// under `--list`), and `verify` runs outside the timed loop.
fn hot_scan<G, P, V>(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    id: String,
    name: &'static str,
    mut prepare: P,
    verify: V,
    n: u32,
) where
    G: DynamicCsr,
    P: FnMut() -> (G, Model),
    V: Fn(&G, &Model),
{
    g.bench_with_input(BenchmarkId::new(id, name), &(), |b, _| {
        let (graph, model) = prepare();
        verify(&graph, &model);
        b.iter(|| black_box(scan_sum(black_box(&graph), n)))
    });
}

fn bench_workload(c: &mut Criterion, w: &Workload) {
    let name = w.name;
    let n = w.n;
    let m = w.m;
    let k = update_k(m);
    let mut g = c.benchmark_group("time");

    // LARA: I, S1, U, S2, SET, S3.
    g.throughput(Throughput::Elements(m));
    g.bench_with_input(BenchmarkId::new("L/I", name), &(), |b, _| {
        let (s, _, _) = data(w);
        b.iter_batched_ref(
            || LaraAdapter::new(n, LaraConfig::default()).unwrap(),
            |a| insert_stream(a, black_box(&s[..])),
            criterion::BatchSize::PerIteration,
        )
    });
    hot_scan(
        &mut g,
        "L/S1".into(),
        name,
        || {
            let (s, _, model) = data(w);
            (inserted_lara(w, &s), model)
        },
        |a, m| check_lara(a, m, "L/S1"),
        n,
    );
    g.throughput(Throughput::Elements(2 * k as u64));
    g.bench_with_input(BenchmarkId::new("L/U", name), &(), |b, _| {
        let (s, pairs, _) = data(w);
        b.iter_batched_ref(
            || inserted_lara(w, &s),
            |a| apply_updates(a, black_box(&pairs[..])),
            criterion::BatchSize::PerIteration,
        )
    });
    g.throughput(Throughput::Elements(m));
    hot_scan(
        &mut g,
        "L/S2".into(),
        name,
        || {
            let (s, pairs, model) = data(w);
            let mut a = inserted_lara(w, &s);
            apply_updates(&mut a, &pairs);
            (a, model)
        },
        |a, m| check_lara(a, m, "L/S2"),
        n,
    );
    // SET is one full-graph maintenance operation, not an edge rate.
    g.throughput(Throughput::Elements(1));
    g.bench_with_input(BenchmarkId::new("L/SET", name), &(), |b, _| {
        let (s, pairs, _) = data(w);
        b.iter_batched_ref(
            || {
                let mut a = inserted_lara(w, &s);
                apply_updates(&mut a, &pairs);
                a
            },
            |a| a.settle(),
            criterion::BatchSize::PerIteration,
        )
    });
    g.throughput(Throughput::Elements(m));
    hot_scan(
        &mut g,
        "L/S3".into(),
        name,
        || {
            let (s, pairs, model) = data(w);
            let mut a = inserted_lara(w, &s);
            apply_updates(&mut a, &pairs);
            a.settle();
            (a, model)
        },
        |a, m| check_lara(a, m, "L/S3"),
        n,
    );

    // petgraph growing and preallocated: I, S1, U, S2.
    for (arm, pre) in [("PG", 0u64), ("PGP", m)] {
        g.throughput(Throughput::Elements(m));
        g.bench_with_input(BenchmarkId::new(format!("{arm}/I"), name), &(), |b, _| {
            let (s, _, _) = data(w);
            b.iter_batched_ref(
                || pg_arm(n, pre),
                |a| insert_stream(a, black_box(&s[..])),
                criterion::BatchSize::PerIteration,
            )
        });
        hot_scan(
            &mut g,
            format!("{arm}/S1"),
            name,
            || {
                let (s, _, model) = data(w);
                (inserted_pg(w, &s, pre), model)
            },
            |a, m| check_rows(a, m, Nodes::Exact, "PG/S1"),
            n,
        );
        g.throughput(Throughput::Elements(2 * k as u64));
        g.bench_with_input(BenchmarkId::new(format!("{arm}/U"), name), &(), |b, _| {
            let (s, pairs, _) = data(w);
            b.iter_batched_ref(
                || inserted_pg(w, &s, pre),
                |a| apply_updates(a, black_box(&pairs[..])),
                criterion::BatchSize::PerIteration,
            )
        });
        g.throughput(Throughput::Elements(m));
        hot_scan(
            &mut g,
            format!("{arm}/S2"),
            name,
            || {
                let (s, pairs, model) = data(w);
                let mut a = inserted_pg(w, &s, pre);
                apply_updates(&mut a, &pairs);
                (a, model)
            },
            |a, m| check_rows(a, m, Nodes::Exact, "PG/S2"),
            n,
        );
    }

    // DGAP port: I and S1 only (no update arm).
    g.throughput(Throughput::Elements(m));
    g.bench_with_input(BenchmarkId::new("D/I", name), &(), |b, _| {
        let (s, _, _) = data(w);
        b.iter_batched_ref(
            || DgapAdapter::new(n).unwrap(),
            |a| insert_stream(a, black_box(&s[..])),
            criterion::BatchSize::PerIteration,
        )
    });
    hot_scan(
        &mut g,
        "D/S1".into(),
        name,
        || {
            let (s, _, model) = data(w);
            (inserted_dgap(w, &s), model)
        },
        |a, m| check_dgap(a, m, "D/S1"),
        n,
    );

    // Read-only CSR reference on the same requested-row loop.
    hot_scan(
        &mut g,
        "CSR/SCAN".into(),
        name,
        || {
            let (s, _, model) = data(w);
            (StaticCsr::new(n, &s), model)
        },
        |a, m| check_rows(a, m, Nodes::Exact, "CSR/SCAN"),
        n,
    );

    g.finish();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--test") {
        smoke();
        return;
    }
    let out = std::env::var_os("LARA_CMP_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/native-comparison")
        });
    let mut c = Criterion::default()
        .output_directory(&out)
        .configure_from_args();
    for w in selected() {
        bench_workload(&mut c, &w);
    }
    c.final_summary();
}
