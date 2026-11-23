//! Scan-gap probe: fair 3-way comparison + core tree-scan breakdown.
//! Run with `-- --nocapture`. Single-shot Instant values (trend use).

use benches::{DgapAdapter, LabeledAdapter, LaraAdapter, LaraConfig, coo_power_law_stream};
use lara_graph::traits::DynamicCsr;
use std::time::Instant;

fn bench_reps<T, F: FnMut() -> T>(reps: u32, mut f: F) -> f64 {
    let t0 = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(f());
    }
    t0.elapsed().as_secs_f64() * 1e6 / reps as f64
}

#[test]
fn scan_probe() {
    // Hub-10k fixtures, T=1024 (tree at every T per scan_sweep guards).
    let n = 20_480u32;
    let d = 10_000u32;
    let cfg = LaraConfig {
        promote_threshold: Some(1024),
        max_log_entries: 170,
        ..LaraConfig::default()
    };
    let mut dgap = DgapAdapter::new(n).unwrap();
    let mut core = LaraAdapter::new(n, cfg).unwrap();
    let mut lab = LabeledAdapter::with_sweep_config(n, Some(1024)).unwrap();
    for t in 0..d {
        dgap.insert_edge(1, t);
        core.insert_edge(1, t);
        lab.insert_labeled(1, 5000 + t, 0);
    }
    assert!(lab.is_tree(1, 0), "labeled hub-10k must be tree");
    assert!(core.promote_count() >= 1, "core hub-10k must promote");
    dgap.verify();
    core.verify();

    let reps = 200u32;
    let a = bench_reps(reps, || {
        let mut sum = 0u64;
        dgap.for_each_neighbor(1, |x| sum += x as u64);
        std::hint::black_box(sum)
    });
    let b_sum = bench_reps(reps, || {
        let mut sum = 0u64;
        core.for_each_neighbor(1, |x| sum += x as u64);
        std::hint::black_box(sum)
    });
    let b_empty = bench_reps(reps, || {
        core.for_each_neighbor(1, |_| {});
    });
    let c_label = bench_reps(reps, || {
        let mut sum = 0u64;
        lab.scan_label(1, 0, |x| sum += x as u64);
        std::hint::black_box(sum)
    });
    let c_vertex = bench_reps(reps, || {
        let mut sum = 0u64;
        lab.scan_vertex(1, |x, _| sum += x as u64);
        std::hint::black_box(sum)
    });
    println!(
        "hub10k/scan-us: dgap={a:.2} core_sum={b_sum:.2} core_empty={b_empty:.2} labeled_of_label={c_label:.2} labeled_vertex={c_vertex:.2}"
    );
    println!(
        "hub10k/ratios: labeled_of_label/core={:.2} core/dgap={:.2}",
        c_label / b_sum,
        b_sum / a
    );

    // Core tree-scan stage breakdown (per-scan us).
    let s_row = bench_reps(2000, || core.scan_row_stage(1));
    let s_raw_empty = bench_reps(reps, || core.scan_tree_raw_empty(1));
    let s_raw_sum = bench_reps(reps, || std::hint::black_box(core.scan_tree_raw_sum(1)));
    let tree_read = s_raw_empty - s_row;
    let callback = b_empty - s_raw_empty;
    let accum = b_sum - b_empty;
    let raw_accum = s_raw_sum - s_raw_empty;
    println!(
        "hub10k/core-stages-us: row_branch={s_row:.3} tree_read_loop={tree_read:.2} callback_call={callback:.2} callback_accum={accum:.2} raw_accum={raw_accum:.2} total_check={:.2}",
        s_row + tree_read + callback + accum
    );
    println!(
        "hub10k/per-edge-ns: tree_read={:.2} callback={:.2}",
        tree_read * 1000.0 / d as f64,
        callback * 1000.0 / d as f64
    );

    // Dispersed settled full scan (smaller scale for labeled build cost).
    let dn = 20_000u32;
    let stream = coo_power_law_stream(dn, 80_000, 99, 1.0);
    let mut dd = DgapAdapter::new(dn).unwrap();
    let mut dc = LaraAdapter::new(dn, cfg).unwrap();
    let mut dl = LabeledAdapter::with_sweep_config(dn, Some(1024)).unwrap();
    for (i, &(u, v)) in stream.iter().enumerate() {
        dd.insert_edge(u, v);
        dc.insert_edge(u, v);
        dl.insert_labeled(u, v, (i % 4) as u16);
    }
    dc.settle();
    dl.settle();
    let r2 = 5u32;
    let sa = bench_reps(r2, || {
        let mut sum = 0u64;
        dd.scan_all(|x| sum += x as u64);
        std::hint::black_box(sum)
    });
    let sb = bench_reps(r2, || {
        let mut sum = 0u64;
        dc.scan_all(|x| sum += x as u64);
        std::hint::black_box(sum)
    });
    let sc = bench_reps(r2, || {
        let mut sum = 0u64;
        dl.scan_all(|x| sum += x as u64);
        std::hint::black_box(sum)
    });
    println!(
        "dispersed80k/scan-ms: dgap={:.2} core={:.2} labeled={:.2}",
        sa / 1000.0,
        sb / 1000.0,
        sc / 1000.0
    );
}
