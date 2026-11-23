//! Criterion benchmarks for the LARA paper: DGAP Rust port vs generalized
//! LARA vs static CSR.
//! Lifecycle v2: constructors/vertex creation and fixture generation are
//! outside insert timing; edge growth and mutation maintenance are inside;
//! returned graphs/buffers are dropped outside. Scans time traversal only.
//! See docs/experiment-contracts-2026-09-11.md before adopting results.
//!
//! Arms (see `benches/src/adapters.rs`):
//! 1. `insert_vs_size` — insertion throughput (MEPS) vs graph size
//! 2. `rebalance_vs_degree` — rebalance cost (ins/edge) vs degree, slab vs
//!    tree-CSR promotion thresholds
//! 3. `tpromote_sweep` — T_promote ∈ {512, 1024, 2048, 4096, 8192, 16384}
//!    (default T_PROMOTE remains 1024; the `lara_tree` arm
//!    promotes at 1024)
//! 4. `segment_sweep` — segment size ∈ {8, 16, 32, 64, 128}
//! 5. `scan_overhead` — scan vs static CSR baseline
//! 6. `ablation` — full vs no-tree vs no-free-span

use benches::{
    CountingLaraAdapter, DgapAdapter, LabeledAdapter, LaraAdapter, LaraConfig, MmapLaraAdapter,
    StaticCsr, coo_power_law, coo_power_law_stream, coo_uniform_stream,
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lara_graph::traits::DynamicCsr;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// 1. Insertion throughput vs graph size
// ---------------------------------------------------------------------------

fn bench_insert_vs_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_vs_size");
    for &n in &[1_000u32, 10_000, 100_000] {
        let m = (n as u64) * 4; // avg degree 4
        let stream: Vec<(u32, u32)> = coo_uniform_stream(n, m, 1234);

        group.throughput(Throughput::Elements(m));

        group.bench_with_input(BenchmarkId::new("dgap", n), &n, |b, &n| {
            b.iter_batched(
                || DgapAdapter::new(n).unwrap(),
                |mut g| {
                    for &(u, v) in &stream {
                        g.insert_edge(u, v);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("lara_slab", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    LaraAdapter::new(
                        n,
                        LaraConfig {
                            promote_threshold: None,
                            ..LaraConfig::default()
                        },
                    )
                    .unwrap()
                },
                |mut g| {
                    for &(u, v) in &stream {
                        g.insert_edge(u, v);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("lara_tree", n), &n, |b, &n| {
            b.iter_batched(
                || LaraAdapter::new(n, LaraConfig::default()).unwrap(),
                |mut g| {
                    for &(u, v) in &stream {
                        g.insert_edge(u, v);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Rebalance cost vs degree (10² → 10⁶), slab vs tree-CSR
// ---------------------------------------------------------------------------

fn bench_rebalance_vs_degree(c: &mut Criterion) {
    let mut group = c.benchmark_group("rebalance_vs_degree");
    group.sample_size(10);
    // One hub vertex grown to degree d: every insert past the slab window
    // triggers leaf rebalance / tree promotion machinery.
    for &d in &[100u32, 10_000, 100_000, 1_000_000] {
        let n = (d * 2).max(1024);

        group.bench_with_input(BenchmarkId::new("dgap_hub", d), &d, |b, &d| {
            b.iter_batched(
                || DgapAdapter::new(n).unwrap(),
                |mut g| {
                    for t in 0..d {
                        g.insert_edge(1, t);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("lara_slab_hub", d), &d, |b, &d| {
            b.iter_batched(
                || {
                    LaraAdapter::new(
                        n,
                        LaraConfig {
                            promote_threshold: None,
                            ..LaraConfig::default()
                        },
                    )
                    .unwrap()
                },
                |mut g| {
                    for t in 0..d {
                        g.insert_edge(1, t);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("lara_tree_hub", d), &d, |b, &d| {
            b.iter_batched(
                || LaraAdapter::new(n, LaraConfig::default()).unwrap(),
                |mut g| {
                    for t in 0..d {
                        g.insert_edge(1, t);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 3. T_promote sweep
// ---------------------------------------------------------------------------

fn bench_tpromote_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("tpromote_sweep");
    group.sample_size(10);
    let n = 200_000u32;
    // Hub degree 2^15 crosses every swept threshold; per-insert cost is the
    // observable (promotion fires mid-stream for the small thresholds).
    let d = 32_768u32;
    for &t in &[512u32, 1024, 2048, 4096, 8192, 16384] {
        group.bench_with_input(BenchmarkId::new("lara", t), &t, |b, &t| {
            b.iter_batched(
                || {
                    LaraAdapter::new(
                        n,
                        LaraConfig {
                            promote_threshold: Some(t),
                            ..LaraConfig::default()
                        },
                    )
                    .unwrap()
                },
                |mut g| {
                    for x in 0..d {
                        g.insert_edge(1, x);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Segment size sweep
// ---------------------------------------------------------------------------

fn bench_segment_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_sweep");
    group.sample_size(10);
    let n = 100_000u32;
    let m = 400_000u64;
    let stream: Vec<(u32, u32)> = coo_power_law_stream(n, m, 99, 1.0);
    for &ss in &[8u32, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("lara", ss), &ss, |b, &ss| {
            b.iter_batched(
                || {
                    LaraAdapter::new(
                        n,
                        LaraConfig {
                            segment_size: Some(ss),
                            ..LaraConfig::default()
                        },
                    )
                    .unwrap()
                },
                |mut g| {
                    for &(u, v) in &stream {
                        g.insert_edge(u, v);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Scan overhead vs static CSR
// ---------------------------------------------------------------------------

fn bench_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan_overhead");
    let n = 100_000u32;
    let m = 400_000u64;
    let edges = coo_power_law(n, m, 99, 1.0);

    let static_g = StaticCsr::new(n, &edges);
    let dgap = DgapAdapter::with_edges(n, &edges).unwrap();
    let lara = LaraAdapter::with_edges(n, &edges, LaraConfig::default()).unwrap();
    // Drain check.
    assert_eq!(static_g.num_edges(), m);
    assert_eq!(dgap.num_edges(), m);
    assert_eq!(lara.num_edges(), m);

    group.throughput(Throughput::Elements(m));

    group.bench_function("static_csr", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for v in 0..n {
                static_g.for_each_neighbor(v, |t| sum += t as u64);
            }
            black_box(sum)
        })
    });

    group.bench_function("dgap", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for v in 0..n {
                dgap.for_each_neighbor(v, |t| sum += t as u64);
            }
            black_box(sum)
        })
    });

    group.bench_function("lara", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for v in 0..n {
                lara.for_each_neighbor(v, |t| sum += t as u64);
            }
            black_box(sum)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. Component ablation: full vs no-tree vs no-free-span
// ---------------------------------------------------------------------------

fn bench_ablation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ablation");
    group.sample_size(10);
    let n = 50_000u32;
    let m = 200_000u64;
    let stream: Vec<(u32, u32)> = coo_power_law_stream(n, m, 55, 1.0);

    group.bench_function("full", |b| {
        b.iter_batched(
            || LaraAdapter::new(n, LaraConfig::default()).unwrap(),
            |mut g| {
                for &(u, v) in &stream {
                    g.insert_edge(u, v);
                }
                g
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("no_tree", |b| {
        b.iter_batched(
            || {
                LaraAdapter::new(
                    n,
                    LaraConfig {
                        promote_threshold: None,
                        ..LaraConfig::default()
                    },
                )
                .unwrap()
            },
            |mut g| {
                for &(u, v) in &stream {
                    g.insert_edge(u, v);
                }
                g
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("no_free_span", |b| {
        b.iter_batched(
            || {
                LaraAdapter::new(
                    n,
                    LaraConfig {
                        use_free_spans: false,
                        ..LaraConfig::default()
                    },
                )
                .unwrap()
            },
            |mut g| {
                for &(u, v) in &stream {
                    g.insert_edge(u, v);
                }
                g
            },
            criterion::BatchSize::LargeInput,
        )
    });

    // Bulk-built labeled fixture; descriptor labels are out of band.
    // Construction and destruction are not part of this scan-only arm.
    let labeled_edges: Vec<(u32, u32, u16)> = stream.iter().map(|&(u, v)| (u, v, 1)).collect();
    let labeled = LabeledAdapter::with_labeled_edges(n, &labeled_edges).unwrap();
    assert_eq!(labeled.num_edges(), m);
    group.bench_function("labeled_scan", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            labeled.scan_all(|t| sum += t as u64);
            black_box(sum)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 7. Threshold sweep: per-row alloc `max(x, min(max(2*x, 1), T))`.
// T varies; no prior optimum is assumed.
// ---------------------------------------------------------------------------

fn bench_promote_threshold_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("promote_threshold_sweep");
    group.sample_size(10);
    // Hub (vertex 1 to degree d) + dispersed power-law stream. Fixed
    // alloc `max(x, min(max(2*x, 1), T))`. T caps headroom, not live data;
    // small T promotes earlier, while large T keeps wider slab rows.
    let hub_n = 65_536u32;
    let hub_d = 32_768u32;
    let disp_n = 100_000u32;
    let disp_m = 400_000u64;
    let disp_stream: Vec<(u32, u32)> = coo_power_law_stream(disp_n, disp_m, 99, 1.0);

    for &t in &[1024u32, 2048, 4096, 8192] {
        let cfg = LaraConfig {
            promote_threshold: Some(t),
            ..LaraConfig::default()
        };
        group.bench_with_input(BenchmarkId::new("hub", t), &t, |b, _| {
            b.iter_batched(
                || LaraAdapter::new(hub_n, cfg).unwrap(),
                |mut gr| {
                    for x in 0..hub_d {
                        gr.insert_edge(1, x);
                    }
                    gr
                },
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("dispersed", t), &t, |b, _| {
            b.iter_batched(
                || LaraAdapter::new(disp_n, cfg).unwrap(),
                |mut gr| {
                    for &(u, v) in &disp_stream {
                        gr.insert_edge(u, v);
                    }
                    gr
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 9. Scan x threshold sweep: slab scan vs tree scan per T.
// ---------------------------------------------------------------------------

fn bench_scan_threshold_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan_threshold_sweep");
    // Fixed alloc `min(max(2*x, 1), T)`. Fixtures are built with
    // sequential inserts (bulk build would lay slab directly and skip
    // promotion). hub-3k straddles the boundary (tree at T<=2048, slab
    // above); hub-10k is tree at every T. Each hub vertex also carries a
    // small label-1 companion bucket (10 edges) for mixed-mode scans.
    let n = 16_384u32;
    for &t in &[1024u32, 2048, 4096, 8192] {
        let mut g3 = LabeledAdapter::with_sweep_config(n, Some(t)).unwrap();
        for d in 0..3000u32 {
            g3.insert_labeled(0, 100 + d, 0);
        }
        for d in 0..10u32 {
            g3.insert_labeled(0, 4000 + d, 1);
        }
        assert_eq!(g3.is_tree(0, 0), t <= 2048, "hub-3k fixture guard T={t}");

        let mut g10 = LabeledAdapter::with_sweep_config(n, Some(t)).unwrap();
        for d in 0..10_000u32 {
            g10.insert_labeled(1, 5000 + d, 0);
        }
        for d in 0..10u32 {
            g10.insert_labeled(1, 4000 + d, 1);
        }
        assert!(g10.is_tree(1, 0), "hub-10k fixture guard T={t}");

        group.bench_with_input(BenchmarkId::new("hub3k_label", t), &t, |b, _| {
            b.iter(|| {
                let mut sum = 0u64;
                g3.scan_label(0, 0, |x| sum += x as u64);
                black_box(sum)
            })
        });
        group.bench_with_input(BenchmarkId::new("hub3k_vertex", t), &t, |b, _| {
            b.iter(|| {
                let mut sum = 0u64;
                g3.scan_vertex(0, |x, _| sum += x as u64);
                black_box(sum)
            })
        });
        group.bench_with_input(BenchmarkId::new("hub3k_all", t), &t, |b, _| {
            b.iter(|| {
                let mut sum = 0u64;
                g3.scan_all(|x| sum += x as u64);
                black_box(sum)
            })
        });
        group.bench_with_input(BenchmarkId::new("hub10k_label", t), &t, |b, _| {
            b.iter(|| {
                let mut sum = 0u64;
                g10.scan_label(1, 0, |x| sum += x as u64);
                black_box(sum)
            })
        });
        group.bench_with_input(BenchmarkId::new("hub10k_vertex", t), &t, |b, _| {
            b.iter(|| {
                let mut sum = 0u64;
                g10.scan_vertex(1, |x, _| sum += x as u64);
                black_box(sum)
            })
        });
        group.bench_with_input(BenchmarkId::new("hub10k_all", t), &t, |b, _| {
            b.iter(|| {
                let mut sum = 0u64;
                g10.scan_all(|x| sum += x as u64);
                black_box(sum)
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 10. Log-cap sweep: per-leaf overflow-log cap ∈ {0, 4, 8, 16, 32, 64,
// 128, 170, 256, 512}. T=1024 fixed, single alloc formula. Scan fixtures
// are settled (`rebalance_all`) first, so scan measures the post-fold
// steady state. cap=0 is the no-log endpoint (overflow rebalances every
// time; at the capped boundary it promotes instead — no infinite loop).
// ---------------------------------------------------------------------------

fn bench_log_cap_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_cap_sweep");
    group.sample_size(10);
    let hub_n = 65_536u32;
    let hub_d = 32_768u32;
    let disp_n = 100_000u32;
    let disp_m = 400_000u64;
    let disp_stream: Vec<(u32, u32)> = coo_power_law_stream(disp_n, disp_m, 99, 1.0);

    for &m in &[0usize, 4, 8, 16, 32, 64, 128, 170, 256, 512] {
        let cfg = LaraConfig {
            promote_threshold: Some(1024),
            max_log_entries: m,
            ..LaraConfig::default()
        };
        group.bench_with_input(BenchmarkId::new("hub", m), &m, |b, _| {
            b.iter_batched(
                || LaraAdapter::new(hub_n, cfg).unwrap(),
                |mut gr| {
                    for x in 0..hub_d {
                        gr.insert_edge(1, x);
                    }
                    gr
                },
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("dispersed", m), &m, |b, _| {
            b.iter_batched(
                || LaraAdapter::new(disp_n, cfg).unwrap(),
                |mut gr| {
                    for &(u, v) in &disp_stream {
                        gr.insert_edge(u, v);
                    }
                    gr
                },
                criterion::BatchSize::LargeInput,
            )
        });
        // Settled scan fixture: hub-3k (tree at T=1024) + companion.
        let mut g3 = LabeledAdapter::with_sweep_config_and_max_log(16_384, Some(1024), m).unwrap();
        for d in 0..3000u32 {
            g3.insert_labeled(0, 100 + d, 0);
        }
        for d in 0..10u32 {
            g3.insert_labeled(0, 4000 + d, 1);
        }
        g3.settle();
        group.bench_with_input(BenchmarkId::new("hub3k_label", m), &m, |b, _| {
            b.iter(|| {
                let mut sum = 0u64;
                g3.scan_label(0, 0, |x| sum += x as u64);
                black_box(sum)
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 11. Uninstrumented insertion timings plus separate backend-write counts.
// Counts omit construction and do not represent complete durable graph
// updates. The mmap arm excludes explicit flush/recovery (file backing only).
// No ranking or cycle-cost inversion is assumed.
// ---------------------------------------------------------------------------

fn bench_persist_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("persist_sweep");
    group.sample_size(10);
    let hub_n = 65_536u32;
    let hub_d = 32_768u32;
    let disp_n = 100_000u32;
    let disp_m = 400_000u64;
    let disp_stream: Vec<(u32, u32)> = coo_power_law_stream(disp_n, disp_m, 99, 1.0);

    // Persist-volume snapshot: build each arm once on the counting
    // adapter and report bytes/insert + ops/insert.
    for &m in &[0usize, 32, 170, 512] {
        let cfg = LaraConfig {
            promote_threshold: Some(1024),
            max_log_entries: m,
            ..LaraConfig::default()
        };
        let mut hub = CountingLaraAdapter::new(hub_n, cfg).unwrap();
        for x in 0..hub_d {
            hub.insert_edge(1, x);
        }
        let hub_counts = (hub.persisted_bytes_per_edge(), hub.write_ops_per_edge());
        hub.verify();
        println!(
            "persist_sweep backend_writes: cap={m} hub: {:.1} bytes/insert {:.2} ops/insert",
            hub_counts.0, hub_counts.1
        );
        let mut disp = CountingLaraAdapter::new(disp_n, cfg).unwrap();
        for &(u, v) in &disp_stream {
            disp.insert_edge(u, v);
        }
        let disp_counts = (disp.persisted_bytes_per_edge(), disp.write_ops_per_edge());
        disp.verify();
        println!(
            "persist_sweep backend_writes: cap={m} rank_zipf: {:.1} bytes/insert {:.2} ops/insert",
            disp_counts.0, disp_counts.1
        );
    }

    for &m in &[0usize, 32, 170, 512] {
        let cfg = LaraConfig {
            promote_threshold: Some(1024),
            max_log_entries: m,
            ..LaraConfig::default()
        };
        group.bench_with_input(BenchmarkId::new("hub", m), &m, |b, _| {
            b.iter_batched(
                || LaraAdapter::new(hub_n, cfg).unwrap(),
                |mut gr| {
                    for x in 0..hub_d {
                        gr.insert_edge(1, x);
                    }
                    gr
                },
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("dispersed", m), &m, |b, _| {
            b.iter_batched(
                || LaraAdapter::new(disp_n, cfg).unwrap(),
                |mut gr| {
                    for &(u, v) in &disp_stream {
                        gr.insert_edge(u, v);
                    }
                    gr
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    // mmap wall-clock arm (hub, cap=170 default config shape).
    let mmap_cfg = LaraConfig {
        promote_threshold: Some(1024),
        max_log_entries: 170,
        ..LaraConfig::default()
    };
    group.bench_function("mmap_hub_170", |b| {
        b.iter_batched(
            || MmapLaraAdapter::new(hub_n, mmap_cfg).unwrap(),
            |mut gr| {
                for x in 0..hub_d {
                    gr.insert_edge(1, x);
                }
                gr
            },
            criterion::BatchSize::LargeInput,
        )
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// 12. dgap vs log0 LARA (pre-log-removal record). Current code shape:
// single alloc formula `min(max(2*x, 1), T)`, T=1024. Arms: `dgap`
// (untouched port), `lara_log0` (cap=0), `lara_170` (cap=170 reference
// isolating the log effect). Fixtures mirror `rebalance_vs_degree`
// (hub), `insert_vs_size` (dispersed uniform), plus a hub-10k neighbor
// scan per arm. `LaraConfig::default()` untouched (still 170).
// ---------------------------------------------------------------------------

fn bench_dgap_vs_log0(c: &mut Criterion) {
    let mut group = c.benchmark_group("dgap_vs_log0");
    group.sample_size(10);
    let cfg_log0 = LaraConfig {
        promote_threshold: Some(1024),
        max_log_entries: 0,
        ..LaraConfig::default()
    };
    let cfg_170 = LaraConfig {
        promote_threshold: Some(1024),
        max_log_entries: 170,
        ..LaraConfig::default()
    };

    // 1. Hub growth: one vertex to degree d.
    for &d in &[100u32, 10_000, 100_000, 1_000_000] {
        let n = (d * 2).max(1024);
        group.bench_with_input(BenchmarkId::new("dgap_hub", d), &d, |b, &d| {
            b.iter_batched(
                || DgapAdapter::new(n).unwrap(),
                |mut g| {
                    for t in 0..d {
                        g.insert_edge(1, t);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("lara_log0_hub", d), &d, |b, &d| {
            b.iter_batched(
                || LaraAdapter::new(n, cfg_log0).unwrap(),
                |mut g| {
                    for t in 0..d {
                        g.insert_edge(1, t);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("lara_170_hub", d), &d, |b, &d| {
            b.iter_batched(
                || LaraAdapter::new(n, cfg_170).unwrap(),
                |mut g| {
                    for t in 0..d {
                        g.insert_edge(1, t);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }

    // 2. Dispersed uniform inserts (same shape as `insert_vs_size`).
    for &n in &[1_000u32, 10_000, 100_000] {
        let m = (n as u64) * 4;
        let stream: Vec<(u32, u32)> = coo_uniform_stream(n, m, 1234);
        group.throughput(Throughput::Elements(m));
        group.bench_with_input(BenchmarkId::new("dgap", n), &n, |b, _| {
            b.iter_batched(
                || DgapAdapter::new(n).unwrap(),
                |mut g| {
                    for &(u, v) in &stream {
                        g.insert_edge(u, v);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("lara_log0", n), &n, |b, _| {
            b.iter_batched(
                || LaraAdapter::new(n, cfg_log0).unwrap(),
                |mut g| {
                    for &(u, v) in &stream {
                        g.insert_edge(u, v);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("lara_170", n), &n, |b, _| {
            b.iter_batched(
                || LaraAdapter::new(n, cfg_170).unwrap(),
                |mut g| {
                    for &(u, v) in &stream {
                        g.insert_edge(u, v);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }

    // 3. Hub-10k neighbor scan per arm (same fixture built on each).
    let sn = 20_480u32;
    let sd = 10_000u32;
    let mut dgap_s = DgapAdapter::new(sn).unwrap();
    let mut log0_s = LaraAdapter::new(sn, cfg_log0).unwrap();
    let mut ref_s = LaraAdapter::new(sn, cfg_170).unwrap();
    for t in 0..sd {
        dgap_s.insert_edge(1, t);
        log0_s.insert_edge(1, t);
        ref_s.insert_edge(1, t);
    }
    assert_eq!(dgap_s.num_edges(), log0_s.num_edges());
    assert_eq!(dgap_s.num_edges(), ref_s.num_edges());
    group.bench_function("dgap_scan_hub10k", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            dgap_s.for_each_neighbor(1, |x| sum += x as u64);
            black_box(sum)
        })
    });
    group.bench_function("lara_log0_scan_hub10k", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            log0_s.for_each_neighbor(1, |x| sum += x as u64);
            black_box(sum)
        })
    });
    group.bench_function("lara_170_scan_hub10k", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            ref_s.for_each_neighbor(1, |x| sum += x as u64);
            black_box(sum)
        })
    });
    group.finish();
}

fn bench_petgraph_vs_lara(c: &mut Criterion) {
    use petgraph::Graph;
    use petgraph::{Directed, graph::NodeIndex};
    let mut group = c.benchmark_group("petgraph_vs_lara");
    group.sample_size(10);
    let lara_cfg = LaraConfig::default();
    for &d in &[100u32, 10_000, 100_000] {
        let n = (d * 2).max(1024);
        group.bench_with_input(BenchmarkId::new("petgraph_growing_hub", d), &d, |b, &d| {
            b.iter_batched(
                || {
                    let mut g: Graph<(), (), Directed> = Graph::with_capacity(n as usize, 0);
                    for _ in 0..n {
                        g.add_node(());
                    }
                    g
                },
                |mut g| {
                    for t in 0..d {
                        g.add_edge(NodeIndex::new(1), NodeIndex::new(t as usize), ());
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("lara_hub", d), &d, |b, &d| {
            b.iter_batched(
                || LaraAdapter::new(n, lara_cfg).unwrap(),
                |mut g| {
                    for t in 0..d {
                        g.insert_edge(1, t);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    for &n in &[1_000u32, 10_000, 100_000] {
        let m = (n as u64) * 4;
        let stream: Vec<(u32, u32)> = coo_uniform_stream(n, m, 1234);
        group.throughput(Throughput::Elements(m));
        group.bench_with_input(BenchmarkId::new("petgraph_growing", n), &n, |b, _| {
            b.iter_batched(
                || {
                    let mut g: Graph<(), (), Directed> = Graph::with_capacity(n as usize, 0);
                    for _ in 0..n {
                        g.add_node(());
                    }
                    g
                },
                |mut g| {
                    for &(u, v) in &stream {
                        g.add_edge(NodeIndex::new(u as usize), NodeIndex::new(v as usize), ());
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("lara", n), &n, |b, _| {
            b.iter_batched(
                || LaraAdapter::new(n, lara_cfg).unwrap(),
                |mut g| {
                    for &(u, v) in &stream {
                        g.insert_edge(u, v);
                    }
                    g
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    let sn = 20_480u32;
    let sd = 10_000u32;
    let mut pg: Graph<(), (), Directed> = Graph::with_capacity(sn as usize, sd as usize);
    for _ in 0..sn {
        pg.add_node(());
    }
    for t in 0..sd {
        pg.add_edge(NodeIndex::new(1), NodeIndex::new(t as usize), ());
    }
    let mut lara_s = LaraAdapter::new(sn, lara_cfg).unwrap();
    for t in 0..sd {
        lara_s.insert_edge(1, t);
    }
    assert_eq!(pg.edge_count() as u64, lara_s.num_edges());
    let hub1 = NodeIndex::new(1);
    group.bench_function("petgraph_scan_hub10k", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for nb in pg.neighbors(hub1) {
                sum += nb.index() as u64;
            }
            black_box(sum)
        })
    });
    group.bench_function("lara_scan_hub10k", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            lara_s.for_each_neighbor(1, |x| sum += x as u64);
            black_box(sum)
        })
    });
    // Allocated memory is measured separately by examples/memory_probe.rs.
    group.finish();
}

// ---------------------------------------------------------------------------
// 13. Ingest (sorted-run batching) vs per-edge insert, release values.
// Uniform n=10k/m=40k (same stream as the ingest probe).
// `sort_only` is an isolated component, not a measured fraction
// of ingest latency. Setup and destruction are excluded for every arm;
// bulk_grouped excludes sorting, bulk_with_sort includes it.
// ---------------------------------------------------------------------------

fn bench_ingest(c: &mut Criterion) {
    use lara_graph::graph::{Edge, LaraGraph, LaraMemories, Vertex};
    use lara_graph::ingest::IngestBuffer;
    use lara_graph::memory::VecMemory;
    fn fresh_core_mem() -> LaraMemories<VecMemory> {
        LaraMemories {
            vertices: VecMemory::new(),
            edges: VecMemory::new(),
            logs: VecMemory::new(),
            counts: VecMemory::new(),
            spans: VecMemory::new(),
            free_spans: VecMemory::new(),
            blocks: VecMemory::new(),
            buckets: VecMemory::new(),
        }
    }
    let mut group = c.benchmark_group("ingest");
    group.sample_size(10);
    let n = 10_000u32;
    let stream: Vec<(u32, u32)> = coo_uniform_stream(n, 40_000, 1234);
    group.bench_function("per_edge_uniform10k", |b| {
        b.iter_batched(
            || LaraGraph::<Vertex, Edge, VecMemory>::new(fresh_core_mem(), n).unwrap(),
            |mut g| {
                for &(u, v) in &stream {
                    g.insert_edge(u, v);
                }
                black_box(g)
            },
            criterion::BatchSize::LargeInput,
        );
    });
    group.bench_function("ingest_uniform10k", |b| {
        b.iter_batched(
            || {
                (
                    LaraGraph::<Vertex, Edge, VecMemory>::new(fresh_core_mem(), n).unwrap(),
                    IngestBuffer::new(n, 4096),
                )
            },
            |(mut g, mut ing)| {
                for &(u, v) in &stream {
                    ing.push(u, v);
                    if ing.len() >= 4096 {
                        ing.flush(&mut g);
                    }
                }
                ing.flush(&mut g);
                black_box((g, ing))
            },
            criterion::BatchSize::LargeInput,
        );
    });
    group.bench_function("sort_only_uniform10k", |b| {
        b.iter_batched(
            || stream.clone(),
            |mut s| {
                for chunk in s.chunks_mut(4096) {
                    chunk.sort_unstable();
                }
                black_box(s)
            },
            criterion::BatchSize::LargeInput,
        );
    });
    let grouped = benches::coo_grouped(stream.clone());
    group.bench_function("bulk_grouped_uniform10k", |b| {
        b.iter_batched(
            || LaraGraph::<Vertex, Edge, VecMemory>::new(fresh_core_mem(), n).unwrap(),
            |mut g| {
                g.build_pairs(black_box(&grouped)).unwrap();
                black_box(g)
            },
            criterion::BatchSize::LargeInput,
        );
    });
    group.bench_function("bulk_with_sort_uniform10k", |b| {
        b.iter_batched(
            || {
                (
                    LaraGraph::<Vertex, Edge, VecMemory>::new(fresh_core_mem(), n).unwrap(),
                    stream.clone(),
                )
            },
            |(mut g, mut edges)| {
                edges.sort_by_key(|&(src, _)| src);
                g.build_pairs(&edges).unwrap();
                black_box((g, edges))
            },
            criterion::BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn criterion_config() -> Criterion {
    let out = std::env::var_os("LARA_CRITERION_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/criterion-contract-v2")
        });
    Criterion::default().sample_size(10).output_directory(&out)
}

criterion_group!(
    name = benches;
    config = criterion_config();
    targets =
    bench_insert_vs_size,
    bench_rebalance_vs_degree,
    bench_tpromote_sweep,
    bench_segment_sweep,
    bench_scan,
    bench_ablation,
    bench_promote_threshold_sweep,
    bench_scan_threshold_sweep,
    bench_log_cap_sweep,
    bench_persist_sweep,
    bench_dgap_vs_log0,
    bench_petgraph_vs_lara,
    bench_ingest
);
criterion_main!(benches);
