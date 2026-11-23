//! Retained requested-heap checkpoints (separate process, `--test` mode).
//!
//! NOT RSS or allocator usable bytes. Peak is not measured here: only the
//! retained requested heap after the graph/backend owners are alive, with the
//! fixture and independent model kept inside the same baseline so the delta is
//! not polluted. The fixture/model are dropped only after the baseline check.
//! The drop-baseline assertion is always active; there is no diagnostic bypass.

use benches::compare::{
    WORKLOADS, Workload, build_any, memory_cases, model_for, stream, stream_fingerprint, update_k,
    update_pairs,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct CountedSystem;
static LIVE: AtomicUsize = AtomicUsize::new(0);

// SAFETY: delegates to System with unchanged layouts/pointers; accounting never
// allocates, panics, or calls back into the allocator.
unsafe impl GlobalAlloc for CountedSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            System.dealloc(ptr, layout);
        }
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, size) };
        if !p.is_null() {
            if size >= layout.size() {
                LIVE.fetch_add(size - layout.size(), Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - size, Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: CountedSystem = CountedSystem;

fn workload(name: &str) -> Workload {
    *WORKLOADS
        .iter()
        .find(|w| w.name == name)
        .expect("known workload")
}

/// Counter self-check (Vec alloc/grow/shrink/zero/drop), reused from memory_probe.
fn allocator_self_check() {
    let before = LIVE.load(Relaxed);
    let mut v = black_box(Vec::<u64>::with_capacity(17));
    v.resize(17, 42);
    assert_eq!(LIVE.load(Relaxed) - before, v.capacity() * 8);
    v.reserve(100);
    black_box(&v);
    assert_eq!(LIVE.load(Relaxed) - before, v.capacity() * 8);
    v.shrink_to_fit();
    black_box(&v);
    assert_eq!(LIVE.load(Relaxed) - before, v.capacity() * 8);
    drop(v);
    let zeroed = black_box(vec![0u64; 19]);
    assert_eq!(LIVE.load(Relaxed) - before, zeroed.capacity() * 8);
    drop(zeroed);
    assert_eq!(LIVE.load(Relaxed), before);
    println!("ALLOCATOR_SELF_CHECK,ok");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.iter().any(|a| a == "--test"),
        "memory_compare only exposes --test for now"
    );
    allocator_self_check();
    let all = memory_cases();
    assert_eq!(all.len(), 40, "frozen memory checkpoint table");
    let (cases, expected) = match std::env::var("LARA_CMP_MEM_CASE") {
        Ok(w) if !w.is_empty() => {
            assert!(
                WORKLOADS.iter().any(|x| x.name == w),
                "unknown workload filter {w}"
            );
            (
                all.into_iter()
                    .filter(|c| c.workload == w)
                    .collect::<Vec<_>>(),
                8usize,
            )
        }
        _ => (all, 40usize),
    };
    assert_eq!(cases.len(), expected, "selected checkpoint count");
    println!(
        "MEMO_HEADER,workload,arm,state,m,req_nodes,act_nodes,fp,obj_bytes,pg_node_cap,pg_edge_cap,retained"
    );
    for case in &cases {
        let w = workload(case.workload);
        let s = stream(&w);
        let mut model = model_for(&s, w.n);
        let pairs = update_pairs(&s, update_k(w.m));
        let apply = case.state != "inserted";
        if apply {
            for &(src, dst) in &pairs {
                assert!(model.remove(src, dst), "{}: model delete", case.workload);
                model.insert(src, dst);
            }
        }
        let fp = stream_fingerprint(&s);
        let baseline = LIVE.load(Relaxed);
        let a = build_any(
            case.arm,
            &w,
            &s,
            if apply { &pairs } else { &[] },
            case.state == "post-SET",
        );
        if let Some((nc, ec)) = a.pg_capacity() {
            let heap = LIVE.load(Relaxed) as i64 - baseline as i64;
            let expected = (nc * std::mem::size_of::<petgraph::graph::Node<()>>()
                + ec * std::mem::size_of::<petgraph::graph::Edge<()>>())
                as i64;
            assert_eq!(
                heap, expected,
                "{}: petgraph allocations must match node/edge capacity",
                case.arm
            );
        }
        {
            // Verification temporary allocations are freed before the snapshot.
            a.check(
                &model,
                &format!("{}/{}/{}", case.workload, case.arm, case.state),
            );
        }
        let retained = LIVE.load(Relaxed) as i64 - baseline as i64;
        assert!(
            retained >= 0,
            "{}: retained must be nonnegative",
            case.workload
        );
        let act_nodes = a.num_nodes();
        let obj_bytes = a.object_bytes();
        let caps = a.pg_capacity();
        drop(a);
        assert_eq!(
            LIVE.load(Relaxed),
            baseline,
            "graph/backend drop must return to baseline"
        );
        // Strings and printing happen after the baseline check: stdout's own
        // buffer is a process-lifetime allocation and must not perturb the delta.
        let (pgc, pge) = caps
            .map(|(n, e)| (n.to_string(), e.to_string()))
            .unwrap_or_default();
        println!(
            "MEMO,{},{},{},m={},req_nodes={},act_nodes={},fp={fp},obj_bytes={obj_bytes},pg_node_cap={pgc},pg_edge_cap={pge},retained={retained}",
            case.workload, case.arm, case.state, w.m, w.n, act_nodes
        );
        drop(model);
        drop(s);
        drop(pairs);
    }
    println!("MEMO_OK,checkpoints={}", cases.len());
}
