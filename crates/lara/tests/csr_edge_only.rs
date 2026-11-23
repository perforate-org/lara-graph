//! Compile-time capability tests, intentionally separate from old-source runtime controls.
#![cfg(feature = "std")]
use lara_graph::graph::LaraMemories;
use lara_graph::memory::{GrowFailed, Memory, PAGE_SIZE, VecMemory};
use lara_graph::{
    CsrEdge, CsrEdgeTombstone, DynamicCsr, FixedWidth, IngestBuffer, LaraGraph, RemovableCsr,
    Vertex,
};
use std::cell::Cell;
use std::rc::Rc;

#[derive(Clone, Copy, Default)]
struct Bare<const N: usize>(u32);
impl<const N: usize> FixedWidth for Bare<N> {
    const BYTES: usize = N;
    fn read_from(b: &[u8]) -> Self {
        assert_eq!(b.len(), N);
        assert!(b[4..].iter().all(|&x| x == 0x5a));
        Self(u32::from_le_bytes(b[..4].try_into().unwrap()) ^ 0x8000_0000)
    }
    fn write_to(&self, b: &mut [u8]) {
        b.fill(0x5a);
        b[..4].copy_from_slice(&(self.0 ^ 0x8000_0000).to_le_bytes());
    }
}
impl<const N: usize> CsrEdge for Bare<N> {
    fn neighbor(&self) -> u32 {
        self.0
    }
    fn with_neighbor(self, n: u32) -> Self {
        Self(n)
    }
}
#[derive(Clone, Copy, Default)]
struct Marked<const N: usize>(Bare<N>);
impl<const N: usize> FixedWidth for Marked<N> {
    const BYTES: usize = N;
    fn read_from(b: &[u8]) -> Self {
        Self(Bare::read_from(b))
    }
    fn write_to(&self, b: &mut [u8]) {
        self.0.write_to(b)
    }
}
impl<const N: usize> CsrEdge for Marked<N> {
    fn neighbor(&self) -> u32 {
        self.0.neighbor()
    }
    fn with_neighbor(self, n: u32) -> Self {
        Self(Bare(n))
    }
}
impl<const N: usize> CsrEdgeTombstone for Marked<N> {
    fn tombstone_edge() -> Self {
        Self(Bare(7)) // also present as a live value in the default graph
    }
    fn is_tombstone(&self) -> bool {
        self.neighbor() == 7
    }
}
#[derive(Clone, Default)]
struct Backing {
    mem: VecMemory,
    ops: Rc<Cell<[u64; 6]>>,
}
impl Memory for Backing {
    fn read(&self, o: u64, b: &mut [u8]) {
        assert!(o + b.len() as u64 <= self.size() * PAGE_SIZE);
        let mut s = self.ops.get();
        s[0] += 1;
        s[2] += b.len() as u64;
        self.ops.set(s);
        self.mem.read(o, b);
    }
    fn read_mapped(&self, o: u64, b: &mut [u8]) {
        assert!(o + b.len() as u64 <= self.size() * PAGE_SIZE);
        let mut s = self.ops.get();
        s[1] += 1;
        s[2] += b.len() as u64;
        self.ops.set(s);
        self.mem.read_mapped(o, b);
    }
    fn write(&self, o: u64, b: &[u8]) {
        assert!(o + b.len() as u64 <= self.size() * PAGE_SIZE);
        let mut s = self.ops.get();
        s[3] += 1;
        s[4] += b.len() as u64;
        self.ops.set(s);
        self.mem.write(o, b);
    }
    fn grow(&self, n: u64) -> Result<u64, GrowFailed> {
        let mut s = self.ops.get();
        s[5] += 1;
        self.ops.set(s);
        self.mem.grow(n)
    }
    fn size(&self) -> u64 {
        self.mem.size()
    }
    fn fence(&self) {
        self.mem.fence();
    }
}
fn generic_remove<G: RemovableCsr>(g: &mut G, src: u32, dst: u32) -> bool {
    g.remove_edge(src, dst)
}
fn generic_insert<G: DynamicCsr>(g: &mut G, src: u32, dst: u32) {
    g.insert_edge(src, dst);
}
fn exercise<E: CsrEdge>() -> Vec<([u64; 6], Vec<u8>)> {
    let r: [Backing; 8] = std::array::from_fn(|_| Backing::default());
    let mem = LaraMemories {
        vertices: r[0].clone(),
        edges: r[1].clone(),
        logs: r[2].clone(),
        counts: r[3].clone(),
        spans: r[4].clone(),
        free_spans: r[5].clone(),
        blocks: r[6].clone(),
        buckets: r[7].clone(),
    };
    let mut g = LaraGraph::<Vertex, E, _>::new_with_segments(mem, 2048, 8).unwrap();
    g.set_promote_threshold(Some(64));
    for x in &r {
        x.ops.set([0; 6]);
    }
    // Public operations being observed; no semantic or storage oracle runs here.
    g.insert_edge(32, 33);
    g.insert_edge(32, 34);
    let tiny_removed = generic_remove(&mut g, 32, 33);
    g.rebalance_leaf(4);
    g.insert_edge(48, 49);
    let pairs: Vec<_> = [1, 2, 2]
        .into_iter()
        .map(|d| (0, d))
        .chain((1..130).map(|d| (8, d)))
        .collect();
    g.build_pairs(&pairs).unwrap();
    generic_insert(&mut g, 8, 130);
    g.insert_sorted_run(8, &[131, 7, 7]);
    let mut buf = IngestBuffer::new(4, 4);
    for d in [132, 8, 133, 8] {
        buf.push(8, d);
    }
    let invisible = g.num_edges();
    buf.flush(&mut g);
    let mut removed = Vec::new();
    for d in [1, 7, 9999, 64, 130] {
        removed.push(generic_remove(&mut g, 8, d));
    }
    let mut want: Vec<_> = (1..=133).chain([7, 7, 8, 8]).collect();
    for d in [1, 7, 64, 130] {
        let p = want.iter().position(|&x| x == d).unwrap();
        want.remove(p);
    }
    let erase = want[..want.len() - 32].to_vec();
    let mut shrunk = Vec::new();
    for d in &erase {
        shrunk.push(g.remove_edge(8, *d));
    }
    want.drain(..erase.len());
    for d in 700..800 {
        g.insert_edge(8, d);
        want.push(d);
    }
    let snapshot: Vec<_> = r
        .iter()
        .map(|r| {
            let s = r.ops.get();
            let mut b = vec![0; (r.size() * PAGE_SIZE) as usize];
            r.mem.read(0, &mut b);
            (s, b)
        })
        .collect();
    // All observations above are frozen before these oracles.
    assert!(tiny_removed);
    assert_eq!(invisible, 136);
    assert!(buf.is_empty());
    assert_eq!(removed, [true, true, false, true, true]);
    assert!(shrunk.iter().all(|x| *x));
    let mut got = g.out_neighbors_vec(8);
    got.sort_unstable();
    want.sort_unstable();
    assert_eq!(got, want);
    assert_eq!(g.out_neighbors_vec(0), [1, 2, 2]);
    assert!(g.out_neighbors_vec(32).is_empty());
    assert!(g.out_neighbors_vec(48).is_empty());
    assert_eq!(g.num_edges(), want.len() as u64 + 3);
    assert_eq!(g.out_degree(8), want.len() as u64);
    g.verify_invariants();
    assert_eq!(g.tree_bucket_count(), usize::from(E::BYTES == 4));
    for (i, (ops, _)) in snapshot.iter().enumerate() {
        println!("CAP,width{},region{i},{ops:?}", E::BYTES);
    }
    snapshot
}
#[test]
fn edge_without_marker_supports_the_public_graph_lifecycle() {
    exercise::<Bare<4>>();
    exercise::<Bare<12>>();
}
#[test]
fn marker_capability_does_not_select_a_different_default() {
    assert_eq!(exercise::<Bare<4>>(), exercise::<Marked<4>>());
    assert_eq!(exercise::<Bare<12>>(), exercise::<Marked<12>>());
}
