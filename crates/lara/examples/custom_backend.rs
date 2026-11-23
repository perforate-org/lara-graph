//! Custom [`Memory`] backend: a region-split wrapper over two `Vec<u8>`s.
//!
//! A backend must obey the [`Memory`] contract, not just implement its
//! methods. This in-memory example exposes one contiguous address space
//! split into "hot" and "cold" buffers. It is not durable tiered storage.
//!
//! Run with: `cargo run --example custom_backend -p lara-graph`
//! Test with: `cargo test --example custom_backend -p lara-graph`

use std::cell::RefCell;
use std::rc::Rc;

use lara_graph::graph::{LaraGraph, LaraMemories};
use lara_graph::memory::{GrowFailed, Memory, PAGE_SIZE};

/// Two shared buffers: `[0, HOT_BYTES)` in `hot`, later bytes in `cold`.
/// Requests crossing the boundary use both buffers. Nonempty writes
/// auto-grow in pages; empty writes do nothing. Reads beyond backing
/// zero-fill. Growth failure preserves mapped lengths and existing bytes,
/// but a successful reserve in one Vec may retain extra capacity.
#[derive(Clone, Default)]
struct TieredMemory {
    hot: Rc<RefCell<Vec<u8>>>,
    cold: Rc<RefCell<Vec<u8>>>,
}

const HOT_BYTES: usize = 1 << 20; // 1 MiB

impl Memory for TieredMemory {
    fn read(&self, offset: u64, buf: &mut [u8]) {
        let split = (HOT_BYTES as u64)
            .saturating_sub(offset)
            .min(buf.len() as u64) as usize;
        let (hot, cold) = buf.split_at_mut(split);
        for (region, local, out) in [
            (&self.hot, offset, hot),
            (&self.cold, offset.saturating_sub(HOT_BYTES as u64), cold),
        ] {
            if out.is_empty() {
                continue;
            }
            let data = region.borrow();
            let start = usize::try_from(local).unwrap_or(usize::MAX).min(data.len());
            let n = out.len().min(data.len() - start);
            out[..n].copy_from_slice(&data[start..start + n]);
            out[n..].fill(0);
        }
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        if buf.is_empty() {
            return;
        }
        let start = usize::try_from(offset).expect("write offset exceeds host address space");
        let end = start.checked_add(buf.len()).expect("write range overflow");
        let needed_pages = (end as u64).div_ceil(PAGE_SIZE);
        let current_pages = self.size();
        if needed_pages > current_pages {
            self.grow(needed_pages - current_pages)
                .expect("tiered memory grow");
        }
        let split = buf.len().min(HOT_BYTES.saturating_sub(start));
        let (hot, cold) = buf.split_at(split);
        for (region, local, data) in [
            (&self.hot, start, hot),
            (&self.cold, start.saturating_sub(HOT_BYTES), cold),
        ] {
            if !data.is_empty() {
                region.borrow_mut()[local..local + data.len()].copy_from_slice(data);
            }
        }
    }

    fn size(&self) -> u64 {
        let hot = self.hot.borrow().len();
        let cold = self.cold.borrow().len();
        (hot + cold).div_ceil(PAGE_SIZE as usize) as u64
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        let current = self.size();
        let failed = GrowFailed {
            current_pages: current,
            delta_pages: pages,
        };
        let new_pages = current.checked_add(pages).ok_or(failed)?;
        let target = new_pages
            .checked_mul(PAGE_SIZE)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(failed)?;
        let hot_len = target.min(HOT_BYTES);
        let cold_len = target - hot_len;
        let mut hot = self.hot.borrow_mut();
        let mut cold = self.cold.borrow_mut();
        let hot_extra = hot_len - hot.len();
        let cold_extra = cold_len - cold.len();
        // Reserve both sides before publishing either new mapped length.
        hot.try_reserve(hot_extra).map_err(|_| failed)?;
        cold.try_reserve(cold_extra).map_err(|_| failed)?;
        hot.resize(hot_len, 0);
        cold.resize(cold_len, 0);
        Ok(new_pages)
    }
}

fn fresh_memories() -> LaraMemories<TieredMemory> {
    LaraMemories {
        vertices: TieredMemory::default(),
        edges: TieredMemory::default(),
        logs: TieredMemory::default(),
        counts: TieredMemory::default(),
        spans: TieredMemory::default(),
        free_spans: TieredMemory::default(),
        blocks: TieredMemory::default(),
        buckets: TieredMemory::default(),
    }
}

fn main() {
    let mem = fresh_memories();
    let mut g: LaraGraph<lara_graph::graph::Vertex, lara_graph::graph::Edge, TieredMemory> =
        LaraGraph::new(mem, 3).unwrap();
    g.insert_edge(0, 1);
    g.insert_edge(0, 2);
    g.insert_edge(1, 2);

    println!("neighbors of 0: {:?}", g.out_neighbors_vec(0));
    println!("neighbors of 1: {:?}", g.out_neighbors_vec(1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use lara_graph::graph::{Edge, Vertex};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    const HOT_PAGES: u64 = HOT_BYTES as u64 / PAGE_SIZE;

    #[test]
    fn grow_is_additive() {
        let m = TieredMemory::default();
        let mut pages = 0;
        for delta in [0, 1, 1, HOT_PAGES, 0] {
            pages += delta;
            assert_eq!(m.grow(delta), Ok(pages));
            assert_eq!(m.size(), pages);
        }
        let mut bytes = vec![0xff; (pages * PAGE_SIZE) as usize];
        m.read(0, &mut bytes);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn mapped_hot_writes_do_not_change_size() {
        let m = TieredMemory::default();
        m.grow(1).unwrap();
        m.write(PAGE_SIZE - 1, b"x");
        assert_eq!(m.size(), 1, "writing mapped hot bytes grew the memory");
        assert_eq!(m.grow(HOT_PAGES), Ok(HOT_PAGES + 1));
        assert_eq!(m.hot.borrow().len(), HOT_BYTES);
        assert_eq!(m.cold.borrow().len(), PAGE_SIZE as usize);
        let mut bytes = [0xff; 3];
        m.read(PAGE_SIZE - 1, &mut bytes);
        assert_eq!(bytes, [b'x', 0, 0]);
    }

    #[test]
    fn crossing_reads_join_hot_and_cold() {
        let m = TieredMemory::default();
        // Populate each side separately so a broken write cannot mask the read bug.
        m.write(HOT_BYTES as u64 - 2, b"ab");
        m.write(HOT_BYTES as u64, b"cd");
        let mut bytes = [0xff; 4];
        m.read(HOT_BYTES as u64 - 2, &mut bytes);
        assert_eq!(&bytes, b"abcd");
        bytes.fill(0xff);
        m.read_mapped(HOT_BYTES as u64 - 2, &mut bytes);
        assert_eq!(&bytes, b"abcd");
    }

    #[test]
    fn crossing_writes_preserve_both_sides() {
        let m = TieredMemory::default();
        m.grow(HOT_PAGES + 1).unwrap();
        m.write(HOT_BYTES as u64 - 3, b"A");
        m.write(HOT_BYTES as u64 + 2, b"Z");
        m.write(HOT_BYTES as u64 - 2, b"bcde");
        // Read each side separately to isolate the write bug.
        let mut bytes = [0xff; 3];
        m.read(HOT_BYTES as u64 - 3, &mut bytes);
        assert_eq!(&bytes, b"Abc");
        m.read(HOT_BYTES as u64, &mut bytes);
        assert_eq!(&bytes, b"deZ");
        assert_eq!(m.size(), HOT_PAGES + 1);
    }

    #[test]
    fn cold_first_write_maps_hot_prefix_and_shares_handles() {
        let m = TieredMemory::default();
        let other = m.clone();
        other.write(HOT_BYTES as u64 + 3, b"cd");
        assert_eq!(m.size(), HOT_PAGES + 1);
        assert_eq!(m.hot.borrow().len(), HOT_BYTES);
        assert_eq!(m.cold.borrow().len(), PAGE_SIZE as usize);
        let mut bytes = [0xff; 4];
        m.read(HOT_BYTES as u64 + 1, &mut bytes);
        assert_eq!(bytes, [0, 0, b'c', b'd']);
        m.write(0, b"a");
        other.read(0, &mut bytes);
        assert_eq!(bytes, [b'a', 0, 0, 0]);
    }

    #[test]
    fn empty_and_out_of_range_accesses() {
        let m = TieredMemory::default();
        for offset in [0, HOT_BYTES as u64 - 1, HOT_BYTES as u64, u64::MAX] {
            let mut bytes = [0xff; 4];
            m.read(offset, &mut bytes);
            assert_eq!(bytes, [0; 4]);
            m.read(offset, &mut []);
            m.write(offset, &[]);
            assert_eq!(m.size(), 0);
        }
        m.write(PAGE_SIZE - 1, b"x");
        let mut bytes = [0xff; 3];
        m.read(PAGE_SIZE - 1, &mut bytes);
        assert_eq!(bytes, [b'x', 0, 0]);
        assert!(catch_unwind(AssertUnwindSafe(|| m.write(u64::MAX, b"xx"))).is_err());
        assert_eq!(m.size(), 1);
        m.read(PAGE_SIZE - 1, &mut bytes);
        assert_eq!(bytes, [b'x', 0, 0]);
    }

    #[test]
    fn failed_growth_keeps_visible_state() {
        let m = TieredMemory::default();
        m.write(0, b"abc");
        // The cold Vec would exceed isize::MAX: deterministic reserve failure,
        // not a request to exhaust physical RAM. A hot reserve may already succeed.
        let too_many_pages = (isize::MAX as u64 + 1 + HOT_BYTES as u64) / PAGE_SIZE;
        for delta in [u64::MAX, too_many_pages - m.size()] {
            let pages = m.size();
            assert_eq!(
                m.grow(delta),
                Err(GrowFailed {
                    current_pages: pages,
                    delta_pages: delta
                })
            );
            assert_eq!(m.size(), pages);
            assert_eq!(m.hot.borrow().len(), PAGE_SIZE as usize);
            assert_eq!(m.cold.borrow().len(), 0);
            let mut bytes = [0xff; 4];
            m.read(0, &mut bytes);
            assert_eq!(bytes, [b'a', b'b', b'c', 0]);
        }
    }

    #[test]
    fn history_matches_flat_bytes() {
        let m = TieredMemory::default();
        let alias = m.clone();
        let mut oracle = Vec::new();
        let starts = [
            0,
            PAGE_SIZE as usize - 3,
            HOT_BYTES - 5,
            HOT_BYTES - 1,
            HOT_BYTES,
            HOT_BYTES + 1,
            2 * HOT_BYTES + 9,
        ];
        for step in 0..256 {
            if step % 17 == 0 {
                oracle.resize(oracle.len() + 2 * PAGE_SIZE as usize, 0);
                assert_eq!(
                    m.grow(2).unwrap(),
                    (oracle.len() / PAGE_SIZE as usize) as u64
                );
            }
            let start = starts[step % starts.len()] + step % 5;
            let data: Vec<u8> = (0..step * 29 % 263)
                .map(|i| (step + i * 37) as u8)
                .collect();
            if !data.is_empty() {
                let end = start + data.len();
                let len = oracle
                    .len()
                    .max(end.div_ceil(PAGE_SIZE as usize) * PAGE_SIZE as usize);
                oracle.resize(len, 0);
                oracle[start..end].copy_from_slice(&data);
            }
            alias.write(start as u64, &data);
            assert_eq!(m.size(), (oracle.len() / PAGE_SIZE as usize) as u64);
            let from = start.saturating_sub(3);
            let mut got = vec![0xff; data.len() + 6];
            m.read(from as u64, &mut got);
            for (i, byte) in got.iter().enumerate() {
                assert_eq!(
                    *byte,
                    oracle.get(from + i).copied().unwrap_or(0),
                    "step {step}, byte {i}"
                );
            }
        }
        let mut got = vec![0xff; oracle.len() + 3];
        m.read(0, &mut got);
        oracle.resize(got.len(), 0);
        assert_eq!(got.iter().zip(&oracle).position(|(a, b)| a != b), None);
    }

    #[test]
    fn graph_slab_and_tree_reach_cold_storage() {
        let n = 64;
        let mut rng = 0x9e37_79b9u32;
        let mut edges = vec![(0, 1)];
        for _ in 0..270_000 {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            edges.push((1, rng % n));
        }
        let mut want: Vec<_> = edges[1..].iter().map(|&(_, dst)| dst).collect();
        want.sort_unstable();
        for tree in [false, true] {
            let mem = fresh_memories();
            let backing = if tree {
                mem.blocks.clone()
            } else {
                mem.edges.clone()
            };
            let mut g: LaraGraph<Vertex, Edge, TieredMemory> = LaraGraph::new(mem, n).unwrap();
            if tree {
                for &(src, dst) in &edges {
                    g.insert_edge(src, dst);
                }
            } else {
                g.set_promote_threshold(None);
                g.build_pairs(&edges).unwrap();
            }
            assert!(
                !backing.cold.borrow().is_empty(),
                "fixture did not use cold storage"
            );
            assert_eq!(g.tree_bucket_count(), usize::from(tree));
            assert_eq!(g.num_edges(), edges.len() as u64);
            g.verify_invariants();
            assert_eq!(g.out_neighbors_vec(0), [1]);
            let mut got = g.out_neighbors_vec(1);
            got.sort_unstable();
            assert_eq!(got.len(), want.len());
            assert_eq!(
                got.iter().zip(&want).position(|(a, b)| a != b),
                None,
                "tree={tree}"
            );
        }
    }
}
