//! Dataset generation and loading: SNAP adjacency lists and synthetic
//! power-law COO graphs.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A graph dataset in grouped COO form (ascending source).
#[derive(Clone, Debug)]
pub struct Dataset {
    pub name: String,
    pub num_vertices: u32,
    pub edges: Vec<(u32, u32)>,
}

/// Loads a SNAP adjacency-list file (`src dst` per line, `#` comments).
/// Returns the grouped COO and the vertex count (max id + 1).
pub fn load_snap(path: &Path) -> std::io::Result<Dataset> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut edges = Vec::new();
    let mut max_id = 0u32;
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(u), Some(v)) = (it.next(), it.next()) else {
            continue;
        };
        let u: u32 = u.parse().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad vertex id: {e}"),
            )
        })?;
        let v: u32 = v.parse().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad vertex id: {e}"),
            )
        })?;
        max_id = max_id.max(u).max(v);
        edges.push((u, v));
    }
    edges.sort_by_key(|&(u, _)| u);
    Ok(Dataset {
        name: path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("snap")
            .to_string(),
        num_vertices: max_id + 1,
        edges,
    })
}

/// Groups raw COO pairs by ascending source (dataset canonicalization for
/// bulk build; both DGAP and LARA builders require grouped input).
pub fn coo_grouped(mut edges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    edges.sort_by_key(|&(u, _)| u);
    edges
}

/// Uniform-random edge stream in **arrival order** (unsorted). Dynamic
/// insert benchmarks consume this order; bulk fixtures group it explicitly.
pub fn coo_uniform_stream(n: u32, m: u64, seed: u64) -> Vec<(u32, u32)> {
    assert!(n > 0 || m == 0, "nonempty stream needs vertices");
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let n = n as u64;
    let mut edges = Vec::with_capacity(m as usize);
    for _ in 0..m {
        let u = (next() % n) as u32;
        let v = (next() % n) as u32;
        edges.push((u, v));
    }
    edges
}

/// Discrete rank-Zipf sources in **arrival order**:
/// `P(src=s) = (s+1)^(-beta) / sum(r^(-beta), r=1..n)`.
/// Destinations are uniform; self-loops and parallel edges are allowed.
/// `beta` must be finite and nonnegative (zero gives uniform source ranks).
///
/// Source IDs are ranks: hot vertices are clustered at low IDs. This is
/// not preferential attachment, nor a degree-frequency exponent claim.
/// The f64 CDF costs O(n) space/setup and O(log n) per sample. Sampling uses
/// seeded xorshift; grouped and arrival forms contain the same stream.
pub fn coo_power_law_stream(n: u32, m: u64, seed: u64, beta: f64) -> Vec<(u32, u32)> {
    assert!(n > 0 || m == 0, "nonempty stream needs vertices");
    assert!(
        beta.is_finite() && beta >= 0.0,
        "beta must be finite and nonnegative"
    );
    if m == 0 {
        return Vec::new();
    }
    let mut total = 0.0;
    let cdf: Vec<f64> = (1..=n)
        .map(|rank| {
            total += f64::from(rank).powf(-beta);
            total
        })
        .collect();
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut edges = Vec::with_capacity(m as usize);
    for _ in 0..m {
        let u = (next() >> 11) as f64 / (1u64 << 53) as f64;
        let pick = u * total;
        // Clamp only protects the upper endpoint against f64 rounding.
        let src = cdf
            .partition_point(|&mass| mass <= pick)
            .min(n as usize - 1);
        let dst = (next() % u64::from(n)) as u32;
        edges.push((src as u32, dst));
    }
    edges
}

/// The same rank-Zipf arrival stream as [`coo_power_law_stream`], grouped
/// by ascending source for bulk construction.
pub fn coo_power_law(n: u32, m: u64, seed: u64, beta: f64) -> Vec<(u32, u32)> {
    coo_grouped(coo_power_law_stream(n, m, seed, beta))
}

/// Uniform-random edge stream (ergodic baseline workload).
pub fn coo_uniform(n: u32, m: u64, seed: u64) -> Vec<(u32, u32)> {
    coo_grouped(coo_uniform_stream(n, m, seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_law_shape_is_heavy_headed() {
        let edges = coo_power_law(10_000, 20_000, 42, 1.0);
        // Verify the head: the top 1% of sources should own a large share.
        let mut counts = vec![0u64; 10_000];
        for &(u, _) in &edges {
            counts[u as usize] += 1;
        }
        counts.sort_unstable_by(|a, b| b.cmp(a));
        let top1pct: u64 = counts[..100].iter().sum();
        assert!(
            top1pct as f64 > edges.len() as f64 * 0.10,
            "top 1% owns only {} of {}",
            top1pct,
            edges.len()
        );
    }

    #[test]
    fn power_law_spreads_at_both_audited_sizes() {
        for n in [10_000u32, 100_000] {
            for seed in [42, 99, 1234] {
                let m = u64::from(n) * 4;
                let edges = coo_power_law_stream(n, m, seed, 1.0);
                let mut counts = vec![0u64; n as usize];
                for &(src, dst) in &edges {
                    assert!(src < n && dst < n);
                    counts[src as usize] += 1;
                }
                let distinct = counts.iter().filter(|&&x| x > 0).count();
                let max_share = *counts.iter().max().unwrap() as f64 / m as f64;
                counts.sort_unstable_by(|a, b| b.cmp(a));
                let head_share = counts[..n as usize / 100].iter().sum::<u64>() as f64 / m as f64;
                assert!(
                    distinct > n as usize / 5,
                    "n={n}, seed={seed}: only {distinct} sources"
                );
                assert!(
                    (0.05..0.15).contains(&max_share),
                    "max source share={max_share}"
                );
                assert!(
                    (0.30..0.65).contains(&head_share),
                    "top 1% share={head_share}"
                );
            }
        }
    }

    #[test]
    fn power_law_matches_discrete_rank_probabilities() {
        let n = 8u32;
        let m = 80_000u64;
        for beta in [0.0, 0.5, 1.0, 2.0] {
            let edges = coo_power_law_stream(n, m, 42, beta);
            let mut counts = [0u64; 8];
            for (src, _) in edges {
                counts[src as usize] += 1;
            }
            let normalizer: f64 = (1..=n).map(|r| f64::from(r).powf(-beta)).sum();
            for (i, &count) in counts.iter().enumerate() {
                let p = ((i + 1) as f64).powf(-beta) / normalizer;
                let expected = m as f64 * p;
                let tolerance = 8.0 * (m as f64 * p * (1.0 - p)).sqrt();
                assert!(
                    (count as f64 - expected).abs() < tolerance,
                    "beta={beta}, rank={}: observed={count}, expected={expected}",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn grouped_power_law_is_the_same_arrival_stream() {
        let stream = coo_power_law_stream(100, 1000, 42, 1.0);
        assert_eq!(stream, coo_power_law_stream(100, 1000, 42, 1.0));
        assert_ne!(stream, coo_power_law_stream(100, 1000, 314, 1.0));
        assert!(stream.windows(2).any(|w| w[0].0 > w[1].0));
        assert_eq!(coo_grouped(stream), coo_power_law(100, 1000, 42, 1.0));
    }

    #[test]
    fn generator_empty_singleton_and_invalid_inputs() {
        assert!(coo_power_law_stream(0, 0, 7, 1.0).is_empty());
        assert!(coo_power_law(10, 0, 7, 1.0).is_empty());
        assert!(coo_uniform_stream(0, 0, 7).is_empty());
        assert_eq!(coo_power_law_stream(1, 10, 7, 1.0), vec![(0, 0); 10]);
        assert!(std::panic::catch_unwind(|| coo_power_law_stream(0, 1, 7, 1.0)).is_err());
        for beta in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| coo_power_law_stream(10, 10, 7, beta)).is_err(),
                "beta={beta}"
            );
        }
    }

    #[test]
    fn uniform_is_balanced() {
        let edges = coo_uniform(1_000, 10_000, 7);
        let mut counts = vec![0u64; 1_000];
        for &(u, _) in &edges {
            counts[u as usize] += 1;
        }
        let max = *counts.iter().max().unwrap();
        assert!(max < 100, "uniform max out-degree {max} too skewed");
    }
}
