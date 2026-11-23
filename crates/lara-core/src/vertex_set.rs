//! Sets of vertices that a scan asks about at every slot it reads.

use crate::id::VertexId;
use alloc::vec::Vec;

/// A set of vertices, built once and asked about cheaply: a bit per id when the ids are close
/// together, else a sorted list. [`crate::Graph::sweep`] asks it about every slot it reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexSet {
    sorted: Vec<VertexId>,
    /// One bit per id from the smallest member on, when that takes 64 bits per member or fewer.
    bits: Option<(u32, Vec<u64>)>,
}

impl VertexSet {
    /// The set of `vertices`.
    #[must_use]
    pub fn new(vertices: impl IntoIterator<Item = VertexId>) -> Self {
        let mut sorted: Vec<VertexId> = vertices.into_iter().collect();
        sorted.sort_unstable();
        sorted.dedup();
        let bits = match (sorted.first(), sorted.last()) {
            (Some(low), Some(high)) => {
                let span = u64::from(high.index() - low.index()) + 1;
                let words = usize::try_from(span.div_ceil(64)).unwrap_or(usize::MAX);
                (words <= sorted.len()).then(|| {
                    let mut words = alloc::vec![0u64; words];
                    for v in &sorted {
                        let at = v.index() - low.index();
                        if let Some(word) = words.get_mut(usize::try_from(at / 64).unwrap_or(0)) {
                            *word |= 1 << (at % 64);
                        }
                    }
                    (low.index(), words)
                })
            }
            _ => None,
        };
        Self { sorted, bits }
    }

    /// Whether `v` is in the set.
    #[must_use]
    pub fn contains(&self, v: VertexId) -> bool {
        match &self.bits {
            Some((low, words)) => v.index().checked_sub(*low).is_some_and(|at| {
                usize::try_from(at / 64)
                    .ok()
                    .and_then(|i| words.get(i))
                    .is_some_and(|word| word >> (at % 64) & 1 == 1)
            }),
            None => self.sorted.binary_search(&v).is_ok(),
        }
    }

    /// The members, in id order.
    pub fn iter(&self) -> impl Iterator<Item = VertexId> + '_ {
        self.sorted.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_and_sparse_sets_answer_alike() {
        let dense = VertexSet::new((100..300).step_by(3).map(VertexId::new));
        let sparse = VertexSet::new([5, 1_000_000, 70, 5].map(VertexId::new));
        assert!(dense.bits.is_some() && sparse.bits.is_none());
        for i in 0..400 {
            assert_eq!(
                dense.contains(VertexId::new(i)),
                (100..300).contains(&i) && (i - 100) % 3 == 0
            );
        }
        let members: Vec<u32> = sparse.iter().map(VertexId::index).collect();
        assert_eq!(members, [5, 70, 1_000_000]);
        assert!(sparse.contains(VertexId::new(70)) && !sparse.contains(VertexId::new(71)));
        assert!(!VertexSet::new([]).contains(VertexId::new(0)));
    }
}
