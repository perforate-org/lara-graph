//! Sets of deleting nodes, checked once when built, that a sweep asks about at every slot it
//! reads.

use crate::id::NodeId;
use alloc::vec::Vec;

/// Nodes known not to be live, built by [`crate::Graph::deleting_set`] and asked about cheaply:
/// a bit per id when the ids are close together, else a sorted list. [`crate::Graph::sweep`]
/// asks it about every slot it reads.
///
/// A node's state only moves from live to deleting to deleted, so the check made when the set
/// was built holds for as long as the set lives, and a sweep need not repeat it. Repeating it on
/// every call made a sweep's cost grow with the set rather than with its budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeletingSet {
    sorted: Vec<NodeId>,
    /// One bit per id from the smallest member on, when that takes 64 bits per member or fewer.
    bits: Option<(u32, Vec<u64>)>,
}

impl DeletingSet {
    /// The set of `nodes`, which the caller has checked are not live.
    pub(crate) fn new(nodes: impl IntoIterator<Item = NodeId>) -> Self {
        let mut sorted: Vec<NodeId> = nodes.into_iter().collect();
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
    pub fn contains(&self, v: NodeId) -> bool {
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
    pub fn iter(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.sorted.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_and_sparse_sets_answer_alike() {
        let dense = DeletingSet::new((100..300).step_by(3).map(NodeId::new));
        let sparse = DeletingSet::new([5, 1_000_000, 70, 5].map(NodeId::new));
        assert!(dense.bits.is_some() && sparse.bits.is_none());
        for i in 0..400 {
            assert_eq!(
                dense.contains(NodeId::new(i)),
                (100..300).contains(&i) && (i - 100) % 3 == 0
            );
        }
        let members: Vec<u32> = sparse.iter().map(NodeId::index).collect();
        assert_eq!(members, [5, 70, 1_000_000]);
        assert!(sparse.contains(NodeId::new(70)) && !sparse.contains(NodeId::new(71)));
        assert!(!DeletingSet::new([]).contains(NodeId::new(0)));
    }
}
