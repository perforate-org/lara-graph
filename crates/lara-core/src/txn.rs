//! Scoped allocation for multi-step updates.
//!
//! An update that allocates several extents and moves data must either finish or leave the
//! allocator as it was. [`SpanTxn`] makes the two easy mistakes impossible:
//!
//! - **Leaking a new extent on failure.** Extents allocated through the transaction go back to
//!   the allocator when it is dropped without [`SpanTxn::commit`].
//!
//! An update that moves data out of an old extent releases it itself, after copying: nothing
//! after the allocations can fail, so the old extent never needs to outlive the transaction.

use alloc::vec::Vec;

/// A source of extents. Releasing must not fail: it runs in `Drop`.
pub(crate) trait Allocator {
    type Request;
    type Extent: Copy;
    type Error;

    fn allocate(&mut self, request: Self::Request) -> Result<Self::Extent, Self::Error>;

    fn release(&mut self, extent: Self::Extent);
}

/// An update in progress. Commit it, or every extent it allocated is released.
#[must_use = "dropping a transaction rolls back its allocations"]
pub(crate) struct SpanTxn<'a, A: Allocator> {
    allocator: &'a mut A,
    acquired: Vec<A::Extent>,
}

impl<'a, A: Allocator> SpanTxn<'a, A> {
    pub(crate) fn begin(allocator: &'a mut A) -> Self {
        Self {
            allocator,
            acquired: Vec::new(),
        }
    }

    pub(crate) fn allocate(&mut self, request: A::Request) -> Result<A::Extent, A::Error> {
        let extent = self.allocator.allocate(request)?;
        self.acquired.push(extent);
        Ok(extent)
    }

    /// Keeps every allocation.
    pub(crate) fn commit(mut self) {
        self.acquired.clear();
    }
}

impl<A: Allocator> Drop for SpanTxn<'_, A> {
    fn drop(&mut self) {
        // Rollback: the new extents go back.
        for extent in self.acquired.drain(..).rev() {
            self.allocator.release(extent);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bump allocator over integers that records releases.
    #[derive(Default)]
    struct Bump {
        next: u32,
        released: Vec<u32>,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Full;

    impl Allocator for Bump {
        type Request = ();
        type Extent = u32;
        type Error = Full;

        fn allocate(&mut self, (): ()) -> Result<u32, Full> {
            if self.next >= 3 {
                return Err(Full);
            }
            self.next += 1;
            Ok(self.next - 1)
        }

        fn release(&mut self, extent: u32) {
            self.released.push(extent);
        }
    }

    #[test]
    fn commit_keeps_new_extents() {
        let mut bump = Bump::default();
        let mut txn = SpanTxn::begin(&mut bump);
        assert_eq!(txn.allocate(()), Ok(0));
        txn.commit();
        assert_eq!(bump.released, []);
    }

    #[test]
    fn drop_without_commit_rolls_back_allocations() {
        let mut bump = Bump::default();
        {
            let mut txn = SpanTxn::begin(&mut bump);
            txn.allocate(()).unwrap();
            txn.allocate(()).unwrap();
        }
        assert_eq!(bump.released, [1, 0]);
    }

    #[test]
    fn failed_allocation_rolls_back_earlier_ones() {
        let mut bump = Bump {
            next: 2,
            ..Bump::default()
        };
        let result = (|| {
            let mut txn = SpanTxn::begin(&mut bump);
            txn.allocate(())?;
            txn.allocate(())?;
            txn.commit();
            Ok::<(), Full>(())
        })();
        assert_eq!(result, Err(Full));
        assert_eq!(bump.released, [2]);
    }
}
