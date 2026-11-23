//! Sizes that tests shrink, so that lists tier and index trees grow deep after a few entries.
//!
//! Tests build shrunk, unless compiled with `--cfg lara_full_size`: measurements use it to run at
//! the sizes the crate ships with, as in
//! `RUSTFLAGS="--cfg lara_full_size" cargo test -p lara-graph --release -- --ignored --nocapture`.

/// How big the shrinkable sizes are in this build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scale {
    /// Test sizes.
    Shrunk,
    /// The sizes the crate ships with.
    Full,
}

impl Scale {
    /// This build's scale.
    pub(crate) const BUILD: Self = if cfg!(all(test, not(lara_full_size))) {
        Self::Shrunk
    } else {
        Self::Full
    };

    /// `shrunk` in a shrunk build, else `full`.
    pub(crate) const fn pick<T: Copy>(shrunk: T, full: T) -> T {
        match Self::BUILD {
            Self::Shrunk => shrunk,
            Self::Full => full,
        }
    }
}
