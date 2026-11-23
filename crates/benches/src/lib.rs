//! Benchmark harness library: datasets, graph adapters, and workloads
//! shared by the criterion benches.
//!
//! Systems compared (the paper's Table-1 arms):
//! - [`StaticCsr`]: optimal read-only baseline;
//! - [`DgapAdapter`]: DGAP Rust port (PMA + per-segment edge log);
//! - [`LaraAdapter`] slab-only LARA (free-span store, no tree promotion);
//! - [`LaraAdapter`] with Tree-CSR promotion enabled;
//! - [`LabeledAdapter`]: labeled LARA.

pub mod adapters;
pub mod compare;
pub mod data;
pub mod probe;

pub use adapters::{
    CountingLaraAdapter, DgapAdapter, LabeledAdapter, LaraAdapter, LaraConfig, MmapLaraAdapter,
    StaticCsr,
};
pub use data::{
    Dataset, coo_grouped, coo_power_law, coo_power_law_stream, coo_uniform, coo_uniform_stream,
    load_snap,
};
