# dgap — faithful Rust port of DGAP

This crate is a **faithful Rust port** of the DGAP (Dynamic Graph Adjacency
Array with Packed Memory Array) reference implementation, used as the
baseline in the LARA comparison.

- **Original paper:** *DGAP: Efficient Dynamic Graph Adjacency Array
  Processing* — arXiv:2403.02665.
- **Reference implementation:** `graph.h` from the DGAP repository
  (DIR Lab, MIT license).
- **License:** MIT (matching the upstream reference).

The port keeps the reference algorithm intact (no optimization) so that
benchmark results are directly comparable to the C++ baseline. It is
published as a baseline artifact for this repository only and is **not**
published to crates.io (`publish = false`).

## Attribution

DGAP is developed by the DIR Lab. This Rust port is derived from the
MIT-licensed reference implementation and is provided for benchmarking and
verification purposes. See the upstream repository and paper for the
original design and authorship.
