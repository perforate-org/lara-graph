// Sample of the paper parts; build with `just paper`. Every number, table, plot and listing
// below is read from the repository at compile time.

#import "lib/bench.typ": num, ratio, change, bench-table, bench-bars
#import "lib/code.typ": item, region, lines

#set page(paper: "a4", margin: 2cm)
#set text(size: 10pt)
#show raw.where(block: true): set text(size: 8pt)
#show raw.where(block: true): block.with(fill: luma(246), inset: 6pt, radius: 2pt, width: 100%)
#show raw.where(block: true): set align(left)

= Evaluation (sample)

Visiting the edges of a 65,536-edge node costs #num("lara_stable_visit_65536") instructions on
stable memory against #num("btree_visit_65536_stable") for a `StableBTreeMap`
(#ratio("btree_visit_65536_stable", "lara_stable_visit_65536") fewer). Batching node insertion
changes the cost by #change("add_nodes_1024_loop", "add_nodes_1024_batched").

#figure(
  bench-table((
    ([Insert 1,024], "lara_stable_insert_1024"),
    ([Insert 65,536], "lara_stable_insert_65536"),
    ([Scan 65,536], "lara_stable_scan_65536"),
    ([Visit 65,536], "lara_stable_visit_65536"),
  )),
  caption: [LARA on stable memory.],
)

#figure(
  bench-bars(
    (
      ([insert 1k], ("lara_stable_insert_1024", "btree_insert_1024_stable")),
      ([insert 64k], ("lara_stable_insert_65536", "btree_insert_65536_stable")),
      ([scan 64k], ("lara_stable_scan_65536", "btree_scan_65536_stable")),
      ([visit 64k], ("lara_stable_visit_65536", "btree_visit_65536_stable")),
    ),
    series: ([LARA], [StableBTreeMap]),
  ),
  caption: [Instructions per operation (log scale).],
)

#figure(
  item("/crates/lara-graph-bench/src/btree.rs", "fn lara_stable_visit_65536"),
  caption: [The benchmark behind the visit numbers.],
)

#figure(
  item("/crates/lara-graph/src/bucket/paged.rs", "struct Leaf", docs: true),
  caption: [A leaf of the paged bucket.],
)
