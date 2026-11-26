// Numbers, tables and plots read straight from canbench's persisted results, so the paper
// follows `canbench --persist` without copying a figure by hand.
//
// Paths are root-relative: compile with `--root .` from the repository (see `just paper`).

#import "@preview/lilaq:0.5.0" as lq

#let results = yaml("/crates/lara-graph-bench/canbench_results.yml").benches

// The `total` record of one bench; panics with the name when it is missing, so a renamed bench
// breaks the build instead of silently printing a stale number.
#let total(name) = {
  assert(name in results, message: "no canbench result named `" + name + "`")
  results.at(name).total
}

/// Raw counter of a bench: `instructions` (default), `heap_increase` or `stable_memory_increase`.
#let value(name, metric: "instructions") = total(name).at(metric)

/// Thousands-separated integer: 1729749 → 1,729,749.
#let group(n, sep: ",") = {
  let s = str(n)
  let out = ""
  for (i, c) in s.clusters().enumerate() {
    if i > 0 and calc.rem(s.len() - i, 3) == 0 { out += sep }
    out += c
  }
  out
}

/// Short SI form: 1729749 → 1.73 M.
#let si(n, digits: 2) = {
  let (div, unit) = if n >= 1e9 { (1e9, "G") } else if n >= 1e6 { (1e6, "M") } else if n >= 1e3 {
    (1e3, "k")
  } else { (1, "") }
  if unit == "" { str(n) } else { str(calc.round(n / div, digits: digits)) + sym.space.thin + unit }
}

/// Instruction count of a bench, formatted for running text.
#let num(name, metric: "instructions", fmt: si) = fmt(value(name, metric: metric))

/// `a / b` of a metric, e.g. how many times fewer instructions `b` needs than `a`.
#let ratio(a, b, metric: "instructions", digits: 1) = {
  str(calc.round(value(a, metric: metric) / value(b, metric: metric), digits: digits)) + sym.times
}

/// Relative change from `base` to `new` in percent, signed: −21 %.
#let change(base, new, metric: "instructions", digits: 0) = {
  let p = (value(new, metric: metric) / value(base, metric: metric) - 1) * 100
  let r = calc.round(p, digits: digits)
  (if r < 0 { sym.minus + str(-r) } else { "+" + str(r) }) + sym.space.thin + "%"
}

/// Table of benches. `rows` is an array of names or `(label, name)` pairs; `metrics` picks the
/// columns.
#let bench-table(
  rows,
  metrics: ("instructions",),
  headers: (instructions: [Instructions], heap_increase: [Heap pages], stable_memory_increase: [Stable pages]),
  fmt: group,
) = {
  let rows = rows.map(r => if type(r) == str { (raw(r), r) } else { r })
  table(
    columns: (auto,) + metrics.map(_ => auto),
    align: (left,) + metrics.map(_ => right),
    stroke: none,
    table.hline(),
    table.header([Benchmark], ..metrics.map(m => headers.at(m, default: raw(m)))),
    table.hline(stroke: 0.5pt),
    ..rows.map(((label, name)) => (label, ..metrics.map(m => fmt(value(name, metric: m))))).flatten(),
    table.hline(),
  )
}

/// Grouped bar chart. `groups` is an array of `(label, (name, name, ...))`, one name per series;
/// `series` names the bars within a group. Log scale by default since counts span decades.
#let bench-bars(
  groups,
  series: (),
  metric: "instructions",
  ylabel: [Instructions],
  yscale: "log",
  width: 8cm,
  height: 4.5cm,
) = {
  let k = groups.first().at(1).len()
  let w = 0.8 / k
  // Bars grow from 0, which a log axis cannot show: start them at the decade below the minimum.
  let base = if yscale == "log" {
    let lo = calc.min(..groups.map(g => g.at(1).map(n => value(n, metric: metric))).flatten())
    calc.pow(10.0, calc.floor(calc.log(lo)))
  } else { 0 }
  // Head room above the tallest bar for the legend: one decade on log, a quarter on linear.
  let hi = calc.max(..groups.map(g => g.at(1).map(n => value(n, metric: metric))).flatten())
  lq.diagram(
    width: width,
    height: height,
    yscale: yscale,
    ylim: (base, if yscale == "log" { hi * 30 } else { hi * 1.25 }),
    ylabel: ylabel,
    xaxis: (ticks: groups.enumerate().map(((i, g)) => (i, g.at(0))), subticks: none),
    legend: (position: top + left),
    ..range(k).map(j => lq.bar(
      range(groups.len()),
      groups.map(g => value(g.at(1).at(j), metric: metric)),
      offset: (j - (k - 1) / 2) * w,
      width: w,
      base: base,
      label: series.at(j, default: none),
    )),
  )
}
