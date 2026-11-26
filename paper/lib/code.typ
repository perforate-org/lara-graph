// Code listings cut from the sources at compile time, so a listing never drifts from the code.
//
// Paths are root-relative ("/crates/..."): compile with `--root .` (see `just paper`).

// Lines of a file, without the trailing newline's empty last element.
#let _lines(path) = {
  let ls = read(path).split("\n")
  if ls.len() > 0 and ls.last() == "" { ls.slice(0, -1) } else { ls }
}

// Strip the common leading indentation so a nested item starts at column 0.
#let _dedent(ls) = {
  let widths = ls.filter(l => l.trim() != "").map(l => l.len() - l.trim(at: start).len())
  let d = if widths.len() == 0 { 0 } else { calc.min(..widths) }
  ls.map(l => if l.len() >= d { l.slice(d) } else { l.trim(at: start) })
}

#let _show(ls, lang) = raw(_dedent(ls).join("\n"), lang: lang, block: true)

/// Lines `from` through `to` (1-based, inclusive). Fragile under edits; prefer `item` or `region`.
#let lines(path, from, to, lang: "rust") = _show(_lines(path).slice(from - 1, to), lang)

/// The block between `// paper:begin <name>` and `// paper:end <name>`, markers excluded.
#let region(path, name, lang: "rust") = {
  let ls = _lines(path)
  let begin = ls.position(l => l.trim() == "// paper:begin " + name)
  assert(begin != none, message: "no `// paper:begin " + name + "` in " + path)
  let end = ls.slice(begin + 1).position(l => l.trim() == "// paper:end " + name)
  assert(end != none, message: "no `// paper:end " + name + "` in " + path)
  _show(ls.slice(begin + 1, begin + 1 + end), lang)
}

/// A Rust item found by its head, e.g. `item("/crates/x/src/a.rs", "fn scan")` or
/// `"struct Page"`: from that line to the brace (or `;`) that closes it. `docs: true` keeps the
/// `///` and `#[...]` lines above it. Braces are counted naively, so a `{` inside a string or
/// char literal in the item can cut it short; use `region` there.
#let item(path, head, docs: false, lang: "rust") = {
  let ls = _lines(path)
  let pat = regex("^\s*(pub(\([^)]*\))?\s+)?((const|unsafe|async|extern\s+\"C\")\s+)*" + head + "\b")
  let start = ls.position(l => l.match(pat) != none)
  assert(start != none, message: "no item `" + head + "` in " + path)
  let first = start
  if docs {
    while first > 0 and {
      let t = ls.at(first - 1).trim()
      t.starts-with("///") or t.starts-with("#[")
    } { first -= 1 }
  }
  let depth = 0
  let opened = false
  let last = start
  for i in range(start, ls.len()) {
    let code = ls.at(i).split("//").first()
    for c in code.clusters() {
      if c == "{" { depth += 1; opened = true } else if c == "}" { depth -= 1 }
    }
    last = i
    if (opened and depth == 0) or (not opened and code.trim().ends-with(";")) { break }
  }
  _show(ls.slice(first, last + 1), lang)
}
