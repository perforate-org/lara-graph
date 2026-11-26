# Gates for `lara-graph`.

crate := "lara-graph"
# The bench crate is a workspace of its own (see its Cargo.toml).
bench := "--manifest-path crates/lara-graph-bench/Cargo.toml"

# Every gate; CI and pre-commit run this.
gate: fmt clippy test doc deslop smells

fmt:
    cargo fmt -p {{crate}} --check
    cargo fmt {{bench}} --check

clippy:
    cargo clippy -p {{crate}} --all-targets --all-features -- -D warnings
    cargo clippy {{bench}} --all-targets --locked -- -D warnings

# The modules' documentation is the specification: every link in it must resolve.
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc -p {{crate}} --all-features --document-private-items --no-deps

# Architecture: modules depend one way, stay deep, and keep their fields to themselves
# (deslop-please: `cargo install deslop-please`).
deslop:
    deslop-please {{crate}} --all-features

# Proofs over every input of the persistent codecs (`#[cfg(kani)] mod proofs`). Needs Kani
# (`cargo install --locked kani-verifier && cargo kani setup`); not part of the gate.
kani:
    cargo kani -p {{crate}}

# Includes the public API snapshot (`tests/public_api.rs`).
test:
    cargo test -p {{crate}} --all-features

# Duplication and complexity (thresholds in `.qlty/qlty.toml`). `qlty smells` exits 0 even with
# findings, so the gate counts the SARIF results and prints them before failing.
smells:
    #!/usr/bin/env bash
    set -euo pipefail
    count=$(qlty smells --all --sarif --no-upgrade-check | jq '[.runs[].results[]] | length')
    if [ "$count" -ne 0 ]; then
        qlty smells --all --no-snippets --no-upgrade-check
        echo "qlty: $count smell(s)" >&2
        exit 1
    fi

# Paper parts (`paper/`): numbers, tables and listings are read from the repository, so the root
# is the repository. `just paper watch` recompiles on every change, including a new canbench run.
paper mode="compile":
    typst {{mode}} --root . paper/main.typ paper/main.pdf

# Use the repository's git hooks (`.githooks`): format and lints before a commit, the gate before a
# push. Once per clone.
hooks:
    git config core.hooksPath .githooks

# Regenerate `tests/public-api.txt` after a deliberate API change; review the diff.
bless-api:
    UPDATE_EXPECT=1 cargo test -p {{crate}} --test public_api
