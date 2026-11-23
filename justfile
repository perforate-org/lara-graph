# Gates for `lara-graph`.

crate := "lara-graph"

# Every gate; CI and pre-commit run this.
gate: fmt clippy test smells

fmt:
    cargo fmt -p {{crate}} --check

clippy:
    cargo clippy -p {{crate}} --all-targets --all-features -- -D warnings

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

# Regenerate `tests/public-api.txt` after a deliberate API change; review the diff.
bless-api:
    UPDATE_EXPECT=1 cargo test -p {{crate}} --test public_api
