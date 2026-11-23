# lara-graph-macros

The `Label` derive macro for `lara-graph`. Requires Rust 1.85 or newer.
Applications normally use the re-export from `lara_graph` with its default
`derive` feature, rather than adding this helper crate directly.

```rust
use lara_graph::{Label, LabelId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Label)]
enum Relation {
    Follows,
    Blocks = 7,
}

// IDs follow declaration order; explicit Rust discriminants are ignored.
assert_eq!(Relation::Blocks.into_id(), LabelId::new(1));
assert_eq!(Relation::from_id(LabelId::new(0)), Relation::Follows);
```

Only fieldless, non-generic enums are supported. The enum must satisfy the
`Label` trait's `Copy` requirement. `from_id` panics for an unknown ID.
Reordering variants changes their assigned IDs.

Generated code refers to `::lara_graph::Label` and `::lara_graph::LabelId`.
The consuming crate must have the graph dependency available as `lara_graph`;
renaming that dependency is not handled automatically by the macro.

## License

MIT OR Apache-2.0; see `LICENSE-MIT` and `LICENSE-APACHE`.
