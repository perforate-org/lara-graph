//! String-name → dense [`LabelId`] registry (requires the `std` feature).
//!
//! The graph stores dense ids only; a catalog is an optional convenience
//! layer that maps human-readable names to those ids. It is append-only:
//! once a name is registered, its id is permanent (edge slots reference
//! it). This module is gated behind the default `std` feature because it
//! uses `std::collections::HashMap`.

use crate::labeled::LabelId;
use core::fmt;
use std::collections::HashMap;
use std::vec::Vec;

/// Error returned when the label catalog is full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabelLimit;

impl fmt::Display for LabelLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "label catalog full (max {} labels)", LabelId::MAX_COUNT)
    }
}

impl std::error::Error for LabelLimit {}

/// Append-only string-name registry mapping names to dense [`LabelId`]s.
///
/// Once registered, a label id is permanent (edge slots reference it).
/// Not part of the graph; the graph stores dense ids only.
pub struct LabelCatalog {
    by_name: HashMap<String, LabelId>,
    names: Vec<Box<str>>, // id -> name
    capacity: usize,
}

impl LabelCatalog {
    /// Creates an empty catalog with the default capacity
    /// ([`LabelId::MAX_COUNT`]).
    pub fn new() -> Self {
        Self::with_capacity(LabelId::MAX_COUNT)
    }

    /// Creates an empty catalog with a custom capacity.
    ///
    /// Used by tests to exercise the limit without registering 65,536
    /// names; the id space is still bounded by [`LabelId::MAX_COUNT`].
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            by_name: HashMap::new(),
            names: Vec::new(),
            capacity,
        }
    }

    /// Returns the existing id for `name`, or registers a new one.
    ///
    /// Errors with [`LabelLimit`] when the catalog is full.
    pub fn register(&mut self, name: &str) -> Result<LabelId, LabelLimit> {
        if let Some(&id) = self.by_name.get(name) {
            return Ok(id);
        }
        if self.names.len() >= self.capacity {
            return Err(LabelLimit);
        }
        let id = LabelId::new(self.names.len() as u16);
        self.by_name.insert(name.to_owned(), id);
        self.names.push(name.into());
        Ok(id)
    }

    /// Returns the name registered for `id`, if any.
    pub fn name_of(&self, id: LabelId) -> Option<&str> {
        self.names.get(id.as_u16() as usize).map(|s| s.as_ref())
    }

    /// Looks up the id for `name`, if registered.
    pub fn lookup(&self, name: &str) -> Option<LabelId> {
        self.by_name.get(name).copied()
    }

    /// Number of registered labels.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns `true` when no labels are registered.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

impl Default for LabelCatalog {
    fn default() -> Self {
        Self::new()
    }
}
