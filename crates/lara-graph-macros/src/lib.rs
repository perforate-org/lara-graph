//! Procedural macros for `lara-graph`.
//!
//! Currently provides [`derive_label`]: `#[derive(Label)]` for fieldless
//! enums. This is a proc-macro crate, so it always runs on the host and has
//! no `no_std`/feature concerns of its own.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Derives `lara_graph::Label` for a fieldless enum.
///
/// Variants are numbered in **declaration order**, starting at 0:
///
/// ```ignore
/// use lara_graph::Label;
///
/// #[derive(Copy, Clone, Debug, PartialEq, Eq, Label)]
/// enum EdgeKind { Follows, Mentions, Blocks }
/// // Follows → LabelId(0), Mentions → LabelId(1), Blocks → LabelId(2)
/// ```
///
/// Explicit discriminants (e.g. `Blocks = 7`) are **ignored** — numbering
/// is always by declaration order. Only fieldless enums are supported;
/// structs, unions, generic enums, and variants with fields are rejected
/// with a `compile_error!`.
///
/// The generated impl references the crate by its absolute path
/// (`::lara_graph::Label` / `::lara_graph::LabelId`), so it does not
/// depend on the caller's `use` statements.
///
/// (The `lara_graph::Label` link above is written as plain text because
/// this proc-macro crate does not depend on `lara-graph`.)
#[proc_macro_derive(Label)]
pub fn derive_label(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Reject generics: Label derive targets fieldless enums only.
    if !input.generics.params.is_empty() {
        return syn::Error::new_spanned(
            &input.generics,
            "Label derive only supports fieldless enums (no generics)",
        )
        .to_compile_error()
        .into();
    }

    let data = match &input.data {
        Data::Enum(data) => data,
        Data::Struct(_) | Data::Union(_) => {
            return syn::Error::new_spanned(
                &input.ident,
                "Label derive only supports fieldless enums",
            )
            .to_compile_error()
            .into();
        }
    };

    // Collect unit variants in declaration order; reject any with fields.
    let mut variants: Vec<(u16, &syn::Ident)> = Vec::new();
    for (index, variant) in data.variants.iter().enumerate() {
        if !matches!(variant.fields, Fields::Unit) {
            return syn::Error::new_spanned(
                variant,
                "Label derive only supports fieldless enums (variant has fields)",
            )
            .to_compile_error()
            .into();
        }
        variants.push((index as u16, &variant.ident));
    }

    let into_arms = variants.iter().map(|(index, ident)| {
        quote! { #name::#ident => ::lara_graph::LabelId::new(#index), }
    });
    let from_arms = variants.iter().map(|(index, ident)| {
        quote! { #index => #name::#ident, }
    });

    let expanded = quote! {
        impl ::lara_graph::Label for #name {
            fn into_id(self) -> ::lara_graph::LabelId {
                match self {
                    #(#into_arms)*
                }
            }
            fn from_id(id: ::lara_graph::LabelId) -> Self {
                match id.as_u16() {
                    #(#from_arms)*
                    _ => panic!(
                        "label id {} out of range for {}",
                        id.as_u16(),
                        stringify!(#name)
                    ),
                }
            }
        }
    };

    expanded.into()
}
