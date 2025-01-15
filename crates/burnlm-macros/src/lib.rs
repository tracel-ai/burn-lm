use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(BurnLM)]
pub fn burnlm(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let input_ident = &input.ident;
    let input_ident_string = input_ident.to_string();
    let input_ident_lc_string = input_ident_string.to_lowercase();
    let config_ident =
        syn::Ident::new(&format!("{}Config", input_ident_string), input_ident.span());
    let expanded = quote! {
        impl #input_ident {
            pub const fn name() -> &'static str { #input_ident_string }
            pub const fn lc_name() -> &'static str { #input_ident_lc_string }
        }

        inventory::submit! {
            InferenceModelPlugin::new(
                #input_ident::name(),
                #input_ident::lc_name(),
                #config_ident::command,
            )
        }
    };
    TokenStream::from(expanded)
}
