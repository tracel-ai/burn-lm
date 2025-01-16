use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[derive(FromDeriveInput, Default)]
#[darling(default, attributes(inference_plugin))]
struct InferencePluginAttributes {
    model_name: Option<String>,
}

#[proc_macro_derive(InferencePlugin, attributes(inference_plugin))]
pub fn inference_plugin(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let input_ident = &input.ident;
    let input_ident_string = input_ident.to_string();
    let (input_generics_impl, input_generics_type, input_generics_where_clause) =
        &input.generics.split_for_impl();
    let config_ident =
        syn::Ident::new(&format!("{}Config", input_ident_string), input_ident.span());
    // retrieve plugin info
    let attributes = InferencePluginAttributes::from_derive_input(&input)
        .expect("Should successfuly parse inference_plugin attributes");
    let model_name = match attributes.model_name {
        Some(value) => value,
        None => {
            let err_msg = "You must provide a 'model_name' using '#[inference_plugin(model_name=\"MyModel\")]'";
            return TokenStream::from(quote! { compile_error!(#err_msg) });
        }
    };
    let model_name_lc = model_name.to_lowercase();

    let expanded = quote! {
        impl #input_generics_impl #input_ident #input_generics_type #input_generics_where_clause {
            pub const fn model_name() -> &'static str { #model_name }
            pub const fn model_name_lc() -> &'static str { #model_name_lc }
        }

        inventory::submit! {
            InferencePluginMetadata::<InferenceBackend>::new(
                #input_ident::<InferenceBackend>::model_name(),
                #input_ident::<InferenceBackend>::model_name_lc(),
                #config_ident::command,
            )
        }
    };
    TokenStream::from(expanded)
}
