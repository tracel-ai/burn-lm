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
    let cli_downcasting_err_msg =
        format!("Args should be from {} struct", config_ident.to_string());
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

        impl #input_generics_impl InferencePluginAssociatedFn for #input_ident #input_generics_type #input_generics_where_clause {
            fn parse_cli_config(args: &clap::ArgMatches) -> Box<dyn Any>
            where
                Self: Sized {
                let config = #config_ident::from_arg_matches(args).expect(#cli_downcasting_err_msg);
                Box::new(config)
            }
        }

        inventory::submit! {
            InferencePluginMetadata::<InferenceBackend>::new(
                // model_name
                #input_ident::<InferenceBackend>::model_name(),
                // model_name_lc
                #input_ident::<InferenceBackend>::model_name_lc(),
                // create_cli_flags_fn
                #config_ident::command,
                // parse_cli_flags_fn
                #input_ident::<InferenceBackend>::parse_cli_config,
                // create_plugin_fn
                #input_ident::<InferenceBackend>::new,
            )
        }
    };
    TokenStream::from(expanded)
}
