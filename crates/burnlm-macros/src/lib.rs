use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Path};
use chrono::NaiveDate;

#[derive(FromDeriveInput, Default)]
#[darling(default, attributes(inference_plugin))]
struct InferencePluginAttributes {
    model_name: Option<String>,
    model_versions: Option<Path>,
    model_creation_date: Option<String>,
    owned_by: Option<String>,
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
    // handle model_name
    let model_name = match attributes.model_name {
        Some(value) => value,
        None => {
            let err_msg = "You must provide a 'model_name' using '#[inference_plugin(model_name=\"MyModel\")]'";
            return TokenStream::from(quote! { compile_error!(#err_msg) });
        }
    };
    let model_name_lc = model_name.to_lowercase();
    // handle model_versions
    let model_versions_impl = if let Some(path) = attributes.model_versions {
        // the enum must derive from clap::ValueEnum
        quote! {
            fn get_model_versions() -> Vec<String>
            where
                Self: Sized {
                <#path as clap::ValueEnum>::value_variants()
                    .iter()
                    .map(|v| v.to_possible_value().unwrap().get_name().to_string())
                    .collect()
            }
        }
    } else {
        // Default to a single version named "default"
        quote! {
            fn get_model_versions() -> Vec<String>
            where
                Self: Sized {
                vec!["default".to_string()]
            }
        }
    };
    // handle model_creation_date
    let model_creation_date = match attributes.model_creation_date {
        Some(ref date_str) => {
            if NaiveDate::parse_from_str(date_str, "%m/%d/%Y").is_err() {
                let err_msg = format!("Invalid 'model_creation_date': {}. Must be in MM/DD/YYYY format.", date_str);
                return TokenStream::from(quote! { compile_error!(#err_msg) });
            }
            quote! { #date_str }
        }
        None => {
            let err_msg = "You must provide a 'model_creation_date' using '#[inference_plugin(model_creation_date=\"MM/DD/YYYY\")]'";
            return TokenStream::from(quote! { compile_error!(#err_msg) });
        }
    };
    // handle owned_by
    let owned_by = match attributes.owned_by {
        Some(ref owner) => quote! { #owner },
        None => {
            let err_msg = "You must provide an 'owned_by' attribute using '#[inference_plugin(owned_by=\"OwnerName\")]'";
            return TokenStream::from(quote! { compile_error!(#err_msg) });
        }
    };

    let expanded = quote! {
        impl #input_generics_impl #input_ident #input_generics_type #input_generics_where_clause {
            pub const fn model_name() -> &'static str { #model_name }
            pub const fn model_name_lc() -> &'static str { #model_name_lc }
            pub const fn model_creation_date() -> &'static str { #model_creation_date }
            pub const fn owned_by() -> &'static str { #owned_by }
        }

        impl #input_generics_impl InferencePluginAssociatedFn for #input_ident #input_generics_type #input_generics_where_clause {
            #model_versions_impl

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
                // model_creation_date
                #input_ident::<InferenceBackend>::model_creation_date(),
                // owned_by
                #input_ident::<InferenceBackend>::owned_by(),
                // create_cli_flags_fn
                #config_ident::command,
                // parse_cli_flags_fn
                #input_ident::<InferenceBackend>::parse_cli_config,
                // create_plugin_fn
                #input_ident::<InferenceBackend>::new,
                // get_model_versions_fn
                #input_ident::<InferenceBackend>::get_model_versions,
            )
        }
    };
    TokenStream::from(expanded)
}
