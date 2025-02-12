use std::collections::BTreeSet;

use chrono::NaiveDate;
use darling::{
    ast::{self, NestedMeta},
    FromDeriveInput, FromField, FromMeta,
};
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, punctuated::Punctuated, DeriveInput, ItemStruct};

// InferenceSeverConfig ------------------------------------------------------

/// This macro consumes the struct, extracts any `#[config(default = ...)]` attributes
/// and regenerates a brand-new struct with:
///   - `#[derive(Parser, Deserialize, Debug)]` derive macros
///   - each field gets the passed default value of the config field attribute for clap and serde with `#[arg(...)]` and `#[serde(...)]`
///   - add marker trait `impl InferenceServerConfig for ... {}`
///   - generated implementation for default values compatible with both clap and serde with the Default trait using generated `fn default_<field>()` functions
#[proc_macro_attribute]
pub fn inference_server_config(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_struct = parse_macro_input!(item as ItemStruct);
    match InferenceServerConfigReceiver::from_item_struct(&input_struct) {
        Ok(receiver) => receiver.expand(),
        Err(e) => e.write_errors().into(),
    }
}

#[derive(FromDeriveInput)]
#[darling(attributes(config), supports(struct_named))]
struct InferenceServerConfigReceiver {
    ident: syn::Ident,
    vis: syn::Visibility,
    generics: syn::Generics,
    attrs: Vec<syn::Attribute>,
    data: ast::Data<(), InferenceServerConfigField>,
}

#[derive(FromField)]
#[darling(attributes(config), forward_attrs(doc))]
struct InferenceServerConfigField {
    ident: Option<syn::Ident>,
    ty: syn::Type,
    attrs: Vec<syn::Attribute>,
    #[darling(default)]
    default: Option<syn::Lit>,
    openwebui_param: Option<syn::LitStr>,
}

impl InferenceServerConfigReceiver {
    /// Darling works with derived structure only
    /// so we convert the struct AST form the attribute macro to a derive input AST
    fn from_item_struct(item: &syn::ItemStruct) -> darling::Result<Self> {
        let di = syn::DeriveInput {
            attrs: item.attrs.clone(),
            vis: item.vis.clone(),
            ident: item.ident.clone(),
            generics: item.generics.clone(),
            data: syn::Data::Struct(syn::DataStruct {
                fields: item.fields.clone(),
                struct_token: item.struct_token,
                semi_token: item.semi_token,
            }),
        };
        // now we can call Darling
        InferenceServerConfigReceiver::from_derive_input(&di)
    }

    fn expand(&self) -> TokenStream {
        let struct_name = &self.ident;
        let vis = &self.vis;
        let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();
        let struct_attrs = &self.attrs;
        // Extract the named fields
        let fields = match &self.data {
            ast::Data::Struct(fields) => &fields.fields,
            _ => unreachable!("Should only be a named struct."),
        };

        // Add clap and serde attributes for default value
        let field_defs = fields.iter().map(|f| {
            let field_ident = f.ident.as_ref().unwrap();
            let field_ty = &f.ty;
            let docs = &f.attrs;
            // We need to wrap the default values into functions so that we can set the default value
            // for both clap and serde
            let default_fn_name =
                syn::Ident::new(&format!("default_{}", field_ident), field_ident.span());
            // we need the serde default value as a string
            let serde_default_string = format!("{}::{}", struct_name, default_fn_name);
            let serde_default_lit_str =
                syn::LitStr::new(&serde_default_string, proc_macro2::Span::call_site());
            // map config to open webui parameter name
            let serde_rename = match &f.openwebui_param {
                Some(lit) => lit.clone(),
                None => syn::LitStr::new(&field_ident.to_string(), proc_macro2::Span::call_site()),
            };
            // rewritten field
            quote! {
                #(#docs)*
                #[arg(long, default_value_t = #struct_name::#default_fn_name())]
                #[serde(default = #serde_default_lit_str, rename = #serde_rename)]
                pub #field_ident: #field_ty,
            }
        });

        // Generate the wrapper functions for default values
        let default_fns = fields.iter().map(|f| {
            let field_ident = f.ident.as_ref().unwrap();
            let fn_name = syn::Ident::new(&format!("default_{}", field_ident), field_ident.span());
            let field_ty = &f.ty;

            if let Some(lit) = &f.default {
                // Example:
                //   #[config(default = 0.9)]
                // generates
                //   fn default_top_p() -> f64 { 0.9 }
                quote! {
                    fn #fn_name() -> #field_ty {
                        #lit
                    }
                }
            } else {
                // fallback to the type's default implementation if there is no config attribute
                quote! {
                    fn #fn_name() -> #field_ty {
                        <#field_ty as ::std::default::Default>::default()
                    }
                }
            }
        });

        // Generate Default trait implementation by making use of the function wrappers
        let default_inits = fields.iter().map(|f| {
            let field_ident = f.ident.as_ref().unwrap();
            let fn_name = syn::Ident::new(&format!("default_{}", field_ident), field_ident.span());
            quote! {
                #field_ident: Self::#fn_name(),
            }
        });

        // Output
        let expanded = quote! {
            // Rewritten struct
            #[derive(Clone, Parser, Deserialize, ::std::fmt::Debug)]
            #(#struct_attrs)*
            #vis struct #struct_name #impl_generics #where_clause {
                #(#field_defs)*
            }
            // Marker trait
            impl #impl_generics InferenceServerConfig for #struct_name #ty_generics #where_clause {}
            // Function wrappers for default values
            impl #impl_generics #struct_name #ty_generics #where_clause {
                #(#default_fns)*
            }
            // Default trait implementation
            impl #impl_generics ::std::default::Default for #struct_name #ty_generics #where_clause {
                fn default() -> Self {
                    Self {
                        #(#default_inits)*
                    }
                }
            }
        };
        expanded.into()
    }
}

// InferenceServer -----------------------------------------------------------

#[derive(FromDeriveInput)]
#[darling(attributes(inference_server))]
struct InferenceServerData {
    model_name: Option<String>,
    model_cli_param_name: Option<String>,
    model_creation_date: Option<String>,
    owned_by: Option<String>,
    data: darling::ast::Data<darling::util::Ignored, InferenceServerField>,
}

#[derive(FromField)]
struct InferenceServerField {
    ident: Option<syn::Ident>,
    ty: syn::Type,
}

#[proc_macro_derive(InferenceServer, attributes(inference_server))]
pub fn inference_server(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let input_ident = &input.ident;
    let (input_generics_impl, input_generics_type, input_generics_where_clause) =
        &input.generics.split_for_impl();
    let receiver = match InferenceServerData::from_derive_input(&input) {
        Ok(r) => r,
        Err(e) => return e.write_errors().into(),
    };

    // Verify that the struct has a 'config' field and retrieve its type
    let config_ty = match receiver.data {
        ast::Data::Struct(fields) => {
            let config = fields
                .fields
                .iter()
                .find(|f| f.ident.as_ref().is_some_and(|i| *i == "config"));
            match config {
                Some(field) => field.ty.clone(),
                None => {
                    let err_msg = "The server struct must have a field named 'config'.";
                    return TokenStream::from(quote! { compile_error!(#err_msg) });
                }
            }
        }
        _ => {
            let err_msg = "The server type must be a struct and not an enum.";
            return TokenStream::from(quote! { compile_error!(#err_msg) });
        }
    };

    // Retrieve the config type

    // retrieve plugin info
    // handle model_name
    let model_name = match receiver.model_name {
        Some(value) => value,
        None => {
            let err_msg = "You must provide a 'model_name' using '#[inference_server(model_name=\"MyModel\")]'";
            return TokenStream::from(quote! { compile_error!(#err_msg) });
        }
    };
    // handle model CLI param name
    let model_cli_param_name = match receiver.model_cli_param_name {
        Some(ref param_name) => {
            let param_name = param_name.to_lowercase().replace(" ", "-");
            quote! { #param_name }
        }
        None => {
            let param_name = model_name.to_lowercase().replace(" ", "-");
            quote! { #param_name }
        }
    };
    // handle model_creation_date
    let model_creation_date = match receiver.model_creation_date {
        Some(ref date_str) => {
            if NaiveDate::parse_from_str(date_str, "%m/%d/%Y").is_err() {
                let err_msg = format!(
                    "Invalid 'model_creation_date': {}. Must be in MM/DD/YYYY format.",
                    date_str
                );
                return TokenStream::from(quote! { compile_error!(#err_msg) });
            }
            quote! { #date_str }
        }
        None => {
            let err_msg = "You must provide a 'model_creation_date' using '#[inference_server(model_creation_date=\"MM/DD/YYYY\")]'";
            return TokenStream::from(quote! { compile_error!(#err_msg) });
        }
    };
    // handle owned_by
    let owned_by = match receiver.owned_by {
        Some(ref owner) => quote! { #owner },
        None => {
            let err_msg = "You must provide an 'owned_by' attribute using '#[inference_server(owned_by=\"OwnerName\")]'";
            return TokenStream::from(quote! { compile_error!(#err_msg) });
        }
    };
    // Generate Sync trait implementation
    let sync_marker_trait = quote! {
        unsafe impl #input_generics_impl Sync for #input_ident #input_generics_type #input_generics_where_clause {}
    };

    let expanded = quote! {
        impl #input_generics_impl #input_ident #input_generics_type #input_generics_where_clause {
            pub const fn model_name() -> &'static str { #model_name }
            pub const fn model_cli_param_name() -> &'static str { #model_cli_param_name }
            pub const fn model_creation_date() -> &'static str { #model_creation_date }
            pub const fn owned_by() -> &'static str { #owned_by }
        }

        impl #input_generics_impl ServerConfigParsing for #input_ident #input_generics_type #input_generics_where_clause {
            type Config = #config_ty;

            fn parse_cli_config(&mut self, args: &clap::ArgMatches) {
                self.config = Self::Config::from_arg_matches(args)
                    .expect("Should be able to parse arguments from CLI");
            }

            fn parse_json_config(&mut self, json: &str) {
                self.config = serde_json::from_str(json)
                    .expect("Should be able to parse JSON");
            }
        }
        // Sync marker
        #sync_marker_trait
    };
    TokenStream::from(expanded)
}

// Register inference servers ------------------------------------------------

#[derive(Debug, FromMeta)]
struct InferenceServerEntry {
    crate_namespace: String,
    #[darling(rename = "server_type")]
    server_ty: String,
}

#[derive(Debug, Default, FromMeta)]
#[darling(default)]
struct InferenceServerEntries {
    #[darling(default, rename = "server", multiple)]
    servers: Vec<InferenceServerEntry>,
}

/// This macro implements the new() function for the Registry struct given a
/// list of Inference Server entries.
/// This macro also defines some type aliases to make the generated code more readable.
/// For instance for "MyModel" server_name the macro will define the following types:
///   - MyModelS = "type of the server passed as server_type"
///   - MyModelC = InferenceClient<MyModelS, Channel<MyModelS>
#[proc_macro_attribute]
pub fn inference_server_registry(attr: TokenStream, item: TokenStream) -> TokenStream {
    let parsed_attr =
        parse_macro_input!(attr with Punctuated::<NestedMeta, syn::Token![,]>::parse_terminated);
    let attributes_meta: Vec<NestedMeta> = parsed_attr.into_iter().collect();
    // Use Darling to parse the attributes meta into InferenceServerEntries
    let registry_args = match InferenceServerEntries::from_list(&attributes_meta) {
        Ok(args) => args,
        Err(e) => {
            return e.write_errors().into();
        }
    };
    let input_struct = parse_macro_input!(item as ItemStruct);
    let struct_ident = &input_struct.ident;
    // generate hash map entry for each server entry
    let mut registry_entries = Vec::new();
    let mut crate_namespaces = BTreeSet::new();
    for server in &registry_args.servers {
        // crate namespaces set
        crate_namespaces.insert(&server.crate_namespace);
        // registry hash map entry
        let server_ty_str = &server.server_ty;
        let server_ty: syn::Type = match syn::parse_str(server_ty_str) {
            Ok(ty) => ty,
            Err(e) => {
                // Server type is unkown
                let msg = format!("Invalid server_type `{}`: {}", server_ty_str, e);
                return syn::Error::new_spanned(
                    syn::Lit::Str(syn::LitStr::new(
                        server_ty_str,
                        proc_macro2::Span::call_site(),
                    )),
                    msg,
                )
                .to_compile_error()
                .into();
            }
        };
        let registry_entry = quote! {
            {
                type S = #server_ty;
                type C = InferenceClient<#server_ty, Channel<#server_ty>>;
                map.insert(
                    S::model_name(),
                    Box::new(C::new(
                        S::model_name(),
                        S::model_cli_param_name(),
                        S::model_creation_date(),
                        S::owned_by(),
                        <S as ServerConfigParsing>::Config::command,
                        Channel::<S>::new(),
                    )),
                );
            }
        };
        registry_entries.push(registry_entry);
    }
    // Imports
    let mut crate_imports = Vec::new();
    for namespace in crate_namespaces {
        let crate_path: syn::Path =
            syn::parse_str(namespace).expect("crate namespace should be a valid path");
        let use_crate = quote! {
            pub use #crate_path::*;
        };
        crate_imports.push(use_crate);
    }

    // Output
    let output = quote! {
        // imports
        #(#crate_imports)*
        // Original struct
        #input_struct
        // new() implementation
        impl #struct_ident {
            pub fn new() -> Self {
                let mut map: DynClients = ::std::collections::HashMap::new();
                #(#registry_entries)*
                Self {
                    clients: ::std::sync::Arc::new(map),
                }
            }
            pub fn get(&self) -> &DynClients {
                &self.clients
            }
        }

        // Implement default to make clippy happy
        impl ::std::default::Default for #struct_ident {
            fn default() -> Self {
                Self::new()
            }
        }
    };

    output.into()
}
