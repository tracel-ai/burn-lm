pub type ConfigFlagsFn = fn() -> clap::Command;

pub struct InferenceModelPlugin {
    pub name: &'static str,
    pub lc_name: &'static str,
    pub config_flags: ConfigFlagsFn,
}

impl InferenceModelPlugin {
    pub const fn new(
        name: &'static str,
        lc_name: &'static str,
        config_flags: ConfigFlagsFn,
    ) -> Self {
        Self {
            name,
            lc_name,
            config_flags,
        }
    }
}

inventory::collect!(InferenceModelPlugin);

// #[derive(Clone, Debug)]
// pub struct InferenceModelPlugin {
//     pub name: &'static str,
// }

// impl InferenceModelPlugin {
//     pub const fn new(name: &'static str) -> Self {
//         Self { name }
//     }
// }

// pub type LazyValue<T> = once_cell::sync::Lazy<T>;
// pub struct Plugin<T: 'static>(pub &'static LazyValue<T>);

// inventory::collect!(Plugin<InferenceModelPlugin>);

// pub const fn make_static_lazy<T: 'static>(func: fn() -> T) -> LazyValue<T> {
//     LazyValue::<T>::new(func)
// }

// pub use gensym;
// pub use inventory;
// pub use paste;

// #[macro_export]
// macro_rules! register_model {
//     ($a:ty, $fn_:expr) => {
//         $crate::plugin::gensym::gensym! { $crate::register_model!{ $a, $fn_ } }
//     };
//     ($gensym:ident, $a:ty, $fn_:expr) => {
//         $crate::plugin::paste::paste! {
//             #[used]
//             static [<$gensym _register_model_>]: $crate::plugin::LazyValue<$a> = $crate::plugin::make_static_lazy(|| {
//                 $fn_
//             });
//             $crate::plugin::inventory::submit!($crate::plugin::Plugin(&[<$gensym _register_model_>]));
//         }
//     };
// }

// pub fn get_models() -> Vec<InferenceModelPlugin> {
//     inventory::iter::<Plugin<InferenceModelPlugin>>
//         .into_iter()
//         .map(|plugin| (*plugin.0).to_owned())
//         .collect()
// }
