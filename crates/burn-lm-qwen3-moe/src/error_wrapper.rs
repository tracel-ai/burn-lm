use burn_lm_inference::{serde_json, InferenceResult};

pub trait Wrap<T> {
    fn w(self) -> InferenceResult<T>;
}
// impl<T> Wrap<T> for Result<T, hf_hub::api::sync::ApiError> {
//     fn w(self) -> InferenceResult<T> {
//         match self {
//             Ok(value) => Ok(value),
//             Err(e) => Err(burn_lm_inference::InferenceError::DownloadError(
//                 "hf-hub".to_string(),
//                 e.to_string(),
//             )),
//         }
//     }
// }

// impl<T> Wrap<T> for std::io::Result<T> {
//     fn w(self) -> InferenceResult<T> {
//         match self {
//             Ok(value) => Ok(value),
//             Err(e) => Err(burn_lm_inference::InferenceError::DownloadError(
//                 "io result".to_string(),
//                 e.to_string(),
//             )),
//         }
//     }
// }
// impl<T> Wrap<T> for serde_json::Result<T> {
//     fn w(self) -> InferenceResult<T> {
//         match self {
//             Ok(value) => Ok(value),
//             Err(e) => Err(burn_lm_inference::InferenceError::Custom(e.to_string())),
//         }
//     }
// }
impl<T, E: core::fmt::Debug + core::fmt::Display> Wrap<T> for std::result::Result<T, E> {
    fn w(self) -> InferenceResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(e) => Err(burn_lm_inference::InferenceError::Custom(e.to_string())),
        }
    }
}
