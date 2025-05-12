use crate::{errors::ServerResult, schemas::model_schemas::ModelSchema};
use burnlm_inference::InferencePlugin;

#[trait_variant::make(ChatController: Send)]
pub trait LocalChatController {
    async fn get_plugin(
        &mut self,
        name: &str,
    ) -> ServerResult<(Box<dyn InferencePlugin>, Option<String>)>;
    async fn list_models(&self) -> ServerResult<Vec<ModelSchema>>;
}
