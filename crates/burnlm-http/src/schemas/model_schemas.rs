use burn::prelude::Backend;
use burnlm_registry::burnlm_plugin::InferencePluginMetadata;
use serde::Serialize;
use utoipa::ToSchema;
use chrono::NaiveDate;

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ModelSchema {
    pub id: String,
    pub created: u32,
    pub object: String,
    pub owned_by: String,
}

impl<B: Backend> From<&InferencePluginMetadata<B>> for ModelSchema {
    fn from(source: &InferencePluginMetadata<B>) -> Self {
        let created_date = NaiveDate::parse_from_str(&source.model_creation_date, "%m/%d/%Y").expect("Valid date format expected (MM/DD/YYYY)");
        let created = created_date
            .and_hms_opt(0, 0, 0)
            .expect("Should be a valid time using MM/DD/YYYY format")
            .and_utc()
            .timestamp() as u32;
        Self {
            id: source.model_name.to_string(),
            object: "model".to_string(),
            owned_by: source.owned_by.to_string(),
            created,
        }
    }
}


#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelResponseSchema {
    pub object: String,
    pub data: Vec<ModelSchema>,
}
