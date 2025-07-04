use burn_lm_inference::InferencePlugin;
use chrono::NaiveDate;
use serde::Serialize;
use utoipa::ToSchema;

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ModelSchema {
    pub id: String,
    pub created: u32,
    pub object: String,
    pub owned_by: String,
}

impl From<&Box<dyn InferencePlugin>> for ModelSchema {
    fn from(plugin: &Box<dyn InferencePlugin>) -> Self {
        let created_date = NaiveDate::parse_from_str(plugin.model_creation_date(), "%m/%d/%Y")
            .expect("Valid date format expected (MM/DD/YYYY)");
        let created = created_date
            .and_hms_opt(0, 0, 0)
            .expect("Should be a valid time using MM/DD/YYYY format")
            .and_utc()
            .timestamp() as u32;
        Self {
            id: plugin.model_name().to_string(),
            object: "model".to_string(),
            owned_by: plugin.owned_by().to_string(),
            created,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelResponseSchema {
    pub object: String,
    pub data: Vec<ModelSchema>,
}
