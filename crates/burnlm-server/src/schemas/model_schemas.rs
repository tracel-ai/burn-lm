use serde::Serialize;
use utoipa::ToSchema;

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct ModelSchema {
    pub id: String,
    pub created: u32,
    pub object: String,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ModelResponseSchema {
    pub object: String,
    pub data: Vec<ModelSchema>,
}
