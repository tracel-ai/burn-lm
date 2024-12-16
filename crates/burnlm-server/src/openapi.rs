use utoipa::OpenApi;

use crate::handlers::model_handlers::{__path_get_model, __path_list_models};
use crate::schemas::model_schemas::ModelResponseSchema;

/// OpenAPI spec
#[derive(OpenApi)]
#[openapi(
    paths(get_model, list_models),
    components(schemas(ModelResponseSchema)),
    )
]
pub(crate) struct ApiDoc;
