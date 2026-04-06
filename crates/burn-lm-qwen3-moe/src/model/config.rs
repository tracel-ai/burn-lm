#[derive(Clone, Debug, serde::Deserialize)]
pub struct BaseConfig {
    pub architecture: String,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    pub bos_token_id: u32,
    pub decoder_sparse_step: u32,
    pub eos_token_id: u32,
    pub head_dim: u32,
    pub hidden_act: String,
    pub hidden_size: u32,
    pub initializer_range: f32,
    pub intermediate_size: u32,
    pub max_position_embeddings: u32,
    pub max_window_layers: u32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    pub num_attention_heads: u32,
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
    pub num_hidden_layers: u32,
    pub num_key_value_heads: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default)]
    pub router_aux_loss_coef: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_cache: bool,
    #[serde(default)]
    pub use_sliding_window: bool,
    pub vocab_size: u32,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Qwen3MoeConfig {
    #[serde(flatten)]
    pub base_config: BaseConfig,
    #[serde(default)]
    pub model_type: String,
    pub moe_intermediate_size: u32,
    #[serde(default)]
    pub mlp_only_layers: Vec<u32>,
    #[serde(default)]
    pub output_router_logits: bool,
}
