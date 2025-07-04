use std::marker::PhantomData;

use crate::{
    channels::InferenceChannel,
    completion::Completion,
    errors::InferenceResult,
    message::Message,
    plugin::{CreateCliFlagsFn, InferencePlugin},
    server::InferenceServer,
    Stats,
};

#[derive(Debug, Clone)]
pub struct InferenceClient<Server: InferenceServer + 'static, Channel: 'static> {
    model_name: &'static str,
    model_cli_param_name: &'static str,
    model_creation_date: &'static str,
    owned_by: &'static str,
    create_cli_flags_fn: CreateCliFlagsFn,
    channel: Channel,
    _phantom_server: PhantomData<Server>,
}

unsafe impl<Server, Channel> Sync for InferenceClient<Server, Channel>
where
    Server: InferenceServer,
    Channel: InferenceChannel<Server>,
{
}

impl<Server, Channel> InferenceClient<Server, Channel>
where
    Server: InferenceServer,
    Channel: InferenceChannel<Server>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_name: &'static str,
        model_cli_param_name: &'static str,
        model_creation_date: &'static str,
        owned_by: &'static str,
        create_cli_flags_fn: CreateCliFlagsFn,
        channel: Channel,
    ) -> Self {
        Self {
            model_name,
            model_cli_param_name,
            model_creation_date,
            owned_by,
            create_cli_flags_fn,
            channel,
            _phantom_server: PhantomData,
        }
    }
}

impl<Server, Channel> InferencePlugin for InferenceClient<Server, Channel>
where
    Server: InferenceServer + 'static,
    Channel: InferenceChannel<Server> + 'static,
{
    fn clone_box(&self) -> Box<dyn InferencePlugin> {
        Box::new(self.clone())
    }

    fn downloader(&self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        self.channel.downloader()
    }

    fn is_downloaded(&self) -> bool {
        self.channel.is_downloaded()
    }

    fn deleter(&self) -> Option<fn() -> InferenceResult<Option<Stats>>> {
        self.channel.deleter()
    }

    fn parse_cli_config(&self, args: &clap::ArgMatches) {
        self.channel.parse_cli_config(args);
    }

    fn parse_json_config(&self, json: &str) {
        self.channel.parse_json_config(json);
    }

    fn load(&self) -> InferenceResult<Option<Stats>> {
        self.channel.load()
    }

    fn is_loaded(&self) -> bool {
        self.channel.is_loaded()
    }

    fn unload(&self) -> InferenceResult<Option<Stats>> {
        self.channel.unload()
    }

    fn run_completion(&self, messages: Vec<Message>) -> InferenceResult<Completion> {
        self.channel.run_completion(messages)
    }

    fn model_name(&self) -> &'static str {
        self.model_name
    }

    fn model_cli_param_name(&self) -> &'static str {
        self.model_cli_param_name
    }

    fn model_creation_date(&self) -> &'static str {
        self.model_creation_date
    }

    fn owned_by(&self) -> &'static str {
        self.owned_by
    }

    fn create_cli_flags_fn(&self) -> CreateCliFlagsFn {
        self.create_cli_flags_fn
    }

    fn clear_state(&self) -> InferenceResult<()> {
        self.channel.clear_state()
    }
}
