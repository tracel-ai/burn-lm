use std::marker::PhantomData;

use crate::{
    channels::InferenceChannel,
    errors::InferenceResult,
    message::Message,
    plugin::{CreateCliFlagsFn, InferencePlugin},
    server::InferenceServer,
    Completion,
};

#[derive(Debug)]
pub struct InferenceClient<Server: InferenceServer, Channel> {
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
    Server: InferenceServer,
    Channel: InferenceChannel<Server>,
{
    fn downloader(&self) -> Option<fn() -> InferenceResult<()>> {
        self.channel.downloader()
    }

    fn is_downloaded(&self) -> bool {
        self.channel.is_downloaded()
    }

    fn parse_cli_config(&self, args: &clap::ArgMatches) {
        self.channel.parse_cli_config(args);
    }

    fn parse_json_config(&self, json: &str) {
        self.channel.parse_json_config(json);
    }

    fn unload(&self) -> InferenceResult<()> {
        self.channel.unload()
    }

    fn complete(&self, messages: Vec<Message>) -> InferenceResult<Completion> {
        self.channel.complete(messages)
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
}
