use std::marker::PhantomData;

#[cfg(not(feature = "legacy-v018"))]
use burn::prelude::Backend;

use crate::{
    channels::InferenceChannel,
    errors::InferenceResult,
    plugin::{CreateCliFlagsFn, InferencePlugin},
    server::InferenceServer,
    InferenceJob, Stats,
};

#[derive(Debug, Clone)]
pub struct InferenceClient<Server: InferenceServer + 'static, Channel: 'static> {
    model_name: &'static str,
    model_cli_param_name: &'static str,
    model_creation_date: &'static str,
    created_by: &'static str,
    create_cli_flags_fn: CreateCliFlagsFn,
    channel: Channel,
    _phantom_server: PhantomData<Server>,
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
        created_by: &'static str,
        create_cli_flags_fn: CreateCliFlagsFn,
        channel: Channel,
    ) -> Self {
        Self {
            model_name,
            model_cli_param_name,
            model_creation_date,
            created_by,
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
        let result = self.channel.deleter();

        #[cfg(not(feature = "legacy-v018"))]
        let device = &crate::INFERENCE_DEVICE;

        #[cfg(not(feature = "legacy-v018"))]
        <crate::InferenceBackend as Backend>::memory_cleanup(device);

        result
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
        let result = self.channel.unload();

        #[cfg(not(feature = "legacy-v018"))]
        {
            let device = &crate::INFERENCE_DEVICE;
            <crate::InferenceBackend as Backend>::memory_cleanup(device);
            // Force pending deallocations to complete
            <crate::InferenceBackend as Backend>::sync(device);
        }

        result
    }

    fn run_job(&self, job: InferenceJob) -> InferenceResult<Stats> {
        self.channel.run_job(job)
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

    fn created_by(&self) -> &'static str {
        self.created_by
    }

    fn create_cli_flags_fn(&self) -> CreateCliFlagsFn {
        self.create_cli_flags_fn
    }

    fn clear_state(&self) -> InferenceResult<()> {
        self.channel.clear_state()
    }
}
