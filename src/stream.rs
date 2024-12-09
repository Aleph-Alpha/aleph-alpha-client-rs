use reqwest::RequestBuilder;
use serde::Deserialize;

use crate::http::MethodJob;

/// A job send to the Aleph Alpha Api using the http client. A job wraps all the knowledge required
/// for the Aleph Alpha API to specify its result. Notably it includes the model(s) the job is
/// executed on. This allows this trait to hold in the presence of services, which use more than one
/// model and task type to achieve their result. On the other hand a bare [`crate::TaskCompletion`]
/// can not implement this trait directly, since its result would depend on what model is chosen to
/// execute it. You can remedy this by turning completion task into a job, calling
/// [`Task::with_model`].
pub trait StreamJob {
    /// Output returned by [`crate::Client::output_of`]
    type Output: Send;

    /// Expected answer of the Aleph Alpha API
    type ResponseBody: for<'de> Deserialize<'de> + Send;

    /// Prepare the request for the Aleph Alpha API. Authentication headers can be assumed to be
    /// already set.
    fn build_request(&self, client: &reqwest::Client, base: &str) -> RequestBuilder;

    /// Parses the response of the server into higher level structs for the user.
    fn body_to_output(response: Self::ResponseBody) -> Self::Output;
}

/// A task send to the Aleph Alpha Api using the http client. Requires to specify a model before it
/// can be executed. Will return a stream of results.
pub trait StreamTask {
    /// Output returned by [`crate::Client::output_of`]
    type Output: Send;

    /// Expected answer of the Aleph Alpha API
    type ResponseBody: for<'de> Deserialize<'de> + Send;

    /// Prepare the request for the Aleph Alpha API. Authentication headers can be assumed to be
    /// already set.
    fn build_request(&self, client: &reqwest::Client, base: &str, model: &str) -> RequestBuilder;

    /// Parses the response of the server into higher level structs for the user.
    fn body_to_output(response: Self::ResponseBody) -> Self::Output;

    /// Turn your task into [`Job`] by annotating it with a model name.
    fn with_model<'a>(&'a self, model: &'a str) -> MethodJob<'a, Self>
    where
        Self: Sized,
    {
        MethodJob { model, task: self }
    }
}

impl<T> StreamJob for MethodJob<'_, T>
where
    T: StreamTask,
{
    type Output = T::Output;

    type ResponseBody = T::ResponseBody;

    fn build_request(&self, client: &reqwest::Client, base: &str) -> RequestBuilder {
        self.task.build_request(client, base, self.model)
    }

    fn body_to_output(response: T::ResponseBody) -> T::Output {
        T::body_to_output(response)
    }
}
