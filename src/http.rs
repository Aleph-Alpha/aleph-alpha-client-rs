use std::{borrow::Cow, time::Duration};

use reqwest::{header, ClientBuilder, RequestBuilder, StatusCode};
use serde::Deserialize;
use thiserror::Error as ThisError;

use crate::How;

/// A job send to the Aleph Alpha Api using the http client. A job wraps all the knowledge required
/// for the Aleph Alpha API to specify its result. Notably it includes the model(s) the job is
/// executed on. This allows this trait to hold in the presence of services, which use more than one
/// model and task type to achieve their result. On the other hand a bare [`crate::TaskCompletion`]
/// can not implement this trait directly, since its result would depend on what model is chosen to
/// execute it. You can remedy this by turning completion task into a job, calling
/// [`Task::with_model`].
pub trait Job {
    /// Output returned by [`crate::Client::output_of`]
    type Output;

    /// Expected answer of the Aleph Alpha API
    type ResponseBody: for<'de> Deserialize<'de>;

    /// Prepare the request for the Aleph Alpha API. Authentication headers can be assumed to be
    /// already set.
    fn build_request(&self, client: &reqwest::Client, base: &str) -> RequestBuilder;

    /// Parses the response of the server into higher level structs for the user.
    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output;
}

/// A task send to the Aleph Alpha Api using the http client. Requires to specify a model before it
/// can be executed.
pub trait Task {
    /// Output returned by [`crate::Client::output_of`]
    type Output;

    /// Expected answer of the Aleph Alpha API
    type ResponseBody: for<'de> Deserialize<'de>;

    /// Prepare the request for the Aleph Alpha API. Authentication headers can be assumed to be
    /// already set.
    fn build_request(&self, client: &reqwest::Client, base: &str, model: &str) -> RequestBuilder;

    /// Parses the response of the server into higher level structs for the user.
    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output;

    /// Turn your task into [`Job`] by annotating it with a model name.
    fn with_model<'a>(&'a self, model: &'a str) -> MethodJob<'a, Self>
    where
        Self: Sized,
    {
        MethodJob { model, task: self }
    }
}

/// Enriches the `Task` to a `Job` by appending the model it should be executed with. Use this as
/// input for [`Client::output_of`].
pub struct MethodJob<'a, T> {
    /// Name of the Aleph Alpha Model. E.g. "luminous-base".
    pub model: &'a str,
    /// Task to be executed against the model.
    pub task: &'a T,
}

impl<'a, T> Job for MethodJob<'a, T>
where
    T: Task,
{
    type Output = T::Output;

    type ResponseBody = T::ResponseBody;

    fn build_request(&self, client: &reqwest::Client, base: &str) -> RequestBuilder {
        self.task.build_request(client, base, self.model)
    }

    fn body_to_output(&self, response: T::ResponseBody) -> T::Output {
        self.task.body_to_output(response)
    }
}

/// Sends HTTP request to the Aleph Alpha API
pub struct HttpClient {
    base: String,
    http: reqwest::Client,
}

impl HttpClient {
    /// In production you typically would want set this to <https://api.aleph-alpha.com>. Yet you
    /// may want to use a different instances for testing.
    pub fn with_base_url(host: String, api_token: &str) -> Result<Self, Error> {
        let mut headers = header::HeaderMap::new();

        let mut auth_value = header::HeaderValue::from_str(&format!("Bearer {api_token}")).unwrap();
        // Consider marking security-sensitive headers with `set_sensitive`.
        auth_value.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, auth_value);

        let http = ClientBuilder::new().default_headers(headers).build()?;

        Ok(Self { base: host, http })
    }

    /// Execute a task with the aleph alpha API and fetch its result.
    ///
    /// ```no_run
    /// use aleph_alpha_client::{Client, How, TaskCompletion, Task, Error};
    ///
    /// async fn print_completion() -> Result<(), Error> {
    ///     // Authenticate against API. Fetches token.
    ///     let client = Client::new("AA_API_TOKEN")?;
    ///
    ///     // Name of the model we we want to use. Large models give usually better answer, but are
    ///     // also slower and more costly.
    ///     let model = "luminous-base";
    ///
    ///     // The task we want to perform. Here we want to continue the sentence: "An apple a day
    ///     // ..."
    ///     let task = TaskCompletion::from_text("An apple a day", 10);
    ///
    ///     // Retrieve answer from API
    ///     let response = client.output_of(&task.with_model(model), &How::default()).await?;
    ///
    ///     // Print entire sentence with completion
    ///     println!("An apple a day{}", response.completion);
    ///     Ok(())
    /// }
    /// ```
    pub async fn output_of<T: Job>(&self, task: &T, how: &How) -> Result<T::Output, Error> {
        let query = if how.be_nice {
            [("nice", "true")].as_slice()
        } else {
            // nice=false is default, so we just omit it.
            [].as_slice()
        };
        let response = task
            .build_request(&self.http, &self.base)
            .query(query)
            .timeout(how.client_timeout)
            .send()
            .await
            .map_err(|reqwest_error| {
                if reqwest_error.is_timeout() {
                    Error::ClientTimeout(how.client_timeout)
                } else {
                    reqwest_error.into()
                }
            })?;
        let response = translate_http_error(response).await?;
        let response_body: T::ResponseBody = response.json().await?;
        let answer = task.body_to_output(response_body);
        Ok(answer)
    }
}

async fn translate_http_error(response: reqwest::Response) -> Result<reqwest::Response, Error> {
    let status = response.status();
    if !status.is_success() {
        // Store body in a variable, so we can use it, even if it is not an Error emitted by
        // the API, but an intermediate Proxy like NGinx, so we can still forward the error
        // message.
        let body = response.text().await?;
        // If the response is an error emitted by the API, this deserialization should succeed.
        let api_error: Result<ApiError, _> = serde_json::from_str(&body);
        let translated_error = match status {
            StatusCode::TOO_MANY_REQUESTS => Error::TooManyRequests,
            StatusCode::SERVICE_UNAVAILABLE => {
                // Presence of `api_error` implies the error originated from the API itself (rather
                // than the intermediate proxy) and so we can decode it as such.
                if api_error.is_ok_and(|error| error.code == "QUEUE_FULL") {
                    Error::Busy
                } else {
                    Error::Unavailable
                }
            }
            _ => Error::Http {
                status: status.as_u16(),
                body,
            },
        };
        Err(translated_error)
    } else {
        Ok(response)
    }
}

/// We are only interested in the status codes of the API.
#[derive(Deserialize, Debug)]
struct ApiError<'a> {
    /// Unique string in capital letters emitted by the API to signal different kinds of errors in a
    /// finer granularity then the HTTP status codes alone would allow for.
    ///
    /// E.g. Differentiating between request rate limiting and parallel tasks limiting which both
    /// are 429 (the former is emitted by NGinx though).
    code: Cow<'a, str>,
}

/// Errors returned by the Aleph Alpha Client
#[derive(ThisError, Debug)]
pub enum Error {
    /// User exceeds his current Task Quota.
    #[error(
        "You are trying to send too many requests to the API in to short an interval. Slow down a \
        bit, otherwise these error will persist. Sorry for this, but we try to prevent DOS attacks."
    )]
    TooManyRequests,
    /// Model is busy. Most likely due to many other users requesting its services right now.
    #[error(
        "Sorry the request to the Aleph Alpha API has been rejected due to the requested model \
        being very busy at the moment. We found it unlikely that your request would finish in a \
        reasonable timeframe, so it was rejected right away, rather than make you wait. You are \
        welcome to retry your request any time."
    )]
    Busy,
    /// The API itself is unavailable, most likely due to restart.
    #[error(
        "The service is currently unavailable. This is likely due to restart. Please try again \
        later."
    )]
    Unavailable,
    #[error("No response received within given timeout: {0:?}")]
    ClientTimeout(Duration),
    /// An error on the Http Protocol level.
    #[error("HTTP request failed with status code {}. Body:\n{}", status, body)]
    Http { status: u16, body: String },
    /// Most likely either TLS errors creating the Client, or IO errors.
    #[error(transparent)]
    Other(#[from] reqwest::Error),
}
