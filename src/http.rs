use std::borrow::Cow;

use reqwest::{header, ClientBuilder, RequestBuilder, StatusCode};
use serde::Deserialize;
use thiserror::Error as ThisError;

use crate::How;

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
    /// An error on the Http Protocl level.
    #[error("HTTP request failed with status code {}. Body:\n{}", status, body)]
    Http { status: u16, body: String },
    /// Most likely either TLS errors creating the Client, or IO errors.
    #[error(transparent)]
    Other(#[from] reqwest::Error),
}

/// A task send to the Aleph Alpha Api using the http client.
pub trait Task {
    /// Output returned by [`Client::execute`]
    type Output;

    /// Expected answer of the Aleph Alpha API
    type ResponseBody: for<'de> Deserialize<'de>;

    /// Prepare the request for the Aleph Alpha API. Authentication headers can be assumed to be
    /// already set.
    fn build_request(&self, client: &reqwest::Client, base: &str, model: &str) -> RequestBuilder;

    /// Parses the response of the server into higher level structs for the user.
    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output;
}

/// Sends HTTP request to the Aleph Alpha API
pub struct Client {
    base: String,
    http: reqwest::Client,
}

impl Client {
    /// A new instance of an Aleph Alpha client helping you interact with the Aleph Alpha API.
    pub fn new(api_token: &str) -> Result<Self, Error> {
        Self::with_base_url("https://api.aleph-alpha.com".to_owned(), api_token)
    }

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
    /// use aleph_alpha_client::{Client, How, TaskCompletion, Error};
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
    ///     let response = client.execute(model, &task, &How::default()).await?;
    /// 
    ///     // Print entire sentence with completion
    ///     println!("An apple a day{}", response.completion);
    ///     Ok(())
    /// }
    /// ```
    pub async fn execute<T: Task>(
        &self,
        model: &str,
        task: &T,
        how: &How,
    ) -> Result<T::Output, Error> {
        let How { be_nice } = how;
        let query = if *be_nice {
            [("nice", "true")].as_slice()
        } else {
            // nice=false is default, so we just ommit it.
            [].as_slice()
        };
        let response = task
            .build_request(&self.http, &self.base, model)
            .query(query)
            .send()
            .await?;
        let response = translate_http_error(response).await?;
        let response_body: T::ResponseBody = response.json().await?;
        let answer = task.body_to_output(response_body);
        Ok(answer)
    }
}

async fn translate_http_error(response: reqwest::Response) -> Result<reqwest::Response, Error> {
    let status = response.status();
    if !status.is_success() {
        // Store body in a variable, so we can use it, even if it is not an Error emmitted by
        // the API, but an intermediate Proxy like NGinx, so we can still forward the error
        // message.
        let body = response.text().await?;
        let translated_error = match status {
            StatusCode::TOO_MANY_REQUESTS => Error::TooManyRequests,
            StatusCode::SERVICE_UNAVAILABLE => Error::Busy,
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

/// We are only interessted in the status codes of the API.
#[derive(Deserialize)]
struct ApiError<'a> {
    /// Unique string in capital letters emitted by the API to signal different kinds of errors in a
    /// finer granualrity then the HTTP status codes alone would allow for.
    ///
    /// E.g. Differentiating between request rate limiting and parallel tasks limiting which both
    /// are 429 (the former is emmited by NGinx though).
    _code: Cow<'a, str>,
}
