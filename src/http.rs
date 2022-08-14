use std::borrow::Cow;

use reqwest::{header, ClientBuilder, RequestBuilder, StatusCode};
use serde::Deserialize;
use thiserror::Error as ThisError;

use crate::{completion::Completion, TaskCompletion};

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

pub trait Task {
    type Output;
    type ResponseBody: for<'de> Deserialize<'de>;

    fn build_request(&self, client: &reqwest::Client, base: &str, model: &str) -> RequestBuilder;

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

    #[deprecated="Use execute instead"]
    pub async fn complete(
        &self,
        model: &str,
        task: &TaskCompletion<'_>,
    ) -> Result<Completion, Error> {
        self.execute(model, task).await
    }

    pub async fn execute<T: Task>(&self, model: &str, task: &T) -> Result<T::Output, Error> {
        let response = task
            .build_request(&self.http, &self.base, model)
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
