use std::borrow::Cow;

use reqwest::{header, ClientBuilder, StatusCode};
use serde::{Deserialize, Serialize};
use thiserror::Error as ThisError;

use crate::{Authentication, Prompt, TaskCompletion};

#[derive(ThisError, Debug)]
pub enum Error {
    /// User exceeds his current Task Quota.
    #[error(
        "You are trying to calculate more tasks in parallel, than you currently can, given how busy
        the API is. Retry the operation again, once one of your other tasks is finished."
    )]
    TooManyTasks,
    /// User exceeds his current Task Quota.
    #[error(
        "You are trying to send too many requests to the API in to short an interval. Slow down a
        bit, otherwise these error will persist. Sorry for this, but we try to prevent DOS attacks."
    )]
    TooManyRequests,
    /// An error on the Http Protocl level.
    #[error("HTTP request failed with status code {}. Body:\n{}", status, body)]
    Http { status: u16, body: String },
    /// Most likely either TLS errors creating the Client, or IO errors.
    #[error(transparent)]
    Other(#[from] reqwest::Error),
}

/// Sends HTTP request to the Aleph Alpha API
pub struct Client {
    base: String,
    http: reqwest::Client,
}

impl Client {
    /// A new instance of an Aleph Alpha client helping you interact with the Aleph Alpha API.
    pub async fn new(auth: Authentication<'_>) -> Result<Self, Error> {
        Self::with_base_url("api.aleph-alpha.com".to_owned(), auth).await
    }

    /// In production you typically would want set this to <https://api.aleph-alpha.com>. Yet you
    /// may want to use a different instances for testing.
    pub async fn with_base_url(host: String, auth: Authentication<'_>) -> Result<Self, Error> {
        let token = auth.api_token(&host).await?;

        let mut headers = header::HeaderMap::new();

        let mut auth_value = header::HeaderValue::from_str(&format!("Bearer {}", token)).unwrap();
        // Consider marking security-sensitive headers with `set_sensitive`.
        auth_value.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, auth_value);

        let http = ClientBuilder::new().default_headers(headers).build()?;

        Ok(Self { base: host, http })
    }



    pub async fn complete(
        &self,
        model: &str,
        task: &TaskCompletion<'_>,
    ) -> Result<Completion, Error> {
        let body = BodyCompletion::new(model, task);
        let response = self
            .http
            .post(format!("{}/complete", self.base))
            .json(&body)
            .send()
            .await?;

        let response = translate_http_error(response).await?;

        let mut answer: ResponseCompletion = response.json().await?;
        Ok(answer.completions.pop().unwrap())
    }
}

async fn translate_http_error(response: reqwest::Response) -> Result<reqwest::Response, Error> {
    let status = response.status();
    if !status.is_success() {
        // Store body in a variable, so we can use it, even if it is not an Error emmitted by
        // the API, but an intermediate Proxy like NGinx, so we can still forward the error
        // message.
        let body = response.text().await?;
        if status == StatusCode::TOO_MANY_REQUESTS {
            // Distinguish between request rate and task quota limiting.

            // For this error to be quota related it must be an API error with status
            // TOO_MANY_TASKS
            if let Some("TOO_MANY_TASKS") = serde_json::from_str::<ApiError>(&body)
                .ok()
                .map(|api_error| api_error.code)
                .as_deref()
            {
                return Err(Error::TooManyTasks);
            } else {
                return Err(Error::TooManyRequests);
            }
        } else {
            // It's a generic Http error
            return Err(Error::Http {
                status: status.as_u16(),
                body,
            });
        }
    }
    Ok(response)
}

/// We are only interessted in the status codes of the API.
#[derive(Deserialize)]
struct ApiError<'a> {
    /// Unique string in capital letters emitted by the API to signal different kinds of errors in a
    /// finer granualrity then the HTTP status codes alone would allow for.
    ///
    /// E.g. Differentiating between request rate limiting and parallel tasks limiting which both
    /// are 429 (the former is emmited by NGinx though).
    code: Cow<'a, str>,
}

/// Body send to the Aleph Alpha API on the POST `/completion` Route
#[derive(Serialize, Debug)]
struct BodyCompletion<'a> {
    /// Name of the model tasked with completing the prompt. E.g. `luminus-base`.
    pub model: &'a str,
    /// Prompt to complete. The modalities supported depend on `model`.
    pub prompt: Prompt<'a>,
    /// Limits the number of tokens, which are generated for the completion.
    pub maximum_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

impl<'a> BodyCompletion<'a> {
    pub fn new(model: &'a str, task: &TaskCompletion<'a>) -> Self {
        Self {
            model,
            prompt: task.prompt,
            maximum_tokens: task.maximum_tokens,
            temperature: task.sampling.temperature(),
            top_k: task.sampling.top_k(),
            top_p: task.sampling.top_p(),
        }
    }
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
pub struct ResponseCompletion {
    pub model_version: String,
    pub completions: Vec<Completion>,
}

impl ResponseCompletion {
    /// The best completion in the answer.
    pub fn best(&self) -> &Completion {
        self.completions
            .first()
            .expect("Response is assumed to always have at least one completion")
    }

    /// Text of the best completion.
    pub fn best_text(&self) -> &str {
        &self.best().completion
    }
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
pub struct Completion {
    pub completion: String,
    pub finish_reason: String,
}
