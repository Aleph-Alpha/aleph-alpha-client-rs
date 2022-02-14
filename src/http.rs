use reqwest::{header, ClientBuilder};
use serde::Serialize;

use crate::TaskCompletion;

use super:: Prompt;

pub type Error = reqwest::Error;

/// Sends HTTP request to the Aleph Alpha API
pub struct Client {
    base: String,
    http: reqwest::Client,
}

impl Client {
    /// In production you typically would want set this to "https://api.aleph-alpha.de". Yet you may
    /// want to use a different instances for testing.
    pub fn with_base_uri(base: String, token: &str) -> Result<Self, Error> {
        let mut headers = header::HeaderMap::new();

        let mut auth_value = header::HeaderValue::from_str(&format!("Bearer {}", token)).unwrap();
        // Consider marking security-sensitive headers with `set_sensitive`.
        auth_value.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, auth_value);

        let http = ClientBuilder::new().default_headers(headers).build()?;

        Ok(Self { base, http })
    }

    pub async fn complete(&self, task: &CompletionBody<'_>) -> Result<String, Error> {
        self.http
            .post(format!("{}/complete", self.base))
            .json(task)
            .send()
            .await?
            .text()
            .await
    }
}

/// Body send to the Aleph Alpha API on the POST `/completion` Route
#[derive(Serialize, Debug)]
pub struct CompletionBody<'a> {
    /// Name of the model tasked with completing the prompt. E.g. `luminus-base`.
    pub model: &'a str,
    /// Prompt to complete. The modalities supported depend on `model`.
    pub prompt: Prompt<'a>,
    /// Limits the number of tokens, which are generated for the completion.
    pub maximum_tokens: u32,
}

impl<'a> CompletionBody<'a> {
    pub fn new(model: &'a str, task: &TaskCompletion<'a>) -> Self {
        Self {
            model,
            prompt: task.prompt,
            maximum_tokens: task.maximum_tokens,
        }
    }
}