use reqwest::{header, ClientBuilder};
use serde::Serialize;

/// Sends HTTP request to the Aleph Alpha API
pub struct Client {
    base: String,
    http: reqwest::Client,
}

impl Client {
    /// In production you typically would want set this to "https://api.aleph-alpha.de". Yet you may
    /// want to use a different instances for testing.
    pub fn with_base_uri(base: String, token: &str) -> Self {
        let mut headers = header::HeaderMap::new();

        let mut auth_value = header::HeaderValue::from_str(&format!("Bearer {}", token)).unwrap();
        // Consider marking security-sensitive headers with `set_sensitive`.
        auth_value.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, auth_value);

        let http = ClientBuilder::new()
            .default_headers(headers)
            .build()
            .unwrap();

        Self { base, http }
    }

    pub async fn complete(&self, task: &CompletionBody<'_>) -> String {
        self.http
            .post(format!("{}/complete", self.base))
            .json(task)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
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

/// A prompt which is passed to the model for inference. Usually it is one text item, but it could
/// also be a combination of several modalities like images and text.
#[derive(Serialize, Debug)]
pub struct Prompt<'a>([Modality<'a>; 1]);

impl<'a> Prompt<'a> {
    /// Create a prompt from a single text item.
    pub fn from_text(text: &'a str) -> Self {
        Self([Modality::from_text(text)])
    }
}

/// The prompt for models can be a combination of different modalities (Text and Image). The type of
/// modalities which are supported depend on the Model in question.
#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Modality<'a> {
    /// The only type of prompt which can be used with pure language models
    Text { data: &'a str },
}

impl<'a> Modality<'a> {
    /// Instantiates a text prompt
    pub fn from_text(text: &'a str) -> Self {
        Modality::Text { data: text }
    }
}