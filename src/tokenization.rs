use crate::Task;
use serde::{Deserialize, Serialize};

/// Input for a [crate::Client::tokenize] request.
pub struct TaskTokenization<'a> {
    /// The text prompt which should be converted into tokens
    pub prompt: &'a str,

    /// Specify `true` to return text-tokens.
    pub tokens: bool,

    /// Specify `true` to return numeric token-ids.
    pub token_ids: bool,
}

impl<'a> From<&'a str> for TaskTokenization<'a> {
    fn from(prompt: &'a str) -> TaskTokenization {
        TaskTokenization {
            prompt,
            tokens: true,
            token_ids: true,
        }
    }
}

impl TaskTokenization<'_> {
    pub fn new(prompt: &str, tokens: bool, token_ids: bool) -> TaskTokenization {
        TaskTokenization {
            prompt,
            tokens,
            token_ids,
        }
    }
}

#[derive(Serialize, Debug)]
struct BodyTokenization<'a> {
    /// Name of the model tasked with completing the prompt. E.g. `luminous-base`.
    pub model: &'a str,
    /// String to tokenize.
    pub prompt: &'a str,
    /// Set this value to `true` to return text-tokens.
    pub tokens: bool,
    /// Set this value to `true` to return numeric token-ids.
    pub token_ids: bool,
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
pub struct ResponseTokenization {
    pub tokens: Option<Vec<String>>,
    pub token_ids: Option<Vec<u32>>,
}

#[derive(Debug, PartialEq)]
pub struct TokenizationOutput {
    pub tokens: Option<Vec<String>>,
    pub token_ids: Option<Vec<u32>>,
}

impl From<ResponseTokenization> for TokenizationOutput {
    fn from(response: ResponseTokenization) -> Self {
        Self {
            tokens: response.tokens,
            token_ids: response.token_ids,
        }
    }
}

impl Task for TaskTokenization<'_> {
    type Output = TokenizationOutput;
    type ResponseBody = ResponseTokenization;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = BodyTokenization {
            model,
            prompt: self.prompt,
            tokens: self.tokens,
            token_ids: self.token_ids,
        };
        client.post(format!("{base}/tokenize")).json(&body)
    }

    fn body_to_output(response: Self::ResponseBody) -> Self::Output {
        TokenizationOutput::from(response)
    }
}
