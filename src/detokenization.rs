use crate::Task;
use serde::{Deserialize, Serialize};

/// Input for a [crate::Client::detokenize] request.
pub struct TaskDetokenization<'a> {
    /// List of token ids which should be detokenized into text.
    pub token_ids: &'a [u32],
}

/// Body send to the Aleph Alpha API on the POST `/detokenize` route
#[derive(Serialize, Debug)]
struct BodyDetokenization<'a> {
    /// Name of the model tasked with completing the prompt. E.g. `luminous-base"`.
    pub model: &'a str,
    /// List of ids to detokenize.
    pub token_ids: &'a [u32],
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
pub struct ResponseDetokenization {
    pub result: String,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DetokenizationOutput {
    pub result: String,
}

impl From<ResponseDetokenization> for DetokenizationOutput {
    fn from(response: ResponseDetokenization) -> Self {
        Self {
            result: response.result,
        }
    }
}

impl<'a> Task for TaskDetokenization<'a> {
    type Output = DetokenizationOutput;
    type ResponseBody = ResponseDetokenization;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = BodyDetokenization {
            model,
            token_ids: &self.token_ids,
        };
        client.post(format!("{base}/detokenize")).json(&body)
    }

    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output {
        DetokenizationOutput::from(response)
    }
}
