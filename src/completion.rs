use serde::{Deserialize, Serialize};

use crate::{http::Task, Prompt};

/// Completes a prompt. E.g. continues a text.
pub struct TaskCompletion<'a> {
    /// The prompt (usually text) to be completed. Unconditional completion can be started with an
    /// empty string. The prompt may contain a zero shot or few shot task.
    pub prompt: Prompt<'a>,
    /// Controls in which circumstances the model will stop generating new tokens.
    pub stopping: Stopping<'a>,
    /// Sampling controls how the tokens ("words") are selected for the completion.
    pub sampling: Sampling,
}

impl<'a> TaskCompletion<'a> {
    /// Convinience constructor leaving most setting to default, just completing a given text and
    /// taking the maximum anticipated length of the completion.
    pub fn from_text(text: &'a str, maximum_tokens: u32) -> Self {
        TaskCompletion {
            prompt: Prompt::from_text(text),
            stopping: Stopping::from_maximum_tokens(maximum_tokens),
            sampling: Sampling::MOST_LIKELY,
        }
    }
}

/// Sampling controls how the tokens ("words") are selected for the completion.
pub struct Sampling {
    /// A temperature encourages teh model to produce less probable outputs ("be more creative").
    /// Values are expected to be between 0 and 1. Try high values for a more random ("creative")
    /// response.
    pub temperature: Option<f64>,
    /// Introduces random sampling for generated tokens by randomly selecting the next token from
    /// the k most likely options. A value larger than 1 encourages the model to be more creative.
    /// Set to 0 to get the same behaviour as `None`.
    pub top_k: Option<u32>,
    /// Introduces random sampling for generated tokens by randomly selecting the next token from
    /// the smallest possible set of tokens whose cumulative probability exceeds the probability
    /// top_p. Set to 0 to get the same behaviour as `None`.
    pub top_p: Option<f64>,
}

impl Sampling {
    /// Always chooses the token most likely to come next. Choose this if you do want close to
    /// deterministic behaviour and do not want to apply any penalties to avoid repetitions.
    pub const MOST_LIKELY: Self = Sampling {
        temperature: None,
        top_k: None,
        top_p: None,
    };
}

/// Controls the conditions under which the language models stops generating text.
pub struct Stopping<'a> {
    /// The maximum number of tokens to be generated. Completion will terminate after the maximum
    /// number of tokens is reached.Increase this value to allow for longer outputs. A text is split
    /// into tokens. Usually there are more tokens than words. The total number of tokens of prompt
    /// and maximum_tokens depends on the model.
    pub maximum_tokens: u32,
    /// List of strings which will stop generation if they are generated. Stop sequences are
    /// helpful in structured texts. E.g.: In a question answering scenario a text may consist of
    /// lines starting with either "Question: " or "Answer: " (alternating). After producing an
    /// answer, the model will be likely to generate "Question: ". "Question: " may therfore be used
    /// as stop sequence in order not to have the model generate more questions but rather restrict
    /// text generation to the answers.
    pub stop_sequences: &'a [&'a str],
}

impl<'a> Stopping<'a> {
    /// Only stop once the model generates end of text, or maximum tokens are reached.
    pub fn from_maximum_tokens(maximum_tokens: u32) -> Self {
        Self { maximum_tokens, stop_sequences: &[]}
    }
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
    /// List of strings which will stop generation if they are generated. Stop sequences are
    /// helpful in structured texts. E.g.: In a question answering scenario a text may consist of
    /// lines starting with either "Question: " or "Answer: " (alternating). After producing an
    /// answer, the model will be likely to generate "Question: ". "Question: " may therfore be used
    /// as stop sequence in order not to have the model generate more questions but rather restrict
    /// text generation to the answers.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub stop_sequences: &'a [&'a str],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

impl<'a> BodyCompletion<'a> {
    pub fn new(model: &'a str, task: &'a TaskCompletion<'a>) -> Self {
        Self {
            model,
            prompt: task.prompt.borrow(),
            maximum_tokens: task.stopping.maximum_tokens,
            stop_sequences: task.stopping.stop_sequences,
            temperature: task.sampling.temperature,
            top_k: task.sampling.top_k,
            top_p: task.sampling.top_p,
        }
    }
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
pub struct ResponseCompletion {
    pub model_version: String,
    pub completions: Vec<CompletionOutput>,
}

impl ResponseCompletion {
    /// The best completion in the answer.
    pub fn best(&self) -> &CompletionOutput {
        self.completions
            .first()
            .expect("Response is assumed to always have at least one completion")
    }

    /// Text of the best completion.
    pub fn best_text(&self) -> &str {
        &self.best().completion
    }
}

/// Completion and metainformation returned by a completion task
#[derive(Deserialize, Debug, PartialEq, Eq)]
pub struct CompletionOutput {
    pub completion: String,
    pub finish_reason: String,
}

impl Task for TaskCompletion<'_> {
    type Output = CompletionOutput;

    type ResponseBody = ResponseCompletion;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = BodyCompletion::new(model, self);
        client.post(format!("{base}/complete")).json(&body)
    }

    fn body_to_output(&self, mut response: Self::ResponseBody) -> Self::Output {
        response.completions.pop().unwrap()
    }
}
