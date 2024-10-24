use serde::{Deserialize, Serialize};

use crate::{http::Task, Prompt, TaskStreamCompletion};

/// Completes a prompt. E.g. continues a text.
pub struct TaskCompletion<'a> {
    /// The prompt (usually text) to be completed. Unconditional completion can be started with an
    /// empty string. The prompt may contain a zero shot or few shot task.
    pub prompt: Prompt<'a>,
    /// Controls in which circumstances the model will stop generating new tokens.
    pub stopping: Stopping<'a>,
    /// Sampling controls how the tokens ("words") are selected for the completion.
    pub sampling: Sampling<'a>,
}

impl<'a> TaskCompletion<'a> {
    /// Convenience constructor leaving most setting to default, just completing a given text
    pub fn from_text(text: &'a str) -> Self {
        TaskCompletion {
            prompt: Prompt::from_text(text),
            stopping: Stopping::NO_TOKEN_LIMIT,
            sampling: Sampling::MOST_LIKELY,
        }
    }

    pub fn with_maximum_tokens(mut self, maximum_tokens: u32) -> Self {
        self.stopping.maximum_tokens = Some(maximum_tokens);
        self
    }

    pub fn with_stop_sequences(mut self, stop_sequences: &'a [&str]) -> Self {
        self.stopping.stop_sequences = stop_sequences;
        self
    }
    pub fn with_streaming(self) -> TaskStreamCompletion<'a> {
        TaskStreamCompletion { task: self }
    }
}

/// Sampling controls how the tokens ("words") are selected for the completion.
pub struct Sampling<'a> {
    /// A temperature encourages the model to produce less probable outputs ("be more creative").
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
    /// Only start the completion with one of the following strings. The model will sample
    /// between these options, and ignore anything else. Once one of the options is generated,
    /// then the model will continue sampling until one of the stop reasons is reached.
    ///
    /// For example, if trying to get the model to answer "Yes" or "No", and your prompt was
    /// "Can this question be answered?" this could be set to `[" Yes", " No"]`. Note the
    /// space in front of each option, since the model would start with a space character.
    pub start_with_one_of: &'a [&'a str],
}

impl Sampling<'_> {
    /// Always chooses the token most likely to come next. Choose this if you do want close to
    /// deterministic behaviour and do not want to apply any penalties to avoid repetitions.
    pub const MOST_LIKELY: Self = Sampling {
        temperature: None,
        top_k: None,
        top_p: None,
        start_with_one_of: &[],
    };
}

impl Default for Sampling<'_> {
    fn default() -> Self {
        Self::MOST_LIKELY
    }
}

/// Controls the conditions under which the language models stops generating text.
pub struct Stopping<'a> {
    /// The maximum number of tokens to be generated. Completion will terminate after the maximum
    /// number of tokens is reached. Increase this value to allow for longer outputs. A text is split
    /// into tokens. Usually there are more tokens than words. The total number of tokens of prompt
    /// and maximum_tokens depends on the model.
    /// If maximum tokens is set to None, no outside limit is opposed on the number of maximum tokens.
    /// The model will generate tokens until it generates one of the specified stop_sequences or it
    /// reaches its technical limit, which usually is its context window.
    pub maximum_tokens: Option<u32>,
    /// List of strings which will stop generation if they are generated. Stop sequences are
    /// helpful in structured texts. E.g.: In a question answering scenario a text may consist of
    /// lines starting with either "Question: " or "Answer: " (alternating). After producing an
    /// answer, the model will be likely to generate "Question: ". "Question: " may therefore be used
    /// as stop sequence in order not to have the model generate more questions but rather restrict
    /// text generation to the answers.
    pub stop_sequences: &'a [&'a str],
}

impl<'a> Stopping<'a> {
    /// Only stop once the model reaches its technical limit, usually the context window.
    pub const NO_TOKEN_LIMIT: Self = Stopping {
        maximum_tokens: None,
        stop_sequences: &[],
    };

    /// Stop once the model has reached maximum_tokens.
    pub fn from_maximum_tokens(maximum_tokens: u32) -> Self {
        Self {
            maximum_tokens: Some(maximum_tokens),
            stop_sequences: &[],
        }
    }
}

impl Default for Stopping<'_> {
    fn default() -> Self {
        Self::NO_TOKEN_LIMIT
    }
}

/// Body send to the Aleph Alpha API on the POST `/completion` Route
#[derive(Serialize, Debug)]
pub struct BodyCompletion<'a> {
    /// Name of the model tasked with completing the prompt. E.g. `luminous-base"`.
    pub model: &'a str,
    /// Prompt to complete. The modalities supported depend on `model`.
    pub prompt: Prompt<'a>,
    /// Limits the number of tokens, which are generated for the completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_tokens: Option<u32>,
    /// List of strings which will stop generation if they are generated. Stop sequences are
    /// helpful in structured texts. E.g.: In a question answering scenario a text may consist of
    /// lines starting with either "Question: " or "Answer: " (alternating). After producing an
    /// answer, the model will be likely to generate "Question: ". "Question: " may therefore be used
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
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub completion_bias_inclusion: &'a [&'a str],
    /// If true, the response will be streamed.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
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
            completion_bias_inclusion: task.sampling.start_with_one_of,
            stream: false,
        }
    }
    pub fn with_streaming(mut self) -> Self {
        self.stream = true;
        self
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

    fn body_to_output(mut response: Self::ResponseBody) -> Self::Output {
        response.completions.pop().unwrap()
    }
}
