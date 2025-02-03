use serde::{Deserialize, Serialize};

use crate::{http::Task, Prompt, StreamTask};

/// Completes a prompt. E.g. continues a text.
pub struct TaskCompletion<'a> {
    /// The prompt (usually text) to be completed. Unconditional completion can be started with an
    /// empty string. The prompt may contain a zero shot or few shot task.
    pub prompt: Prompt<'a>,
    /// Controls in which circumstances the model will stop generating new tokens.
    pub stopping: Stopping<'a>,
    /// Sampling controls how the tokens ("words") are selected for the completion.
    pub sampling: Sampling,
    /// Whether to include special tokens (e.g. <|endoftext|>, <|python_tag|>) in the completion.
    pub special_tokens: bool,
}

impl<'a> TaskCompletion<'a> {
    /// Convenience constructor leaving most setting to default, just completing a given text
    pub fn from_text(text: &'a str) -> Self {
        TaskCompletion {
            prompt: Prompt::from_text(text),
            stopping: Stopping::NO_TOKEN_LIMIT,
            sampling: Sampling::MOST_LIKELY,
            special_tokens: false,
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

    /// Include special tokens (e.g. <|endoftext|>, <|python_tag|>) in the completion.
    pub fn with_special_tokens(mut self) -> Self {
        self.special_tokens = true;
        self
    }
}

/// Sampling controls how the tokens ("words") are selected for the completion.
pub struct Sampling {
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
    /// When specified, this number will decrease (or increase) the likelihood of repeating tokens
    /// that were mentioned prior in the completion. The penalty is cumulative. The more a token
    /// is mentioned in the completion, the more its probability will decrease.
    /// A negative value will increase the likelihood of repeating tokens.
    pub frequency_penalty: Option<f64>,
}

impl Sampling {
    /// Always chooses the token most likely to come next. Choose this if you do want close to
    /// deterministic behaviour and do not want to apply any penalties to avoid repetitions.
    pub const MOST_LIKELY: Self = Sampling {
        temperature: None,
        top_k: None,
        top_p: None,
        frequency_penalty: None,
    };
}

impl Default for Sampling {
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

impl Stopping<'_> {
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
struct BodyCompletion<'a> {
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
    /// If true, the response will be streamed.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// Forces the raw completion of the model to be returned.
    /// For some models, the completion that was generated by the model may be optimized and
    /// returned in the completion field of the CompletionResponse.
    /// The raw completion, if returned, will contain the un-optimized completion.
    /// Setting tokens to true or log_probs to any value will also trigger the raw completion to be returned.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub raw_completion: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
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
            stream: false,
            raw_completion: task.special_tokens,
            frequency_penalty: task.sampling.frequency_penalty,
        }
    }
    pub fn with_streaming(mut self) -> Self {
        self.stream = true;
        self
    }
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
pub struct ResponseCompletion {
    model_version: String,
    completions: Vec<DeserializedCompletion>,
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
struct DeserializedCompletion {
    completion: String,
    finish_reason: String,
    raw_completion: Option<String>,
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
        let deserialized = response.completions.pop().unwrap();
        let completion = if self.special_tokens {
            deserialized.raw_completion.unwrap()
        } else {
            deserialized.completion
        };
        CompletionOutput {
            completion,
            finish_reason: deserialized.finish_reason,
        }
    }
}

/// Describes a chunk of a completion stream
#[derive(Deserialize, Debug)]
pub struct StreamChunk {
    /// The index of the stream that this chunk belongs to.
    /// This is relevant if multiple completion streams are requested (see parameter n).
    pub index: u32,
    /// The completion of the stream.
    pub completion: String,
}

/// Denotes the end of a completion stream.
///
/// The index of the stream that is being terminated is not deserialized.
/// It is only relevant if multiple completion streams are requested, (see parameter n),
/// which is not supported by this crate yet.
#[derive(Deserialize)]
pub struct StreamSummary {
    /// Model name and version (if any) of the used model for inference.
    pub model_version: String,
    /// The reason why the model stopped generating new tokens.
    pub finish_reason: String,
}

/// Denotes the end of all completion streams.
#[derive(Deserialize)]
pub struct CompletionSummary {
    /// Number of tokens combined across all completion tasks.
    /// In particular, if you set best_of or n to a number larger than 1 then we report the
    /// combined prompt token count for all best_of or n tasks.
    pub num_tokens_prompt_total: u32,
    /// Number of tokens combined across all completion tasks.
    /// If multiple completions are returned or best_of is set to a value greater than 1 then
    /// this value contains the combined generated token count.
    pub num_tokens_generated: u32,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum CompletionEvent {
    StreamChunk(StreamChunk),
    StreamSummary(StreamSummary),
    CompletionSummary(CompletionSummary),
}

impl StreamTask for TaskCompletion<'_> {
    type Output = CompletionEvent;

    type ResponseBody = CompletionEvent;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = BodyCompletion::new(model, self).with_streaming();
        client.post(format!("{base}/complete")).json(&body)
    }

    fn body_to_output(response: Self::ResponseBody) -> Self::Output {
        response
    }
}
