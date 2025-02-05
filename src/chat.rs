use core::str;
use std::{borrow::Cow, str::Utf8Error};

use serde::{Deserialize, Serialize};

use crate::{Stopping, StreamTask, Task};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message<'a> {
    pub role: Cow<'a, str>,
    pub content: Cow<'a, str>,
}

impl<'a> Message<'a> {
    pub fn new(role: impl Into<Cow<'a, str>>, content: impl Into<Cow<'a, str>>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<Cow<'a, str>>) -> Self {
        Self::new("user", content)
    }
    pub fn assistant(content: impl Into<Cow<'a, str>>) -> Self {
        Self::new("assistant", content)
    }
    pub fn system(content: impl Into<Cow<'a, str>>) -> Self {
        Self::new("system", content)
    }
}

pub struct TaskChat<'a> {
    /// The list of messages comprising the conversation so far.
    pub messages: Vec<Message<'a>>,
    /// Controls in which circumstances the model will stop generating new tokens.
    pub stopping: Stopping<'a>,
    /// Sampling controls how the tokens ("words") are selected for the completion.
    pub sampling: ChatSampling,
    /// Use this to control the logarithmic probabilities you want to have returned. This is useful
    /// to figure out how likely it had been that this specific token had been sampled.
    pub logprobs: Logprobs,
}

impl<'a> TaskChat<'a> {
    /// Creates a new TaskChat containing one message with the given role and content.
    /// All optional TaskChat attributes are left unset.
    pub fn with_message(message: Message<'a>) -> Self {
        Self::with_messages(vec![message])
    }

    /// Creates a new TaskChat containing the given messages.
    /// All optional TaskChat attributes are left unset.
    pub fn with_messages(messages: Vec<Message<'a>>) -> Self {
        TaskChat {
            messages,
            sampling: ChatSampling::default(),
            stopping: Stopping::default(),
            logprobs: Logprobs::No,
        }
    }

    /// Pushes a new Message to this TaskChat.
    pub fn push_message(mut self, message: Message<'a>) -> Self {
        self.messages.push(message);
        self
    }

    /// Sets the maximum token attribute of this TaskChat.
    pub fn with_maximum_tokens(mut self, maximum_tokens: u32) -> Self {
        self.stopping.maximum_tokens = Some(maximum_tokens);
        self
    }
}

#[derive(Clone, Copy)]
pub enum Logprobs {
    /// Do not return any logprobs
    No,
    /// Return only the logprob of the tokens which have actually been sampled into the completion.
    Sampled,
    /// Request between 0 and 20 tokens
    Top(u8),
}

impl Logprobs {
    /// Representation for serialization in request body, for `logprobs` parameter
    fn logprobs(self) -> bool {
        match self {
            Logprobs::No => false,
            Logprobs::Sampled | Logprobs::Top(_)=> true,
        }
    }

    /// Representation for serialization in request body, for `top_logprobs` parameter
    fn top_logprobs(self) -> Option<u8> {
        match self {
            Logprobs::No | Logprobs::Sampled => None,
            Logprobs::Top(n) => Some(n)
        }
    }
}

/// Sampling controls how the tokens ("words") are selected for the completion. This is different
/// from [`crate::Sampling`], because it does **not** supprot the `top_k` parameter.
pub struct ChatSampling {
    /// A temperature encourages the model to produce less probable outputs ("be more creative").
    /// Values are expected to be between 0 and 1. Try high values for a more random ("creative")
    /// response.
    pub temperature: Option<f64>,
    /// Introduces random sampling for generated tokens by randomly selecting the next token from
    /// the k most likely options. A value larger than 1 encourages the model to be more creative.
    /// Set to 0 to get the same behaviour as `None`.
    pub top_p: Option<f64>,
    /// When specified, this number will decrease (or increase) the likelihood of repeating tokens
    /// that were mentioned prior in the completion. The penalty is cumulative. The more a token
    /// is mentioned in the completion, the more its probability will decrease.
    /// A negative value will increase the likelihood of repeating tokens.
    pub frequency_penalty: Option<f64>,
    /// The presence penalty reduces the likelihood of generating tokens that are already present
    /// in the generated text (repetition_penalties_include_completion=true) respectively the
    /// prompt (repetition_penalties_include_prompt=true). Presence penalty is independent of the
    /// number of occurrences. Increase the value to reduce the likelihood of repeating text.
    /// An operation like the following is applied:
    ///
    /// logits[t] -> logits[t] - 1 * penalty
    ///
    /// where logits[t] is the logits for any given token. Note that the formula is independent
    /// of the number of times that a token appears.
    pub presence_penalty: Option<f64>,
}

impl ChatSampling {
    /// Always chooses the token most likely to come next. Choose this if you do want close to
    /// deterministic behaviour and do not want to apply any penalties to avoid repetitions.
    pub const MOST_LIKELY: Self = ChatSampling {
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
    };
}

impl Default for ChatSampling {
    fn default() -> Self {
        Self::MOST_LIKELY
    }
}

#[derive(Debug, PartialEq)]
pub struct ChatOutput {
    pub message: Message<'static>,
    pub finish_reason: String,
    /// Contains the logprobs for the sampled and top n tokens, given that [`crate::Logprobs`] has
    /// been set to [`crate::Logprobs::Sampled`] or [`crate::Logprobs::Top`].
    pub logprobs: Vec<Logprob>,
}

#[derive(Deserialize, Debug, PartialEq)]
pub struct ResponseChoice {
    pub message: Message<'static>,
    pub finish_reason: String,
    pub logprobs: Option<LogprobContent>,
}

#[derive(Deserialize, Debug, PartialEq, Default)]
pub struct LogprobContent {
    content: Vec<Logprob>,
}

impl ResponseChoice {
    fn into_chat_output(self) -> ChatOutput {
        let ResponseChoice {
            message,
            finish_reason,
            logprobs,
        } = self;
        ChatOutput {
            message,
            finish_reason,
            logprobs: logprobs.unwrap_or_default().content,
        }
    }
}

/// Logprob information for a single token
#[derive(Deserialize, Debug, PartialEq)]
pub struct Logprob {
    // The API returns both a UTF-8 String token and bytes as an array of numbers. We only
    // deserialize bytes as it is the better source of truth.
    /// Binary represtantation of the token, usually these bytes are UTF-8.
    #[serde(rename = "bytes")]
    pub token: Vec<u8>,
    pub logprob: f64,
    pub top_logprobs: Vec<TopLogprob>
}

impl Logprob {
    pub fn token_as_str(&self) -> Result<&str, Utf8Error> {
        str::from_utf8(&self.token)
    }
}

#[derive(Deserialize, Debug, PartialEq)]
pub struct TopLogprob {
    // The API returns both a UTF-8 String token and bytes as an array of numbers. We only
    // deserialize bytes as it is the better source of truth.
    /// Binary represtantation of the token, usually these bytes are UTF-8.
    #[serde(rename = "bytes")]
    pub token: Vec<u8>,
    pub logprob: f64,
}

impl TopLogprob {
    pub fn token_as_str(&self) -> Result<&str, Utf8Error> {
        str::from_utf8(&self.token)
    }
}

#[derive(Deserialize, Debug, PartialEq)]
pub struct ResponseChat {
    choices: Vec<ResponseChoice>,
}

#[derive(Serialize)]
struct ChatBody<'a> {
    /// Name of the model tasked with completing the prompt. E.g. `luminous-base"`.
    pub model: &'a str,
    /// The list of messages comprising the conversation so far.
    messages: &'a [Message<'a>],
    /// Limits the number of tokens, which are generated for the completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub stop: &'a [&'a str],
    /// Controls the randomness of the model. Lower values will make the model more deterministic and higher values will make it more random.
    /// Mathematically, the temperature is used to divide the logits before sampling. A temperature of 0 will always return the most likely token.
    /// When no value is provided, the default value of 1 will be used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// "nucleus" parameter to dynamically adjust the number of choices for each predicted token based on the cumulative probabilities. It specifies a probability threshold, below which all less likely tokens are filtered out.
    /// When no value is provided, the default value of 1 will be used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Whether to stream the response or not.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub logprobs: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
}

impl<'a> ChatBody<'a> {
    pub fn new(model: &'a str, task: &'a TaskChat) -> Self {
        let TaskChat {
            messages,
            stopping:
                Stopping {
                    maximum_tokens,
                    stop_sequences,
                },
            sampling:
                ChatSampling {
                    temperature,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                },
            logprobs,
        } = task;

        Self {
            model,
            messages,
            max_tokens: *maximum_tokens,
            stop: stop_sequences,
            temperature: *temperature,
            top_p: *top_p,
            frequency_penalty: *frequency_penalty,
            presence_penalty: *presence_penalty,
            stream: false,
            logprobs: logprobs.logprobs(),
            top_logprobs: logprobs.top_logprobs(),
        }
    }

    pub fn with_streaming(mut self) -> Self {
        self.stream = true;
        self
    }
}

impl Task for TaskChat<'_> {
    type Output = ChatOutput;

    type ResponseBody = ResponseChat;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = ChatBody::new(model, self);
        client.post(format!("{base}/chat/completions")).json(&body)
    }

    fn body_to_output(&self, mut response: Self::ResponseBody) -> Self::Output {
        response.choices.pop().unwrap().into_chat_output()
    }
}

#[derive(Deserialize)]
pub struct StreamMessage {
    /// The role of the current chat completion. Will be assistant for the first chunk of every
    /// completion stream and missing for the remaining chunks.
    pub role: Option<String>,
    /// The content of the current chat completion. Will be empty for the first chunk of every
    /// completion stream and non-empty for the remaining chunks.
    pub content: String,
}

/// One chunk of a chat completion stream.
#[derive(Deserialize)]
pub struct ChatStreamChunk {
    /// The reason the model stopped generating tokens.
    /// The value is only set in the last chunk of a completion and null otherwise.
    pub finish_reason: Option<String>,
    /// Chat completion chunk generated by the model when streaming is enabled.
    pub delta: StreamMessage,
}

/// Event received from a chat completion stream. As the crate does not support multiple
/// chat completions, there will always exactly one choice item.
#[derive(Deserialize)]
pub struct ChatEvent {
    pub choices: Vec<ChatStreamChunk>,
}

impl StreamTask for TaskChat<'_> {
    type Output = ChatStreamChunk;

    type ResponseBody = ChatEvent;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = ChatBody::new(model, self).with_streaming();
        client.post(format!("{base}/chat/completions")).json(&body)
    }

    fn body_to_output(mut response: Self::ResponseBody) -> Self::Output {
        // We always expect there to be exactly one choice, as the `n` parameter is not
        // supported by this crate.
        response
            .choices
            .pop()
            .expect("There must always be at least one choice")
    }
}
