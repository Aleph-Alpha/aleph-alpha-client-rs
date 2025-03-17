use core::str;
use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::{
    logprobs::{Logprob, Logprobs},
    Stopping, StreamTask, Task,
};

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

    /// Sets the logprobs attribute of this TaskChat.
    pub fn with_logprobs(mut self, logprobs: Logprobs) -> Self {
        self.logprobs = logprobs;
        self
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

#[derive(Debug, PartialEq, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Debug, PartialEq)]
pub struct ChatOutput {
    pub message: Message<'static>,
    pub finish_reason: String,
    /// Contains the logprobs for the sampled and top n tokens, given that [`crate::Logprobs`] has
    /// been set to [`crate::Logprobs::Sampled`] or [`crate::Logprobs::Top`].
    pub logprobs: Vec<Distribution>,
    pub usage: Usage,
}

impl ChatOutput {
    pub fn new(
        message: Message<'static>,
        finish_reason: String,
        logprobs: Vec<Distribution>,
        usage: Usage,
    ) -> Self {
        Self {
            message,
            finish_reason,
            logprobs,
            usage,
        }
    }
}

#[derive(Deserialize, Debug, PartialEq)]
pub struct ResponseChoice {
    pub message: Message<'static>,
    pub finish_reason: String,
    pub logprobs: Option<LogprobContent>,
}

#[derive(Deserialize, Debug, PartialEq, Default)]
pub struct LogprobContent {
    content: Vec<Distribution>,
}

/// Logprob information for a single token
#[derive(Deserialize, Debug, PartialEq)]
pub struct Distribution {
    // Logarithmic probability of the token returned in the completion
    #[serde(flatten)]
    pub sampled: Logprob,
    // Logarithmic probabilities of the most probable tokens, filled if user has requested [`crate::Logprobs::Top`]
    #[serde(rename = "top_logprobs")]
    pub top: Vec<Logprob>,
}

#[derive(Deserialize, Debug, PartialEq)]
pub struct ChatResponse {
    choices: Vec<ResponseChoice>,
    usage: Usage,
}

/// Additional options to affect the streaming behavior.
#[derive(Serialize)]
struct StreamOptions {
    /// If set, an additional chunk will be streamed before the data: [DONE] message.
    /// The usage field on this chunk shows the token usage statistics for the entire request,
    /// and the choices field will always be an empty array.
    include_usage: bool,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
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
            stream_options: None,
        }
    }

    pub fn with_streaming(mut self) -> Self {
        self.stream = true;
        // Always set the `include_usage` to true, as currently we have not seen a
        // case where this information might hurt.
        self.stream_options = Some(StreamOptions {
            include_usage: true,
        });
        self
    }
}

impl Task for TaskChat<'_> {
    type Output = ChatOutput;

    type ResponseBody = ChatResponse;

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
        let ResponseChoice {
            message,
            finish_reason,
            logprobs,
        } = response.choices.pop().unwrap();
        ChatOutput::new(
            message,
            finish_reason,
            logprobs.unwrap_or_default().content,
            response.usage,
        )
    }
}

#[derive(Debug, Deserialize)]
pub struct StreamMessage {
    /// The role of the current chat completion. Will be assistant for the first chunk of every
    /// completion stream and missing for the remaining chunks.
    pub role: Option<String>,
    /// The content of the current chat completion. Will be empty for the first chunk of every
    /// completion stream and non-empty for the remaining chunks.
    pub content: String,
}

/// One chunk of a chat completion stream.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum DeserializedChatChunk {
    Delta {
        /// Chat completion chunk generated by the model when streaming is enabled.
        delta: StreamMessage,
        logprobs: Option<LogprobContent>,
    },
    /// The last chunk of a chat completion stream.
    Finished {
        /// The reason the model stopped generating tokens.
        finish_reason: String,
    },
}

/// Response received from a chat completion stream.
/// Will either have Some(Usage) or choices of length 1.
///
/// While we could deserialize directly into an enum, deserializing into a struct and
/// only having the enum on the output type seems to be the simpler solution.
#[derive(Deserialize)]
pub struct StreamChatResponse {
    pub choices: Vec<DeserializedChatChunk>,
    pub usage: Option<Usage>,
}

#[derive(Debug)]
pub enum ChatEvent {
    Delta {
        /// Chat completion chunk generated by the model when streaming is enabled.
        /// The role is always "assistant".
        content: String,
        /// Log probabilities of the completion tokens if requested via logprobs parameter in request.
        logprobs: Vec<Distribution>,
    },
    /// The last chunk of a chat completion stream.
    Finished {
        /// The reason the model stopped generating tokens.
        reason: String,
    },
    /// Summary of the chat completion stream.
    Summary { usage: Usage },
}

impl StreamTask for TaskChat<'_> {
    type Output = ChatEvent;

    type ResponseBody = StreamChatResponse;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = ChatBody::new(model, self).with_streaming();
        client.post(format!("{base}/chat/completions")).json(&body)
    }

    fn body_to_output(&self, mut response: Self::ResponseBody) -> Option<Self::Output> {
        if let Some(usage) = response.usage {
            Some(ChatEvent::Summary { usage })
        } else {
            // We always expect there to be exactly one choice, as the `n` parameter is not
            // supported by this crate.
            let chunk = response
                .choices
                .pop()
                .expect("There must always be at least one choice");

            match chunk {
                // Skip the role message
                DeserializedChatChunk::Delta {
                    delta: StreamMessage { role: Some(_), .. },
                    ..
                } => None,
                DeserializedChatChunk::Delta {
                    delta:
                        StreamMessage {
                            role: None,
                            content,
                        },
                    logprobs,
                } => Some(ChatEvent::Delta {
                    content,
                    logprobs: logprobs.unwrap_or_default().content,
                }),
                DeserializedChatChunk::Finished { finish_reason } => Some(ChatEvent::Finished {
                    reason: finish_reason,
                }),
            }
        }
    }
}

impl Logprobs {
    /// Representation for serialization in request body, for `logprobs` parameter
    pub fn logprobs(self) -> bool {
        match self {
            Logprobs::No => false,
            Logprobs::Sampled | Logprobs::Top(_) => true,
        }
    }

    /// Representation for serialization in request body, for `top_logprobs` parameter
    pub fn top_logprobs(self) -> Option<u8> {
        match self {
            Logprobs::No | Logprobs::Sampled => None,
            Logprobs::Top(n) => Some(n),
        }
    }
}
