use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::{Sampling, Stopping, StreamTask, Task};

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
    pub sampling: Sampling,
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
            sampling: Sampling::default(),
            stopping: Stopping::default(),
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

#[derive(Deserialize, Debug, PartialEq, Eq)]
pub struct ChatOutput {
    pub message: Message<'static>,
    pub finish_reason: String,
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
pub struct ResponseChat {
    pub choices: Vec<ChatOutput>,
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
}

impl<'a> ChatBody<'a> {
    pub fn new(model: &'a str, task: &'a TaskChat) -> Self {
        Self {
            model,
            messages: &task.messages,
            max_tokens: task.stopping.maximum_tokens,
            stop: task.stopping.stop_sequences,
            temperature: task.sampling.temperature,
            top_p: task.sampling.top_p,
            frequency_penalty: task.sampling.frequency_penalty,
            presence_penalty: task.sampling.presence_penalty,
            stream: false,
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
        response.choices.pop().unwrap()
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
