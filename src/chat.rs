use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::{stream::TaskStreamChat, Task};

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
    /// The maximum number of tokens to be generated. Completion will terminate after the maximum
    /// number of tokens is reached. Increase this value to allow for longer outputs. A text is split
    /// into tokens. Usually there are more tokens than words. The total number of tokens of prompt
    /// and maximum_tokens depends on the model.
    /// If maximum tokens is set to None, no outside limit is opposed on the number of maximum tokens.
    /// The model will generate tokens until it generates one of the specified stop_sequences or it
    /// reaches its technical limit, which usually is its context window.
    pub maximum_tokens: Option<u32>,
    /// A temperature encourages the model to produce less probable outputs ("be more creative").
    /// Values are expected to be between 0 and 1. Try high values for a more random ("creative")
    /// response.
    pub temperature: Option<f64>,
    /// Introduces random sampling for generated tokens by randomly selecting the next token from
    /// the smallest possible set of tokens whose cumulative probability exceeds the probability
    /// top_p. Set to 0 to get the same behaviour as `None`.
    pub top_p: Option<f64>,
}

impl<'a> TaskChat<'a> {
    /// Creates a new TaskChat containing one message with the given role and content.
    /// All optional TaskChat attributes are left unset.
    pub fn with_message(message: Message<'a>) -> Self {
        TaskChat {
            messages: vec![message],
            maximum_tokens: None,
            temperature: None,
            top_p: None,
        }
    }

    /// Creates a new TaskChat containing the given messages.
    /// All optional TaskChat attributes are left unset.
    pub fn with_messages(messages: Vec<Message<'a>>) -> Self {
        TaskChat {
            messages,
            maximum_tokens: None,
            temperature: None,
            top_p: None,
        }
    }

    /// Pushes a new Message to this TaskChat.
    pub fn push_message(mut self, message: Message<'a>) -> Self {
        self.messages.push(message);
        self
    }

    /// Sets the maximum token attribute of this TaskChat.
    pub fn with_maximum_tokens(mut self, maximum_tokens: u32) -> Self {
        self.maximum_tokens = Some(maximum_tokens);
        self
    }

    /// Sets the temperature attribute of this TaskChat.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the top_p attribute of this TaskChat.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Creates a wrapper `TaskStreamChat` for this TaskChat.
    pub fn with_streaming(self) -> TaskStreamChat<'a> {
        TaskStreamChat { task: self }
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
pub struct ChatBody<'a> {
    /// Name of the model tasked with completing the prompt. E.g. `luminous-base"`.
    pub model: &'a str,
    /// The list of messages comprising the conversation so far.
    messages: &'a [Message<'a>],
    /// Limits the number of tokens, which are generated for the completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_tokens: Option<u32>,
    /// Controls the randomness of the model. Lower values will make the model more deterministic and higher values will make it more random.
    /// Mathematically, the temperature is used to divide the logits before sampling. A temperature of 0 will always return the most likely token.
    /// When no value is provided, the default value of 1 will be used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// "nucleus" parameter to dynamically adjust the number of choices for each predicted token based on the cumulative probabilities. It specifies a probability threshold, below which all less likely tokens are filtered out.
    /// When no value is provided, the default value of 1 will be used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Whether to stream the response or not.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
}

impl<'a> ChatBody<'a> {
    pub fn new(model: &'a str, task: &'a TaskChat) -> Self {
        Self {
            model,
            messages: &task.messages,
            maximum_tokens: task.maximum_tokens,
            temperature: task.temperature,
            top_p: task.top_p,
            stream: false,
        }
    }

    pub fn with_streaming(mut self) -> Self {
        self.stream = true;
        self
    }
}

impl<'a> Task for TaskChat<'a> {
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

    fn body_to_output(mut response: Self::ResponseBody) -> Self::Output {
        response.choices.pop().unwrap()
    }
}
