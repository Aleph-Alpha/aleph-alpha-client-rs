use serde::Serialize;

mod http;

pub use self::http::{Client, Completion, Error, ResponseCompletion};

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
    /// Always chooses the token most likely to come next.
    pub const MOST_LIKELY: Self = Sampling {
        temperature: None,
        top_k: None,
        top_p: None,
    };
}

/// Completes a prompt. E.g. continues a text.
pub struct TaskCompletion<'a> {
    /// The prompt (usually text) to be completed. Unconditional completion can be started with an
    /// empty string. The prompt may contain a zero shot or few shot task.
    pub prompt: Prompt<'a>,
    /// The maximum number of tokens to be generated. Completion will terminate after the maximum
    /// number of tokens is reachedIncrease this value to allow for longer outputs. A text is split
    /// into tokens. Usually there are more tokens than words. The total number of tokens of prompt
    /// and maximum_tokens depends on the model.
    pub maximum_tokens: u32,
    /// Sampling controls how the tokens ("words") are selected for the completion.
    pub sampling: Sampling,
}

/// A prompt which is passed to the model for inference. Usually it is one text item, but it could
/// also be a combination of several modalities like images and text.
#[derive(Serialize, Debug, Clone, Copy)]
pub struct Prompt<'a>([Modality<'a>; 1]);

impl<'a> Prompt<'a> {
    /// Create a prompt from a single text item.
    pub fn from_text(text: &'a str) -> Self {
        Self([Modality::from_text(text)])
    }
}

/// The prompt for models can be a combination of different modalities (Text and Image). The type of
/// modalities which are supported depend on the Model in question.
#[derive(Serialize, Debug, Clone, Copy)]
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
