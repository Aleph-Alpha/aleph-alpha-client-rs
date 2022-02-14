use serde::Serialize;

mod http;

pub use self::http::{Client, Error, CompletionBody};

/// Completes a prompt. E.g. continues a text.
pub struct TaskCompletion<'a> {
    /// The prompt (usually text) to be completed. Unconditional completion can be started with an
    /// empty string. The prompt may contain a zero shot or few shot task.
    pub prompt: Prompt<'a>,
    /// The maximum number of tokens to be generated. Completion will terminate after the maximum
    /// number of tokens is reached. Increase this value to generate longer texts. A text is split
    /// into tokens.  Usually there are more tokens than words. The total number of tokens of prompt
    /// and maximum_tokens depends on the model.
    pub maximum_tokens: u32,
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
