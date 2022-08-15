use serde::Serialize;

mod completion;
mod http;
mod semantic_embedding;

pub use self::{
    completion::{Completion, Sampling, TaskCompletion},
    http::{Client, Error},
    semantic_embedding::{SemanticRepresentation, TaskSemanticEmbedding},
};

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

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let ab: f32 = a.iter().zip(b).map(|(a,b)| a * b).sum();
    let aa: f32 = a.iter().map(|a| a * a).sum();
    let bb: f32 = b.iter().map(|b| b * b).sum();
    let prod_len = (aa * bb).sqrt();
    ab / prod_len
}