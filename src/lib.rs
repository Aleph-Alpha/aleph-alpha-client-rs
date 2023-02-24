use std::{
    borrow::{Borrow, Cow},
    io,
    path::Path,
};

use base64::{prelude::BASE64_STANDARD, Engine};
use serde::Serialize;

mod completion;
mod http;
mod semantic_embedding;

pub use self::{
    completion::{CompletionOutput, Sampling, Stopping, TaskCompletion},
    http::{Client, Error, Task},
    semantic_embedding::{SemanticRepresentation, TaskSemanticEmbedding},
};

/// A prompt which is passed to the model for inference. Usually it is one text item, but it could
/// also be a combination of several modalities like images and text.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct Prompt<'a>(Vec<Modality<'a>>);

impl<'a> Prompt<'a> {
    /// Create a prompt from a single text item.
    pub fn from_text(text: impl Into<Cow<'a, str>>) -> Self {
        Self(vec![Modality::from_text(text)])
    }

    /// Create a multimodal prompt from a list of individual items with any modality.
    pub fn from_vec(items: Vec<Modality<'a>>) -> Self {
        Self(items)
    }

    /// Allows you to borrow the contents of the prompt without allocating a new one.
    pub fn borrow(&'a self) -> Prompt<'a> {
        Self(self.0.iter().map(|item| item.borrow()).collect())
    }
}

/// The prompt for models can be a combination of different modalities (Text and Image). The type of
/// modalities which are supported depend on the Model in question.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Modality<'a> {
    /// The only type of prompt which can be used with pure language models
    Text { data: Cow<'a, str> },
    /// An image input into the model. See [`Modality::from_image`].
    Image { data: Cow<'a, str> },
}

impl<'a> Modality<'a> {
    /// Instantiates a text prompt
    pub fn from_text(text: impl Into<Cow<'a, str>>) -> Self {
        Modality::Text { data: text.into() }
    }

    /// Image input for model, from file path.
    ///
    /// The model can only see squared pictures. Images are centercropped.
    pub fn from_image_path(path: &Path) -> io::Result<Self> {
        let image = std::fs::read(path)?;
        Ok(Self::from_image_bytes(&image))
    }

    /// Generates an image input from the binary representation of the image.
    ///
    /// Using this constructor you must use a binary representation compatible with the API. Png is
    /// guaranteed to be supported, and all others formats are converted into it. Furthermore, the
    /// model can only look at square shaped pictures. If the picture is not square shaped it will
    /// be center cropped.
    fn from_image_bytes(image: &[u8]) -> Self {
        Modality::Image {
            data: BASE64_STANDARD.encode(image).into(),
        }
    }

    /// Create a semantically idetical entry of modality which borrows the contents of this one.
    ///
    /// It is very practical to allow Modality of e.g. Text to take both ownership of the string it
    /// contains as well as borrow a slice. However then we are creating a body from the user input
    /// we want to avoid copying everything and needing to allocate for that modality again. This is
    /// there this borrow function really shines.
    pub fn borrow(&self) -> Modality<'_> {
        match self {
            Modality::Text { data } => Modality::Text {
                data: Cow::Borrowed(data.borrow()),
            },
            Modality::Image { data } => Modality::Image {
                data: Cow::Borrowed(data.borrow()),
            },
        }
    }
}

/// Controls of how to execute a task
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct How {
    /// Set this to `true` if you want to not put any load on the API in case it is already pretty
    /// busy for the models you intend to use. All this does from the user perspective is that it
    /// makes it more likely you get a `Busy` response from the server. One of the reasons you may
    /// want to set is that you are an employee or associate of Aleph Alpha and want to perform
    /// experiments without hurting paying customers.
    pub be_nice: bool,
}

/// Intended to compare embeddings.
///
/// ```no_run
/// use aleph_alpha_client::{
///     Client, Prompt, TaskSemanticEmbedding, cosine_similarity, SemanticRepresentation, How
/// };
///
/// async fn semanitc_search_with_luminous_base(client: &Client) {
///     // Given
///     let robot_fact = Prompt::from_text(
///         "A robot is a machine—especially one programmable by a computer—capable of carrying out a \
///         complex series of actions automatically.",
///     );
///     let pizza_fact = Prompt::from_text(
///         "Pizza (Italian: [ˈpittsa], Neapolitan: [ˈpittsə]) is a dish of Italian origin consisting \
///         of a usually round, flat base of leavened wheat-based dough topped with tomatoes, cheese, \
///         and often various other ingredients (such as various types of sausage, anchovies, \
///         mushrooms, onions, olives, vegetables, meat, ham, etc.), which is then baked at a high \
///         temperature, traditionally in a wood-fired oven.",
///     );
///     let query = Prompt::from_text("What is Pizza?");
///     let model = "luminous-base";
///     let how = How::default();
///     
///     // When
///     let robot_embedding_task = TaskSemanticEmbedding {
///         prompt: robot_fact,
///         representation: SemanticRepresentation::Document,
///         compress_to_size: Some(128),
///     };
///     let robot_embedding = client.execute(
///         model,
///         &robot_embedding_task,
///         &how,
///     ).await.unwrap().embedding;
///     
///     let pizza_embedding_task = TaskSemanticEmbedding {
///         prompt: pizza_fact,
///         representation: SemanticRepresentation::Document,
///         compress_to_size: Some(128),
///     };
///     let pizza_embedding = client.execute(
///         model,
///         &pizza_embedding_task,
///         &how,
///     ).await.unwrap().embedding;
///     
///     let query_embedding_task = TaskSemanticEmbedding {
///         prompt: query,
///         representation: SemanticRepresentation::Query,
///         compress_to_size: Some(128),
///     };
///     let query_embedding = client.execute(
///         model,
///         &query_embedding_task,
///         &how,
///     ).await.unwrap().embedding;
///     let similarity_pizza = cosine_similarity(&query_embedding, &pizza_embedding);
///     println!("similarity pizza: {similarity_pizza}");
///     let similarity_robot = cosine_similarity(&query_embedding, &robot_embedding);
///     println!("similarity robot: {similarity_robot}");
///     
///     // Then
///     
///     // The fact about pizza should be more relevant to the "What is Pizza?" question than a fact
///     // about robots.
///     assert!(similarity_pizza > similarity_robot);
/// }
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let ab: f32 = a.iter().zip(b).map(|(a, b)| a * b).sum();
    let aa: f32 = a.iter().map(|a| a * a).sum();
    let bb: f32 = b.iter().map(|b| b * b).sum();
    let prod_len = (aa * bb).sqrt();
    ab / prod_len
}

#[cfg(test)]
mod tests {
    use crate::Prompt;

    #[test]
    fn ability_to_generate_prompt_in_local_function() {
        fn local_function() -> Prompt<'static> {
            Prompt::from_text(String::from("My test prompt"))
        }

        assert_eq!(Prompt::from_text("My test prompt"), local_function())
    }
}
