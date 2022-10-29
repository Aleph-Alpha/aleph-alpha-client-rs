use serde::Serialize;

mod completion;
mod http;
mod semantic_embedding;

pub use self::{
    completion::{CompletionOutput, Sampling, TaskCompletion, Stopping},
    http::{Client, Error, Task},
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

/// Intended to compare embeddings.
/// 
/// ```no_run
/// use aleph_alpha_client::{
///     Client, Prompt, TaskSemanticEmbedding, cosine_similarity, SemanticRepresentation
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
///     
///     // When
///     let robot_embedding_task = TaskSemanticEmbedding {
///         prompt: robot_fact,
///         representation: SemanticRepresentation::Document,
///         compress_to_size: Some(128),
///     };
///     let robot_embedding = client.execute(model, &robot_embedding_task).await.unwrap().embedding;
///     
///     let pizza_embedding_task = TaskSemanticEmbedding {
///         prompt: pizza_fact,
///         representation: SemanticRepresentation::Document,
///         compress_to_size: Some(128),
///     };
///     let pizza_embedding = client.execute(model, &pizza_embedding_task).await.unwrap().embedding;
///     
///     let query_embedding_task = TaskSemanticEmbedding {
///         prompt: query,
///         representation: SemanticRepresentation::Query,
///         compress_to_size: Some(128),
///     };
///     let query_embedding = client.execute(model, &query_embedding_task).await.unwrap().embedding;
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
    let ab: f32 = a.iter().zip(b).map(|(a,b)| a * b).sum();
    let aa: f32 = a.iter().map(|a| a * a).sum();
    let bb: f32 = b.iter().map(|b| b * b).sum();
    let prod_len = (aa * bb).sqrt();
    ab / prod_len
}