//! Usage sample
//!
//! ```no_run
//! use aleph_alpha_client::{Client, TaskCompletion, How};
//!
//! #[tokio::main(flavor = "current_thread")]
//! async fn main() {
//!     // Authenticate against API. Fetches token.
//!     let client = Client::new("AA_API_TOKEN").unwrap();
//!
//!     // Name of the model we we want to use. Large models give usually better answer, but are also
//!     // more costly.
//!     let model = "luminous-base";
//!
//!     // The task we want to perform. Here we want to continue the sentence: "An apple a day ..."
//!     let task = TaskCompletion::from_text("An apple a day", 10);
//!     
//!     // Retrieve the answer from the API
//!     let response = client.execute(model, &task, &How::default()).await.unwrap();
//!
//!     // Print entire sentence with completion
//!     println!("An apple a day{}", response.completion);
//! }
//! ```

mod completion;
mod http;
mod image_preprocessing;
mod prompt;
mod semantic_embedding;

pub use self::{
    completion::{CompletionOutput, Sampling, Stopping, TaskCompletion},
    http::{Client, Error, Job, Task},
    prompt::{Modality, Prompt},
    semantic_embedding::{SemanticRepresentation, TaskSemanticEmbedding},
};

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
