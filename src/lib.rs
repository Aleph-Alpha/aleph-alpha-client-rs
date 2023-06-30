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
mod explanation;
mod http;
mod image_preprocessing;
mod prompt;
mod semantic_embedding;

use http::HttpClient;
use semantic_embedding::SemanticEmbeddingOutput;

pub use self::{
    completion::{CompletionOutput, Sampling, Stopping, TaskCompletion},
    explanation::{
        Explanation, ExplanationOutput, ExplanationScore, ItemExplanation, PromptGranularity,
        TaskExplanation,
    },
    http::{Error, Job, Task},
    prompt::{Modality, Prompt},
    semantic_embedding::{SemanticRepresentation, TaskSemanticEmbedding},
};

/// Execute Jobs against the Aleph Alpha API
pub struct Client {
    /// This client does all the work of sending the requests and talking to the AA API. The only
    /// additional knowledge added by this layer is that it knows about the individual jobs which
    /// can be executed, which allows for an alternative non generic interface which might produce
    /// easier to read code for the end user in many use cases.
    http_client: HttpClient,
}

impl Client {
    /// A new instance of an Aleph Alpha client helping you interact with the Aleph Alpha API.
    pub fn new(api_token: &str) -> Result<Self, Error> {
        Self::with_base_url("https://api.aleph-alpha.com".to_owned(), api_token)
    }

    /// In production you typically would want set this to <https://api.aleph-alpha.com>. Yet
    /// you may want to use a different instances for testing.
    pub fn with_base_url(host: String, api_token: &str) -> Result<Self, Error> {
        let http_client = HttpClient::with_base_url(host, api_token)?;
        Ok(Self { http_client })
    }

    /// Execute a task with the aleph alpha API and fetch its result.
    ///
    /// ```no_run
    /// use aleph_alpha_client::{Client, How, TaskCompletion, Error};
    ///
    /// async fn print_completion() -> Result<(), Error> {
    ///     // Authenticate against API. Fetches token.
    ///     let client = Client::new("AA_API_TOKEN")?;
    ///
    ///     // Name of the model we we want to use. Large models give usually better answer, but are
    ///     // also slower and more costly.
    ///     let model = "luminous-base";
    ///
    ///     // The task we want to perform. Here we want to continue the sentence: "An apple a day
    ///     // ..."
    ///     let task = TaskCompletion::from_text("An apple a day", 10);
    ///
    ///     // Retrieve answer from API
    ///     let response = client.execute(model, &task, &How::default()).await?;
    ///
    ///     // Print entire sentence with completion
    ///     println!("An apple a day{}", response.completion);
    ///     Ok(())
    /// }
    /// ```
    #[deprecated = "Please use output_of instead."]
    pub async fn execute<T: Task>(
        &self,
        model: &str,
        task: &T,
        how: &How,
    ) -> Result<T::Output, Error> {
        self.output_of(&task.with_model(model), how).await
    }

    /// Execute any task with the aleph alpha API and fetch its result. This is most usefull in
    /// generic code then you want to execute arbitrary task types. Otherwise prefer methods taking
    /// concrete tasks like [`Self::completion`] for improved readability.
    pub async fn output_of<T: Job>(&self, task: &T, how: &How) -> Result<T::Output, Error> {
        self.http_client.output_of(task, how).await
    }

    /// An embedding trying to capture the semantic meaning of a text. Cosine similarity can be used
    /// find out how well two texts (or multimodal prompts) match. Useful for search usecases.
    ///
    /// See the example for [`cosine_similarity`].
    pub async fn semantic_embedding(
        &self,
        task: &TaskSemanticEmbedding<'_>,
        how: &How,
    ) -> Result<SemanticEmbeddingOutput, Error> {
        self.http_client.output_of(task, how).await
    }

    /// Instruct a model served by the aleph alpha API to continue writing a piece of text (or
    /// multimodal document).
    ///
    /// ```no_run
    /// use aleph_alpha_client::{Client, How, TaskCompletion, Task, Error};
    ///
    /// async fn print_completion() -> Result<(), Error> {
    ///     // Authenticate against API. Fetches token.
    ///     let client = Client::new("AA_API_TOKEN")?;
    ///
    ///     // Name of the model we we want to use. Large models give usually better answer, but are
    ///     // also slower and more costly.
    ///     let model = "luminous-base";
    ///
    ///     // The task we want to perform. Here we want to continue the sentence: "An apple a day
    ///     // ..."
    ///     let task = TaskCompletion::from_text("An apple a day", 10);
    ///
    ///     // Retrieve answer from API
    ///     let response = client.completion(&task, model, &How::default()).await?;
    ///
    ///     // Print entire sentence with completion
    ///     println!("An apple a day{}", response.completion);
    ///     Ok(())
    /// }
    /// ```
    pub async fn completion(
        &self,
        task: &TaskCompletion<'_>,
        model: &str,
        how: &How,
    ) -> Result<CompletionOutput, Error> {
        self.http_client
            .output_of(&task.with_model(model), how)
            .await
    }

    pub async fn explanation(
        &self,
        task: &TaskExplanation<'_>,
        model: &str,
        how: &How,
    ) -> Result<ExplanationOutput, Error> {
        self.http_client
            .output_of(&task.with_model(model), how)
            .await
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

impl How {
    pub fn be_nice(self) -> Self {
        Self { be_nice: true }
    }
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
///     let how = How::default();
///     
///     // When
///     let robot_embedding_task = TaskSemanticEmbedding {
///         prompt: robot_fact,
///         representation: SemanticRepresentation::Document,
///         compress_to_size: Some(128),
///     };
///     let robot_embedding = client.semantic_embedding(
///         &robot_embedding_task,
///         &how,
///     ).await.unwrap().embedding;
///     
///     let pizza_embedding_task = TaskSemanticEmbedding {
///         prompt: pizza_fact,
///         representation: SemanticRepresentation::Document,
///         compress_to_size: Some(128),
///     };
///     let pizza_embedding = client.semantic_embedding(
///         &pizza_embedding_task,
///         &how,
///     ).await.unwrap().embedding;
///     
///     let query_embedding_task = TaskSemanticEmbedding {
///         prompt: query,
///         representation: SemanticRepresentation::Query,
///         compress_to_size: Some(128),
///     };
///     let query_embedding = client.semantic_embedding(
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
