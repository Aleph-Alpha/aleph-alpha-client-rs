mod embedding;
mod embedding_batch;
mod embedding_with_instruction;

pub use embedding::{SemanticEmbeddingOutput, TaskSemanticEmbedding};
pub use embedding_batch::{BatchSemanticEmbeddingOutput, TaskBatchSemanticEmbedding};
pub use embedding_with_instruction::TaskSemanticEmbeddingWithInstruction;

use serde::Serialize;
use std::fmt::Debug;

const DEFAULT_EMBEDDING_MODEL: &str = "luminous-base";
const DEFAULT_EMBEDDING_MODEL_WITH_INSTRUCTION: &str = "pharia-1-embedding-4608-control";

/// Appends model and hosting to the bare task
/// `T` stands for [`TaskSemanticEmbedding`], [`TaskSemanticEmbeddingWithInstruction`] or
/// [`TaskBatchSemanticEmbedding`].
#[derive(Serialize, Debug)]
struct RequestBody<'a, T: Serialize + Debug> {
    /// Currently semantic embedding still requires a model parameter, even though "luminous-base"
    /// is the only model to support it. This makes Semantic embedding both a Service and a Method.
    model: &'a str,
    #[serde(flatten)]
    semantic_embedding_task: &'a T,
}

/// Allows you to choose a semantic representation fitting for your use case.
#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum SemanticRepresentation {
    /// Useful for comparing prompts to each other, in use cases such as clustering, classification,
    /// similarity, etc. `Symmetric` embeddings are intended to be compared with other `Symmetric`
    /// embeddings.
    Symmetric,
    /// `Document` and `Query` are used together in use cases such as search where you want to
    /// compare shorter queries against larger documents. `Document` embeddings are optimized for
    /// larger pieces of text to compare queries against.
    Document,
    /// `Document` and `Query` are used together in use cases such as search where you want to
    /// compare shorter queries against larger documents. `Query` embeddings are optimized for
    /// shorter texts, such as questions or keywords.
    Query,
}
