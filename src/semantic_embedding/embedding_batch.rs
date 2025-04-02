use crate::semantic_embedding::{RequestBody, DEFAULT_EMBEDDING_MODEL};
use crate::{Job, Prompt, SemanticRepresentation};
use serde::{Deserialize, Serialize};

const ENDPOINT: &str = "batch_semantic_embed";

/// Create embeddings for multiple prompts
#[derive(Serialize, Debug)]
pub struct TaskBatchSemanticEmbedding<'a> {
    /// The prompt (usually text) to be embedded.
    pub prompts: Vec<Prompt<'a>>,
    /// Semantic representation to embed the prompt with. This parameter is governed by the specific
    /// use case in mind.
    pub representation: SemanticRepresentation,
    /// Default behaviour is to return the full embedding, but you can optionally request an
    /// embedding compressed to a smaller set of dimensions. A size of `128` is supported for every
    /// model.
    ///
    /// The 128 size is expected to have a small drop in accuracy performance (4-6%), with the
    /// benefit of being much smaller, which makes comparing these embeddings much faster for use
    /// cases where speed is critical.
    ///
    /// The 128 size can also perform better if you are embedding short texts or documents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compress_to_size: Option<u32>,
}

/// Heap allocated vec of embeddings. Can hold full embeddings or compressed ones
#[derive(Deserialize)]
pub struct BatchSemanticEmbeddingOutput {
    pub embeddings: Vec<Vec<f32>>,
}

impl Job for TaskBatchSemanticEmbedding<'_> {
    type Output = BatchSemanticEmbeddingOutput;
    type ResponseBody = BatchSemanticEmbeddingOutput;

    fn build_request(&self, client: &reqwest::Client, base: &str) -> reqwest::RequestBuilder {
        let body = RequestBody {
            model: DEFAULT_EMBEDDING_MODEL,
            semantic_embedding_task: self,
        };
        client.post(format!("{base}/{ENDPOINT}")).json(&body)
    }

    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output {
        response
    }
}
