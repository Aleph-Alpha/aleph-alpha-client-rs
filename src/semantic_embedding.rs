use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::{http::Task, Job, Prompt};

const DEFAULT_EMBEDDING_MODEL: &str = "luminous-base";
const DEFAULT_EMBEDDING_MODEL_WITH_INSTRUCTION: &str = "pharia-1-embedding-4608-control";

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

/// Create embeddings for prompts which can be used for downstream tasks. E.g. search, classifiers
#[derive(Serialize, Debug)]
pub struct TaskSemanticEmbedding<'a> {
    /// The prompt (usually text) to be embedded.
    pub prompt: Prompt<'a>,
    /// Semantic representation to embed the prompt with. This parameter is governed by the specific
    /// usecase in mind.
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

/// Appends model and hosting to the bare task
/// T stands for TaskSemanticEmbedding or TaskBatchSemanticEmbedding
#[derive(Serialize, Debug)]
struct RequestBody<'a, T: Serialize + Debug> {
    /// Currently semantic embedding still requires a model parameter, even though "luminous-base"
    /// is the only model to support it. This makes Semantic embedding both a Service and a Method.
    model: &'a str,
    #[serde(flatten)]
    semantic_embedding_task: &'a T,
}

/// Heap allocated embedding. Can hold full embeddings or compressed ones
#[derive(Deserialize)]
pub struct SemanticEmbeddingOutput {
    pub embedding: Vec<f32>,
}

impl Task for TaskSemanticEmbedding<'_> {
    type Output = SemanticEmbeddingOutput;
    type ResponseBody = SemanticEmbeddingOutput;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = RequestBody {
            model,
            semantic_embedding_task: self,
        };
        client.post(format!("{base}/semantic_embed")).json(&body)
    }

    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output {
        response
    }
}

impl Job for TaskSemanticEmbedding<'_> {
    type Output = SemanticEmbeddingOutput;
    type ResponseBody = SemanticEmbeddingOutput;

    fn build_request(&self, client: &reqwest::Client, base: &str) -> reqwest::RequestBuilder {
        let body = RequestBody {
            model: DEFAULT_EMBEDDING_MODEL,
            semantic_embedding_task: self,
        };
        client.post(format!("{base}/semantic_embed")).json(&body)
    }

    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output {
        response
    }
}

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
        client
            .post(format!("{base}/batch_semantic_embed"))
            .json(&body)
    }

    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output {
        response
    }
}

/// Allows you to choose a semantic representation fitting for your use case.
///
/// By providing instructions, you can help the model better understand the nuances of your specific
/// data, leading to embeddings that are more useful for your use case.
#[derive(Serialize, Debug)]
pub struct TaskSemanticEmbeddingWithInstruction<'a> {
    /// To further improve performance by steering the model, you can use instructions.
    ///
    /// Instructions can help the model understand nuances of your specific data and ultimately lead
    /// to embeddings that are more useful for your use-case. In this case, we aim to further
    /// increase the absolute difference between the cosine similarities. Instruction can also be
    /// the empty string.
    pub instruction: &'a str,
    /// The prompt (usually text) to be embedded.
    #[serde(rename = "input")]
    pub prompt: Prompt<'a>,
    /// Return normalized embeddings. This can be used to save on additional compute when applying a
    /// cosine similarity metric.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,
}

impl Task for TaskSemanticEmbeddingWithInstruction<'_> {
    type Output = SemanticEmbeddingOutput;
    type ResponseBody = SemanticEmbeddingOutput;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = RequestBody {
            model,
            semantic_embedding_task: self,
        };
        client
            .post(format!("{base}/instructable_embed"))
            .json(&body)
    }

    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output {
        response
    }
}

impl Job for TaskSemanticEmbeddingWithInstruction<'_> {
    type Output = SemanticEmbeddingOutput;
    type ResponseBody = SemanticEmbeddingOutput;

    fn build_request(&self, client: &reqwest::Client, base: &str) -> reqwest::RequestBuilder {
        let body = RequestBody {
            model: DEFAULT_EMBEDDING_MODEL_WITH_INSTRUCTION,
            semantic_embedding_task: self,
        };
        client
            .post(format!("{base}/instructable_embed"))
            .json(&body)
    }

    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output {
        response
    }
}
