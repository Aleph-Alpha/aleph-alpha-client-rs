use crate::{
    semantic_embedding::{
        embedding::SemanticEmbeddingOutput, RequestBody, DEFAULT_EMBEDDING_MODEL_WITH_INSTRUCTION,
    },
    Job, Prompt, Task,
};
use serde::Serialize;

const ENDPOINT: &str = "/instructable_embed";

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
        client.post(format!("{base}/{ENDPOINT}")).json(&body)
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
        client.post(format!("{base}/{ENDPOINT}")).json(&body)
    }

    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output {
        response
    }
}
