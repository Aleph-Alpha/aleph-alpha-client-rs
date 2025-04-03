use serde::Deserialize;

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    /// The model is configured and a matching worker is connected to serve it.
    Available,
    /// The model is configured but no worker has shown recent activity to serve
    /// it.
    Unavailable,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum WorkerType {
    /// The model is served by a Luminous worker.
    Luminous,
    /// The model is served by a vLLM worker.
    Vllm,
    /// Worker type to serve translation requests.
    Translation,
    /// Worker type to serve transcription requests.
    Transcription,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum CompletionType {
    /// The model has not been trained to support completions. Trying to trigger a completion
    /// request will lead to a validation error.
    None,
    /// The model has been trained to support completions.
    Full,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingType {
    /// The model cannot be used for embeddings. The scheduler will reject requests for embeddings
    /// to this model.
    None,
    /// The model has not explicitly been trained to support embeddings. However, it is possible to
    /// retrieve the embedding details technically. This option maps to the `/embed` endpoint.
    Raw,
    /// The model has been trained with a switchable set of weights usable for semantic embedding
    /// retrieval. This option maps to the `/semantic_embed` endpoint.
    Semantic,
    /// The model has been trained to support any custom instruction for embedding retrieval. This
    /// option maps to the `/instructable_embed` endpoint.
    Instructable,
}

#[derive(Deserialize, Debug)]
pub struct ModelSettings {
    pub name: String,
    /// A description of the model.
    pub description: String,
    /// The current availability status of the model.
    pub status: ModelStatus,
    /// The embedding type supported by the model.
    pub embedding_type: EmbeddingType,
    /// Whether this model is supported by the chat endpoint.
    pub chat: bool,
    /// Whether the model is aligned s.t. end users can be warned about the model's limitations.
    pub aligned: bool,
    /// The completion type supported by the model.
    pub completion_type: CompletionType,
    /// A prompt template that can be used for this model.
    pub prompt_template: String,
    /// Whether this model supports semantic embeddings.
    pub semantic_embedding: bool,
    /// The maximum context size of this model.
    pub max_context_size: u32,
    /// Feature flag for whether multimodal prompts are available to users.
    pub multimodal: bool,
    /// The worker type that is used to serve the configured model.
    pub worker_type: WorkerType,
}
