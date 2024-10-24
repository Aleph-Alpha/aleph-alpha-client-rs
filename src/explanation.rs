use serde::{Deserialize, Serialize};

use crate::{Prompt, Task};

/// Input for a [crate::Client::explanation] request.
pub struct TaskExplanation<'a> {
    /// The prompt that typically was the input of a previous completion request
    pub prompt: Prompt<'a>,
    /// The target string that should be explained. The influence of individual parts
    /// of the prompt for generating this target string will be indicated in the response.
    pub target: &'a str,
    /// Granularity parameters for the explanation
    pub granularity: Granularity,
}

/// Granularity parameters for the [TaskExplanation]
#[derive(Default)]
pub struct Granularity {
    /// The granularity of the parts of the prompt for which a single
    /// score is computed.
    prompt: PromptGranularity,
}

impl Granularity {
    /// Returns a new [Granularity] based on the given one with the [Granularity::prompt]
    /// being set to `prompt_granularity`.
    pub fn with_prompt_granularity(self, prompt_granularity: PromptGranularity) -> Self {
        Self {
            prompt: prompt_granularity,
        }
    }
}

/// At which granularity should the target be explained in terms of the prompt.
/// If you choose, for example, [PromptGranularity::Sentence] then we report the importance score of each
/// sentence in the prompt towards generating the target output.
/// The default is [PromptGranularity::Auto] which means we will try to find the granularity that
/// brings you closest to around 30 explanations. For large prompts, this would likely
/// be sentences. For short prompts this might be individual words or even tokens.
#[derive(Serialize, Copy, Clone, PartialEq, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PromptGranularity {
    /// Let the system decide which granularity is most suitable for the given input.
    /// This will result
    #[default]
    Auto,
    Word,
    Sentence,
    Paragraph,
}

impl PromptGranularity {
    fn is_auto(&self) -> bool {
        self == &PromptGranularity::Auto
    }
}

/// Body sent to the Aleph Alpha API for an explanation request
#[derive(Serialize)]
struct BodyExplanation<'a> {
    prompt: Prompt<'a>,
    target: &'a str,
    #[serde(skip_serializing_if = "PromptGranularity::is_auto")]
    prompt_granularity: PromptGranularity,
    model: &'a str,
}

/// Body received by the Aleph Alpha API from an explanation request
#[derive(Deserialize, Debug, PartialEq)]
pub struct ResponseExplanation {
    /// The Body contains an array of [Explanation]s one for each
    /// part of the target being explained.
    explanations: Vec<Explanation>,
}

/// The result of an explanation request.
#[derive(Debug, PartialEq)]
pub struct ExplanationOutput {
    /// Explanation scores for different parts of the prompt or target.
    pub items: Vec<ItemExplanation>,
}

impl ExplanationOutput {
    fn from(mut response: ResponseExplanation) -> ExplanationOutput {
        ExplanationOutput {
            items: response.explanations.pop().unwrap().items,
        }
    }
}

/// The explanation for the target.
#[derive(Debug, Deserialize, PartialEq)]
pub struct Explanation {
    /// Explanation scores for different parts of the prompt or target.
    pub items: Vec<ItemExplanation>,
}

/// Explanation scores for a [crate::prompt::Modality] or the target.
/// There is one score
/// for each part of a `modality` respectively the target with the parts being choosen according to
/// the [PromptGranularity]
#[derive(PartialEq, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ItemExplanation {
    Text { scores: Vec<TextScore> },
    Image { scores: Vec<ImageScore> },
    Target { scores: Vec<TextScore> },
}

/// Score for the part of a text-modality
#[derive(Debug, PartialEq, Deserialize, Clone)]
pub struct TextScore {
    pub start: u32,
    pub length: u32,
    pub score: f32,
}

/// Resembles the actual response from the API for deserialization
#[derive(Deserialize)]
struct BoundingBox {
    top: f32,
    left: f32,
    width: f32,
    height: f32,
}

/// Resembles the actual response from the API for deserialization
#[derive(Deserialize)]
struct ImageScoreWithRect {
    rect: BoundingBox,
    score: f32,
}

/// Score for a part of an image.
#[derive(Debug, PartialEq, Deserialize, Clone)]
#[serde(from = "ImageScoreWithRect")]
pub struct ImageScore {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub score: f32,
}

impl From<ImageScoreWithRect> for ImageScore {
    fn from(value: ImageScoreWithRect) -> Self {
        Self {
            left: value.rect.left,
            top: value.rect.top,
            width: value.rect.width,
            height: value.rect.height,
            score: value.score,
        }
    }
}

impl Task for TaskExplanation<'_> {
    type Output = ExplanationOutput;

    type ResponseBody = ResponseExplanation;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = BodyExplanation {
            model,
            prompt: self.prompt.borrow(),
            target: self.target,
            prompt_granularity: self.granularity.prompt,
        };
        client.post(format!("{base}/explain")).json(&body)
    }

    fn body_to_output(response: Self::ResponseBody) -> Self::Output {
        ExplanationOutput::from(response)
    }
}
