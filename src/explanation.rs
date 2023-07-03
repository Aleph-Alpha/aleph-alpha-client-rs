use serde::{Deserialize, Serialize};

use crate::{Prompt, Task};

pub struct TaskExplanation<'a> {
    pub prompt: Prompt<'a>,
    pub target: &'a str,
    pub granularity: Granularity,
}

#[derive(Default, Serialize)]
pub struct Granularity {
    #[serde(skip_serializing_if = "PromptGranularity::is_auto")]
    prompt: PromptGranularity,
}

impl Granularity {
    pub fn with_prompt_granularity(self, prompt_granularity: PromptGranularity) -> Self {
        Self {
            prompt: prompt_granularity,
        }
    }
}

#[derive(Serialize, Copy, Clone, PartialEq, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PromptGranularity {
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

#[derive(Serialize)]
struct BodyExplanation<'a> {
    prompt: Prompt<'a>,
    target: &'a str,
    prompt_granularity: PromptGranularity,
    model: &'a str,
}

#[derive(Deserialize, Debug, PartialEq)]
pub struct ResponseExplanation {
    explanations: Vec<Explanation>,
}
#[derive(Debug, PartialEq)]
pub struct ExplanationOutput {
    pub explanation: Explanation,
}

impl ExplanationOutput {
    fn from(mut response: ResponseExplanation) -> ExplanationOutput {
        ExplanationOutput {
            explanation: response.explanations.pop().unwrap(),
        }
    }
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct Explanation {
    pub target: String,
    pub items: Vec<ItemExplanation>,
}

#[derive(PartialEq, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ItemExplanation {
    Text { scores: Vec<ExplanationScore> },
    Target { scores: Vec<ExplanationScore> },
}

#[derive(Debug, PartialEq, Deserialize)]
pub struct ExplanationScore {
    pub start: u32,
    pub length: u32,
    pub score: f32,
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

    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output {
        ExplanationOutput::from(response)
    }
}
