use serde::{Deserialize, Serialize};

use crate::{Prompt, Task};

pub struct TaskExplanation<'a> {
    pub prompt: Prompt<'a>,
    pub target: &'a str,
    pub prompt_granularity: &'a PromptGranularity,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PromptGranularity {
    Word,
    Sentence,
    Paragraph,
}

#[derive(Serialize)]
struct TaskExplanationBody<'a> {
    prompt: Prompt<'a>,
    target: &'a str,
    prompt_granularity: &'a PromptGranularity,
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
        let body = TaskExplanationBody {
            model,
            prompt: self.prompt.borrow(),
            target: self.target,
            prompt_granularity: self.prompt_granularity,
        };
        client.post(format!("{base}/explain")).json(&body)
    }

    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output {
        ExplanationOutput::from(response)
    }
}
