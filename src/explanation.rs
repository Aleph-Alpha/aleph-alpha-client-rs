use serde::{Deserialize, Serialize};

use crate::{Prompt, Task};

pub struct TaskExplanation<'a> {
    pub prompt: Prompt<'a>,
    pub target: &'a str,
}

#[derive(Serialize)]
struct TaskExplanationBody<'a> {
    prompt: Prompt<'a>,
    target: &'a str,
    model: &'a str,
}

#[derive(Deserialize, Debug, PartialEq)]
pub struct ExplanationOutput {
    pub model_version: String,
    pub explanations: Vec<Explanation>,
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
    start: u32,
    length: u32,
    score: f32,
}

impl Task for TaskExplanation<'_> {
    type Output = ExplanationOutput;

    type ResponseBody = ExplanationOutput;

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
        };
        client.post(format!("{base}/explain")).json(&body)
    }

    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output {
        response
    }
}
