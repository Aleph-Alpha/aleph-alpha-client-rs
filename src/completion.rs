use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{http::Task, Distribution, Logprob, Logprobs, Prompt, StreamTask, Usage};

/// Completes a prompt. E.g. continues a text.
pub struct TaskCompletion<'a> {
    /// The prompt (usually text) to be completed. Unconditional completion can be started with an
    /// empty string. The prompt may contain a zero shot or few shot task.
    pub prompt: Prompt<'a>,
    /// Controls in which circumstances the model will stop generating new tokens.
    pub stopping: Stopping<'a>,
    /// Sampling controls how the tokens ("words") are selected for the completion.
    pub sampling: Sampling,
    /// Whether to include special tokens (e.g. <|endoftext|>, <|python_tag|>) in the completion.
    pub special_tokens: bool,
    /// Wether you are interessted in the probabilities of the sampled tokens, or most likely
    /// tokens.
    pub logprobs: Logprobs,
}

impl<'a> TaskCompletion<'a> {
    /// Convenience constructor leaving most setting to default, just completing a given text
    pub fn from_text(text: &'a str) -> Self {
        TaskCompletion {
            prompt: Prompt::from_text(text),
            stopping: Stopping::NO_TOKEN_LIMIT,
            sampling: Sampling::MOST_LIKELY,
            special_tokens: false,
            logprobs: Logprobs::No,
        }
    }

    pub fn with_maximum_tokens(mut self, maximum_tokens: u32) -> Self {
        self.stopping.maximum_tokens = Some(maximum_tokens);
        self
    }

    pub fn with_stop_sequences(mut self, stop_sequences: &'a [&str]) -> Self {
        self.stopping.stop_sequences = stop_sequences;
        self
    }

    /// Include special tokens (e.g. <|endoftext|>, <|python_tag|>) in the completion.
    pub fn with_special_tokens(mut self) -> Self {
        self.special_tokens = true;
        self
    }

    pub fn with_logprobs(mut self, logprobs: Logprobs) -> Self {
        self.logprobs = logprobs;
        self
    }
}

/// Sampling controls how the tokens ("words") are selected for the completion.
pub struct Sampling {
    /// A temperature encourages the model to produce less probable outputs ("be more creative").
    /// Values are expected to be between 0 and 1. Try high values for a more random ("creative")
    /// response.
    pub temperature: Option<f64>,
    /// Introduces random sampling for generated tokens by randomly selecting the next token from
    /// the k most likely options. A value larger than 1 encourages the model to be more creative.
    /// Set to 0 to get the same behaviour as `None`.
    pub top_k: Option<u32>,
    /// Introduces random sampling for generated tokens by randomly selecting the next token from
    /// the smallest possible set of tokens whose cumulative probability exceeds the probability
    /// top_p. Set to 0 to get the same behaviour as `None`.
    pub top_p: Option<f64>,
    /// When specified, this number will decrease (or increase) the likelihood of repeating tokens
    /// that were mentioned prior in the completion. The penalty is cumulative. The more a token
    /// is mentioned in the completion, the more its probability will decrease.
    /// A negative value will increase the likelihood of repeating tokens.
    pub frequency_penalty: Option<f64>,
    /// The presence penalty reduces the likelihood of generating tokens that are already present
    /// in the generated text (repetition_penalties_include_completion=true) respectively the
    /// prompt (repetition_penalties_include_prompt=true). Presence penalty is independent of the
    /// number of occurrences. Increase the value to reduce the likelihood of repeating text.
    /// An operation like the following is applied:
    ///
    /// logits[t] -> logits[t] - 1 * penalty
    ///
    /// where logits[t] is the logits for any given token. Note that the formula is independent
    /// of the number of times that a token appears.
    pub presence_penalty: Option<f64>,
}

impl Sampling {
    /// Always chooses the token most likely to come next. Choose this if you do want close to
    /// deterministic behaviour and do not want to apply any penalties to avoid repetitions.
    pub const MOST_LIKELY: Self = Sampling {
        temperature: None,
        top_k: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
    };
}

impl Default for Sampling {
    fn default() -> Self {
        Self::MOST_LIKELY
    }
}

/// Controls the conditions under which the language models stops generating text.
pub struct Stopping<'a> {
    /// The maximum number of tokens to be generated. Completion will terminate after the maximum
    /// number of tokens is reached. Increase this value to allow for longer outputs. A text is split
    /// into tokens. Usually there are more tokens than words. The total number of tokens of prompt
    /// and maximum_tokens depends on the model.
    /// If maximum tokens is set to None, no outside limit is opposed on the number of maximum tokens.
    /// The model will generate tokens until it generates one of the specified stop_sequences or it
    /// reaches its technical limit, which usually is its context window.
    pub maximum_tokens: Option<u32>,
    /// List of strings which will stop generation if they are generated. Stop sequences are
    /// helpful in structured texts. E.g.: In a question answering scenario a text may consist of
    /// lines starting with either "Question: " or "Answer: " (alternating). After producing an
    /// answer, the model will be likely to generate "Question: ". "Question: " may therefore be used
    /// as stop sequence in order not to have the model generate more questions but rather restrict
    /// text generation to the answers.
    pub stop_sequences: &'a [&'a str],
}

impl<'a> Stopping<'a> {
    /// Only stop once the model reaches its technical limit, usually the context window.
    pub const NO_TOKEN_LIMIT: Self = Stopping {
        maximum_tokens: None,
        stop_sequences: &[],
    };

    /// Stop once the model has reached maximum_tokens.
    pub fn from_maximum_tokens(maximum_tokens: u32) -> Self {
        Self {
            maximum_tokens: Some(maximum_tokens),
            stop_sequences: &[],
        }
    }

    pub fn from_stop_sequences(stop_sequences: &'a [&'a str]) -> Self {
        Self {
            maximum_tokens: None,
            stop_sequences,
        }
    }
}

impl Default for Stopping<'_> {
    fn default() -> Self {
        Self::NO_TOKEN_LIMIT
    }
}

/// Body send to the Aleph Alpha API on the POST `/completion` Route
#[derive(Serialize, Debug)]
struct BodyCompletion<'a> {
    /// Name of the model tasked with completing the prompt. E.g. `luminous-base"`.
    pub model: &'a str,
    /// Prompt to complete. The modalities supported depend on `model`.
    pub prompt: Prompt<'a>,
    /// Limits the number of tokens, which are generated for the completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_tokens: Option<u32>,
    /// List of strings which will stop generation if they are generated. Stop sequences are
    /// helpful in structured texts. E.g.: In a question answering scenario a text may consist of
    /// lines starting with either "Question: " or "Answer: " (alternating). After producing an
    /// answer, the model will be likely to generate "Question: ". "Question: " may therefore be used
    /// as stop sequence in order not to have the model generate more questions but rather restrict
    /// text generation to the answers.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pub stop_sequences: &'a [&'a str],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// If true, the response will be streamed.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// Forces the raw completion of the model to be returned.
    /// For some models, the completion that was generated by the model may be optimized and
    /// returned in the completion field of the CompletionResponse.
    /// The raw completion, if returned, will contain the un-optimized completion.
    /// Setting tokens to true or log_probs to any value will also trigger the raw completion to be returned.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub raw_completion: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_probs: Option<u8>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub tokens: bool,
}

impl<'a> BodyCompletion<'a> {
    pub fn new(model: &'a str, task: &'a TaskCompletion<'a>) -> Self {
        let TaskCompletion {
            prompt,
            stopping,
            sampling,
            special_tokens,
            logprobs,
        } = task;
        Self {
            model,
            prompt: prompt.borrow(),
            maximum_tokens: stopping.maximum_tokens,
            stop_sequences: stopping.stop_sequences,
            temperature: sampling.temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
            stream: false,
            raw_completion: *special_tokens,
            frequency_penalty: sampling.frequency_penalty,
            presence_penalty: sampling.presence_penalty,
            log_probs: logprobs.to_logprobs_num(),
            tokens: logprobs.to_tokens(),
        }
    }
    pub fn with_streaming(mut self) -> Self {
        self.stream = true;
        self
    }
}

#[derive(Deserialize, Debug, PartialEq)]
pub struct ResponseCompletion {
    model_version: String,
    completions: Vec<DeserializedCompletion>,
    num_tokens_prompt_total: u32,
    num_tokens_generated: u32,
}

#[derive(Deserialize, Debug, PartialEq)]
struct DeserializedCompletion {
    completion: String,
    finish_reason: String,
    raw_completion: Option<String>,
    #[serde(default)]
    log_probs: Vec<HashMap<String, f64>>,
    #[serde(default)]
    completion_tokens: Vec<String>,
}

/// Completion and metainformation returned by a completion task
#[derive(Deserialize, Debug, PartialEq)]
pub struct CompletionOutput {
    pub completion: String,
    pub finish_reason: String,
    pub logprobs: Vec<Distribution>,
    pub usage: Usage,
}

impl Task for TaskCompletion<'_> {
    type Output = CompletionOutput;

    type ResponseBody = ResponseCompletion;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = BodyCompletion::new(model, self);
        client.post(format!("{base}/complete")).json(&body)
    }

    fn body_to_output(&self, mut response: Self::ResponseBody) -> Self::Output {
        // We expect the API to return exactly one completion, despite them being modled as an array
        let DeserializedCompletion {
            completion,
            finish_reason,
            raw_completion,
            log_probs,
            completion_tokens,
        } = response.completions.pop().unwrap();
        let completion = if self.special_tokens {
            raw_completion.unwrap()
        } else {
            completion
        };
        CompletionOutput {
            completion,
            finish_reason,
            logprobs: completion_logprobs_to_canonical(
                log_probs,
                completion_tokens,
                self.logprobs.top_logprobs().unwrap_or_default(),
            ),
            usage: Usage {
                prompt_tokens: response.num_tokens_prompt_total,
                completion_tokens: response.num_tokens_generated,
            },
        }
    }
}

fn completion_logprobs_to_canonical(
    log_probs: Vec<HashMap<String, f64>>,
    completion_tokens: Vec<String>,
    num_expected_top_logprobs: u8,
) -> Vec<Distribution> {
    let mut logprobs = Vec::new();
    for (token, map) in completion_tokens.into_iter().zip(log_probs) {
        let logprob = *map.get(&token).unwrap_or(&f64::NAN);
        let mut top_logprobs = map
            .into_iter()
            .map(|(token, logprob)| Logprob {
                token: token.into_bytes(),
                logprob,
            })
            .collect::<Vec<_>>();
        // We want to make sure the most likely tokens are first in the array
        top_logprobs.sort_by(|a, b| b.logprob.total_cmp(&a.logprob));
        // The aa api always makes the sampled token part of the array, even if not in the top n
        // elements. Since we translate into a representation with the sampled token separate, we
        // can keep the top n elements constant. In case the sampled token has not been in the top
        // n, the below line will shorten the array by one.
        top_logprobs.resize_with(num_expected_top_logprobs as usize, || {
            unreachable!("Vec should only shorten")
        });
        logprobs.push(Distribution {
            sampled: Logprob {
                token: token.into_bytes(),
                logprob,
            },
            top: top_logprobs,
        });
    }
    logprobs
}

#[derive(Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum DeserializedCompletionEvent {
    StreamChunk {
        /// The completion of the stream.
        completion: String,
        /// Completion with special tokens still included
        raw_completion: Option<String>,
        #[serde(default)]
        log_probs: Vec<HashMap<String, f64>>,
        #[serde(default)]
        completion_tokens: Vec<String>,
    },
    StreamSummary {
        /// The reason why the model stopped generating new tokens.
        finish_reason: String,
    },
    CompletionSummary {
        /// Number of tokens combined across all completion tasks.
        /// In particular, if you set best_of or n to a number larger than 1 then we report the
        /// combined prompt token count for all best_of or n tasks.
        num_tokens_prompt_total: u32,
        /// Number of tokens combined across all completion tasks.
        /// If multiple completions are returned or best_of is set to a value greater than 1 then
        /// this value contains the combined generated token count.
        num_tokens_generated: u32,
    },
}

#[derive(Debug, PartialEq)]
pub enum CompletionEvent {
    StreamChunk {
        /// The completion of the stream.
        completion: String,
        /// Log probabilities of the completion tokens if requested via logprobs parameter in request.
        logprobs: Vec<Distribution>,
    },
    StreamSummary {
        /// The reason why the model stopped generating new tokens.
        finish_reason: String,
    },
    CompletionSummary {
        usage: Usage,
    },
}

impl StreamTask for TaskCompletion<'_> {
    type Output = CompletionEvent;

    type ResponseBody = DeserializedCompletionEvent;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = BodyCompletion::new(model, self).with_streaming();
        client.post(format!("{base}/complete")).json(&body)
    }

    fn body_to_output(&self, response: Self::ResponseBody) -> Option<Self::Output> {
        Some(match response {
            DeserializedCompletionEvent::StreamChunk {
                completion,
                raw_completion,
                log_probs,
                completion_tokens,
            } => CompletionEvent::StreamChunk {
                completion: if self.special_tokens {
                    raw_completion.expect("Missing raw completion")
                } else {
                    completion
                },
                logprobs: completion_logprobs_to_canonical(
                    log_probs,
                    completion_tokens,
                    self.logprobs.top_logprobs().unwrap_or_default(),
                ),
            },
            DeserializedCompletionEvent::StreamSummary { finish_reason } => {
                CompletionEvent::StreamSummary { finish_reason }
            }
            DeserializedCompletionEvent::CompletionSummary {
                num_tokens_prompt_total,
                num_tokens_generated,
            } => CompletionEvent::CompletionSummary {
                usage: Usage {
                    prompt_tokens: num_tokens_prompt_total,
                    completion_tokens: num_tokens_generated,
                },
            },
        })
    }
}

impl Logprobs {
    /// Convert into a number for completion endpoint
    fn to_logprobs_num(self) -> Option<u8> {
        match self {
            Logprobs::No => None,
            Logprobs::Sampled => Some(0),
            Logprobs::Top(n) => Some(n),
        }
    }

    /// Wether or not we want to return the completion tokens
    fn to_tokens(self) -> bool {
        match self {
            Logprobs::No => false,
            Logprobs::Sampled | Logprobs::Top(_) => true,
        }
    }
}
