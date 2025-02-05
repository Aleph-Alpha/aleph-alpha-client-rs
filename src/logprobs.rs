use std::str::{self, Utf8Error};

use serde::Deserialize;

#[derive(Clone, Copy)]
pub enum Logprobs {
    /// Do not return any logprobs
    No,
    /// Return only the logprob of the tokens which have actually been sampled into the completion.
    Sampled,
    /// Request between 0 and 20 tokens
    Top(u8),
}

#[derive(Deserialize, Debug, PartialEq)]
pub struct TopLogprob {
    // The API returns both a UTF-8 String token and bytes as an array of numbers. We only
    // deserialize bytes as it is the better source of truth.
    /// Binary represtantation of the token, usually these bytes are UTF-8.
    #[serde(rename = "bytes")]
    pub token: Vec<u8>,
    pub logprob: f64,
}

impl TopLogprob {
    pub fn token_as_str(&self) -> Result<&str, Utf8Error> {
        str::from_utf8(&self.token)
    }
}
