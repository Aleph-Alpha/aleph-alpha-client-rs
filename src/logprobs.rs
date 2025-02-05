#[derive(Clone, Copy)]
pub enum Logprobs {
    /// Do not return any logprobs
    No,
    /// Return only the logprob of the tokens which have actually been sampled into the completion.
    Sampled,
    /// Request between 0 and 20 tokens
    Top(u8),
}
