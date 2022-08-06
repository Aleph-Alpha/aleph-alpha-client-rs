# Aleph Alpha API client for Rust

Interact with large language models provided by the Aleph Alpha API in Rust code.

## Usage

```rust
use aleph_alpha_client::{Client, Authentication};

#[tokio::main]
fn main() {
    // Authenticate against API. Fetches token.
    let client = Client::new("ALEPH_ALPHA_API_TOKEN").await;

    // Name of the model we we want to use. Large models give usually better answer, but are also
    // more costly.
    let model = "luminous-base";

    // The task we want to perform. Here we want to continue the sentence: "The most important thing
    // is ..."
    let task = TaskCompletion {
        prompt: Prompt::from_text("The most important thing is"),
        // The maximum number of tokens within the completion. A token is very roughly something
        // like a word. The bigger this number, the longer the completion **might** be.
        maximum_tokens: 64,
        sampling: Sampling::MOST_LIKELY,
    };
    
    // Send the task to the client.
    let response = client.complete(model, &task).await.unwrap();

    // Print entire sentence with completion
    println!("The most important thing is{}", response.completion);
}
```

**Work in Progress**

Currently the Rust client is not a priority on our Roadmap, so expect this client to be incomplete. If we work on it expect intefaces to break.