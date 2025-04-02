# Aleph Alpha API client for Rust

Interact with large language models provided by the Aleph Alpha API in Rust code.

## Usage

```rust
use aleph_alpha_client::{Client, TaskCompletion, How, Task};

#[tokio::main]
fn main() {
    // Authenticate against API. Fetches token.
    let client = Client::from_env().unwrap();

    // Name of the model we want to use. Large models give usually better answers, but are also
    // more costly.
    let model = "luminous-base";

    // The task we want to perform. Here we want to continue the sentence: "An apple a day ..."
    let task = TaskCompletion::from_text("An apple a day").with_maximum_tokens(20);

    // Retrieve the answer from the API
    let response = client.completion(&task, model, &How::default()).await.unwrap();

    // Print entire sentence with completion
    println!("An apple a day{}", response.completion);
}
```

This is a **work in progress** currently the Rust client is not a priority on our Roadmap, so expect this client to be
incomplete. If we work on it expect interfaces to break.
