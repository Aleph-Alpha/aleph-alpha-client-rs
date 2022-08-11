use aleph_alpha_client::{Client, Prompt, Sampling, TaskCompletion};
use lazy_static::lazy_static;

lazy_static! {
    static ref AA_API_TOKEN: String = std::env::var("AA_API_TOKEN")
        .expect("AA_API_TOKEN environment variable must be specified to run tests.");
}

#[tokio::test]
async fn completion_with_luminous_base() {
    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text("Hello"),
        maximum_tokens: 1,
        sampling: Sampling::MOST_LIKELY,
    };

    let model = "luminous-base";

    let client = Client::new(&AA_API_TOKEN).unwrap();
    let response = client.complete(model, &task).await.unwrap();

    eprintln!("{}", response.completion);

    // Then
    assert!(!response.completion.is_empty())
}
