use aleph_alpha_client::{Client, Prompt, TaskCompletion};
use wiremock::{
    matchers::{body_json_string, header, method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test]
async fn completion_with_luminous_base() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let token = "dummy-token";
    let answer = r#"{"id":"273a3698-876e-49cb-af71-dbe7a249df92","model_version":"2021-12","completions":[{"completion":"\n","finish_reason":"maximum_tokens"}]}"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", format!("Bearer {token}").as_str()))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(200).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text("Hello,"), 
        maximum_tokens: 1
    };

    let model = "luminous-base";

    let client = Client::with_base_uri(mock_server.uri(), token).unwrap();
    let response = client.complete(model, &task).await.unwrap();

    // Then
    eprintln!("{}", response);
    assert_eq!(answer, response)
}

/// If we open too many tasks at once, at one point the API, will tell us that we exceeded our
/// quota. This might change very dynamically, depending on how much compute is available and
/// how many other users are trying to utilize the API.
/// 
/// It should be easy for client code to detect this scenario.
#[tokio::test]
async fn detect_too_many_tasks() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let token = "dummy-token";
    let answer = r#"{"error":"There are already running too many tasks for you","code":"TOO_MANY_TASKS"}"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", format!("Bearer {token}").as_str()))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(429).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = TaskCompletion {
        prompt: Prompt::from_text("Hello,"), 
        maximum_tokens: 1
    };

    let model = "luminous-base";

    let client = Client::with_base_uri(mock_server.uri(), token).unwrap();
    let response = client.complete(model, &task).await.unwrap_err();

    // Then
    eprintln!("{}", response);
}