use std::time::Duration;

use aleph_alpha_client::{Client, Error, How, Task, TaskCompletion};
use reqwest::StatusCode;
use wiremock::{
    matchers::{any, body_json_string, header, method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test]
async fn completion_with_luminous_base() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let answer = r#"{"model_version":"2021-12","completions":[{"completion":"\n","finish_reason":"maximum_tokens"}]}"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", "Bearer dummy-token"))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(200).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = TaskCompletion::from_text("Hello,", 1);
    let model = "luminous-base";
    let client = Client::with_base_url(mock_server.uri(), "dummy-token").unwrap();
    let response = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap();
    let actual = response.completion;

    // Then
    assert_eq!("\n", actual)
}

/// If we open too many requests at once, we may trigger rate limiting. We want this scenario to be
/// easily detectible by the user, so he/she/it can start sending requests slower.
#[tokio::test]
async fn detect_rate_limiting() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let answer = r#"Too many requests"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", "Bearer dummy-token"))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(429).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = TaskCompletion::from_text("Hello,", 1);
    let model = "luminous-base";
    let client = Client::with_base_url(mock_server.uri(), "dummy-token").unwrap();
    let error = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap_err();

    // Then
    assert!(matches!(error, Error::TooManyRequests));
}

/// Even if we do not open too many requests at once ourselves, the API may just be busy. We also
/// want this scenario to be easily detectable by users.
#[tokio::test]
async fn detect_queue_full() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    let answer = r#"{
        "error":"Sorry we had to reject your request because we could not guarantee to finish it in
            a reasonable timeframe. This specific model is very busy at this moment. Try again later
            or use another model.",
        "code":"QUEUE_FULL"
    }"#;
    let body = r#"{
        "model": "luminous-base",
        "prompt": [{"type": "text", "data": "Hello,"}],
        "maximum_tokens": 1
    }"#;

    Mock::given(method("POST"))
        .and(path("/complete"))
        .and(header("Authorization", "Bearer dummy-token"))
        .and(header("Content-Type", "application/json"))
        .and(body_json_string(body))
        .respond_with(ResponseTemplate::new(503).set_body_string(answer))
        // Mounting the mock on the mock server - it's now effective!
        .mount(&mock_server)
        .await;

    // When
    let task = TaskCompletion::from_text("Hello,", 1);
    let model = "luminous-base";
    let client = Client::with_base_url(mock_server.uri(), "dummy-token").unwrap();
    let error = client
        .output_of(&task.with_model(model), &How::default())
        .await
        .unwrap_err();

    assert!(matches!(error, Error::Busy));
}

/// Should set `nice=true` in query URL in order to tell the server we do not need our result right
/// now in a high stress situation.
#[tokio::test]
async fn be_nice() {
    // Given

    // Start a background HTTP server on a random local part
    let mock_server = MockServer::start().await;

    // When
    let task = TaskCompletion::from_text("Hello,", 1);
    let model = "luminous-base";
    let client = Client::with_base_url(mock_server.uri(), "dummy-token").unwrap();
    // Drop result, answer is meaningless anyway
    let _ = client
        .output_of(
            &task.with_model(model),
            &How {
                be_nice: true,
                ..Default::default()
            },
        )
        .await;

    // Then
    let last_request = &mock_server.received_requests().await.unwrap()[0];
    assert!(last_request
        .url
        .query_pairs()
        .any(|(k, v)| k == "nice" && v == "true"));
}

#[tokio::test]
async fn client_timeout() {
    // Given
    let mock_server = MockServer::start().await;
    let response_time = Duration::from_millis(20);
    Mock::given(any())
        .respond_with(ResponseTemplate::new(StatusCode::OK).set_delay(response_time))
        .mount(&mock_server)
        .await;
    let client = Client::with_base_url(mock_server.uri(), "dummy-token").unwrap();

    // When
    let result = client
        .output_of(
            &TaskCompletion::from_text("Hello,", 1).with_model("any"),
            &How {
                client_timeout: response_time / 2,
                ..Default::default()
            },
        )
        .await
        .unwrap_err();

    // Then
    assert!(matches!(result, Error::ClientTimeout(..)));
}
